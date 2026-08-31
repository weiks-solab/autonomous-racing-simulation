"""Matplotlib visualization for a completed or live simulation."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core.pure_pursuit import PursuitTarget
from core.track import Track
from core.vehicle_state import VehicleState


@dataclass(frozen=True)
class VisualizationConfig:
    show_vehicle_heading: bool = True
    equal_axis: bool = True
    realtime: bool = False
    pause_seconds: float = 0.001

    def __post_init__(self) -> None:
        if not math.isfinite(self.pause_seconds) or self.pause_seconds < 0.0:
            raise ValueError("pause_seconds must be non-negative and finite")


@dataclass(frozen=True)
class _VehicleArtists:
    trajectory: Line2D
    marker: Line2D
    target: Line2D
    heading: Line2D


class SimulationVisualizer:
    def __init__(self, track: Track, config: VisualizationConfig) -> None:
        self.config = config
        if config.realtime:
            plt.ion()
        self.figure, self.axes = plt.subplots()
        self.axes.plot(
            track.centerline[:, 0],
            track.centerline[:, 1],
            "--",
            color="0.5",
            label="centerline",
        )
        closing_x = (track.centerline[-1, 0], track.centerline[0, 0])
        closing_y = (track.centerline[-1, 1], track.centerline[0, 1])
        if track.closed:
            self.axes.plot(closing_x, closing_y, "--", color="0.5")
        self._vehicle_artists: dict[str, _VehicleArtists] = {}
        if config.equal_axis:
            self.axes.set_aspect("equal", adjustable="box")
        x_min = float(track.centerline[:, 0].min())
        x_max = float(track.centerline[:, 0].max())
        y_min = float(track.centerline[:, 1].min())
        y_max = float(track.centerline[:, 1].max())
        margin = 0.1 * max(x_max - x_min, y_max - y_min)
        self.axes.set_xlim(x_min - margin, x_max + margin)
        self.axes.set_ylim(y_min - margin, y_max + margin)
        self.axes.set_xlabel("x [m]")
        self.axes.set_ylabel("y [m]")
        self.axes.set_title("Pure Pursuit on an Ellipse")
        self.axes.legend()

    def update(
        self,
        state: VehicleState,
        trajectory: Sequence[VehicleState],
        target: PursuitTarget | None = None,
    ) -> None:
        """Preserve the Milestone 1 single-vehicle update API."""
        self.update_vehicles(
            states={"vehicle": state},
            trajectories={"vehicle": trajectory},
            targets={"vehicle": target},
        )

    def update_vehicles(
        self,
        states: Mapping[str, VehicleState],
        trajectories: Mapping[str, Sequence[VehicleState]],
        targets: Mapping[str, PursuitTarget | None],
    ) -> None:
        """Update reusable artists for an arbitrary vehicle collection."""
        vehicle_ids = set(states)
        if set(trajectories) != vehicle_ids or set(targets) != vehicle_ids:
            raise ValueError("states, trajectories, and targets must have matching IDs")

        for vehicle_id, state in states.items():
            artists = self._vehicle_artists.get(vehicle_id)
            if artists is None:
                artists = self._create_vehicle_artists(vehicle_id)
                self._vehicle_artists[vehicle_id] = artists

            trajectory = trajectories[vehicle_id]
            artists.trajectory.set_data(
                [sample.x for sample in trajectory],
                [sample.y for sample in trajectory],
            )
            artists.marker.set_data([state.x], [state.y])
            target = targets[vehicle_id]
            if target is None:
                artists.target.set_data([], [])
            else:
                artists.target.set_data([target.x], [target.y])
            if self.config.show_vehicle_heading:
                heading_length = 1.0
                artists.heading.set_data(
                    [state.x, state.x + heading_length * math.cos(state.yaw)],
                    [state.y, state.y + heading_length * math.sin(state.yaw)],
                )
            else:
                artists.heading.set_data([], [])

        if self.config.realtime:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(self.config.pause_seconds)

    def _create_vehicle_artists(self, vehicle_id: str) -> _VehicleArtists:
        color = plt.get_cmap("tab10")(len(self._vehicle_artists) % 10)
        (trajectory,) = self.axes.plot([], [], color=color, label=vehicle_id)
        (marker,) = self.axes.plot([], [], "o", color=color)
        (target,) = self.axes.plot([], [], "x", color=color)
        (heading,) = self.axes.plot([], [], color=color, linewidth=2.0)
        self.axes.legend()
        return _VehicleArtists(trajectory, marker, target, heading)

    def show(self) -> None:
        self.figure.tight_layout()
        if self.config.realtime:
            plt.ioff()
        plt.show()
