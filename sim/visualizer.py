"""Matplotlib visualization for a completed or live simulation."""

from dataclasses import dataclass
import math
from typing import Sequence

import matplotlib.pyplot as plt

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
        (self.trajectory_line,) = self.axes.plot(
            [], [], color="tab:blue", label="trajectory"
        )
        (self.vehicle_marker,) = self.axes.plot(
            [], [], "o", color="tab:red", label="vehicle"
        )
        (self.target_marker,) = self.axes.plot(
            [], [], "x", color="tab:green", label="target"
        )
        (self.heading_artist,) = self.axes.plot(
            [], [], color="tab:red", linewidth=2.0, label="heading"
        )
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
        self.trajectory_line.set_data(
            [sample.x for sample in trajectory], [sample.y for sample in trajectory]
        )
        self.vehicle_marker.set_data([state.x], [state.y])
        if target is not None:
            self.target_marker.set_data([target.x], [target.y])
        else:
            self.target_marker.set_data([], [])
        if self.config.show_vehicle_heading:
            heading_length = 1.0
            self.heading_artist.set_data(
                [state.x, state.x + heading_length * math.cos(state.yaw)],
                [state.y, state.y + heading_length * math.sin(state.yaw)],
            )
        else:
            self.heading_artist.set_data([], [])

        if self.config.realtime:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(self.config.pause_seconds)

    def show(self) -> None:
        self.figure.tight_layout()
        if self.config.realtime:
            plt.ioff()
        plt.show()
