"""Matplotlib visualization for a completed or live simulation."""

from dataclasses import dataclass
import math
from typing import Collection, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.text import Text
import numpy as np

from core.pure_pursuit import PursuitTarget
from core.track import Track
from core.track_geometry import TrackPosition, track_boundaries
from core.vehicle_geometry import VehicleGeometry, oriented_rectangle_corners
from core.vehicle_state import VehicleState


@dataclass(frozen=True)
class VisualizationConfig:
    show_vehicle_heading: bool = True
    equal_axis: bool = True
    realtime: bool = False
    pause_seconds: float = 0.001
    show_centerline: bool = True
    show_track_boundaries: bool = True
    show_vehicle_footprints: bool = True
    show_track_positions: bool = False
    show_vehicle_markers: bool = True
    show_trajectories: bool = True
    show_pursuit_targets: bool = True
    show_collision_status: bool = True
    show_strategy_status: bool = True
    show_temporary_paths: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.pause_seconds) or self.pause_seconds < 0.0:
            raise ValueError("pause_seconds must be non-negative and finite")


@dataclass(frozen=True)
class _VehicleArtists:
    trajectory: Line2D
    marker: Line2D
    target: Line2D
    heading: Line2D
    footprint: Polygon
    track_position_text: Text
    collision_text: Text
    strategy_text: Text
    temporary_path: Line2D
    color: tuple[float, float, float, float]


class SimulationVisualizer:
    def __init__(self, track: Track, config: VisualizationConfig) -> None:
        self.config = config
        if config.realtime:
            plt.ion()
        self.figure, self.axes = plt.subplots()
        if config.show_centerline:
            self._plot_closed_line(
                track.centerline,
                linestyle="--",
                color="0.5",
                label="centerline",
                closed=track.closed,
            )
        left_boundary, right_boundary = track_boundaries(track)
        if config.show_track_boundaries:
            self._plot_closed_line(
                left_boundary,
                color="black",
                linewidth=1.0,
                label="left boundary",
                closed=track.closed,
            )
            self._plot_closed_line(
                right_boundary,
                color="black",
                linewidth=1.0,
                label="right boundary",
                closed=track.closed,
            )
        self._vehicle_artists: dict[str, _VehicleArtists] = {}
        if config.equal_axis:
            self.axes.set_aspect("equal", adjustable="box")
        displayed_track = np.vstack((left_boundary, right_boundary))
        x_min = float(displayed_track[:, 0].min())
        x_max = float(displayed_track[:, 0].max())
        y_min = float(displayed_track[:, 1].min())
        y_max = float(displayed_track[:, 1].max())
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
        geometries: Mapping[str, VehicleGeometry] | None = None,
        track_positions: Mapping[str, TrackPosition] | None = None,
        collision_pairs: Collection[tuple[str, str]] | None = None,
        strategy_labels: Mapping[str, str] | None = None,
        temporary_paths: Mapping[str, Track] | None = None,
    ) -> None:
        """Update reusable artists for an arbitrary vehicle collection."""
        vehicle_ids = set(states)
        if set(trajectories) != vehicle_ids or set(targets) != vehicle_ids:
            raise ValueError("states, trajectories, and targets must have matching IDs")
        if geometries is not None and set(geometries) != vehicle_ids:
            raise ValueError("geometries must have the same vehicle IDs as states")
        if track_positions is not None and set(track_positions) != vehicle_ids:
            raise ValueError("track_positions must have the same vehicle IDs as states")
        if strategy_labels is not None and not set(strategy_labels) <= vehicle_ids:
            raise ValueError("strategy_labels contain an unknown vehicle ID")
        if temporary_paths is not None and not set(temporary_paths) <= vehicle_ids:
            raise ValueError("temporary_paths contain an unknown vehicle ID")
        colliding_ids: set[str] = set()
        for first_id, second_id in collision_pairs or ():
            if first_id not in vehicle_ids or second_id not in vehicle_ids:
                raise ValueError("collision_pairs contain an unknown vehicle ID")
            colliding_ids.update((first_id, second_id))

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
            artists.trajectory.set_visible(self.config.show_trajectories)
            artists.marker.set_data([state.x], [state.y])
            artists.marker.set_visible(self.config.show_vehicle_markers)
            target = targets[vehicle_id]
            if target is None:
                artists.target.set_data([], [])
            else:
                artists.target.set_data([target.x], [target.y])
            artists.target.set_visible(
                self.config.show_pursuit_targets and target is not None
            )
            if self.config.show_vehicle_heading:
                heading_length = 1.0
                artists.heading.set_data(
                    [state.x, state.x + heading_length * math.cos(state.yaw)],
                    [state.y, state.y + heading_length * math.sin(state.yaw)],
                )
                artists.heading.set_visible(True)
            else:
                artists.heading.set_data([], [])
                artists.heading.set_visible(False)

            geometry = None if geometries is None else geometries[vehicle_id]
            if self.config.show_vehicle_footprints and geometry is not None:
                artists.footprint.set_xy(
                    oriented_rectangle_corners(state, geometry)
                )
                artists.footprint.set_visible(True)
            else:
                artists.footprint.set_visible(False)

            position = (
                None if track_positions is None else track_positions[vehicle_id]
            )
            if self.config.show_track_positions and position is not None:
                artists.track_position_text.set_position((state.x, state.y))
                artists.track_position_text.set_text(
                    f"{vehicle_id}\ns={position.s:.1f} d={position.d:+.2f}"
                )
                artists.track_position_text.set_visible(True)
            else:
                artists.track_position_text.set_visible(False)

            is_colliding = vehicle_id in colliding_ids
            if is_colliding:
                artists.footprint.set_facecolor("red")
                artists.footprint.set_edgecolor("red")
                artists.footprint.set_alpha(0.65)
            else:
                artists.footprint.set_facecolor(artists.color)
                artists.footprint.set_edgecolor(artists.color)
                artists.footprint.set_alpha(0.25)

            if self.config.show_collision_status and is_colliding:
                artists.collision_text.set_position((state.x, state.y + 0.45))
                artists.collision_text.set_text("COLLISION")
                artists.collision_text.set_visible(True)
            else:
                artists.collision_text.set_visible(False)

            strategy_label = (
                None if strategy_labels is None else strategy_labels.get(vehicle_id)
            )
            if self.config.show_strategy_status and strategy_label is not None:
                artists.strategy_text.set_position((state.x, state.y + 0.9))
                artists.strategy_text.set_text(strategy_label)
                artists.strategy_text.set_visible(True)
            else:
                artists.strategy_text.set_visible(False)

            temporary_path = (
                None if temporary_paths is None else temporary_paths.get(vehicle_id)
            )
            if self.config.show_temporary_paths and temporary_path is not None:
                artists.temporary_path.set_data(
                    temporary_path.centerline[:, 0],
                    temporary_path.centerline[:, 1],
                )
                artists.temporary_path.set_visible(True)
            else:
                artists.temporary_path.set_data([], [])
                artists.temporary_path.set_visible(False)

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
        footprint = Polygon(
            np.empty((0, 2)),
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.25,
        )
        self.axes.add_patch(footprint)
        track_position_text = self.axes.text(
            0.0,
            0.0,
            "",
            color=color,
            fontsize=8,
            visible=False,
        )
        collision_text = self.axes.text(
            0.0,
            0.0,
            "",
            color="red",
            fontsize=10,
            fontweight="bold",
            horizontalalignment="center",
            visible=False,
            zorder=10,
        )
        strategy_text = self.axes.text(
            0.0,
            0.0,
            "",
            color=color,
            fontsize=8,
            horizontalalignment="center",
            visible=False,
            zorder=9,
        )
        (temporary_path,) = self.axes.plot(
            [],
            [],
            linestyle=":",
            linewidth=1.5,
            color=color,
            visible=False,
        )
        self.axes.legend()
        return _VehicleArtists(
            trajectory,
            marker,
            target,
            heading,
            footprint,
            track_position_text,
            collision_text,
            strategy_text,
            temporary_path,
            color,
        )

    def _plot_closed_line(
        self,
        points: np.ndarray,
        *,
        closed: bool,
        **plot_options: object,
    ) -> None:
        plotted_points = (
            np.vstack((points, points[0])) if closed else points
        )
        self.axes.plot(
            plotted_points[:, 0],
            plotted_points[:, 1],
            **plot_options,
        )

    def show(self) -> None:
        self.figure.tight_layout()
        if self.config.realtime:
            plt.ioff()
        plt.show()
