"""Matplotlib visualization for a completed or live simulation."""

from dataclasses import dataclass
import math
from typing import Collection, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.text import Annotation, Text
import numpy as np

from core.pure_pursuit import PursuitTarget
from core.track import Track
from core.track_geometry import TrackPosition, track_boundaries
from core.vehicle_geometry import VehicleGeometry, oriented_rectangle_corners
from core.vehicle_state import VehicleState
from sim.candidate_paths import candidate_path_label, candidate_path_points
from strategy.base import CandidateSnapshot


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
    show_candidate_paths: bool = True
    show_candidate_labels: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.pause_seconds) or self.pause_seconds < 0.0:
            raise ValueError("pause_seconds must be non-negative and finite")


@dataclass(frozen=True)
class _CandidateArtists:
    line: Line2D
    label: Annotation


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
    candidate_paths: list[_CandidateArtists]
    candidate_time: Text
    color: tuple[float, float, float, float]


class SimulationVisualizer:
    def __init__(self, track: Track, config: VisualizationConfig) -> None:
        self.config = config
        self.track = track
        if config.realtime:
            plt.ion()
        self.figure, self.axes = plt.subplots(figsize=(10, 7))
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
        self._candidate_snapshots: dict[str, CandidateSnapshot] = {}
        if config.equal_axis:
            self.axes.set_aspect("equal", adjustable="box")
        displayed_track = np.vstack((left_boundary, right_boundary))
        x_min = float(displayed_track[:, 0].min())
        x_max = float(displayed_track[:, 0].max())
        self._label_mid_x = (x_min + x_max) / 2.0
        y_min = float(displayed_track[:, 1].min())
        y_max = float(displayed_track[:, 1].max())
        margin = 0.1 * max(x_max - x_min, y_max - y_min)
        self.axes.set_xlim(x_min - margin, x_max + margin)
        self.axes.set_ylim(y_min - margin, y_max + margin)
        self.axes.set_xlabel("x [m]")
        self.axes.set_ylabel("y [m]")
        self.axes.set_title("Pure Pursuit on an Ellipse")
        self._update_legend()

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
        candidate_snapshots: Mapping[str, CandidateSnapshot] | None = None,
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
        if candidate_snapshots is not None and not set(candidate_snapshots) <= vehicle_ids:
            raise ValueError("candidate_snapshots contain an unknown vehicle ID")
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

            snapshot = (
                None if candidate_snapshots is None else candidate_snapshots.get(vehicle_id)
            )
            self._update_candidate_artists(vehicle_id, artists, snapshot, state)

        if self.config.realtime:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(self.config.pause_seconds)

    def _create_vehicle_artists(self, vehicle_id: str) -> _VehicleArtists:
        color = plt.get_cmap("tab10")(len(self._vehicle_artists) % 10)
        (trajectory,) = self.axes.plot(
            [], [], color=color, label=vehicle_id, linewidth=1.0, alpha=0.45,
        )
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
            linestyle="-",
            linewidth=2.5,
            color=color,
            visible=False,
            zorder=4,
        )
        candidate_time = self.axes.text(
            0.0, 0.0, "", color=color, fontsize=7,
            horizontalalignment="center", visible=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        self._update_legend()
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
            [],
            candidate_time,
            color,
        )

    def _update_candidate_artists(
        self,
        vehicle_id: str,
        artists: _VehicleArtists,
        snapshot: CandidateSnapshot | None,
        state: VehicleState,
    ) -> None:
        visible = self.config.show_candidate_paths and snapshot is not None
        candidates = snapshot.candidates if visible else ()
        while len(artists.candidate_paths) < len(candidates):
            (line,) = self.axes.plot([], [], visible=False, zorder=3)
            label = self.axes.annotate(
                "", xy=(0.0, 0.0), xytext=(8, 0), textcoords="offset points",
                fontsize=7, verticalalignment="center", visible=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                arrowprops={"arrowstyle": "-", "linewidth": 0.5, "alpha": 0.4},
                zorder=8,
            )
            artists.candidate_paths.append(_CandidateArtists(line, label))

        # 同一份快照沿用世界座標與圖元，不以新車位重畫舊評估結果。
        if visible and self._candidate_snapshots.get(vehicle_id) is not snapshot:
            self._draw_candidate_snapshot(artists, snapshot)
            self._candidate_snapshots[vehicle_id] = snapshot
        if snapshot is None:
            self._candidate_snapshots.pop(vehicle_id, None)
        for index, candidate_artists in enumerate(artists.candidate_paths):
            active = index < len(candidates)
            candidate_artists.line.set_visible(active)
            candidate_artists.label.set_visible(active and self.config.show_candidate_labels)
        artists.candidate_time.set_visible(
            bool(candidates) and self.config.show_candidate_labels,
        )
        if candidates:
            artists.candidate_time.set_position((state.x, state.y - 1.0))
            artists.candidate_time.set_text(f"candidates @ {snapshot.evaluated_at:.2f}s")

    def _draw_candidate_snapshot(
        self, artists: _VehicleArtists, snapshot: CandidateSnapshot,
    ) -> None:
        points = [
            candidate_path_points(snapshot, candidate, self.track)
            for candidate in snapshot.candidates
        ]
        label_order = sorted(range(len(points)), key=lambda index: points[index][-1, 1])
        for rank, index in enumerate(label_order):
            candidate = snapshot.candidates[index]
            candidate_artists = artists.candidate_paths[index]
            color = artists.color if candidate.feasible else "0.5"
            line = candidate_artists.line
            line.set_data(points[index][:, 0], points[index][:, 1])
            line.set_color(color)
            line.set_linestyle(":" if not candidate.feasible else "-" if candidate.selected else "--")
            line.set_linewidth(2.0 if candidate.selected else 1.0)
            line.set_alpha(0.75 if candidate.selected else 0.4)
            label = candidate_artists.label
            label.xy = tuple(points[index][-1])
            label.set_text(candidate_path_label(candidate))
            label.set_color(color)
            left = label.xy[0] > self._label_mid_x
            label.set_horizontalalignment("right" if left else "left")
            label.set_position((-8 if left else 8, 13 * (rank - (len(points) - 1) / 2)))

    def _update_legend(self) -> None:
        handles, _ = self.axes.get_legend_handles_labels()
        if self.config.show_candidate_paths:
            handles.extend((
                Line2D([], [], color="0.4", linestyle="--", label="candidate snapshot"),
                Line2D([], [], color="0.4", linewidth=2.5, label="selected / tracking"),
                Line2D([], [], color="0.5", linestyle=":", label="rejected candidate"),
            ))
        self.axes.legend(handles=handles, fontsize=8, loc="upper right")

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
