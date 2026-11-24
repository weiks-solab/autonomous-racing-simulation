"""Matplotlib visualization for a completed or live simulation."""

from dataclasses import dataclass, replace
from itertools import combinations
import math
from typing import Collection, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.text import Annotation, Text
import numpy as np

from core.pure_pursuit import PursuitTarget
from core.track import Track
from core.track_geometry import TrackPosition, track_boundaries
from core.vehicle_geometry import VehicleGeometry, oriented_rectangle_corners, rectangles_overlap
from core.vehicle_state import VehicleState
from sim.candidate_paths import candidate_path_label, candidate_path_points
from sim.map_background import MapBackground
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
    show_gp_predictions: bool = True
    show_gp_prediction_labels: bool = True
    show_map_background: bool = True
    show_envelope_overlap: bool = True

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
    def __init__(
        self, track: Track, config: VisualizationConfig,
        background: MapBackground | None = None,
    ) -> None:
        self.config = config
        self.track = track
        if config.realtime:
            plt.ion()
        self.figure, self.axes = plt.subplots(figsize=(10, 7))
        self.map_artist = (
            background.draw(self.axes)
            if background is not None and config.show_map_background else None
        )
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
        # 以「觀測者、對手」區分，多台進攻車的預測不互相覆蓋。
        self._gp_prediction_artists: dict[tuple[str, str], _CandidateArtists] = {}
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
        self.axes.set_title("Racing Overtake Simulator")
        self._update_legend()
        self._background = None
        # 線條仍依原 zorder 重畫，避免車身遮住本來位於上層的賽道邊界。
        self._static_artists = set(self.axes.get_children()) - set(self.axes.lines)
        self._use_blit = config.realtime and self.figure.canvas.supports_blit
        if self._use_blit:
            self.figure.canvas.mpl_connect("draw_event", self._cache_background)
            self.figure.canvas.mpl_connect("resize_event", self._invalidate_background)
            self.axes.callbacks.connect("xlim_changed", self._invalidate_background)
            self.axes.callbacks.connect("ylim_changed", self._invalidate_background)

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

        # 畫面另查實體車身；不改規劃器傳入的碰撞判定結果。
        body_colliding_ids: set[str] = set()
        if geometries is not None:
            bodies = {key: replace(geometry, collision=None) for key, geometry in geometries.items()}
            for first_id, second_id in combinations(states, 2):
                if rectangles_overlap(states[first_id], bodies[first_id], states[second_id], bodies[second_id]):
                    body_colliding_ids.update((first_id, second_id))

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

            is_colliding = vehicle_id in body_colliding_ids
            envelope_overlap = self.config.show_envelope_overlap and vehicle_id in colliding_ids
            color = "red" if is_colliding else "darkorange" if envelope_overlap else artists.color
            artists.footprint.set_facecolor(color)
            artists.footprint.set_edgecolor(color)
            artists.footprint.set_alpha(0.65 if is_colliding or envelope_overlap else 0.25)

            if self.config.show_collision_status and (is_colliding or envelope_overlap):
                artists.collision_text.set_position((state.x, state.y + 0.45))
                artists.collision_text.set_text("COLLISION" if is_colliding else "ENVELOPE OVERLAP")
                artists.collision_text.set_color(color)
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

        self._update_gp_predictions(candidate_snapshots or {})
        if self.config.realtime:
            if self._use_blit:
                # 新增的車輛、候選與圖例只重畫自身，PNG 與座標軸沿用快取。
                for artist in self._dynamic_artists():
                    artist.set_animated(True)
                if self._background is None:
                    self.figure.canvas.draw()
                else:
                    self.figure.canvas.restore_region(self._background)
                    self._draw_dynamic_artists()
                self.figure.canvas.blit(self.figure.bbox)
            else:
                self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            if self._use_blit:
                # pause() 會再次 show，TkAgg 因而要求完整重畫；此處只處理事件。
                if self.config.pause_seconds > 0.0:
                    self.figure.canvas.start_event_loop(self.config.pause_seconds)
            else:
                plt.pause(self.config.pause_seconds)

    def wait(self, seconds: float) -> None:
        """播放時等到下一幀；等待期間仍處理視窗事件，視窗不會沒有回應。"""
        if not math.isfinite(seconds) or seconds <= 0.0:
            # TkAgg 的 start_event_loop(0) 會一直等下去，所以不足一幀時直接返回。
            return
        if self._use_blit:
            # 與 update_vehicles 相同，避開 plt.pause 觸發的完整重畫。
            self.figure.canvas.start_event_loop(seconds)
        else:
            plt.pause(seconds)

    def _dynamic_artists(self) -> list[Artist]:
        return sorted(
            (artist for artist in self.axes.get_children() if artist not in self._static_artists),
            key=lambda artist: artist.get_zorder(),
        )

    def _draw_dynamic_artists(self) -> None:
        for artist in self._dynamic_artists():
            if artist.get_visible():
                self.axes.draw_artist(artist)

    def _invalidate_background(self, _event: object = None) -> None:
        self._background = None

    def _cache_background(self, _event: object) -> None:
        # 存圖使用完整畫面；縮放後則以新尺寸重新建立背景。
        self._background = None
        if self._use_blit and not self.figure.canvas.is_saving():
            self._background = self.figure.canvas.copy_from_bbox(self.figure.bbox)
            self._draw_dynamic_artists()

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

    def _update_gp_predictions(self, snapshots: Mapping[str, CandidateSnapshot]) -> None:
        active = set()
        for observer_id, snapshot in snapshots.items():
            for prediction in snapshot.gp_predictions:
                if not self.config.show_gp_predictions or not prediction.states:
                    continue
                key = (observer_id, prediction.vehicle_id)
                active.add(key)
                if key not in self._gp_prediction_artists:
                    (line,) = self.axes.plot(
                        [], [], color="tab:purple", linestyle="-.", linewidth=2.5,
                        marker="o", markersize=5, markevery=[-1], zorder=6,
                    )
                    label = self.axes.annotate(
                        "", xy=(0, 0), xytext=(8, 8), textcoords="offset points",
                        color="tab:purple", fontsize=8, zorder=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                    )
                    self._gp_prediction_artists[key] = _CandidateArtists(line, label)
                artists = self._gp_prediction_artists[key]
                artists.line.set_data([state.x for state in prediction.states],
                                      [state.y for state in prediction.states])
                artists.line.set_visible(True)
                endpoint = prediction.states[-1]
                artists.label.xy = (endpoint.x, endpoint.y)
                # 靠右時把文字放到左邊，讓標籤留在畫面內。
                on_right = endpoint.x > self._label_mid_x
                artists.label.set_position((-8 if on_right else 8, 8))
                artists.label.set_horizontalalignment("right" if on_right else "left")
                text = (f"GP {observer_id} -> {prediction.vehicle_id}\n"
                        f"@ {snapshot.evaluated_at:.2f}s, +{prediction.times[-1]:.2f}s")
                if prediction.delta_d_mean is not None and prediction.delta_d_std is not None:
                    text += (f"\nΔd(+{prediction.horizon_seconds:g}s)={prediction.delta_d_mean:+.2f}"
                             f"±{prediction.delta_d_std:.2f} m")
                artists.label.set_text(text)
                artists.label.set_visible(self.config.show_gp_prediction_labels)
        # 沒有新快照或停用 GP 時，隱藏舊預測；圖元保留供下次重用。
        for key, artists in self._gp_prediction_artists.items():
            if key not in active:
                artists.line.set_visible(False)
                artists.label.set_visible(False)

    def _update_legend(self) -> None:
        self._invalidate_background()
        handles, _ = self.axes.get_legend_handles_labels()
        if self.config.show_candidate_paths:
            handles.extend((
                Line2D([], [], color="0.4", linestyle="--", label="candidate snapshot"),
                Line2D([], [], color="0.4", linewidth=2.5, label="selected / tracking"),
                Line2D([], [], color="0.5", linestyle=":", label="rejected candidate"),
            ))
        if self.config.show_gp_predictions:
            handles.append(Line2D([], [], color="tab:purple", linestyle="-.", linewidth=2.5,
                                  marker="o", label="GP opponent prediction (end: circle)"))
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
        self._use_blit = False
        for artist in self._dynamic_artists():
            artist.set_animated(False)
        self.figure.tight_layout()
        if self.config.realtime:
            plt.ioff()
        plt.show()
