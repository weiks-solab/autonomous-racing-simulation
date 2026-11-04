"""Small geometric predictions shared by overtaking and defending."""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from core.track import Track
from core.track_geometry import track_length, track_to_world, track_to_world_points
from core.vehicle_state import VehicleState
from strategy.observations import VehicleObservation
from strategy.offset_path import (
    lateral_offset_at_distance,
    wrapped_forward_distance,
)


@dataclass(frozen=True)
class PathPrediction:
    """依行駛距離取樣的幾何軌跡；progress 是不繞回的賽道縱向進度。"""

    progress: NDArray[np.float64]
    offsets: NDArray[np.float64]
    states: tuple[VehicleState, ...]


def predict_path_trajectory(
    vehicle: VehicleObservation,
    track: Track,
    times: Sequence[float],
    sample_spacing: float = 0.2,
) -> PathPrediction:
    """沿候選折線走滿 v*t，再插值取得位置與 s；不執行 bicycle rollout。"""
    times_array = np.asarray(times, dtype=float)
    if (times_array.ndim != 1 or times_array.size == 0
            or not np.all(np.isfinite(times_array)) or times_array[0] != 0.0
            or np.any(np.diff(times_array) < 0.0)):
        raise ValueError("times must be finite, nondecreasing and start at zero")
    if not track.closed:
        raise ValueError("path prediction requires a closed track")
    if not math.isfinite(sample_spacing) or sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be positive and finite")
    travel = max(vehicle.state.velocity, 0.0) * times_array
    if travel[-1] == 0.0:
        return PathPrediction(
            np.zeros_like(travel), np.full_like(travel, vehicle.track_position.d),
            (vehicle.state,) * len(travel),
        )

    maneuver = vehicle.maneuver
    origin = 0.0 if maneuver is None else wrapped_forward_distance(
        maneuver.start_s, vehicle.track_position.s, track_length(track),
    )
    span = float(travel[-1])
    while True:
        progress = np.arange(math.ceil(span / sample_spacing) + 2) * sample_spacing
        offsets = np.full_like(progress, vehicle.track_position.d)
        if maneuver is not None:
            fraction = np.clip((origin + progress) / maneuver.transition_length, 0.0, 1.0)
            offsets = maneuver.start_d + (maneuver.target_d - maneuver.start_d) * (
                3.0 * fraction**2 - 2.0 * fraction**3
            )
        offsets[0] = vehicle.track_position.d
        points = track_to_world_points(vehicle.track_position.s + progress, offsets, track)
        points[0] = (vehicle.state.x, vehicle.state.y)
        segments = np.diff(points, axis=0)
        lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(segments, axis=1))))
        if lengths[-1] > travel[-1]:
            break
        if lengths[-1] <= 1e-9:
            raise ValueError("candidate path has no usable length")
        # 內側路徑較短時補取樣，直到能走滿指定距離。
        span *= 2.0

    positions = np.column_stack([
        np.interp(travel, lengths, points[:, axis]) for axis in (0, 1)
    ])
    indices = np.clip(np.searchsorted(lengths, travel, side="right") - 1, 0, len(segments) - 1)
    headings = np.arctan2(segments[indices, 1], segments[indices, 0])
    states = tuple(
        vehicle.state if time == 0.0 else VehicleState(
            float(point[0]), float(point[1]), float(yaw), vehicle.state.velocity,
        )
        for time, point, yaw in zip(times_array, positions, headings)
    )
    return PathPrediction(
        np.interp(travel, lengths, progress), np.interp(travel, lengths, offsets), states,
    )


def predicted_offset(
    vehicle: VehicleObservation,
    distance: float,
    track: Track,
) -> float:
    """Follow the current maneuver, or keep d when there is no maneuver."""
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance must be non-negative and finite")
    maneuver = vehicle.maneuver
    if maneuver is None:
        return vehicle.track_position.d
    progress = wrapped_forward_distance(
        maneuver.start_s, vehicle.track_position.s, track_length(track)
    )
    return lateral_offset_at_distance(
        progress + distance,
        maneuver.start_d,
        maneuver.target_d,
        maneuver.transition_length,
    )


def predict_vehicle_state(
    vehicle: VehicleObservation,
    time: float,
    track: Track,
) -> VehicleState:
    """Keep speed fixed and follow the supplied high-level path.

    This is a geometric approximation, not a bicycle-model rollout. The
    actual state is used at time zero so existing overlaps are not hidden.
    """
    if not math.isfinite(time) or time < 0.0:
        raise ValueError("time must be non-negative and finite")
    if time == 0.0:
        return vehicle.state
    distance = max(vehicle.state.velocity, 0.0) * time
    s = vehicle.track_position.s + distance
    d = predicted_offset(vehicle, distance, track)
    d_ahead = predicted_offset(vehicle, distance + 0.05, track)
    point = track_to_world(s, d, track)
    point_ahead = track_to_world(s + 0.05, d_ahead, track)
    yaw = math.atan2(point_ahead[1] - point[1], point_ahead[0] - point[0])
    return VehicleState(
        float(point[0]), float(point[1]), yaw, vehicle.state.velocity
    )
