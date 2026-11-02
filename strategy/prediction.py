"""Small geometric predictions shared by overtaking and defending."""

import math

from core.track import Track
from core.track_geometry import track_length, track_to_world
from core.vehicle_state import VehicleState
from strategy.observations import VehicleObservation
from strategy.offset_path import (
    lateral_offset_at_distance,
    wrapped_forward_distance,
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
