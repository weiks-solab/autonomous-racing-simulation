"""Simulator-independent vehicle, track, and controller components."""

from core.pure_pursuit import PurePursuitConfig, PurePursuitController, PursuitTarget
from core.track import Track
from core.track_geometry import (
    RelativeTrackPosition,
    TrackBoundaryStatus,
    TrackPosition,
    classify_reference_point_boundary,
    relative_track_position,
    signed_longitudinal_gap,
    track_boundaries,
    track_length,
    track_to_world,
    world_to_track,
)
from core.vehicle_geometry import (
    VehicleGeometry,
    oriented_rectangle_corners,
    rectangles_overlap,
)
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import ControlCommand, VehicleState

__all__ = [
    "ControlCommand",
    "KinematicBicycleModel",
    "PurePursuitConfig",
    "PurePursuitController",
    "PursuitTarget",
    "RelativeTrackPosition",
    "Track",
    "TrackBoundaryStatus",
    "TrackPosition",
    "VehicleGeometry",
    "VehicleModelConfig",
    "VehicleState",
    "classify_reference_point_boundary",
    "oriented_rectangle_corners",
    "rectangles_overlap",
    "relative_track_position",
    "signed_longitudinal_gap",
    "track_boundaries",
    "track_length",
    "track_to_world",
    "world_to_track",
]
