"""Simulator-independent input values for strategy algorithms."""

from dataclasses import dataclass

from core.track import Track
from core.track_geometry import TrackPosition, world_to_track
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.offset_path import LateralManeuver


@dataclass(frozen=True)
class VehicleObservation:
    vehicle_id: str
    state: VehicleState
    track_position: TrackPosition
    geometry: VehicleGeometry
    maneuver: LateralManeuver | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_id, str) or not self.vehicle_id.strip():
            raise ValueError("vehicle_id must be a non-empty string")


def create_vehicle_observation(
    vehicle_id: str,
    state: VehicleState,
    geometry: VehicleGeometry,
    track: Track,
    maneuver: LateralManeuver | None = None,
) -> VehicleObservation:
    """Build an observation without exposing simulator runtime objects."""
    return VehicleObservation(
        vehicle_id=vehicle_id,
        state=state,
        track_position=world_to_track(state, track),
        geometry=geometry,
        maneuver=maneuver,
    )
