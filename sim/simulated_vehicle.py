"""Per-vehicle runtime state for an independent simulation participant."""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

from core.pure_pursuit import PurePursuitController, PursuitTarget
from core.track import Track
from core.vehicle_geometry import VehicleGeometry, rectangles_overlap
from core.vehicle_model import KinematicBicycleModel
from core.vehicle_state import ControlCommand, VehicleState


@dataclass
class SimulatedVehicle:
    """Own one vehicle's model, controller, state, target, and history."""

    vehicle_id: str
    state: VehicleState
    model: KinematicBicycleModel
    controller: PurePursuitController
    geometry: VehicleGeometry = field(
        default_factory=lambda: VehicleGeometry(
            length=0.5,
            width=0.3,
            rear_axle_to_center=0.1,
        )
    )
    trajectory: list[VehicleState] = field(default_factory=list, init=False)
    current_target: PursuitTarget | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.vehicle_id, str) or not self.vehicle_id.strip():
            raise ValueError("vehicle_id must be a non-empty string")
        self.trajectory.append(self.state)

    def step(self, track: Track) -> tuple[ControlCommand, PursuitTarget]:
        """Advance this vehicle by one independent control and model step."""
        command, target = self.controller.compute_command(
            self.state,
            track,
            self.model.config.wheelbase,
        )
        self.state = self.model.step(self.state, command)
        self.trajectory.append(self.state)
        self.current_target = target
        return command, target


def validate_unique_vehicle_ids(vehicles: Iterable[SimulatedVehicle]) -> None:
    """Reject duplicate identifiers in a vehicle collection."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for vehicle in vehicles:
        if vehicle.vehicle_id in seen:
            duplicates.add(vehicle.vehicle_id)
        seen.add(vehicle.vehicle_id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate vehicle IDs: {duplicate_list}")


def find_vehicle_collisions(
    vehicles: Iterable[SimulatedVehicle],
) -> set[tuple[str, str]]:
    """Return overlapping ID pairs without changing any vehicle behavior."""
    vehicle_list = list(vehicles)
    validate_unique_vehicle_ids(vehicle_list)
    collisions: set[tuple[str, str]] = set()
    for first, second in combinations(vehicle_list, 2):
        if rectangles_overlap(
            first.state,
            first.geometry,
            second.state,
            second.geometry,
        ):
            collisions.add(tuple(sorted((first.vehicle_id, second.vehicle_id))))
    return collisions
