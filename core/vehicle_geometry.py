"""Oriented rectangular vehicle-body geometry and overlap detection."""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from core.vehicle_state import VehicleState


@dataclass(frozen=True)
class CollisionConfig:
    """Collision rectangle dimensions, centred on the physical body."""

    length: float
    width: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0.0 for value in (self.length, self.width)):
            raise ValueError("collision length and width must be positive and finite")


@dataclass(frozen=True)
class VehicleGeometry:
    """Vehicle body dimensions relative to the rear-axle state reference."""

    length: float
    width: float
    rear_axle_to_center: float
    collision: CollisionConfig | None = None

    def __post_init__(self) -> None:
        values = (self.length, self.width, self.rear_axle_to_center)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vehicle geometry values must be finite")
        if self.length <= 0.0:
            raise ValueError("vehicle length must be positive")
        if self.width <= 0.0:
            raise ValueError("vehicle width must be positive")
        if not 0.0 <= self.rear_axle_to_center <= self.length / 2.0:
            raise ValueError(
                "rear_axle_to_center must place the rear axle within the body"
            )


def oriented_rectangle_corners(
    state: VehicleState,
    geometry: VehicleGeometry,
    *,
    collision: bool = False,
) -> NDArray[np.float64]:
    """Return physical or collision corners using the same body centre."""
    cosine = math.cos(state.yaw)
    sine = math.sin(state.yaw)
    body_center = np.array(
        (
            state.x + geometry.rear_axle_to_center * cosine,
            state.y + geometry.rear_axle_to_center * sine,
        )
    )
    dimensions = geometry.collision if collision and geometry.collision is not None else geometry
    half_length = dimensions.length / 2.0
    half_width = dimensions.width / 2.0
    local_corners = np.array(
        (
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        )
    )
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    return local_corners @ rotation.T + body_center


def rectangles_overlap(
    state_a: VehicleState,
    geometry_a: VehicleGeometry,
    state_b: VehicleState,
    geometry_b: VehicleGeometry,
) -> bool:
    """Report oriented-body overlap using the Separating Axis Theorem."""
    corners_a = oriented_rectangle_corners(state_a, geometry_a, collision=True)
    corners_b = oriented_rectangle_corners(state_b, geometry_b, collision=True)
    axes = (
        _forward_axis(state_a.yaw),
        _left_axis(state_a.yaw),
        _forward_axis(state_b.yaw),
        _left_axis(state_b.yaw),
    )
    tolerance = 1e-12
    for axis in axes:
        projection_a = corners_a @ axis
        projection_b = corners_b @ axis
        if (
            float(np.max(projection_a)) < float(np.min(projection_b)) - tolerance
            or float(np.max(projection_b))
            < float(np.min(projection_a)) - tolerance
        ):
            return False
    return True


def _forward_axis(yaw: float) -> NDArray[np.float64]:
    return np.array((math.cos(yaw), math.sin(yaw)))


def _left_axis(yaw: float) -> NDArray[np.float64]:
    return np.array((-math.sin(yaw), math.cos(yaw)))
