"""Value objects exchanged by controllers and vehicle models."""

from dataclasses import dataclass
import math


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class VehicleState:
    """Rear-axle vehicle state in SI units."""

    x: float
    y: float
    yaw: float
    velocity: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "yaw", "velocity"):
            _require_finite(name, getattr(self, name))


@dataclass(frozen=True)
class ControlCommand:
    """Steering and longitudinal acceleration command in SI units."""

    steering_angle: float
    acceleration: float

    def __post_init__(self) -> None:
        _require_finite("steering_angle", self.steering_angle)
        _require_finite("acceleration", self.acceleration)
