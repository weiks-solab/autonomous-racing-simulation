"""Rear-axle kinematic bicycle vehicle model."""

from dataclasses import dataclass
import math

from core.vehicle_state import ControlCommand, VehicleState


@dataclass(frozen=True)
class VehicleModelConfig:
    wheelbase: float
    time_step: float
    max_steering_angle: float
    min_velocity: float
    max_velocity: float

    def __post_init__(self) -> None:
        values = (
            self.wheelbase,
            self.time_step,
            self.max_steering_angle,
            self.min_velocity,
            self.max_velocity,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vehicle model configuration must be finite")
        if self.wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive")
        if self.time_step <= 0.0:
            raise ValueError("time_step must be positive")
        if not 0.0 < self.max_steering_angle < math.pi / 2.0:
            raise ValueError("max_steering_angle must be between 0 and pi/2")
        if self.min_velocity > self.max_velocity:
            raise ValueError("min_velocity must not exceed max_velocity")


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _normalize_angle(angle: float) -> float:
    if -math.pi <= angle < math.pi:
        return angle
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class KinematicBicycleModel:
    """Advance a rear-axle state with explicit Euler integration."""

    def __init__(self, config: VehicleModelConfig) -> None:
        self.config = config

    def step(self, state: VehicleState, command: ControlCommand) -> VehicleState:
        config = self.config
        steering = _clamp(
            command.steering_angle,
            -config.max_steering_angle,
            config.max_steering_angle,
        )

        # Explicit Euler: every pose derivative uses the current velocity.
        next_x = state.x + state.velocity * math.cos(state.yaw) * config.time_step
        next_y = state.y + state.velocity * math.sin(state.yaw) * config.time_step
        next_yaw = _normalize_angle(
            state.yaw
            + state.velocity
            / config.wheelbase
            * math.tan(steering)
            * config.time_step
        )
        next_velocity = _clamp(
            state.velocity + command.acceleration * config.time_step,
            config.min_velocity,
            config.max_velocity,
        )
        return VehicleState(next_x, next_y, next_yaw, next_velocity)
