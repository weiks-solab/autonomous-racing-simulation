"""Pure Pursuit lateral control with proportional speed regulation."""

from dataclasses import dataclass
import math

from core.track import Track
from core.vehicle_state import ControlCommand, VehicleState


@dataclass(frozen=True)
class PurePursuitConfig:
    lookahead_distance: float
    target_speed: float
    speed_gain: float
    max_acceleration: float
    max_steering_angle: float
    max_deceleration: float | None = None

    def __post_init__(self) -> None:
        # 舊設定只有加速度上限，省略減速度時保留對稱限制。
        if self.max_deceleration is None:
            object.__setattr__(self, "max_deceleration", self.max_acceleration)
        values = (
            self.lookahead_distance,
            self.target_speed,
            self.speed_gain,
            self.max_acceleration,
            self.max_steering_angle,
            self.max_deceleration,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("controller configuration must be finite")
        if self.lookahead_distance <= 0.0:
            raise ValueError("lookahead_distance must be positive")
        if self.target_speed < 0.0:
            raise ValueError("target_speed must be non-negative")
        if self.speed_gain < 0.0:
            raise ValueError("speed_gain must be non-negative")
        if self.max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be positive")
        if self.max_deceleration <= 0.0:
            raise ValueError("max_deceleration must be positive")
        if not 0.0 < self.max_steering_angle < math.pi / 2.0:
            raise ValueError("max_steering_angle must be between 0 and pi/2")


@dataclass(frozen=True)
class PursuitTarget:
    nearest_index: int
    lookahead_index: int
    x: float
    y: float


class PurePursuitController:
    def __init__(self, config: PurePursuitConfig) -> None:
        self.config = config

    def compute_command(
        self,
        state: VehicleState,
        track: Track,
        wheelbase: float,
        speed_limit: float | None = None,
    ) -> tuple[ControlCommand, PursuitTarget]:
        if not math.isfinite(wheelbase) or wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive and finite")
        target = self._find_target(state, track)
        dx = target.x - state.x
        dy = target.y - state.y
        target_distance = math.hypot(dx, dy)
        if target_distance <= 0.0:
            steering = 0.0
        else:
            alpha = _normalize_angle(math.atan2(dy, dx) - state.yaw)
            steering = math.atan2(
                2.0 * wheelbase * math.sin(alpha), target_distance
            )
        steering = _clamp(
            steering,
            -self.config.max_steering_angle,
            self.config.max_steering_angle,
        )
        acceleration = compute_speed_acceleration(state.velocity, self.config, speed_limit)
        return ControlCommand(steering, acceleration), target

    def _find_target(self, state: VehicleState, track: Track) -> PursuitTarget:
        nearest_index = track.nearest_point_index(state.x, state.y)
        lookahead_index = track.forward_point_index(
            nearest_index, self.config.lookahead_distance
        )
        point = track.point(lookahead_index)
        return PursuitTarget(
            nearest_index=nearest_index,
            lookahead_index=lookahead_index,
            x=float(point[0]),
            y=float(point[1]),
        )


def validate_speed_limit(speed_limit: float | None) -> None:
    """速度要求以 m/s 表示。None 表示恢復原巡航目標。"""
    if speed_limit is not None and (not math.isfinite(speed_limit) or speed_limit < 0.0):
        raise ValueError("speed_limit must be non-negative and finite, or None")


def compute_speed_acceleration(
    velocity: float,
    config: PurePursuitConfig,
    speed_limit: float | None = None,
) -> float:
    """預測與執行共用的比例速度控制，每次依巡航設定與速度要求計算。"""
    validate_speed_limit(speed_limit)
    target_speed = config.target_speed
    if speed_limit is not None:
        target_speed = min(target_speed, speed_limit)
    return _clamp(
        config.speed_gain * (target_speed - velocity),
        -config.max_deceleration,
        config.max_acceleration,
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
