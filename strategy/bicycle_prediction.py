"""以既有控制器與自行車模型預測單車，不存取模擬器或其他車的策略。"""

from dataclasses import dataclass, replace
import math

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length, world_to_track
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import ControlCommand, VehicleState
from strategy.observations import VehicleObservation
from strategy.offset_path import (
    LateralManeuver, OffsetPathConfig, continue_lateral_maneuver, generate_offset_track,
)


@dataclass(frozen=True)
class BicyclePredictionConfig:
    model: VehicleModelConfig
    controller: PurePursuitConfig
    path: OffsetPathConfig


@dataclass(frozen=True)
class BicyclePrediction:
    states: tuple[VehicleState, ...]
    progress: tuple[float, ...]
    times: tuple[float, ...]


def predict_bicycle_trajectory(
    vehicle: VehicleObservation,
    track: Track,
    maneuver: LateralManeuver,
    config: BicyclePredictionConfig,
    horizon: float,
    constant_speed: bool = False,
) -> BicyclePrediction:
    """逐步重建前方參考線並追蹤；定速模式只將加速度設為零。"""
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("prediction horizon must be positive and finite")
    if not track.closed:
        raise ValueError("bicycle prediction requires a closed track")
    if not config.model.min_velocity <= vehicle.state.velocity <= config.model.max_velocity:
        raise ValueError("initial velocity must be within prediction model limits")
    controller = PurePursuitController(config.controller)
    model = KinematicBicycleModel(config.model)
    state = vehicle.state
    position = vehicle.track_position
    length = track_length(track)
    states, progress, times = [state], [0.0], [0.0]
    step_count = math.ceil(horizon / config.model.time_step)
    for step in range(step_count):
        maneuver = continue_lateral_maneuver(maneuver, position.s, length)
        reference = generate_offset_track(track, position, maneuver, config.path)
        command, _ = controller.compute_command(state, reference, config.model.wheelbase)
        if constant_speed:
            command = ControlCommand(command.steering_angle, 0.0)
        # 最後一步可較短，讓所有候選都在相同的預測終點評分。
        active_model = model
        if step == step_count - 1:
            final_step = horizon - step * config.model.time_step
            active_model = KinematicBicycleModel(replace(config.model, time_step=final_step))
        state = active_model.step(state, command)
        next_position = world_to_track(state, track)
        progress.append(progress[-1] + signed_longitudinal_gap(position, next_position, length))
        position = next_position
        states.append(state)
        times.append(min((step + 1) * config.model.time_step, horizon))
    return BicyclePrediction(tuple(states), tuple(progress), tuple(times))
