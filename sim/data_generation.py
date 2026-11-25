"""Headless training episodes using the existing strategies and vehicle model."""

from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.track_geometry import track_length, track_to_world, world_to_track
from core.vehicle_geometry import VehicleGeometry, oriented_rectangle_corners
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import ControlCommand, VehicleState
from strategy.base import StrategyDecision
from sim.simulated_vehicle import SimulatedVehicle, find_vehicle_collisions, validate_unique_vehicle_ids
from strategy.assignment import (
    StrategyAssignment, StrategyConfigs, StrategyManager, create_strategy_factories,
)
from strategy.bicycle_prediction import BicyclePredictionConfig
from strategy.cost_overtaking import CostOvertakingConfig
from strategy.defending import DefenseConfig
from strategy.interaction_gp import InteractionGpModel
from strategy.interaction_overtaking import InteractionOvertakingConfig
from strategy.rollout_overtaking import RolloutOvertakingConfig
from strategy.paper_gp import PaperGpModel
from strategy.paper_rollout_overtaking import PaperRolloutOvertakingConfig
from strategy.observations import VehicleObservation, create_vehicle_observation
from strategy.offset_path import LateralManeuver, OffsetPathConfig, generate_offset_track
from strategy.overtaking import OvertakingConfig
from strategy.progress import ProgressConfig
from strategy.probe_following import ProbeFollowingConfig
from strategy.relative_overtaking import RelativeOvertakingConfig
from training_data.collector import TrainingDataCollector
from training_data.schema import require_id, require_integer


@dataclass(frozen=True)
class VehicleInitialCondition:
    vehicle_id: str
    start_s: float
    start_d: float
    initial_speed: float
    target_speed: float
    heading_error: float = 0.0

    def __post_init__(self) -> None:
        require_id("vehicle_id", self.vehicle_id)
        values = (self.start_s, self.start_d, self.initial_speed, self.target_speed, self.heading_error)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("initial vehicle conditions must be finite")
        if self.target_speed < 0.0:
            raise ValueError("target_speed must be non-negative")


@dataclass(frozen=True)
class EpisodeScenario:
    episode_id: str
    vehicles: tuple[VehicleInitialCondition, ...]
    assignments: Mapping[str, StrategyAssignment]

    def __post_init__(self) -> None:
        require_id("episode_id", self.episode_id)
        object.__setattr__(self, "vehicles", tuple(self.vehicles))
        ids = [vehicle.vehicle_id for vehicle in self.vehicles]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("scenario vehicle IDs must be non-empty and unique")
        if set(self.assignments) - set(ids):
            raise ValueError("scenario assignments reference unknown vehicles")
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))


@dataclass(frozen=True)
class SimulationSettings:
    track: Track
    model_config: VehicleModelConfig
    geometry: VehicleGeometry
    controller_config: PurePursuitConfig
    overtaking_config: OvertakingConfig = field(default_factory=OvertakingConfig)
    defense_config: DefenseConfig = field(default_factory=DefenseConfig)
    cost_overtaking_config: CostOvertakingConfig = field(default_factory=CostOvertakingConfig)
    offset_path_config: OffsetPathConfig = field(default_factory=OffsetPathConfig)
    defense_enabled: bool = True
    progress_config: ProgressConfig = field(default_factory=ProgressConfig)
    relative_overtaking_config: RelativeOvertakingConfig = field(default_factory=RelativeOvertakingConfig)
    probe_following_config: ProbeFollowingConfig = field(default_factory=ProbeFollowingConfig)
    interaction_overtaking_config: InteractionOvertakingConfig = field(
        default_factory=InteractionOvertakingConfig)
    rollout_overtaking_config: RolloutOvertakingConfig = field(default_factory=RolloutOvertakingConfig)
    # interaction_overtake 使用的互動 GP；None 時該策略退回定速等 d 預測。
    interaction_model: InteractionGpModel | None = None
    paper_rollout_overtaking_config: PaperRolloutOvertakingConfig = field(
        default_factory=PaperRolloutOvertakingConfig)
    # paper_rollout_overtake 使用的論文格式一步 GP；None 時該策略退回定速等 d 預測。
    paper_model: PaperGpModel | None = None

    @property
    def strategy_configs(self) -> StrategyConfigs:
        """依策略名稱整理上方的各策略設定，供共用工廠與參考路徑長度規則使用。"""
        return StrategyConfigs(
            overtake=self.overtaking_config, defend=self.defense_config,
            cost_overtake=self.cost_overtaking_config, progress=self.progress_config,
            relative_overtake=self.relative_overtaking_config,
            probe_follow=self.probe_following_config,
            interaction_overtake=self.interaction_overtaking_config,
            rollout_overtake=self.rollout_overtaking_config,
            paper_rollout_overtake=self.paper_rollout_overtaking_config,
        )


@dataclass(frozen=True)
class RandomizationConfig:
    longitudinal_jitter: float = 1.5
    lateral_limit: float = 0.5
    target_speed_jitter: float = 0.5
    initial_speed_deficit: float = 2.0
    boundary_margin: float = 0.15
    max_attempts: int = 100

    def __post_init__(self) -> None:
        values = (
            self.longitudinal_jitter, self.lateral_limit, self.target_speed_jitter,
            self.initial_speed_deficit, self.boundary_margin,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("randomization ranges must be non-negative and finite")
        require_integer("max_attempts", self.max_attempts, 1)


def simulation_step_count(seconds: float, time_step: float) -> int:
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("seconds must be positive and finite")
    step_count = round(seconds / time_step)
    if step_count < 1 or not math.isclose(step_count * time_step, seconds, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("seconds must be a positive integer multiple of the simulation time_step")
    return step_count


def _create_vehicle(initial: VehicleInitialCondition, settings: SimulationSettings) -> SimulatedVehicle:
    config = settings.model_config
    if not config.min_velocity <= initial.initial_speed <= config.max_velocity:
        raise ValueError("initial speed must lie within the vehicle model limits")
    if not config.min_velocity <= initial.target_speed <= config.max_velocity:
        raise ValueError("target speed must lie within the vehicle model limits")
    point = track_to_world(initial.start_s, initial.start_d, settings.track)
    position = world_to_track(VehicleState(float(point[0]), float(point[1]), 0.0, 0.0), settings.track)
    index = position.centerline_index
    points = settings.track.centerline
    tangent = points[(index + 1) % len(points)] - points[index]
    state = VehicleState(
        float(point[0]), float(point[1]),
        math.atan2(float(tangent[1]), float(tangent[0])) + initial.heading_error,
        initial.initial_speed,
    )
    return SimulatedVehicle(
        initial.vehicle_id, state, KinematicBicycleModel(config),
        PurePursuitController(replace(settings.controller_config, target_speed=initial.target_speed)),
        settings.geometry,
    )


def _create_vehicles(scenario: EpisodeScenario, settings: SimulationSettings) -> list[SimulatedVehicle]:
    vehicles = [_create_vehicle(initial, settings) for initial in scenario.vehicles]
    validate_unique_vehicle_ids(vehicles)
    if find_vehicle_collisions(vehicles):
        raise ValueError("initial vehicle footprints must not overlap")
    for vehicle in vehicles:
        for x, y in oriented_rectangle_corners(vehicle.state, vehicle.geometry):
            corner = replace(vehicle.state, x=float(x), y=float(y))
            if abs(world_to_track(corner, settings.track).d) > settings.track.width / 2.0 + 1e-9:
                raise ValueError("initial vehicle footprints must stay inside track boundaries")
    return vehicles


def _create_manager(
    scenario: EpisodeScenario, vehicles: list[SimulatedVehicle], settings: SimulationSettings,
    offset_config: OffsetPathConfig,
) -> StrategyManager:
    prediction_configs = {
        vehicle.vehicle_id: BicyclePredictionConfig(
            vehicle.model.config, vehicle.controller.config, offset_config,
        )
        for vehicle in vehicles
    }
    configs = settings.strategy_configs
    return StrategyManager(
        [vehicle.vehicle_id for vehicle in vehicles], scenario.assignments,
        configs.overtake, configs.defend,
        defense_enabled=settings.defense_enabled,
        strategy_factories=create_strategy_factories(
            configs, prediction_configs, interaction_model=settings.interaction_model,
            paper_model=settings.paper_model),
        prediction_configs=prediction_configs,
    )


def _observe(
    vehicles: list[SimulatedVehicle], track: Track, maneuvers: Mapping[str, LateralManeuver],
) -> dict[str, VehicleObservation]:
    return {
        vehicle.vehicle_id: create_vehicle_observation(
            vehicle.vehicle_id, vehicle.state, vehicle.geometry, track,
            maneuver=maneuvers.get(vehicle.vehicle_id),
        )
        for vehicle in vehicles
    }


def run_episode(
    scenario: EpisodeScenario, settings: SimulationSettings,
    collector: TrainingDataCollector, seconds: float,
    step_observer: Callable[[int, VehicleObservation, StrategyDecision, ControlCommand], None] | None = None,
) -> None:
    """Run actual controls and motion; record initial and terminal frames too."""
    time_step = settings.model_config.time_step
    if collector.time_step != time_step:
        raise ValueError("collector and vehicle model must use the same time_step")
    steps = simulation_step_count(seconds, time_step)
    vehicles = _create_vehicles(scenario, settings)
    # 資料入口也支援原本的 overtake；參考線必須涵蓋完整橫移。
    offset_config = settings.strategy_configs.path_config_for(
        settings.offset_path_config, scenario.assignments.values(),
    )
    manager = _create_manager(scenario, vehicles, settings, offset_config)
    observations = _observe(vehicles, settings.track, {})
    collector.record(scenario.episode_id, 0, 0.0, observations)
    for step_index in range(steps):
        decisions = manager.update(observations, settings.track, step_index * time_step)
        maneuvers = {
            vehicle_id: decision.maneuver for vehicle_id, decision in decisions.items()
            if decision.maneuver is not None
        }
        temporary_tracks = {
            vehicle_id: generate_offset_track(
                settings.track, observations[vehicle_id].track_position, maneuver, offset_config,
            )
            for vehicle_id, maneuver in maneuvers.items()
        }
        for vehicle in vehicles:
            command, _ = vehicle.step(
                temporary_tracks.get(vehicle.vehicle_id, settings.track),
                speed_limit=decisions[vehicle.vehicle_id].speed_limit,
            )
            # 只記錄控制前狀態與本步動作；觀察器不參與規劃。
            if step_observer is not None:
                step_observer(step_index, observations[vehicle.vehicle_id],
                              decisions[vehicle.vehicle_id], command)
        # 所有車先前進，再統一觀察；不會混用更新前後的狀態。
        observations = _observe(vehicles, settings.track, maneuvers)
        collector.record(
            scenario.episode_id, step_index + 1, (step_index + 1) * time_step,
            observations, find_vehicle_collisions(vehicles),
        )


def randomized_scenarios(
    settings: SimulationSettings,
    vehicle_specs: Sequence[tuple[str, float, float]],
    assignments: Mapping[str, StrategyAssignment],
    episode_count: int,
    seed: int,
    randomization: RandomizationConfig = RandomizationConfig(),
) -> tuple[EpisodeScenario, ...]:
    """Vary physical initial conditions while keeping strategy parameters fixed."""
    require_integer("episode_count", episode_count, 1)
    require_integer("seed", seed)
    if not settings.track.closed:
        raise ValueError("randomized scenarios require a closed track")
    if not vehicle_specs:
        raise ValueError("at least one vehicle specification is required")
    for vehicle_id, start_s, speed in vehicle_specs:
        VehicleInitialCondition(vehicle_id, start_s, 0.0, speed, speed)
    ids = [spec[0] for spec in vehicle_specs]
    if len(set(ids)) != len(ids):
        raise ValueError("vehicle IDs must be unique")
    max_d = min(
        randomization.lateral_limit,
        settings.track.width / 2.0 - settings.geometry.width / 2.0 - randomization.boundary_margin,
    )
    if max_d < 0.0 or settings.model_config.max_velocity < max(settings.model_config.min_velocity, 0.0):
        raise ValueError("track width and speed limits cannot accommodate randomized vehicles")
    rng = np.random.default_rng(seed)
    length = track_length(settings.track)
    scenarios = []
    for episode_index in range(episode_count):
        for _ in range(randomization.max_attempts):
            origin = float(rng.uniform(0.0, length))
            initials = []
            for vehicle_id, base_s, base_speed in vehicle_specs:
                target_speed = float(np.clip(
                    base_speed + rng.uniform(-randomization.target_speed_jitter, randomization.target_speed_jitter),
                    max(settings.model_config.min_velocity, 0.0), settings.model_config.max_velocity,
                ))
                initial_speed = float(rng.uniform(
                    max(settings.model_config.min_velocity, target_speed - randomization.initial_speed_deficit),
                    target_speed,
                ))
                initials.append(VehicleInitialCondition(
                    vehicle_id,
                    (origin + base_s + float(rng.uniform(
                        -randomization.longitudinal_jitter, randomization.longitudinal_jitter,
                    ))) % length,
                    float(rng.uniform(-max_d, max_d)), initial_speed, target_speed,
                ))
            scenario = EpisodeScenario(f"episode-{episode_index:05d}", tuple(initials), assignments)
            try:
                _create_vehicles(scenario, settings)
            except ValueError:
                continue
            scenarios.append(scenario)
            break
        else:
            raise ValueError("could not generate valid initial vehicle footprints within max_attempts")
    return tuple(scenarios)
