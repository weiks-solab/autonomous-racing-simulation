"""Assemble vehicles and run their individually assigned driving strategies."""

import argparse
import math

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.track_geometry import (
    TrackPosition,
    track_to_world,
    world_to_track,
)
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import VehicleState
from sim.simulated_vehicle import (
    SimulatedVehicle,
    find_vehicle_collisions,
    validate_unique_vehicle_ids,
)
from sim.visualizer import SimulationVisualizer, VisualizationConfig
from sim.strategy_history import StrategyHistory
from strategy.assignment import StrategyAssignment, StrategyManager
from strategy.observations import VehicleObservation, create_vehicle_observation
from strategy.defending import DefenseConfig
from strategy.offset_path import LateralManeuver, OffsetPathConfig, generate_offset_track
from strategy.overtaking import OvertakingConfig


# ===== 常用參數集中在這裡：距離 m、速度 m/s、時間 s =====
DEFENSE_ENABLED = True
SIMULATION_SECONDS = 21.0

TRACK_SEMI_MAJOR_AXIS = 10.0
TRACK_SEMI_MINOR_AXIS = 5.0
TRACK_POINT_COUNT = 500
TRACK_WIDTH = 2.0

# (車輛 ID, 起始 s, 目標速度)。可增加其他車；負 s 會繞回封閉賽道。
VEHICLE_SPECS = (
    ("vehicle-1", 0.0, 6.0),
    ("vehicle-2", 6.0, 5.0),
    ("vehicle-3", -20.0, 4.0),
)

# 此次模擬的策略對照表：只改這裡，就能把策略換到另一台車。
# centerline：跟隨中心線；overtake：進攻；defend：防守指定對象。
# 未列出的車也會自動使用 centerline。每台車都有自己的策略物件。
STRATEGY_ASSIGNMENTS = {
    "vehicle-1": StrategyAssignment("overtake"),
    "vehicle-2": StrategyAssignment("defend", target_vehicle_id="vehicle-1"),
    "vehicle-3": StrategyAssignment("centerline"),
}

MODEL_CONFIG = VehicleModelConfig(
    wheelbase=0.33,
    time_step=0.02,
    max_steering_angle=math.radians(50.0),
    min_velocity=0.0,
    max_velocity=10.0,
)
VEHICLE_GEOMETRY = VehicleGeometry(
    length=0.55, width=0.30, rear_axle_to_center=0.12,
)
LOOKAHEAD_DISTANCE = 2.0
SPEED_GAIN = 1.5
MAX_ACCELERATION = 3.0

OVERTAKING_CONFIG = OvertakingConfig(
    detection_range=6.0,
    minimum_speed_advantage=0.5,
    lateral_offset=0.6,
    safety_margin=0.1,
    pass_clearance=1.0,
    return_tolerance=0.08,
    transition_length=6.0,
    prediction_horizon=5.0,
    prediction_step=0.25,
)
DEFENSE_CONFIG = DefenseConfig(
    q_block=10.0,                    # 越大越想與 ego 對齊；0 表示不主動防守
    center_weight=1.0,             # 越大越偏好中心線
    smooth_weight=1.0,             # 越大越不喜歡急促橫移
    detection_range=8.0,           # 後車在此範圍內才考慮防守
    minimum_speed_advantage=0.2,  # 後車至少快多少才構成威脅
    minimum_gap=1.2,              # 太近或並排時停止追逐橫向位置
    return_tolerance=0.08,
    transition_length=1.0,         # 越長，橫移越緩慢
    decision_period=0.2,           # 每隔多久比較一次候選路徑
    hold_time=0.6,                 # 選定目標後，至少維持多久（原路徑仍可行時）
    switch_improvement=0.05,       # 新路徑至少改善多少成本才切換
    candidate_offsets=(-0.6, -0.3, 0.0, 0.3, 0.6),
    prediction_horizon=1.5,
    prediction_step=0.1,
    safety_margin=0.1,             # 車身與賽道邊界保留的距離
)
OFFSET_PATH_CONFIG = OffsetPathConfig(sample_spacing=0.2, path_length=18.0)


def create_display_options(realtime: bool) -> VisualizationConfig:
    """Choose which elements are visible in the simulation figure."""
    return VisualizationConfig(
        realtime=realtime,

        # Track
        show_centerline=True,
        show_track_boundaries=True,

        # Vehicles
        show_vehicle_markers=False,
        show_vehicle_footprints=True,
        show_vehicle_heading=False,
        show_collision_status=True,
        show_strategy_status=True,
        show_temporary_paths=True,

        # Controller and history information
        show_trajectories=True,
        show_pursuit_targets=True,
        show_track_positions=True,
    )


def _create_vehicle(
    vehicle_id: str,
    start_s: float,
    target_speed: float,
    model_config: VehicleModelConfig,
    track: Track,
) -> SimulatedVehicle:
    point = track_to_world(start_s, 0.0, track)
    point_ahead = track_to_world(start_s + 0.05, 0.0, track)
    tangent_yaw = math.atan2(
        point_ahead[1] - point[1],
        point_ahead[0] - point[0],
    )
    controller_config = PurePursuitConfig(
        lookahead_distance=LOOKAHEAD_DISTANCE,
        target_speed=target_speed,
        speed_gain=SPEED_GAIN,
        max_acceleration=MAX_ACCELERATION,
        max_steering_angle=model_config.max_steering_angle,
    )
    return SimulatedVehicle(
        vehicle_id=vehicle_id,
        state=VehicleState(
            x=float(point[0]),
            y=float(point[1]),
            yaw=tangent_yaw,
            velocity=0.0,
        ),
        model=KinematicBicycleModel(model_config),
        controller=PurePursuitController(controller_config),
        geometry=VEHICLE_GEOMETRY,
    )


def _visualize_vehicles(
    visualizer: SimulationVisualizer,
    vehicles: list[SimulatedVehicle],
    track: Track,
    collision_pairs: set[tuple[str, str]],
    strategy_labels: dict[str, str],
    temporary_tracks: dict[str, Track],
) -> None:
    track_positions: dict[str, TrackPosition] = {
        vehicle.vehicle_id: world_to_track(vehicle.state, track)
        for vehicle in vehicles
    }
    visualizer.update_vehicles(
        states={vehicle.vehicle_id: vehicle.state for vehicle in vehicles},
        trajectories={
            vehicle.vehicle_id: vehicle.trajectory for vehicle in vehicles
        },
        targets={
            vehicle.vehicle_id: vehicle.current_target for vehicle in vehicles
        },
        geometries={
            vehicle.vehicle_id: vehicle.geometry for vehicle in vehicles
        },
        track_positions=track_positions,
        collision_pairs=collision_pairs,
        strategy_labels=strategy_labels,
        temporary_paths=temporary_tracks,
    )


def _create_observations(
    vehicles: list[SimulatedVehicle],
    track: Track,
    maneuvers: dict[str, LateralManeuver] | None = None,
) -> dict[str, VehicleObservation]:
    return {
        vehicle.vehicle_id: create_vehicle_observation(
            vehicle.vehicle_id,
            vehicle.state,
            vehicle.geometry,
            track,
            maneuver=None if maneuvers is None else maneuvers.get(vehicle.vehicle_id),
        )
        for vehicle in vehicles
    }


def run_simulation(
    realtime: bool = False,
    defense_enabled: bool = DEFENSE_ENABLED,
) -> None:
    track = Track.create_oval(
        semi_major_axis=TRACK_SEMI_MAJOR_AXIS,
        semi_minor_axis=TRACK_SEMI_MINOR_AXIS,
        point_count=TRACK_POINT_COUNT,
        width=TRACK_WIDTH,
    )
    model_config = MODEL_CONFIG
    vehicles = [
        _create_vehicle(vehicle_id, start_s, target_speed, model_config, track)
        for vehicle_id, start_s, target_speed in VEHICLE_SPECS
    ]
    validate_unique_vehicle_ids(vehicles)
    manager = StrategyManager(
        [vehicle.vehicle_id for vehicle in vehicles],
        STRATEGY_ASSIGNMENTS,
        OVERTAKING_CONFIG,
        DEFENSE_CONFIG,
        defense_enabled=defense_enabled,
    )
    visualizer = SimulationVisualizer(track, create_display_options(realtime))
    histories = {vehicle.vehicle_id: StrategyHistory() for vehicle in vehicles}
    maneuvers: dict[str, LateralManeuver] = {}
    temporary_tracks: dict[str, Track] = {}
    strategy_labels: dict[str, str] = {}
    detected_collisions: set[tuple[str, str]] = set()
    current_collisions: set[tuple[str, str]] = set()

    step_count = int(SIMULATION_SECONDS / model_config.time_step)
    for step_index in range(step_count):
        time_seconds = step_index * model_config.time_step
        observations = _create_observations(vehicles, track, maneuvers)
        # 不必在這裡判斷哪台是進攻／防守車，統一交給策略管理器。
        decisions = manager.update(observations, track, time_seconds)
        maneuvers = {}
        strategy_labels = {}
        for vehicle_id, decision in decisions.items():
            histories[vehicle_id].record(time_seconds, decision)
            if decision.maneuver is not None:
                maneuvers[vehicle_id] = decision.maneuver
            if manager.assignments[vehicle_id].name != "centerline":
                strategy_labels[vehicle_id] = decision.display_text

        temporary_tracks = {
            vehicle_id: generate_offset_track(
                track, observations[vehicle_id].track_position,
                maneuver, OFFSET_PATH_CONFIG,
            )
            for vehicle_id, maneuver in maneuvers.items()
        }
        for vehicle in vehicles:
            vehicle.step(temporary_tracks.get(vehicle.vehicle_id, track))
        current_collisions = find_vehicle_collisions(vehicles)
        detected_collisions.update(current_collisions)
        if realtime:
            _visualize_vehicles(
                visualizer, vehicles, track, current_collisions,
                strategy_labels, temporary_tracks,
            )

    if not realtime:
        _visualize_vehicles(
            visualizer, vehicles, track, current_collisions,
            strategy_labels, temporary_tracks,
        )
    for vehicle in vehicles:
        vehicle_id = vehicle.vehicle_id
        histories[vehicle_id].print_summary(
            vehicle_id, manager.assignments[vehicle_id].name
        )
        final_position = world_to_track(vehicle.state, track)
        print(f"Final {vehicle_id} d: {final_position.d:+.3f} m")
    if detected_collisions:
        formatted_pairs = ", ".join(
            f"{first}/{second}" for first, second in sorted(detected_collisions)
        )
        print(f"Detected collision pairs: {formatted_pairs}")
    else:
        print("Detected collision pairs: none")
    visualizer.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="animate the simulation as it runs",
    )
    parser.add_argument(
        "--no-defense",
        action="store_true",
        help="replace every defend assignment with centerline following",
    )
    arguments = parser.parse_args()
    run_simulation(
        realtime=arguments.realtime,
        defense_enabled=DEFENSE_ENABLED and not arguments.no_defense,
    )
