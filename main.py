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
from strategy.base import CandidateSnapshot
from strategy.bicycle_prediction import BicyclePredictionConfig
from strategy.cost_overtaking import CostBasedOvertakingStrategy, CostOvertakingConfig
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
    ("vehicle-1", 0.0, 10.0),
    ("vehicle-2", 6.0, 8.0),
##    ("vehicle-3", -20.0, 4.0),
)

# 此次模擬的策略對照表：只改這裡，就能把策略換到另一台車。
# cost_overtake：五候選成本進攻；overtake：原本的規則式進攻。
# centerline：跟隨中心線；defend：防守指定對象。
# 未列出的車也會自動使用 centerline。每台車都有自己的策略物件。
STRATEGY_ASSIGNMENTS = {
    "vehicle-1": StrategyAssignment("cost_overtake"),
    "vehicle-2": StrategyAssignment("defend", target_vehicle_id="vehicle-1"),
 ##   "vehicle-3": StrategyAssignment("centerline"),
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

# 規則式 overtake 專用；cost_overtake 使用下方的 COST_OVERTAKING_CONFIG。
OVERTAKING_CONFIG = OvertakingConfig(
    detection_range=5.0,          # 前方縱向距離 5 m 內，才考慮較慢車輛
    minimum_speed_advantage=0.5,  # 當下車速至少比前車快 0.5 m/s，才觸發超車
    lateral_offset=0.6,           # 左右超車目標為 d=±0.6 m，相對於中心線
    safety_margin=0.1,            # 篩選橫向候選時，車寬之外預留的賽道邊界距離（m）
    pass_clearance=1.0,           # 自車的 s 領先被超車輛超過 1 m 後，開始返回中心線
    return_tolerance=0.08,        # 回程時 |d|≤0.08 m，就判定已回到中心線
    transition_length=6.0,       # 以賽道縱向 6 m 完成平滑橫移；越長越平緩
    prediction_horizon=4.0,      # 左右候選的幾何碰撞檢查，向前預測 4 s
    prediction_step=0.25,        # 每 0.25 s 取樣檢查預測碰撞；不是決策更新週期
)
DEFENSE_CONFIG = DefenseConfig(
    q_block=15.0,                  # 越大越想與 ego 當下位置對齊；0 關閉對齊成本
    progress_weight=2.0,           # 越大越重視預估前進進度
    smooth_weight=1.0,             # 越大越不喜歡急促橫移
    detection_range=6.0,           # 後車在此範圍內才考慮防守
    return_tolerance=0.08,        # 接回共同中心線的橫向容許誤差
    transition_length=4.0,         # 越長，橫移越緩慢
    decision_period=0.3,           # 每隔多久比較一次候選路徑
    candidate_offsets=(-0.6, -0.3, 0.0, 0.3, 0.6),  # 固定候選；超出可用路寬就排除
    prediction_horizon=1,       # 自身候選路徑的成本評估時間
    prediction_step=0.1,          # 成本取樣的時間間距
    path_sample_spacing=0.2,      # 幾何路徑的取樣距離；越小越細
    collision_check_enabled=False, # False 可停用獨立的碰撞條件
    collision_horizon=1.5,        # 幾何碰撞預測時長；間距沿用 prediction_step
    safety_margin=0.1,            # 限制候選目標與賽道邊界的留距
)
COST_OVERTAKING_CONFIG = CostOvertakingConfig(
    progress_weight=1.0,           # 前進距離越多，成本越低
    overtaking_weight=20.0,        # 預測終點超過前車時給一次獎勵
    candidate_offsets=(-0.6, -0.3, 0.0, 0.3, 0.6),
    detection_range=6.0,           # 對此範圍內最近的前車預測
    transition_length=3.0,         # 候選參考路徑的橫移過渡長度
    decision_period=0.1,           # 每次重新比較五條預測軌跡的間隔
    prediction_horizon=1.5,        # 預測使用 MODEL_CONFIG 的時間步長
    safety_margin=0.1,             # 只依可用路寬篩選橫向目標
)
OFFSET_PATH_CONFIG = OffsetPathConfig(sample_spacing=0.2, path_length=4.0)


def create_display_options(realtime: bool) -> VisualizationConfig:
    """Choose which elements are visible in the simulation figure."""
    return VisualizationConfig(
        realtime=realtime,

        # Track
        show_centerline=False,
        show_track_boundaries=True,

        # Vehicles
        show_vehicle_markers=False,
        show_vehicle_footprints=True,
        show_vehicle_heading=False,
        show_collision_status=True,
        show_strategy_status=False,
        show_temporary_paths=True,
        show_candidate_paths=True,
        show_candidate_labels=False,

        # Controller and history information
        show_trajectories=True,
        show_pursuit_targets=True,
        show_track_positions=False,
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
    candidate_snapshots: dict[str, CandidateSnapshot],
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
        candidate_snapshots=candidate_snapshots,
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
    prediction_configs = {
        vehicle.vehicle_id: BicyclePredictionConfig(
            vehicle.model.config, vehicle.controller.config, OFFSET_PATH_CONFIG,
        )
        for vehicle in vehicles
    }
    # 工廠每次建立獨立策略；新增策略可在此註冊，再由對照表選用。
    strategy_factories = {
        "cost_overtake": lambda vehicle_id: CostBasedOvertakingStrategy(
            COST_OVERTAKING_CONFIG, prediction_configs[vehicle_id], prediction_configs,
        ),
    }
    manager = StrategyManager(
        [vehicle.vehicle_id for vehicle in vehicles],
        STRATEGY_ASSIGNMENTS,
        OVERTAKING_CONFIG,
        DEFENSE_CONFIG,
        defense_enabled=defense_enabled,
        strategy_factories=strategy_factories,
    )
    visualizer = SimulationVisualizer(track, create_display_options(realtime))
    histories = {vehicle.vehicle_id: StrategyHistory() for vehicle in vehicles}
    maneuvers: dict[str, LateralManeuver] = {}
    temporary_tracks: dict[str, Track] = {}
    strategy_labels: dict[str, str] = {}
    candidate_snapshots: dict[str, CandidateSnapshot] = {}
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
        candidate_snapshots = {}
        for vehicle_id, decision in decisions.items():
            histories[vehicle_id].record(time_seconds, decision)
            if decision.maneuver is not None:
                maneuvers[vehicle_id] = decision.maneuver
            if decision.candidate_snapshot is not None:
                candidate_snapshots[vehicle_id] = decision.candidate_snapshot
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
                candidate_snapshots,
            )

    if not realtime:
        _visualize_vehicles(
            visualizer, vehicles, track, current_collisions,
            strategy_labels, temporary_tracks,
            candidate_snapshots,
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
