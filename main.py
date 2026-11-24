"""Assemble vehicles and run their individually assigned driving strategies."""

import argparse
from dataclasses import replace
import math

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.track_factory import BUILTIN_TRACKS, TrackConfig, create_track
from core.track_loader import find_centerline
from core.track_geometry import (
    track_to_world,
    world_to_track,
)
from core.vehicle_geometry import CollisionConfig, VehicleGeometry
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import VehicleState
from sim.simulated_vehicle import (
    SimulatedVehicle,
    VehicleSpec,
    find_vehicle_collisions,
    validate_unique_vehicle_ids,
)
from sim.visualizer import SimulationVisualizer, VisualizationConfig
from sim.strategy_history import StrategyHistory
from sim.map_background import MapBackground, load_map_background
from sim.replay import ReplayFrame, frame_kwargs, play_frames, precise_timer, record_frame
from strategy.assignment import (
    StrategyAssignment,
    StrategyConfigs,
    StrategyManager,
    create_strategy_factories,
)
from strategy.base import CandidateSnapshot
from strategy.bicycle_prediction import BicyclePredictionConfig
from strategy.cost_overtaking import CostOvertakingConfig
from strategy.observations import VehicleObservation, create_vehicle_observation
from strategy.defending import DefenseConfig
from strategy.offset_path import LateralManeuver, OffsetPathConfig, generate_offset_track
from strategy.overtaking import OvertakingConfig
from strategy.progress import ProgressConfig
from strategy.probe_following import ProbeFollowingConfig
from strategy.relative_overtaking import RelativeOvertakingConfig
from strategy.responsibility import ResponsibilityConfig
from strategy.gp_prediction import GpRoute, load_gp_routes


# =====================================================================
# 參數區（距離 m、速度 m/s、時間 s）。常用設定都在這裡，下方函式只負責組裝與執行。
#   ① 場景：模擬時長、賽道、車輛、策略指派
#   ② 車輛與控制器：模型、車身與碰撞框、Pure Pursuit、臨時路徑
#   ③ 避撞責任約束：defend 與 relative_overtake 共用
#   ④ 各策略參數：relative_overtake、defend、cost_overtake、progress、overtake
#   ⑤ GP 對手預測
#   ⑥ 畫面顯示
#   ⑦ --probe-follow 示範場景
# =====================================================================

# ===== ① 場景 =====
SIMULATION_SECONDS = 21.0
DEFENSE_ENABLED = True  # False 時所有 defend 改為跟隨中心線；命令列 --no-defense 效果相同。

# 只改 TRACK_NAME 即可換圖；下載圖沿用原本的多車與策略設定。
TRACK_NAME = "rounded_rectangle"  # "rounded_rectangle"、"oval" 或 "Oschersleben" 等下載圖名。
TRACK_MAP_ROOT = "external/f1tenth_racetracks"
TRACK_SAMPLE_SPACING = 0.4  # 外部圖以公尺間距取樣；內建圖仍使用 TRACK_POINT_COUNT。
TRACK_RECTANGLE_LENGTH = 20.0    # 中心線左右跨度，不含路寬
TRACK_RECTANGLE_HEIGHT = 12.0    # 中心線上下跨度，不含路寬
TRACK_CORNER_RADIUS = 3.0        # 四個 90 度彎的中心線半徑
TRACK_SEMI_MAJOR_AXIS = 10.0     # oval 專用
TRACK_SEMI_MINOR_AXIS = 5.0      # oval 專用
TRACK_POINT_COUNT = 500
TRACK_WIDTH = 2.0

# 每台車的 ID、起始 s 與巡航目標速度。可增加其他車；負 s 會繞回封閉賽道。
VEHICLE_SPECS = (
    VehicleSpec("vehicle-1", start_s=0.0, target_speed=10.0),
    VehicleSpec("vehicle-2", start_s=3.0, target_speed=9.0),
 ##   VehicleSpec("vehicle-3", start_s=4.0, target_speed=9.0),
)

# 此次模擬的策略對照表：只改這裡，就能把策略換到另一台車。每台車都有自己的策略物件。
# relative_overtake：比較預測終點的相對間距（可加安全距離與偏離中心線成本），保留碰撞篩選。
# cost_overtake：固定候選的成本進攻（進度與超車獎勵）；overtake：原本的規則式進攻。
# progress：固定候選只比較自車進度，不預測其他車或篩選碰撞。
# probe_follow：追蹤目標車距，並輪流左右試探前車。
# defend：防守指定對象，例如 StrategyAssignment("defend", target_vehicle_id="vehicle-2")。
# centerline：跟隨中心線；未列出的車也會自動使用 centerline。
STRATEGY_ASSIGNMENTS = {
    "vehicle-1": StrategyAssignment("probe_follow"),
    "vehicle-2": StrategyAssignment("defend", target_vehicle_id="vehicle-1"),
 ##   "vehicle-3": StrategyAssignment("progress"),
}

# ===== ② 車輛與控制器 =====
MODEL_CONFIG = VehicleModelConfig(
    wheelbase=0.33,
    time_step=0.02,  # 決策只在模擬步發生，實際決策間隔是此值的整數倍。
    max_steering_angle=math.radians(50.0),
    min_velocity=0.0,
    max_velocity=30.0,
)
COLLISION_CONFIG = CollisionConfig(length=0.6, width=0.4)  # 只調整碰撞判定矩形，單位 m。
VEHICLE_GEOMETRY = VehicleGeometry(
    length=0.55, width=0.30, rear_axle_to_center=0.12,
    collision=COLLISION_CONFIG,
)
LOOKAHEAD_DISTANCE = 1.0
SPEED_GAIN = 1.5
MAX_ACCELERATION = 5.0
MAX_DECELERATION = 5.0  # 正值，控制器輸出的減速度最多為此大小，單位 m/s²。
# 交給 Pure Pursuit 的臨時路徑；path_length 比被指派策略的 transition_length 短時，執行時會自動延長。
OFFSET_PATH_CONFIG = OffsetPathConfig(sample_spacing=0.2, path_length=4.0)

# ===== ③ 避撞責任約束（defend 與 relative_overtake 共用）=====
# g=‖Δc‖²−r²−ℓ(Δs)≥0，違反就剔除候選。
# 後車負責保持 r；並排時兩車都負責；領先越多越寬鬆（ℓ 依 Δs=對方 s−自車 s 的 sigmoid）。
# False：恢復原本設定（進攻車矩形重疊剔除、防守車依 collision_check_enabled）。
RESPONSIBILITY_ENABLED = True
RESPONSIBILITY_CONFIG = ResponsibilityConfig(
    collision_radius=0.6,  # r_col：兩車車身中心最小距離 m；=碰撞框長，尚未調參
    slope=40.0,            # a：1/m；前車領先超過 b/a≈0.11 m 即免責，只在完全並排時負責（論文為 5）
    offset=4.5,            # b：沿用論文；後車額外留距約 1/(1+e^b)≈0.011 m²
    # leader_soft=True：決策當下在前的車不剔除，改以 leader_penalty_weight×mean(max(0,r−‖Δc‖)²) 加入成本；
    # 後車與完全並排仍為硬約束。與 overlap_rule 只能擇一。
    leader_soft=False,
    # overlap_rule=True：依決策當下狀態，縱向重疊中（|Δs|<碰撞框長）兩車都負責，
    # 脫離重疊後只剩後車負責；負責者要保持 ‖Δc‖≥r_col，不負責者只計 leader 懲罰。此模式不使用 slope、offset。
    overlap_rule=True,
)
_RESPONSIBILITY = RESPONSIBILITY_CONFIG if RESPONSIBILITY_ENABLED else None

# ===== ④ 各策略參數：每種策略一組，指派到多台車時共用同一組參數 =====
RELATIVE_OVERTAKING_CONFIG = RelativeOvertakingConfig(
    candidate_offsets=(-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6),
    detection_range=6.0,  # 新前車的選取範圍；超車後仍保留原比較對象。
    transition_length=3.0,
    decision_period=0.05,
    prediction_horizon=1.0,
    safety_margin=0.1,
    # 啟用速度候選時，每條橫向路徑都搭配巡航與下列低速，一起比較成本。
    speed_candidates_enabled=True,
    speed_candidate_delta=1.0,       # 第二個速度要求為 max(前車速度 - 此差值, 0)，m/s。
    extra_speed_candidate_deltas=(4.0,),  # 再加前車 -4 m/s；可填 (2.0, 4.0)，最低為 0。
    safety_weight=10.0,    # 安全距離成本權重（1/m）；0 關閉，只比預測終點間距。
    safety_distance=0.5,   # 碰撞矩形間隙小於 0.5 m 時開始懲罰，越近越大。
    deviation_weight=3.0,  # 偏離中心線成本 w×mean(d²)（1/m）；0 關閉，尚未調參。
    responsibility=_RESPONSIBILITY,  # 啟用時取代矩形重疊剔除
    leader_penalty_weight=0.0,  # leader_soft 或 overlap_rule 下攻方不負責時的避碰懲罰（1/m）；尚未調參
)
DEFENSE_CONFIG = DefenseConfig(
    q_block=15.0,                  # 越大越想與 ego 當下位置對齊；0 關閉對齊成本
    progress_weight=2.0,           # 越大越重視預估前進進度
    smooth_weight=1.0,             # 越大越不喜歡急促橫移
    detection_range=6.0,           # 後車在此範圍內才考慮防守
    return_tolerance=0.08,         # 接回共同中心線的橫向容許誤差
    transition_length=2.0,         # 越長，橫移越緩慢
    decision_period=0.05,          # 每隔多久比較一次候選路徑
    candidate_offsets=(-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6),  # 固定候選；超出可用路寬就排除
    prediction_horizon=1.0,        # 自身候選路徑的成本評估時間
    prediction_step=0.1,           # 成本取樣的時間間距
    path_sample_spacing=0.2,       # 幾何路徑的取樣距離；越小越細
    collision_check_enabled=False, # False 可停用獨立的碰撞條件
    collision_horizon=1.0,         # 碰撞／責任檢查時長；越長越會因預測被追上並排而提前避讓
    safety_margin=0.1,             # 限制候選目標與賽道邊界的留距
    deviation_weight=3.0,          # 偏離中心線成本 w×Σd²，併入 J_race；0 關閉，尚未調參
    responsibility=_RESPONSIBILITY,  # 啟用時檢查所有車，取代上方碰撞條件；時長沿用 collision_horizon
    leader_penalty_weight=0.0,     # leader_soft 或 overlap_rule 下守方不負責時的避碰懲罰；0 表示不避讓，尚未調參
)
COST_OVERTAKING_CONFIG = CostOvertakingConfig(
    progress_weight=2.0,           # 前進距離越多，成本越低
    overtaking_weight=10.0,        # 預測終點超過前車時給一次獎勵
    candidate_offsets=(-0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6),
    detection_range=6.0,           # 對此範圍內最近的前車預測
    transition_length=3.0,         # 候選參考路徑的橫移過渡長度
    decision_period=0.1,           # 每隔多久重新比較所有候選的預測軌跡
    prediction_horizon=0.8,        # 預測使用 MODEL_CONFIG 的時間步長
    safety_margin=0.1,             # 只依可用路寬篩選橫向目標
)
PROGRESS_CONFIG = ProgressConfig(
    candidate_offsets=(-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6),
    transition_length=3.0,
    decision_period=0.1,
    prediction_horizon=1.5,
    safety_margin=0.1,  # 只限制候選目標與路邊的距離，不是車間碰撞留距。
)
# 原本的規則式 overtake 專用。
OVERTAKING_CONFIG = OvertakingConfig(
    detection_range=5.0,          # 前方縱向距離 5 m 內，才考慮較慢車輛
    minimum_speed_advantage=0.5,  # 當下車速至少比前車快 0.5 m/s，才觸發超車
    lateral_offset=0.6,           # 左右超車目標為 d=±0.6 m，相對於中心線
    safety_margin=0.1,            # 篩選橫向候選時，車寬之外預留的賽道邊界距離（m）
    pass_clearance=1.0,           # 自車的 s 領先被超車輛超過 1 m 後，開始返回中心線
    return_tolerance=0.08,        # 回程時 |d|≤0.08 m，就判定已回到中心線
    transition_length=6.0,        # 以賽道縱向 6 m 完成平滑橫移；越長越平緩
    prediction_horizon=4.0,       # 左右候選的幾何碰撞檢查，向前預測 4 s
    prediction_step=0.25,         # 每 0.25 s 取樣檢查預測碰撞；不是決策更新週期
)

# ===== ⑤ GP 對手預測（只用於 relative_overtake；下載圖自動停用）=====
# 字典的 ID 是「要用 GP 預測的對手」；未列出的車沿用定速、等 d 的預測。
GP_PREDICTION_ENABLED = False
GP_MODEL_PATHS = {
    "vehicle-2": "models/gp/20261111T064250Z_dbc5bc95/model.pkl",
}

# ===== ⑥ 畫面顯示：True 顯示、False 隱藏 =====
DISPLAY_OPTIONS = VisualizationConfig(
    # Track
    show_centerline=False,
    show_track_boundaries=True,
    show_map_background=True,       # 下載圖的 PNG 背景。

    # Vehicles
    show_vehicle_markers=False,
    show_vehicle_footprints=True,
    show_vehicle_heading=False,
    show_collision_status=True,
    show_envelope_overlap=False,  # False 隱藏橘色重疊提示，保留實體碰撞紅色標示。
    show_strategy_status=False,
    show_temporary_paths=True,
    show_candidate_paths=True,
    show_candidate_labels=False,
    show_gp_predictions=True,       # 紫色點虛線：進攻車估計的對手 GP 軌跡。
    show_gp_prediction_labels=True,

    # Controller and history information
    show_trajectories=True,
    show_pursuit_targets=True,
    show_track_positions=False,
)
# python main.py --replay：先算完整場，再依實際時間平順播放。1.0 為實際速度，0.5 為半速；--speed 可覆寫。
PLAYBACK_SPEED = 1.0

# ===== ⑦ python main.py --probe-follow：圓角矩形雙車場景 =====
PROBE_VEHICLE_SPECS = (
    VehicleSpec("rear", start_s=0.0, target_speed=4.0),
    VehicleSpec("front", start_s=4.0, target_speed=3.0),
)
PROBE_STRATEGY_ASSIGNMENTS = {
    "rear": StrategyAssignment("probe_follow"),
    "front": StrategyAssignment("defend", target_vehicle_id="rear"),
}
# 一般場景指派 probe_follow 時也使用這組參數。
PROBE_FOLLOWING_CONFIG = ProbeFollowingConfig(
    target_gap=2.5,          # 期望的後軸縱向車距，m。
    speed_step=1.0,          # 前車速度 ±1 m/s 與速度匹配，另比較原巡航速度。
    probe_offset=0.6,        # 左右試探偏移，m。
    hold_seconds=2.0,        # 首次到達目標後，維持該橫向目標 2 s；不可行時可中止。
    gap_weight=2.0, speed_weight=1.0, probe_weight=2.0, switch_weight=0.5,
)


def create_demo_track(name: str | None = None) -> Track:
    """Share the selected map with the simulation and training-data entry points."""
    return create_track(TrackConfig(
        name=TRACK_NAME if name is None else name,
        map_root=TRACK_MAP_ROOT,
        sample_spacing=TRACK_SAMPLE_SPACING,
        point_count=TRACK_POINT_COUNT,
        width=TRACK_WIDTH,
        rectangle_length=TRACK_RECTANGLE_LENGTH,
        rectangle_height=TRACK_RECTANGLE_HEIGHT,
        corner_radius=TRACK_CORNER_RADIUS,
        semi_major_axis=TRACK_SEMI_MAJOR_AXIS,
        semi_minor_axis=TRACK_SEMI_MINOR_AXIS,
    ))


def create_demo_background() -> MapBackground | None:
    """Keep image metadata out of Track and the strategy interfaces."""
    if TRACK_NAME in BUILTIN_TRACKS:
        return None
    path = find_centerline(TRACK_NAME, TRACK_MAP_ROOT)
    return load_map_background(path.with_name(path.name.replace("_centerline.csv", "_map.yaml")))


def create_display_options(realtime: bool) -> VisualizationConfig:
    """參數區的 DISPLAY_OPTIONS，加上這次是否即時播放。"""
    return replace(DISPLAY_OPTIONS, realtime=realtime)


def _strategy_configs() -> StrategyConfigs:
    """每次執行時才讀取上方常數，測試與量測腳本可直接覆寫 main 的設定。"""
    return StrategyConfigs(
        overtake=OVERTAKING_CONFIG,
        defend=DEFENSE_CONFIG,
        cost_overtake=COST_OVERTAKING_CONFIG,
        progress=PROGRESS_CONFIG,
        relative_overtake=RELATIVE_OVERTAKING_CONFIG,
        probe_follow=PROBE_FOLLOWING_CONFIG,
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
        max_deceleration=MAX_DECELERATION,
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


def _load_gp_routes(
    vehicles: list[SimulatedVehicle],
    assignments: dict[str, StrategyAssignment],
    imported_track: bool,
) -> dict[str, GpRoute]:
    """只在內建賽道、啟用 GP 且有 relative_overtake 時載入模型；舊模型屬於內建賽道。"""
    if not GP_PREDICTION_ENABLED:
        return {}
    if imported_track:
        print("GP prediction disabled on imported tracks: the existing s-to-d model belongs to the old track.")
        return {}
    if not any(assignment.name == "relative_overtake" for assignment in assignments.values()):
        return {}
    vehicle_ids = {vehicle.vehicle_id for vehicle in vehicles}
    paths = {key: path for key, path in GP_MODEL_PATHS.items() if key in vehicle_ids}
    gp_routes = load_gp_routes(paths)
    for vehicle_id, path in paths.items():
        print(f"GP opponent prediction: {vehicle_id} ({path})")
    return gp_routes


def _print_replay_progress(done_steps: int, total_steps: int) -> None:
    """--replay 計算期間每 10% 印一次進度。"""
    if done_steps == total_steps or done_steps % max(1, total_steps // 10) == 0:
        print(f"Computing replay: {done_steps * 100 // total_steps}% "
              f"({done_steps}/{total_steps} steps)", flush=True)


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


def _print_summary(
    vehicles: list[SimulatedVehicle],
    histories: dict[str, StrategyHistory],
    assignments: dict[str, StrategyAssignment],
    track: Track,
    detected_collisions: set[tuple[str, str]],
) -> None:
    for vehicle in vehicles:
        vehicle_id = vehicle.vehicle_id
        histories[vehicle_id].print_summary(vehicle_id, assignments[vehicle_id].name)
        final_position = world_to_track(vehicle.state, track)
        print(f"Final {vehicle_id} d: {final_position.d:+.3f} m")
    if detected_collisions:
        formatted_pairs = ", ".join(
            f"{first}/{second}" for first, second in sorted(detected_collisions)
        )
        print(f"Detected collision pairs: {formatted_pairs}")
    else:
        print("Detected collision pairs: none")


def run_simulation(
    realtime: bool = False,
    defense_enabled: bool = DEFENSE_ENABLED,
    probe_follow_demo: bool = False,
    replay: bool = False,
) -> None:
    """realtime：邊算邊畫。replay：先算完整場、不開視窗，再依實際時間平順播放（優先於 realtime）。"""
    track = (create_demo_track("rounded_rectangle") if probe_follow_demo
             else create_demo_track())
    vehicle_specs = PROBE_VEHICLE_SPECS if probe_follow_demo else VEHICLE_SPECS
    assignments = PROBE_STRATEGY_ASSIGNMENTS if probe_follow_demo else STRATEGY_ASSIGNMENTS
    model_config = MODEL_CONFIG
    vehicles = [
        _create_vehicle(vehicle_id, start_s, target_speed, model_config, track)
        for vehicle_id, start_s, target_speed in vehicle_specs
    ]
    validate_unique_vehicle_ids(vehicles)
    imported_track = not probe_follow_demo and TRACK_NAME not in BUILTIN_TRACKS
    gp_routes = _load_gp_routes(vehicles, assignments, imported_track)
    configs = _strategy_configs()
    # 參考路徑不夠涵蓋被指派策略的橫移時自動延長，例如規則式 overtake 的 6 m。
    path_config = configs.path_config_for(OFFSET_PATH_CONFIG, assignments.values())
    prediction_configs = {
        vehicle.vehicle_id: BicyclePredictionConfig(
            vehicle.model.config, vehicle.controller.config, path_config,
        )
        for vehicle in vehicles
    }
    # 策略名稱與建構方式集中在 strategy/assignment.py；新增策略不必修改這裡。
    manager = StrategyManager(
        [vehicle.vehicle_id for vehicle in vehicles],
        assignments,
        configs.overtake,
        configs.defend,
        defense_enabled=defense_enabled,
        strategy_factories=create_strategy_factories(configs, prediction_configs, gp_routes),
        prediction_configs=prediction_configs,  # defend 以 bicycle model 預測自車候選
    )
    live = realtime and not replay
    display_options = create_display_options(realtime or replay)
    if probe_follow_demo:
        display_options = replace(display_options, show_centerline=True,
                                  show_strategy_status=True, show_gp_predictions=False)
    if imported_track:
        display_options = replace(display_options, show_gp_predictions=False)
    # PNG 背景只屬於下載圖；--probe-follow 固定使用圓角矩形。
    background = (create_demo_background()
                  if imported_track and display_options.show_map_background else None)
    # --replay 算完才開視窗，計算期間不會出現「沒有回應」的視窗。
    visualizer = None if replay else SimulationVisualizer(track, display_options, background=background)
    replay_frames: list[ReplayFrame] = []
    # 候選路徑與 GP 預測都不顯示時，播放不必保存候選快照，以節省記憶體。
    keep_snapshots = display_options.show_candidate_paths or display_options.show_gp_predictions
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
                maneuver, path_config,
            )
            for vehicle_id, maneuver in maneuvers.items()
        }
        for vehicle in vehicles:
            vehicle.step(
                temporary_tracks.get(vehicle.vehicle_id, track),
                speed_limit=decisions[vehicle.vehicle_id].speed_limit,
            )
        current_collisions = find_vehicle_collisions(vehicles)
        detected_collisions.update(current_collisions)
        if live or replay:
            # 即時、最後一幀與 --replay 共用同一份畫面資料。
            frame = record_frame(
                (step_index + 1) * model_config.time_step, vehicles, track, current_collisions,
                strategy_labels, temporary_tracks,
                candidate_snapshots if live or keep_snapshots else {},
            )
            if live:
                visualizer.update_vehicles(**frame_kwargs(frame, vehicles))
            else:
                replay_frames.append(frame)
                _print_replay_progress(step_index + 1, step_count)

    if not (realtime or replay):
        final_frame = record_frame(
            step_count * model_config.time_step, vehicles, track, current_collisions,
            strategy_labels, temporary_tracks, candidate_snapshots,
        )
        visualizer.update_vehicles(**frame_kwargs(final_frame, vehicles))
    _print_summary(vehicles, histories, manager.assignments, track, detected_collisions)
    if replay:
        visualizer = SimulationVisualizer(track, display_options, background=background)
        with precise_timer():
            play_frames(visualizer, replay_frames, vehicles, speed=PLAYBACK_SPEED)
    visualizer.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="animate the simulation as it runs",
    )
    parser.add_argument(
        "--probe-follow",
        action="store_true",
        help="run the rounded-rectangle probe-following and defense demo",
    )
    parser.add_argument(
        "--no-defense",
        action="store_true",
        help="replace every defend assignment with centerline following",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="compute the whole run first, then play it back smoothly at real-time speed",
    )
    parser.add_argument("--speed", type=float, help="playback speed for --replay (default: PLAYBACK_SPEED)")
    parser.add_argument("--track", help="track for this run only (default: TRACK_NAME)")
    parser.add_argument("--seconds", type=float, help="simulated seconds for this run only (default: SIMULATION_SECONDS)")
    arguments = parser.parse_args()
    # 命令列只覆寫這次執行的設定，不修改檔案。
    for option in ("speed", "seconds"):
        value = getattr(arguments, option)
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            parser.error(f"--{option} must be positive and finite")
    if arguments.speed is not None:
        PLAYBACK_SPEED = arguments.speed
    if arguments.seconds is not None:
        SIMULATION_SECONDS = arguments.seconds
    if arguments.track is not None:
        TRACK_NAME = arguments.track
    run_simulation(
        realtime=arguments.realtime,
        defense_enabled=DEFENSE_ENABLED and not arguments.no_defense,
        probe_follow_demo=arguments.probe_follow,
        replay=arguments.replay,
    )
