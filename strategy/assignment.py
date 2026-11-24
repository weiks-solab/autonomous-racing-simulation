"""把策略名稱轉成各車獨立的策略，統一呼叫既有攻防演算法。

新增策略：演算法放在 strategy/ 的獨立檔案，再於本檔的 StrategyConfigs 加入同名設定欄位，
並在 create_strategy_factories() 加一行工廠；模擬迴圈與 StrategyManager 都不必修改。
"""

from dataclasses import dataclass, field, fields, replace
import math
from typing import Callable, Iterable, Mapping, Sequence

from core.track import Track
from strategy.base import CandidatePath, CandidateSnapshot, StrategyDecision, VehicleStrategy
from strategy.bicycle_prediction import BicyclePredictionConfig
from strategy.centerline import CenterlineStrategy
from strategy.cost_overtaking import CostBasedOvertakingStrategy, CostOvertakingConfig
from strategy.defending import CostBasedDefenseStrategy, DefenseConfig, DefenseMode
from strategy.gp_prediction import GpRoute
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver, OffsetPathConfig
from strategy.overtaking import OvertakingConfig, OvertakingMode, RuleBasedOvertakingStrategy
from strategy.probe_following import ProbeFollowingConfig, ProbeFollowingStrategy
from strategy.progress import ProgressConfig, ProgressStrategy
from strategy.relative_overtaking import RelativeOvertakingConfig, RelativeOvertakingStrategy

# StrategyManager 直接建立的策略；其他名稱由 create_strategy_factories() 的工廠建立。
BUILTIN_STRATEGY_NAMES = ("centerline", "overtake", "defend")


@dataclass(frozen=True)
class StrategyAssignment:
    """name 選策略；只有 defend 需要 target_vehicle_id 指定防守對象。"""

    name: str
    target_vehicle_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("strategy name must be a non-empty string")
        if self.name == "defend":
            if not isinstance(self.target_vehicle_id, str) or not self.target_vehicle_id.strip():
                raise ValueError("defend requires a target_vehicle_id")
        elif self.target_vehicle_id is not None:
            raise ValueError("only defend accepts target_vehicle_id")


class OvertakingStrategy:
    """把既有超車策略的輸出轉成共用格式，不改變超車邏輯。"""

    def __init__(self, config: OvertakingConfig) -> None:
        self.algorithm = RuleBasedOvertakingStrategy(config)
        self.candidate_snapshot: CandidateSnapshot | None = None

    def update(
        self,
        vehicle: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> StrategyDecision:
        previous_evaluations = self.algorithm.last_candidate_evaluations
        result = self.algorithm.update(vehicle, others, track)
        evaluations = self.algorithm.last_candidate_evaluations
        if result.mode in (OvertakingMode.FOLLOW, OvertakingMode.RETURN):
            self.candidate_snapshot = None
        elif evaluations and evaluations is not previous_evaluations:
            # 只記錄真正評估過的左右候選；鎖定側邊後沿用這份快照。
            config = self.algorithm.config
            count = math.ceil(config.prediction_horizon / config.prediction_step)
            self.candidate_snapshot = CandidateSnapshot(
                evaluated_at=time_seconds,
                position=vehicle.track_position,
                distances=tuple(
                    max(vehicle.state.velocity, 0.0)
                    * min(step * config.prediction_step, config.prediction_horizon)
                    for step in range(count + 1)
                ),
                candidates=tuple(
                    CandidatePath(
                        maneuver=LateralManeuver(
                            vehicle.track_position.s, vehicle.track_position.d,
                            candidate.target_d, config.transition_length,
                        ),
                        selected=candidate.side is result.side,
                        name=candidate.side.name,
                        feasible=candidate.feasible,
                        reason=candidate.reason,
                    )
                    for candidate in evaluations
                ),
            )
        details = () if result.side is None else (f"side={result.side.name}",)
        return StrategyDecision(
            mode=result.mode.value,
            target_d=result.target_d,
            target_vehicle_id=result.target_vehicle_id,
            maneuver=result.maneuver,
            reason=result.reason,
            details=details,
            candidate_snapshot=self.candidate_snapshot,
        )


class DefendingStrategy:
    """把既有防守策略的輸出轉成同一格式。"""

    def __init__(
        self,
        config: DefenseConfig,
        target_vehicle_id: str,
        prediction_config: BicyclePredictionConfig | None = None,
    ) -> None:
        self.algorithm = CostBasedDefenseStrategy(config, target_vehicle_id, prediction_config)
        self.candidate_snapshot: CandidateSnapshot | None = None

    def update(
        self,
        vehicle: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> StrategyDecision:
        result = self.algorithm.update(vehicle, others, track, time_seconds)
        if (self.candidate_snapshot is None
                or self.candidate_snapshot.evaluated_at != self.algorithm.last_evaluation_time):
            self.candidate_snapshot = CandidateSnapshot(
                evaluated_at=self.algorithm.last_evaluation_time,
                position=vehicle.track_position,
                distances=(),
                candidates=tuple(
                    CandidatePath(
                        maneuver=candidate.maneuver,
                        selected=(candidate.feasible and result.mode is not DefenseMode.NO_FEASIBLE_PATH
                                  and candidate.target_d == result.target_d),
                        feasible=candidate.feasible,
                        reason="" if candidate.collision is None else candidate.collision.reason,
                        race_cost=candidate.race_cost,
                        blocking_cost=candidate.blocking_cost,
                        predicted_states=candidate.predicted_states,
                    )
                    for candidate in self.algorithm.last_candidates
                ),
            )
        details = ("continuing previous path",)
        if result.race_cost is not None and result.blocking_cost is not None:
            details = (f"race={result.race_cost:.2f} block={result.blocking_cost:.2f}",)
        return StrategyDecision(
            mode=result.mode.value,
            target_d=result.target_d,
            target_vehicle_id=result.target_vehicle_id,
            maneuver=result.maneuver,
            reason=result.reason,
            details=details,
            candidate_snapshot=self.candidate_snapshot,
        )


def create_strategy(
    assignment: StrategyAssignment,
    overtaking_config: OvertakingConfig,
    defense_config: DefenseConfig,
    prediction_config: BicyclePredictionConfig | None = None,
) -> VehicleStrategy:
    """每次建立新物件，讓各車的策略狀態互相獨立。"""
    if assignment.name == "centerline":
        return CenterlineStrategy()
    if assignment.name == "overtake":
        return OvertakingStrategy(overtaking_config)
    if assignment.name == "defend" and assignment.target_vehicle_id is not None:
        return DefendingStrategy(defense_config, assignment.target_vehicle_id, prediction_config)
    raise ValueError("unknown strategy assignment")


@dataclass(frozen=True)
class StrategyConfigs:
    """每種策略一份參數；欄位名稱就是 StrategyAssignment 使用的策略名稱。"""

    overtake: OvertakingConfig = field(default_factory=OvertakingConfig)
    defend: DefenseConfig = field(default_factory=DefenseConfig)
    cost_overtake: CostOvertakingConfig = field(default_factory=CostOvertakingConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    relative_overtake: RelativeOvertakingConfig = field(default_factory=RelativeOvertakingConfig)
    probe_follow: ProbeFollowingConfig = field(default_factory=ProbeFollowingConfig)

    def path_config_for(
        self, base: OffsetPathConfig, assignments: Iterable[StrategyAssignment],
    ) -> OffsetPathConfig:
        """臨時參考路徑要涵蓋每個被指派策略的完整橫移；不足時延長 path_length。"""
        configs = {item.name: getattr(self, item.name) for item in fields(self)}
        transitions = (
            getattr(configs.get(assignment.name), "transition_length", 0.0)
            for assignment in assignments
        )
        return replace(base, path_length=max((base.path_length, *transitions)))


def create_strategy_factories(
    configs: StrategyConfigs,
    prediction_configs: Mapping[str, BicyclePredictionConfig],
    gp_routes: Mapping[str, GpRoute] | None = None,
) -> dict[str, Callable[[str], VehicleStrategy]]:
    """以 bicycle 模型預測的策略。每次呼叫工廠都建立新物件，各車狀態互相獨立。"""
    return {
        "probe_follow": lambda vehicle_id: ProbeFollowingStrategy(
            configs.probe_follow, prediction_configs[vehicle_id], prediction_configs,
        ),
        "cost_overtake": lambda vehicle_id: CostBasedOvertakingStrategy(
            configs.cost_overtake, prediction_configs[vehicle_id], prediction_configs,
        ),
        "progress": lambda vehicle_id: ProgressStrategy(
            configs.progress, prediction_configs[vehicle_id],
        ),
        "relative_overtake": lambda vehicle_id: RelativeOvertakingStrategy(
            configs.relative_overtake, prediction_configs[vehicle_id], prediction_configs,
            gp_routes=gp_routes,
        ),
    }


class StrategyManager:
    """依照對照表，替任意數量的車建立與更新策略。"""

    def __init__(
        self,
        vehicle_ids: Sequence[str],
        assignments: Mapping[str, StrategyAssignment],
        overtaking_config: OvertakingConfig,
        defense_config: DefenseConfig,
        defense_enabled: bool = True,
        strategy_factories: Mapping[str, Callable[[str], VehicleStrategy]] | None = None,
        prediction_configs: Mapping[str, BicyclePredictionConfig] | None = None,
    ) -> None:
        """prediction_configs 提供時，defend 依自己的 ID 取用並改用 bicycle 預測自車。"""
        if not vehicle_ids or len(set(vehicle_ids)) != len(vehicle_ids):
            raise ValueError("vehicle_ids must be non-empty and unique")
        unknown_ids = sorted(set(assignments) - set(vehicle_ids))
        if unknown_ids:
            raise ValueError(
                f"strategy assignments refer to unknown vehicle IDs {unknown_ids}; "
                f"vehicles in this run: {list(vehicle_ids)}"
            )
        factories = dict(strategy_factories or {})
        known_names = sorted({*BUILTIN_STRATEGY_NAMES, *factories})
        for vehicle_id, assignment in assignments.items():
            if assignment.name not in known_names:
                raise ValueError(
                    f"unknown strategy {assignment.name!r} for {vehicle_id}; "
                    f"available strategies: {', '.join(known_names)}"
                )

        self.assignments: dict[str, StrategyAssignment] = {}
        self.strategies: dict[str, VehicleStrategy] = {}
        for vehicle_id in vehicle_ids:
            # 沒有設定策略的車，自動跟隨中心線。
            assignment = assignments.get(vehicle_id, StrategyAssignment("centerline"))
            if not defense_enabled and assignment.name == "defend":
                assignment = StrategyAssignment("centerline")
            if assignment.name == "defend":
                target_id = assignment.target_vehicle_id
                if target_id not in vehicle_ids or target_id == vehicle_id:
                    raise ValueError(
                        f"defense target {target_id!r} for {vehicle_id} must be another "
                        f"vehicle in this run: {list(vehicle_ids)}"
                    )
            self.assignments[vehicle_id] = assignment
            if assignment.name in factories:
                self.strategies[vehicle_id] = factories[assignment.name](vehicle_id)
            else:
                self.strategies[vehicle_id] = create_strategy(
                    assignment, overtaking_config, defense_config,
                    None if prediction_configs is None else prediction_configs.get(vehicle_id),
                )

    def update(
        self,
        observations: Mapping[str, VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> dict[str, StrategyDecision]:
        if set(observations) != set(self.strategies):
            raise ValueError("observations must match the assigned vehicle IDs")
        if any(key != value.vehicle_id for key, value in observations.items()):
            raise ValueError("observation keys must match their vehicle IDs")

        decisions: dict[str, StrategyDecision] = {}
        # 所有車讀取同一步觀測；防守車不接收其他車的規劃路徑。
        for vehicle_id, strategy in self.strategies.items():
            is_defender = self.assignments[vehicle_id].name == "defend"
            others = tuple(
                replace(other, maneuver=None) if is_defender else other
                for other_id, other in observations.items() if other_id != vehicle_id
            )
            decisions[vehicle_id] = strategy.update(
                observations[vehicle_id], others, track, time_seconds
            )
        return decisions
