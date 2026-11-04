"""把策略名稱轉成各車獨立的策略，統一呼叫既有攻防演算法。"""

from dataclasses import dataclass, replace
import math
from typing import Callable, Mapping, Sequence

from core.track import Track
from strategy.base import CandidatePath, CandidateSnapshot, StrategyDecision, VehicleStrategy
from strategy.centerline import CenterlineStrategy
from strategy.defending import CostBasedDefenseStrategy, DefenseConfig, DefenseMode
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver
from strategy.overtaking import OvertakingConfig, OvertakingMode, RuleBasedOvertakingStrategy


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

    def __init__(self, config: DefenseConfig, target_vehicle_id: str) -> None:
        self.algorithm = CostBasedDefenseStrategy(config, target_vehicle_id)
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
) -> VehicleStrategy:
    """每次建立新物件，讓各車的策略狀態互相獨立。"""
    if assignment.name == "centerline":
        return CenterlineStrategy()
    if assignment.name == "overtake":
        return OvertakingStrategy(overtaking_config)
    if assignment.name == "defend" and assignment.target_vehicle_id is not None:
        return DefendingStrategy(defense_config, assignment.target_vehicle_id)
    raise ValueError("unknown strategy assignment")


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
    ) -> None:
        if not vehicle_ids or len(set(vehicle_ids)) != len(vehicle_ids):
            raise ValueError("vehicle_ids must be non-empty and unique")
        if set(assignments) - set(vehicle_ids):
            raise ValueError("strategy assignment contains an unknown vehicle ID")
        factories = dict(strategy_factories or {})
        known_names = {"centerline", "overtake", "defend"} | set(factories)
        if any(assignment.name not in known_names for assignment in assignments.values()):
            raise ValueError("strategy assignment contains an unregistered strategy name")

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
                    raise ValueError("defense target must be an existing, different vehicle")
            self.assignments[vehicle_id] = assignment
            if assignment.name in factories:
                self.strategies[vehicle_id] = factories[assignment.name](vehicle_id)
            else:
                self.strategies[vehicle_id] = create_strategy(
                    assignment, overtaking_config, defense_config,
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
