"""把策略名稱轉成各車獨立的策略，統一呼叫既有攻防演算法。"""

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from core.track import Track
from strategy.base import StrategyDecision, VehicleStrategy
from strategy.centerline import CenterlineStrategy
from strategy.defending import CostBasedDefenseStrategy, DefenseConfig
from strategy.observations import VehicleObservation
from strategy.overtaking import OvertakingConfig, RuleBasedOvertakingStrategy


@dataclass(frozen=True)
class StrategyAssignment:
    """name 選策略；只有 defend 需要 target_vehicle_id 指定防守對象。"""

    name: str
    target_vehicle_id: str | None = None

    def __post_init__(self) -> None:
        if self.name not in ("centerline", "overtake", "defend"):
            raise ValueError("strategy must be centerline, overtake or defend")
        if self.name == "defend":
            if not isinstance(self.target_vehicle_id, str) or not self.target_vehicle_id.strip():
                raise ValueError("defend requires a target_vehicle_id")
        elif self.target_vehicle_id is not None:
            raise ValueError("only defend accepts target_vehicle_id")


class OvertakingStrategy:
    """把既有超車策略的輸出轉成共用格式，不改變超車邏輯。"""

    def __init__(self, config: OvertakingConfig) -> None:
        self.algorithm = RuleBasedOvertakingStrategy(config)

    def update(
        self,
        vehicle: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> StrategyDecision:
        result = self.algorithm.update(vehicle, others, track)
        details = () if result.side is None else (f"side={result.side.name}",)
        return StrategyDecision(
            mode=result.mode.value,
            target_d=result.target_d,
            target_vehicle_id=result.target_vehicle_id,
            maneuver=result.maneuver,
            reason=result.reason,
            details=details,
        )


class DefendingStrategy:
    """把既有防守策略的輸出轉成同一格式。"""

    def __init__(self, config: DefenseConfig, target_vehicle_id: str) -> None:
        self.algorithm = CostBasedDefenseStrategy(config, target_vehicle_id)

    def update(
        self,
        vehicle: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> StrategyDecision:
        result = self.algorithm.update(vehicle, others, track, time_seconds)
        return StrategyDecision(
            mode=result.mode.value,
            target_d=result.target_d,
            target_vehicle_id=result.target_vehicle_id,
            maneuver=result.maneuver,
            reason=result.reason,
            details=(f"race={result.race_cost:.2f} block={result.blocking_cost:.2f}",),
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
    ) -> None:
        if not vehicle_ids or len(set(vehicle_ids)) != len(vehicle_ids):
            raise ValueError("vehicle_ids must be non-empty and unique")
        if set(assignments) - set(vehicle_ids):
            raise ValueError("strategy assignment contains an unknown vehicle ID")

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
            self.strategies[vehicle_id] = create_strategy(
                assignment, overtaking_config, defense_config
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

        shared = dict(observations)
        decisions: dict[str, StrategyDecision] = {}
        # 保留既有資訊順序：一般／進攻車先決策，防守車再讀取新的進攻路徑。
        # 同一階段內都讀相同快照，避免多車結果受字典排列順序影響。
        for defense_phase in (False, True):
            phase_decisions: dict[str, StrategyDecision] = {}
            for vehicle_id, strategy in self.strategies.items():
                is_defender = self.assignments[vehicle_id].name == "defend"
                if is_defender != defense_phase:
                    continue
                others = tuple(
                    other for other_id, other in shared.items() if other_id != vehicle_id
                )
                phase_decisions[vehicle_id] = strategy.update(
                    shared[vehicle_id], others, track, time_seconds
                )
            for vehicle_id, decision in phase_decisions.items():
                shared[vehicle_id] = replace(shared[vehicle_id], maneuver=decision.maneuver)
            decisions.update(phase_decisions)
        return decisions
