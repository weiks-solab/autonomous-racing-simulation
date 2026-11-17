"""所有車輛策略共用的輸入方式與輸出格式。"""

from dataclasses import dataclass
from typing import Protocol, Sequence

from core.pure_pursuit import validate_speed_limit
from core.track import Track
from core.track_geometry import TrackPosition
from core.vehicle_state import VehicleState
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver


@dataclass(frozen=True)
class CandidatePath:
    """候選路徑的顯示資料，不參與控制或其他車的觀測。"""

    maneuver: LateralManeuver
    selected: bool = False
    name: str = ""
    feasible: bool = True
    reason: str = ""
    race_cost: float | None = None
    blocking_cost: float | None = None
    cost_terms: tuple[tuple[str, float], ...] = ()
    predicted_states: tuple[VehicleState, ...] = ()
    speed_limit: float | None = None

    def __post_init__(self) -> None:
        validate_speed_limit(self.speed_limit)


@dataclass(frozen=True)
class GpPredictionSnapshot:
    """實際用於碰撞檢查的對手 GP 預測，只供視覺化讀取。"""

    vehicle_id: str
    states: tuple[VehicleState, ...]
    times: tuple[float, ...]


@dataclass(frozen=True)
class CandidateSnapshot:
    """保留評估位置與結果；候選可提供取樣距離或完整預測狀態。"""

    evaluated_at: float
    position: TrackPosition
    distances: tuple[float, ...]
    candidates: tuple[CandidatePath, ...]
    gp_predictions: tuple[GpPredictionSnapshot, ...] = ()


@dataclass(frozen=True)
class StrategyDecision:
    """描述路徑與可選速度上限，方向盤與加速度仍由 Pure Pursuit 計算。"""

    mode: str
    target_d: float = 0.0
    target_vehicle_id: str | None = None
    maneuver: LateralManeuver | None = None
    reason: str = ""
    details: tuple[str, ...] = ()
    candidate_snapshot: CandidateSnapshot | None = None
    speed_limit: float | None = None  # m/s，None 沿用巡航目標，0 逐步減速至停止。

    def __post_init__(self) -> None:
        validate_speed_limit(self.speed_limit)

    @property
    def display_text(self) -> str:
        lines = [self.mode.upper(), f"target_d={self.target_d:+.2f}"]
        if self.target_vehicle_id is not None:
            lines.append(f"target={self.target_vehicle_id}")
        if self.speed_limit is not None:
            lines.append(f"speed_limit={self.speed_limit:.2f} m/s")
        return "\n".join((*lines, *self.details))


class VehicleStrategy(Protocol):
    """新策略只需提供相同的 update()，不必接收整個模擬器。"""

    def update(
        self,
        vehicle: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> StrategyDecision:
        ...
