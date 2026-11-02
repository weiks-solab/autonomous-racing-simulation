"""所有車輛策略共用的輸入方式與輸出格式。"""

from dataclasses import dataclass
from typing import Protocol, Sequence

from core.track import Track
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver


@dataclass(frozen=True)
class StrategyDecision:
    """只描述想去的位置，方向盤與加速度仍由 Pure Pursuit 計算。"""

    mode: str
    target_d: float = 0.0
    target_vehicle_id: str | None = None
    maneuver: LateralManeuver | None = None
    reason: str = ""
    details: tuple[str, ...] = ()

    @property
    def display_text(self) -> str:
        lines = [self.mode.upper(), f"target_d={self.target_d:+.2f}"]
        if self.target_vehicle_id is not None:
            lines.append(f"target={self.target_vehicle_id}")
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
