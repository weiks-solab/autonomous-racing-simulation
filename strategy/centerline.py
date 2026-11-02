"""沒有攻防行為，持續跟隨共同中心線。"""

from typing import Sequence

from core.track import Track
from strategy.base import StrategyDecision
from strategy.observations import VehicleObservation


class CenterlineStrategy:
    def update(
        self,
        vehicle: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> StrategyDecision:
        # 沒有臨時橫移路徑時，模擬器就使用共同中心線。
        return StrategyDecision(mode="centerline", reason="following shared centerline")
