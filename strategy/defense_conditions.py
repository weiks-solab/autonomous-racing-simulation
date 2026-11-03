"""可替換的防守候選條件：只用幾何預測，不執行 bicycle rollout。"""

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from core.track import Track
from core.vehicle_geometry import VehicleGeometry, rectangles_overlap
from core.vehicle_state import VehicleState
from strategy.collision_check import PredictedCollision
from strategy.observations import VehicleObservation
from strategy.prediction import predict_vehicle_state


@dataclass(frozen=True)
class RearPrediction:
    vehicle_id: str
    geometry: VehicleGeometry
    times: tuple[float, ...]
    states: tuple[VehicleState, ...]


def predict_rear(
    rear: VehicleObservation, track: Track, horizon: float, step: float,
) -> RearPrediction:
    """指定後車維持當下 d 與縱向速度，忽略其已選路徑。"""
    snapshot = replace(rear, maneuver=None)
    count = math.ceil(horizon / step)
    times = tuple(float(time) for time in np.linspace(0.0, horizon, count + 1))
    return RearPrediction(
        rear.vehicle_id, rear.geometry, times,
        tuple(predict_vehicle_state(snapshot, time, track) for time in times),
    )


def has_predicted_collision(
    defender: VehicleObservation,
    candidate_states: Sequence[VehicleState],
    rear: RearPrediction,
) -> PredictedCollision | None:
    """None 表示通過；回傳碰撞資訊時剔除候選。可單獨替換此函式。"""
    if len(candidate_states) != len(rear.times) or len(rear.states) != len(rear.times):
        raise ValueError("defender and rear predictions must share the same sample times")
    for time, own_state, rear_state in zip(rear.times, candidate_states, rear.states):
        if rectangles_overlap(own_state, defender.geometry, rear_state, rear.geometry):
            return PredictedCollision(rear.vehicle_id, time)
    return None
