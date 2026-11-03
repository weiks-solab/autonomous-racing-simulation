"""在共同預測時間點檢查車身矩形，不重新執行任何車輛策略。"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.vehicle_geometry import VehicleGeometry, rectangles_overlap
from core.vehicle_state import VehicleState
from strategy.bicycle_prediction import BicyclePrediction


@dataclass(frozen=True)
class PredictedVehicle:
    vehicle_id: str
    geometry: VehicleGeometry
    prediction: BicyclePrediction


@dataclass(frozen=True)
class PredictedCollision:
    vehicle_id: str
    time_seconds: float

    @property
    def reason(self) -> str:
        return f"predicted overlap with {self.vehicle_id} @ {self.time_seconds:.2f}s"


def first_predicted_collision(
    prediction: BicyclePrediction,
    geometry: VehicleGeometry,
    others: Sequence[PredictedVehicle],
) -> PredictedCollision | None:
    """任一共同時間點重疊就回報，包含當下與最後一個預測時間點。"""
    if not others:
        return None
    ordered = sorted(others, key=lambda other: other.vehicle_id)
    all_predictions = [prediction, *(other.prediction for other in ordered)]
    for item in all_predictions:
        if (not item.times or len(item.times) != len(item.states)
                or item.times[0] != prediction.times[0]
                or item.times[-1] != prediction.times[-1]):
            raise ValueError("collision predictions must cover the same time interval")
    times = tuple(sorted({time for item in all_predictions for time in item.times}))
    ego_states = _states_at_times(prediction, times)
    other_states = [_states_at_times(other.prediction, times) for other in ordered]
    for index, time in enumerate(times):
        for other, states in zip(ordered, other_states):
            if rectangles_overlap(ego_states[index], geometry, states[index], other.geometry):
                return PredictedCollision(other.vehicle_id, time)
    return None


def _states_at_times(
    prediction: BicyclePrediction, times: tuple[float, ...],
) -> tuple[VehicleState, ...]:
    """步長相同時直接沿用狀態；不同時插值，yaw 沿最短角度銜接。"""
    if times == prediction.times:
        return prediction.states
    values = np.array([(state.x, state.y, state.yaw, state.velocity)
                       for state in prediction.states])
    values[:, 2] = np.unwrap(values[:, 2])
    columns = [np.interp(times, prediction.times, values[:, index]) for index in range(4)]
    columns[2] = (columns[2] + np.pi) % (2.0 * np.pi) - np.pi
    return tuple(VehicleState(float(x), float(y), float(yaw), float(speed))
                 for x, y, yaw, speed in zip(*columns))
