"""定速推算對手 s，再用已訓練的 GP 查 d；不執行對手策略。"""

from dataclasses import dataclass
import math
from pathlib import Path
import pickle
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from core.track import Track
from core.track_geometry import track_length, track_to_world_points
from core.vehicle_state import VehicleState
from strategy.bicycle_prediction import BicyclePrediction
from strategy.observations import VehicleObservation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MeanPredictor(Protocol):
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class GpRoute:
    """保存含輸入標準化的 Pipeline，只使用 GP 的平均 d。"""

    model: MeanPredictor

    @classmethod
    def load(cls, path: str | Path) -> "GpRoute":
        path = (PROJECT_ROOT / path).resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError("GP model must remain inside this repository")
        if not path.is_file():
            raise FileNotFoundError(f"GP 模型不存在：{path}；請修改 GP_MODEL_PATHS 或關閉 GP。")
        # 只載入自己訓練、可信任的 pickle；不在模擬時重新訓練。
        with path.open("rb") as stream:
            saved = pickle.load(stream)
        if (tuple(saved.get("feature_names", ())) != ("opponent_s",)
                or saved.get("target_spec", {}).get("name") != "opponent_current_d"
                or saved.get("target_timing") != "current"
                or saved.get("label_horizon_seconds") != 0.0):
            raise ValueError("GP 必須是同時刻的 opponent_s → opponent_current_d 模型。")
        return cls(saved["pipeline"])

    def predict_d(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        d = np.asarray(self.model.predict(s.reshape(-1, 1)), dtype=float).reshape(-1)
        if d.shape != s.shape or not np.all(np.isfinite(d)):
            raise ValueError("GP must return one finite d for each s")
        return d


def load_gp_routes(paths: Mapping[str, str | Path]) -> dict[str, GpRoute]:
    """多台對手可共用同一模型；相同檔案只載入一次。"""
    loaded: dict[Path, GpRoute] = {}
    routes = {}
    for vehicle_id, path in paths.items():
        key = (PROJECT_ROOT / path).resolve()
        if key not in loaded:
            loaded[key] = GpRoute.load(key)
        routes[vehicle_id] = loaded[key]
    return routes


def predict_gp_trajectory(
    vehicle: VehicleObservation, track: Track, route: GpRoute,
    longitudinal_speed: float, horizon: float, time_step: float,
) -> BicyclePrediction:
    """s(t)=s0+縱向速度*t，d(t)=GP(s(t))；GP 本身不預測時間。"""
    if not track.closed:
        raise ValueError("GP prediction requires a closed track")
    if any(not math.isfinite(value) or value <= 0.0 for value in (horizon, time_step)):
        raise ValueError("prediction horizon and time_step must be positive and finite")
    if not math.isfinite(longitudinal_speed) or longitudinal_speed < 0.0:
        raise ValueError("longitudinal_speed must be non-negative and finite")
    times = np.minimum(np.arange(math.ceil(horizon / time_step) + 1) * time_step, horizon)
    progress = longitudinal_speed * times
    # t=0 一律保留觀測狀態，不能讓 GP 隱藏目前已存在的重疊。
    states = [vehicle.state]
    if longitudinal_speed == 0.0:
        return BicyclePrediction(tuple(states * len(times)), tuple(progress), tuple(times))

    length = track_length(track)
    s = (vehicle.track_position.s + progress[1:]) % length
    ahead_s = (s + 0.05) % length
    # 一次查詢所有未來點及其前方點；後者用來估計 GP 路線的航向。
    d, ahead_d = np.split(route.predict_d(np.concatenate((s, ahead_s))), 2)
    points = track_to_world_points(s, d, track)
    directions = track_to_world_points(ahead_s, ahead_d, track) - points
    yaws = np.arctan2(directions[:, 1], directions[:, 0])
    states.extend(VehicleState(float(x), float(y), float(yaw), vehicle.state.velocity)
                  for (x, y), yaw in zip(points, yaws))
    return BicyclePrediction(tuple(states), tuple(progress), tuple(times))
