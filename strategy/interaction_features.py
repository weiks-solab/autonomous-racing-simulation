"""互動 GP 的輸入特徵：訓練資料與線上策略共用同一套計算，避免兩邊算法不一致。"""

from collections import deque
import math
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.track import Track
from core.track_geometry import (
    signed_longitudinal_gap, track_length, track_tangent_angle, track_to_world,
)
from strategy.observations import VehicleObservation

# 順序即模型輸入順序；最後一欄是後車在當下採用的目標 d。
INTERACTION_FEATURES = (
    "gap", "rear_d", "rear_heading_error", "rear_speed",
    "front_d", "front_heading_error", "front_speed", "front_curvature",
    "rear_dd_early", "front_dd_early", "rear_dd_late", "front_dd_late", "d_target",
)

# 論文格式（Lee et al., RA-L 2024）：前 10 欄依論文 z_k 的順序（前車在前、後車在後、前車位置與前方 ζ、2ζ 的曲率）；
# 後 4 欄沿用 v2 的歷史 d 變化，取代論文的 M=10 序列。不含 d_target。
PAPER_STATE_FEATURES = (
    "gap", "front_d", "front_heading_error", "front_speed",
    "rear_d", "rear_heading_error", "rear_speed",
    "front_curvature_0", "front_curvature_1", "front_curvature_2",
)
HISTORY_DELTA_FEATURES = ("rear_dd_early", "front_dd_early", "rear_dd_late", "front_dd_late")
PAPER_FEATURES = PAPER_STATE_FEATURES + HISTORY_DELTA_FEATURES
PAPER_FEATURE_UNITS = ("m", "m", "rad", "m/s", "m", "rad", "m/s", "1/m", "1/m", "1/m", "m", "m", "m", "m")
# 前車一步的狀態變化，依論文 y_k 的順序。
PAPER_TARGETS = ("front_delta_s", "front_delta_d", "front_delta_heading_error", "front_delta_speed")
PAPER_TARGET_UNITS = ("m", "m", "rad", "m/s")
CURVATURE_LOOKAHEAD = 1.0  # 論文的 ζ，m；論文未給數值，尚未調參


def centerline_curvature(track: Track, s: float, spacing: float = 0.5) -> float:
    """用中心線前後各 0.5 m 的三點估計有號曲率，左彎為正。"""
    a, b, c = (np.asarray(track_to_world(s + offset, 0.0, track))
               for offset in (-spacing, 0.0, spacing))
    u, v = b - a, c - a
    denominator = np.linalg.norm(u) * np.linalg.norm(v) * np.linalg.norm(c - b)
    return 0.0 if denominator < 1e-12 else float(2 * (u[0] * v[1] - u[1] * v[0]) / denominator)


def heading_error(observation: VehicleObservation, track: Track) -> float:
    """車頭角減所在中心線線段的切線角，包裝到 [-π, π]，左轉為正。"""
    tangent = track_tangent_angle(observation.track_position.s, track)
    return math.remainder(observation.state.yaw - tangent, 2.0 * math.pi)


def history_state_features(
    rear: VehicleObservation, front: VehicleObservation,
    rear_past_d: tuple[float, float], front_past_d: tuple[float, float], track: Track,
) -> tuple[float, ...]:
    """前 12 欄（不含 d_target）。past_d 依序為 t−2 段與 t−1 段的 d。

    歷史四欄依序為後車前段、前車前段、後車後段、前車後段的 d 變化。
    """
    gap = signed_longitudinal_gap(rear.track_position, front.track_position, track_length(track))
    rear_d = (*rear_past_d, rear.track_position.d)
    front_d = (*front_past_d, front.track_position.d)
    return (gap, rear.track_position.d, heading_error(rear, track), rear.state.velocity,
            front.track_position.d, heading_error(front, track), front.state.velocity,
            centerline_curvature(track, front.track_position.s),
            rear_d[1] - rear_d[0], front_d[1] - front_d[0],
            rear_d[2] - rear_d[1], front_d[2] - front_d[1])


def paper_state_features(
    rear: VehicleObservation, front: VehicleObservation,
    rear_past_d: tuple[float, float], front_past_d: tuple[float, float], track: Track,
    zeta: float = CURVATURE_LOOKAHEAD,
) -> tuple[float, ...]:
    """PAPER_FEATURES 的 14 欄。past_d 依序為 t−2 段與 t−1 段的 d，歷史 4 欄算法與 history_state_features 相同。"""
    if not math.isfinite(zeta) or zeta <= 0.0:
        raise ValueError("curvature lookahead must be positive and finite")
    gap = signed_longitudinal_gap(rear.track_position, front.track_position, track_length(track))
    s = front.track_position.s
    rear_d = (*rear_past_d, rear.track_position.d)
    front_d = (*front_past_d, front.track_position.d)
    return (gap, front.track_position.d, heading_error(front, track), front.state.velocity,
            rear.track_position.d, heading_error(rear, track), rear.state.velocity,
            *(centerline_curvature(track, s + k * zeta) for k in range(3)),
            rear_d[1] - rear_d[0], front_d[1] - front_d[0],
            rear_d[2] - rear_d[1], front_d[2] - front_d[1])


def paper_feature_rows(
    rear: VehicleObservation, rear_past_d: tuple[float, float],
    front_s: ArrayLike, front_d: ArrayLike, front_heading_error: ArrayLike, front_speed: ArrayLike,
    front_past_d: ArrayLike, track: Track, zeta: float = CURVATURE_LOOKAHEAD,
) -> NDArray[np.float64]:
    """推演用：一次產生 Q 列 PAPER_FEATURES，前車以 (s, d, 航向誤差, 車速) 給定，不經投影。

    front_past_d 為 (Q, 2)，依序是 t−2 段與 t−1 段的 d。其餘算法與 paper_state_features 相同，
    同一組狀態兩者逐欄一致（見測試）。
    """
    if not math.isfinite(zeta) or zeta <= 0.0:
        raise ValueError("curvature lookahead must be positive and finite")
    s, d, psi, speed = (np.asarray(values, dtype=float).reshape(-1)
                        for values in (front_s, front_d, front_heading_error, front_speed))
    past = np.asarray(front_past_d, dtype=float).reshape(-1, 2)
    count = len(s)
    if not len(d) == len(psi) == len(speed) == len(past) == count:
        raise ValueError("front sample arrays must have the same length")
    length = track_length(track)
    half = length / 2.0
    # 與 signed_longitudinal_gap 同式：結果落在 [-L/2, L/2)。
    gap = (s - rear.track_position.s + half) % length - half
    curvature = np.array([[centerline_curvature(track, float(value) + k * zeta) for k in range(3)]
                          for value in s], dtype=float).reshape(count, 3)
    rear_d = (*rear_past_d, rear.track_position.d)

    def column(value: float) -> NDArray[np.float64]:
        return np.full(count, float(value))

    return np.column_stack((
        gap, d, psi, speed,
        column(rear.track_position.d), column(heading_error(rear, track)), column(rear.state.velocity),
        curvature,
        column(rear_d[1] - rear_d[0]), past[:, 1] - past[:, 0],
        column(rear_d[2] - rear_d[1]), d - past[:, 1],
    ))


def front_state_delta(
    now: VehicleObservation, later: VehicleObservation, track: Track,
) -> tuple[float, float, float, float]:
    """同一台車 now→later 的 PAPER_TARGETS：Δs 取最短有號距離（跨圈正確），航向誤差差值包裝到 [-π, π]。"""
    return (signed_longitudinal_gap(now.track_position, later.track_position, track_length(track)),
            later.track_position.d - now.track_position.d,
            math.remainder(heading_error(later, track) - heading_error(now, track), 2.0 * math.pi),
            later.state.velocity - now.state.velocity)


class LateralHistory:
    """各車在各時間點的 d；只保留最近 window 秒，查詢時可在兩筆紀錄間線性內插。"""

    def __init__(self, window: float, tolerance: float = 1e-6) -> None:
        if not math.isfinite(window) or window <= 0.0:
            raise ValueError("history window must be positive and finite")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("history tolerance must be non-negative and finite")
        self.window = window
        self.tolerance = tolerance
        self._records: dict[str, deque[tuple[float, float]]] = {}

    def record(self, time_seconds: float, observations: Iterable[VehicleObservation]) -> None:
        if not math.isfinite(time_seconds):
            raise ValueError("history time must be finite")
        for observation in observations:
            records = self._records.setdefault(observation.vehicle_id, deque())
            if records and time_seconds < records[-1][0] - self.tolerance:
                raise ValueError("history time must not go backwards")
            entry = (time_seconds, observation.track_position.d)
            if records and abs(records[-1][0] - time_seconds) <= self.tolerance:
                records[-1] = entry   # 同一時刻重複記錄時以最新值為準
            else:
                records.append(entry)
            while records and records[0][0] < time_seconds - self.window - self.tolerance:
                records.popleft()

    def d_at(self, vehicle_id: str, time_seconds: float) -> float | None:
        """查不到（早於最舊紀錄、晚於最新紀錄或沒有此車）時回傳 None。"""
        records = self._records.get(vehicle_id)
        if not records:
            return None
        previous = None
        for time, d in records:
            if abs(time - time_seconds) <= self.tolerance:
                return d
            if time > time_seconds:
                if previous is None:
                    return None
                ratio = (time_seconds - previous[0]) / (time - previous[0])
                return previous[1] + ratio * (d - previous[1])
            previous = (time, d)
        return None
