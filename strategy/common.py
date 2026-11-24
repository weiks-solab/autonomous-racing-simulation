"""多個策略完全相同的小工具：輸入檢查、決策時機、候選橫向目標、前車與跨圈間距。

各策略的選擇規則、成本與退路仍留在各自的檔案。
"""

from dataclasses import dataclass
import math
from typing import Sequence

from core.track import Track
from core.track_geometry import TrackPosition, signed_longitudinal_gap, track_length
from core.vehicle_geometry import VehicleGeometry
from strategy.observations import VehicleObservation


def check_track_and_time(
    name: str, track: Track, time_seconds: float, last_evaluation_time: float,
) -> None:
    """策略只在封閉賽道上運作；時間須為非負有限值，且不可倒退。"""
    if not track.closed:
        raise ValueError(f"{name} requires a closed track")
    if not math.isfinite(time_seconds) or time_seconds < 0.0:
        raise ValueError("time_seconds must be non-negative and finite")
    if time_seconds < last_evaluation_time:
        raise ValueError("time_seconds must not move backwards")


def check_other_ids(
    vehicle: VehicleObservation, others: Sequence[VehicleObservation],
) -> list[str]:
    """其他車的 ID 不可重複，也不可包含自己；回傳其他車的 ID。"""
    ids = [other.vehicle_id for other in others]
    if len(set(ids)) != len(ids) or vehicle.vehicle_id in ids:
        raise ValueError("vehicle IDs must be unique; exclude self from others")
    return ids


def decision_due(
    time_seconds: float, last_evaluation_time: float, period: float, tolerance: float = 1e-9,
) -> bool:
    """距上次評估已滿一個決策週期。

    模擬時間由步長累加，例如 0.30000000000000004 − 0.2 會略小於 0.1；
    tolerance 吸收這類浮點誤差，避免決策偶爾延後一步。
    """
    return time_seconds - last_evaluation_time >= period - tolerance


def lateral_targets(
    offsets: Sequence[float], track: Track, geometry: VehicleGeometry, safety_margin: float,
) -> tuple[float, ...]:
    """中心線 0 一定列入；只保留路寬扣掉半車寬與留距後仍可到達的目標，並去除重複。"""
    maximum_d = track.width / 2.0 - geometry.width / 2.0 - safety_margin
    if maximum_d < 0.0:
        raise ValueError("track is too narrow for vehicle width and safety_margin")
    return tuple(dict.fromkeys(target for target in (0.0, *offsets) if abs(target) <= maximum_d))


def nearest_vehicle_ahead(
    vehicle: VehicleObservation, others: Sequence[VehicleObservation],
    track: Track, detection_range: float,
) -> tuple[VehicleObservation | None, float]:
    """偵測範圍內最近的前車與其間距；同距離時依車輛 ID。沒有前車時回傳 (None, 0.0)。"""
    length = track_length(track)
    ahead = [
        (signed_longitudinal_gap(vehicle.track_position, other.track_position, length),
         other.vehicle_id, other)
        for other in others
    ]
    ahead = [item for item in ahead if 0.0 < item[0] <= detection_range]
    if not ahead:
        return None, 0.0
    gap, _, target = min(ahead, key=lambda item: item[:2])
    return target, gap


@dataclass(frozen=True)
class RelativeTargetState:
    """保存同一對象的累積間距，避免領先半圈或一圈時跳回最短距離。"""

    vehicle_id: str
    vehicle_position: TrackPosition
    target_position: TrackPosition
    gap: float


def accumulated_gap(
    previous: RelativeTargetState, vehicle: VehicleObservation,
    target: VehicleObservation, track: Track,
) -> float:
    """同一對象：上次間距＋對象前進量－自車前進量，跨起終點仍連續。"""
    length = track_length(track)
    target_progress = signed_longitudinal_gap(previous.target_position, target.track_position, length)
    own_progress = signed_longitudinal_gap(previous.vehicle_position, vehicle.track_position, length)
    return previous.gap + target_progress - own_progress
