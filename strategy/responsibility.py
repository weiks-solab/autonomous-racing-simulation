"""避撞責任約束：依兩車相對縱向位置，決定自車要負多少避撞責任。

依使用者提供的論文截圖（Fig. 2 與其公式），對每台其他車 j、每個預測時間點 k≥1：

    g_k = ‖c_i − c_j‖² − r_col² − ℓ(Δs_k) ≥ 0,   Δs_k = s_j − s_i（對方減自車，m）
    ℓ(Δs) = 1/(1+e^b) − 1/(1+e^{aΔs+b})

- 自車在後（Δs>0）：0 ≤ ℓ ≤ 1/(1+e^b)，約束略嚴，主要由後車負責避撞。
- 並排（Δs≈0）：ℓ≈0，兩車都要保持 r_col。
- 自車在前（Δs<0）：ℓ 逐漸降到約 1/(1+e^b) − 1，約束放寬，甚至不再限制。

截圖印出的 ℓ1、ℓ2 正負號與 Fig. 2 及內文「後車負責」相反；這裡依圖與內文的語意實作。
c 是碰撞矩形中心，距離用圓形（歐氏距離），與模擬中的 SAT 矩形判定不同。

leader_soft=True 時，依決策當下的 Δs₀ 判定每一對車的角色：
- Δs₀ ≥ 0（自車在後或完全並排）：維持上述硬約束。
- Δs₀ < 0（自車在前）：不剔除，改回傳懲罰 mean_k max(0, r_col − ‖Δc_k‖)²，由策略乘上權重加入成本。

overlap_rule=True 時，依決策當下兩車中心的縱向差 Δs₀ᶜ 與碰撞框長半和 L̄ 判定：
- |Δs₀ᶜ| < L̄（縱向重疊中）：兩車都負責。
- Δs₀ᶜ ≥ L̄（對方完全在前）：自車是後車，負責。
- Δs₀ᶜ ≤ −L̄（自車完全在前，已脫離重疊）：自車不負責，只計 leader 懲罰（權重 0 即不限制）。
負責時要求 ‖Δc_k‖ ≥ r_col，不套 ℓ，避免重疊期間因預測領先而被放寬。
"""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.collision_check import PredictedCollision


@dataclass(frozen=True)
class ResponsibilityConfig:
    collision_radius: float = 0.6  # r_col：兩車中心的最小距離，m
    slope: float = 5.0             # a：並排過渡區的斜率，1/m
    offset: float = 4.5            # b：決定過渡區位置與後車的額外留距
    leader_soft: bool = False      # True：當下在前的車改為成本懲罰，後車仍為硬約束
    overlap_rule: bool = False     # True：重疊中兩車都負責，脫離後只剩後車；不使用 a、b

    def __post_init__(self) -> None:
        if not isinstance(self.leader_soft, bool):
            raise ValueError("leader_soft must be a bool")
        if not isinstance(self.overlap_rule, bool):
            raise ValueError("overlap_rule must be a bool")
        if self.leader_soft and self.overlap_rule:
            raise ValueError("leader_soft and overlap_rule cannot both be enabled")
        if not math.isfinite(self.collision_radius) or self.collision_radius <= 0.0:
            raise ValueError("collision_radius must be positive and finite")
        if not math.isfinite(self.slope) or self.slope <= 0.0:
            raise ValueError("responsibility slope must be positive and finite")
        if not math.isfinite(self.offset):
            raise ValueError("responsibility offset must be finite")


@dataclass(frozen=True)
class PairPrediction:
    """對方在自車檢查時間點上的預測狀態，與對應的 Δs（對方 s 減自車 s）。"""

    vehicle_id: str
    geometry: VehicleGeometry
    states: tuple[VehicleState, ...]
    delta_s: NDArray[np.float64]


@dataclass(frozen=True)
class ResponsibilityResult:
    violation: PredictedCollision | None  # 後車角色的硬約束；非 None 表示剔除候選
    leader_penalty: float = 0.0           # 前車角色的未加權懲罰，m²


def _inverse_logistic(values: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """1/(1+e^x)，以 tanh 計算避免大數值溢位。"""
    return 0.5 * (1.0 - np.tanh(0.5 * np.asarray(values, dtype=float)))


def responsibility_bound(delta_s: ArrayLike, config: ResponsibilityConfig) -> NDArray[np.float64]:
    """ℓ(Δs)，單位 m²。Δs>0 表示對方在前、自車在後。"""
    values = np.asarray(delta_s, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("delta_s must be finite")
    return _inverse_logistic(config.offset) - _inverse_logistic(config.slope * values + config.offset)


def body_centers(
    states: Sequence[VehicleState], geometry: VehicleGeometry,
) -> NDArray[np.float64]:
    """後軸狀態沿車頭方向移到車身中心，與碰撞矩形使用同一個中心。"""
    values = np.array([(state.x, state.y, state.yaw) for state in states], dtype=float)
    if values.size == 0:
        return np.zeros((0, 2))
    offset = geometry.rear_axle_to_center
    return np.column_stack((
        values[:, 0] + offset * np.cos(values[:, 2]),
        values[:, 1] + offset * np.sin(values[:, 2]),
    ))


def responsibility_constraint(
    own_states: Sequence[VehicleState],
    own_geometry: VehicleGeometry,
    other: PairPrediction,
    config: ResponsibilityConfig,
    use_bound: bool = True,
) -> NDArray[np.float64]:
    """回傳每個時間點的 g_k；g_k<0 表示自車違反應負的避撞責任。use_bound=False 時 ℓ=0。"""
    delta_s = np.asarray(other.delta_s, dtype=float)
    if not len(own_states) == len(other.states) == len(delta_s):
        raise ValueError("responsibility check requires the same sample times for both vehicles")
    gaps = body_centers(own_states, own_geometry) - body_centers(other.states, other.geometry)
    distance_squared = np.sum(gaps**2, axis=1)
    values = distance_squared - config.collision_radius**2
    return values - responsibility_bound(delta_s, config) if use_bound else values


def first_responsibility_violation(
    times: Sequence[float],
    own_states: Sequence[VehicleState],
    own_geometry: VehicleGeometry,
    others: Sequence[PairPrediction],
    config: ResponsibilityConfig,
    use_bound: bool = True,
) -> PredictedCollision | None:
    """依論文只檢查未來時間點 k≥1；回傳最早的違反，同時間依車輛 ID 排序。"""
    if len(times) != len(own_states):
        raise ValueError("times and own_states must have the same length")
    earliest: PredictedCollision | None = None
    for other in sorted(others, key=lambda item: item.vehicle_id):
        values = responsibility_constraint(own_states, own_geometry, other, config, use_bound)
        violated = np.flatnonzero(values[1:] < 0.0)
        if violated.size == 0:
            continue
        time = float(times[int(violated[0]) + 1])
        if earliest is None or time < earliest.time_seconds:
            earliest = PredictedCollision(other.vehicle_id, time, kind="responsibility")
    return earliest


def leader_penalty(
    own_states: Sequence[VehicleState],
    own_geometry: VehicleGeometry,
    other: PairPrediction,
    config: ResponsibilityConfig,
) -> float:
    """前車的未加權懲罰：k≥1 各點中心距離不足 r_col 的量取平方再平均，單位 m²。"""
    if not len(own_states) == len(other.states) == len(other.delta_s):
        raise ValueError("responsibility check requires the same sample times for both vehicles")
    if len(own_states) < 2:
        return 0.0
    gaps = body_centers(own_states[1:], own_geometry) - body_centers(other.states[1:], other.geometry)
    shortage = np.maximum(config.collision_radius - np.linalg.norm(gaps, axis=1), 0.0)
    return float(np.mean(shortage**2))


def evaluate_responsibility(
    times: Sequence[float],
    own_states: Sequence[VehicleState],
    own_geometry: VehicleGeometry,
    others: Sequence[PairPrediction],
    config: ResponsibilityConfig,
) -> ResponsibilityResult:
    """依設定分配責任：負責的對象硬剔除，不負責的對象只計 leader 懲罰。"""
    if config.overlap_rule:
        return _evaluate_overlap_rule(times, own_states, own_geometry, others, config)
    if not config.leader_soft:
        return ResponsibilityResult(first_responsibility_violation(
            times, own_states, own_geometry, others, config,
        ))
    # Δs₀ 為決策當下的對方 s − 自車 s；負值表示自車在前。
    leading = [other for other in others if float(other.delta_s[0]) < 0.0]
    following = [other for other in others if float(other.delta_s[0]) >= 0.0]
    return ResponsibilityResult(
        first_responsibility_violation(times, own_states, own_geometry, following, config),
        sum(leader_penalty(own_states, own_geometry, other, config) for other in leading),
    )


def collision_length(geometry: VehicleGeometry) -> float:
    """碰撞框長；未另設碰撞框時用車身長。"""
    return (geometry.collision or geometry).length


def center_gap_now(own_geometry: VehicleGeometry, other: PairPrediction) -> float:
    """決策當下兩車碰撞框中心的縱向差（對方 − 自車）；delta_s 以後軸 s 計算，需補中心偏移。"""
    return (float(other.delta_s[0]) + other.geometry.rear_axle_to_center
            - own_geometry.rear_axle_to_center)


def _evaluate_overlap_rule(
    times: Sequence[float],
    own_states: Sequence[VehicleState],
    own_geometry: VehicleGeometry,
    others: Sequence[PairPrediction],
    config: ResponsibilityConfig,
) -> ResponsibilityResult:
    """重疊中或自車在後：負責（‖Δc‖≥r_col，不套 ℓ）；自車完全在前：只計懲罰。"""
    responsible, free = [], []
    for other in others:
        half_sum = (collision_length(own_geometry) + collision_length(other.geometry)) / 2.0
        # 自車領先至少一個 L̄ 才算已脫離重疊、不再負責。
        (responsible if center_gap_now(own_geometry, other) > -half_sum else free).append(other)
    return ResponsibilityResult(
        first_responsibility_violation(
            times, own_states, own_geometry, responsible, config, use_bound=False,
        ),
        sum(leader_penalty(own_states, own_geometry, other, config) for other in free),
    )
