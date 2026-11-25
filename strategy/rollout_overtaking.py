"""沿自車上一步的預測軌跡，用互動 GP 逐步推演對手軌跡，再依這條對手軌跡選候選。

每次決策：
1. 取上一次決策選中候選的預測軌跡（不從目前狀態重算），時間對齊到現在。
2. 沿這條軌跡每 rollout_step 秒查一次 GP：自車狀態取軌跡上的值，對手用推演到該節點的位置，
   特徵的歷史欄位早於現在用實際紀錄、晚於現在用軌跡或推演值，d_target 為上一步選中的目標。
   GP 預測 0.5 s 的 Δd，換算成每步增量 Δd×step/0.5；σ 以相同比例累加。
3. 所有候選共用這條對手軌跡；候選、成本、避撞責任與 ±kσ 邊界檢查沿用 interaction_overtake。

沒有上一步軌跡、第一個節點歷史不足或超出 GP 訓練範圍時，退回定速等 d（與 relative_overtake 相同）；
之後的節點超出範圍時，其餘時段維持最後推演到的 d。
"""

from dataclasses import dataclass, field
import math
from typing import Mapping

import numpy as np

from core.track import Track
from core.track_geometry import track_length, track_tangent_angle, track_to_world, world_to_track
from core.validation import require_nonnegative, require_positive
from core.vehicle_state import VehicleState
from strategy.bicycle_prediction import BicyclePrediction, BicyclePredictionConfig, prediction_config_for
from strategy.collision_check import states_at_times
from strategy.interaction_features import history_state_features
from strategy.interaction_gp import InteractionGpModel, predict_profile_trajectory
from strategy.interaction_overtaking import (
    GpQuery, InteractionOvertakingConfig, InteractionOvertakingStrategy, TRAINED_OFFSETS,
)
from strategy.observations import VehicleObservation, create_vehicle_observation
from strategy.relative_overtaking import RelativeCandidate, RelativeOvertakingConfig, RelativeOvertakingStrategy

TIME_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RolloutOvertakingConfig:
    # 候選只用 GP 訓練過的目標；橫移距離 3.0 m 與訓練資料相同。
    relative: RelativeOvertakingConfig = field(default_factory=lambda: RelativeOvertakingConfig(
        candidate_offsets=TRAINED_OFFSETS, transition_length=3.0))
    rollout_step: float = 0.1     # 推演步長，s；不可超過 GP 預測時域
    band_sigma: float = 2.0       # 邊界軌跡＝平均 ± band_sigma·累加 σ；0 表示只用平均
    history_margin: float = 0.1   # 歷史多保留的秒數

    def __post_init__(self) -> None:
        if not isinstance(self.relative, RelativeOvertakingConfig):
            raise ValueError("RolloutOvertakingConfig.relative must be a RelativeOvertakingConfig")
        require_positive(self, "rollout_step", "history_margin")
        require_nonnegative(self, "band_sigma")

    @property
    def transition_length(self) -> float:
        """供參考路徑長度規則讀取。"""
        return self.relative.transition_length


@dataclass(frozen=True)
class PreviousPlan:
    """上一次決策選中候選的預測軌跡；prediction.times 從 time_seconds 起算。"""

    time_seconds: float
    prediction: BicyclePrediction
    target_d: float


@dataclass(frozen=True)
class RolloutResult:
    """節點時間（相對現在，含時域終點）、推演的對手 d、累加 σ，以及各查詢節點的原因（空字串＝已查詢）。"""

    node_times: tuple[float, ...]
    offsets: tuple[float, ...]
    sigmas: tuple[float, ...]
    reasons: tuple[str, ...]


def plan_state_at(plan: BicyclePrediction, time: float, track: Track) -> VehicleState:
    """取軌跡在 time 的狀態；超過末端時，以末端速度沿賽道、維持末端 d 延伸。"""
    end = plan.times[-1]
    if time <= end + TIME_TOLERANCE:
        return states_at_times(plan, (min(max(time, 0.0), end),))[0]
    last = plan.states[-1]
    position = world_to_track(last, track)
    s = position.s + max(last.velocity, 0.0) * (time - end)
    if track.closed:
        s %= track_length(track)
    x, y = track_to_world(s, position.d, track)
    return VehicleState(float(x), float(y), track_tangent_angle(s, track), last.velocity)


def plan_d_at(plan: BicyclePrediction, time: float) -> float:
    """軌跡在 time 的 d；超過末端時維持末端值。"""
    return float(np.interp(time, plan.times, plan.offsets))


class RolloutOvertakingStrategy(InteractionOvertakingStrategy):
    """每台車各自保存上一步軌跡、觀測歷史與推演結果；GP 模型可共用。"""

    def __init__(
        self,
        config: RolloutOvertakingConfig,
        prediction_config: BicyclePredictionConfig,
        other_prediction_configs: Mapping[str, BicyclePredictionConfig],
        model: InteractionGpModel | None = None,
    ) -> None:
        # 所有候選共用一份對手軌跡，等同 interaction_overtake 的 previous 模式。
        super().__init__(InteractionOvertakingConfig(config.relative, "previous", config.band_sigma,
                                                     config.history_margin),
                         prediction_config, other_prediction_configs, model)
        self.rollout_config = config
        if model is not None and config.rollout_step > model.horizon_seconds + TIME_TOLERANCE:
            raise ValueError("rollout_step must not exceed the GP prediction horizon")
        self.previous_plan: PreviousPlan | None = None
        self.last_plan: PreviousPlan | None = None
        self.last_rollout: RolloutResult | None = None

    def _predict_other(self, other: VehicleObservation, track: Track, time_seconds: float) -> BicyclePrediction:
        # 退回用的定速等 d 預測（relative_overtake 原本的作法）。
        fallback = RelativeOvertakingStrategy._predict_other(self, other, track, time_seconds)
        target = self.target_state
        if target is None or other.vehicle_id != target.vehicle_id or self._vehicle is None:
            return fallback
        plan = self.previous_plan
        self.last_plan, self.last_rollout = plan, None
        if self.model is None or plan is None:
            d_target = self.decision.target_d if plan is None else plan.target_d
            reason = "no_model" if self.model is None else "no_previous_plan"
            self.gp_queries = {d_target: GpQuery(d_target, reason)}
            return fallback
        query = self._rollout(self._vehicle, other, track, time_seconds, plan)
        self.gp_queries = {plan.target_d: query}
        return query.prediction if query.active else fallback

    def _rollout(
        self, vehicle: VehicleObservation, front: VehicleObservation, track: Track,
        now: float, plan: PreviousPlan,
    ) -> GpQuery:
        model = self.model
        segment, gp_horizon = model.segment_seconds, model.horizon_seconds
        step, horizon = self.rollout_config.rollout_step, self.config.prediction_horizon
        nodes = np.minimum(np.arange(math.ceil(horizon / step - TIME_TOLERANCE) + 1) * step, horizon)
        lag = now - plan.time_seconds   # 上一步軌跡上對應「現在」的時間
        length = track_length(track)
        speed = max(0.0, front.state.velocity)
        offsets, sigmas, reasons = [front.track_position.d], [0.0], []
        hold = False
        for index, tau in enumerate(nodes[:-1]):
            increment = (nodes[index + 1] - tau) / gp_horizon
            if hold:
                offsets.append(offsets[-1])
                sigmas.append(sigmas[-1])
                continue
            # 歷史欄位：早於現在用實際紀錄，晚於現在用上一步軌跡（自車）或已推演的值（對手）。
            ego_past, front_past = [], []
            for k in (2, 1):
                when = tau - k * segment
                if when < -TIME_TOLERANCE:
                    ego_past.append(self.history.d_at(vehicle.vehicle_id, now + when))
                    front_past.append(self.history.d_at(front.vehicle_id, now + when))
                else:
                    ego_past.append(plan_d_at(plan.prediction, lag + when))
                    front_past.append(float(np.interp(when, nodes[:index + 1], offsets[:index + 1])))
            if any(value is None for value in (*ego_past, *front_past)):
                reason = "no_history"
            else:
                ego = create_vehicle_observation(
                    vehicle.vehicle_id, plan_state_at(plan.prediction, lag + tau, track), vehicle.geometry, track)
                opponent = front if index == 0 else self._front_observation(
                    front, track, tau, nodes[index - 1], offsets[index], offsets[index - 1], speed, length)
                features = (*history_state_features(ego, opponent, tuple(ego_past), tuple(front_past), track),
                            plan.target_d)
                reason = model.check(features)
            if reason is not None:
                if index == 0:
                    label = reason if reason == "no_history" else f"out_of_range:{reason}"
                    return GpQuery(plan.target_d, label)
                # 之後的節點超出範圍：其餘時段維持最後推演到的 d。
                reasons.append(reason)
                hold = True
                offsets.append(offsets[-1])
                sigmas.append(sigmas[-1])
                continue
            mean, std = model.predict(np.asarray(features, dtype=float))
            offsets.append(offsets[-1] + float(mean[0]) * increment)
            sigmas.append(sigmas[-1] + float(std[0]) * increment)
            reasons.append("")
        self.last_rollout = RolloutResult(tuple(float(t) for t in nodes), tuple(offsets), tuple(sigmas),
                                          tuple(reasons))
        return self._query_from_rollout(front, track, plan.target_d, nodes, offsets, sigmas, speed)

    @staticmethod
    def _front_observation(
        front: VehicleObservation, track: Track, tau: float, previous_tau: float,
        d: float, previous_d: float, speed: float, length: float,
    ) -> VehicleObservation:
        """對手在節點的觀測：s 定速前進，d 為推演值，航向取上一節點到此節點的方向。"""
        s = (front.track_position.s + speed * tau) % length
        previous_s = (front.track_position.s + speed * previous_tau) % length
        x, y = track_to_world(s, d, track)
        px, py = track_to_world(previous_s, previous_d, track)
        yaw = math.atan2(y - py, x - px) if speed > 0.0 else front.state.yaw
        return create_vehicle_observation(front.vehicle_id, VehicleState(float(x), float(y), yaw, front.state.velocity),
                                          front.geometry, track)

    def _query_from_rollout(
        self, front: VehicleObservation, track: Track, d_target: float,
        nodes: np.ndarray, offsets: list[float], sigmas: list[float], speed: float,
    ) -> GpQuery:
        """節點間線性內插到預測格點，組成平均軌跡與 ±kσ 邊界。"""
        config = prediction_config_for(self.other_prediction_configs, front.vehicle_id)
        horizon, dt = self.config.prediction_horizon, config.model.time_step
        times = np.minimum(np.arange(math.ceil(horizon / dt) + 1) * dt, horizon)
        mean_d = np.interp(times, nodes, offsets)
        band = self.rollout_config.band_sigma * np.interp(times, nodes, sigmas)

        def build(values: np.ndarray) -> BicyclePrediction:
            return predict_profile_trajectory(front, track, values, speed, horizon, dt)

        bands = () if not np.any(band > 0.0) else (build(mean_d + band), build(mean_d - band))
        gp_horizon = self.model.horizon_seconds
        # 快照沿用 0.5 s 的 Δd 與 σ，與單次查詢的數字可直接比較。
        return GpQuery(d_target, mean=float(np.interp(gp_horizon, nodes, offsets) - offsets[0]),
                       std=float(np.interp(gp_horizon, nodes, sigmas)), prediction=build(mean_d), bands=bands)

    def _snapshot(self, vehicle, time_seconds, best):
        snapshot = super()._snapshot(vehicle, time_seconds, best)
        # 保存本步選中候選的預測軌跡，下一步沿它推演對手。
        if best is not None:
            self.previous_plan = PreviousPlan(time_seconds, best.prediction, best.maneuver.target_d)
        return snapshot

    def _details(self, no_target: bool, best: RelativeCandidate) -> tuple[str, ...]:
        details = RelativeOvertakingStrategy._details(self, no_target, best)
        query = self.last_gp_query
        if self.model is None or no_target or query is None:
            return details
        if not query.active:
            return (*details, f"gp=off({query.reason})")
        rollout = self.last_rollout
        text = (f"rollout d_target={query.d_target:+.2f} d(+0.5)={rollout.offsets[0] + query.mean:+.2f} "
                f"d(+{rollout.node_times[-1]:g})={rollout.offsets[-1]:+.2f} std_end={rollout.sigmas[-1]:.3f}")
        held = next((index for index, reason in enumerate(rollout.reasons) if reason), None)
        if held is not None:
            text += f" hold@{rollout.node_times[held]:.1f}s({rollout.reasons[held]})"
        return (*details, text)
