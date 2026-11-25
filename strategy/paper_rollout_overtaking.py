"""依論文（Lee et al., RA-L 2024）的方式使用一步 GP 預測前車，接到超車候選的選擇。

每次決策（只對目標前車）：
1. 上一步計畫：上一次決策選中候選的預測軌跡，時間對齊到現在（同 rollout_overtake）。
2. 取樣推演：Q 個樣本從前車目前狀態出發，每 0.1 s 以 14 維特徵查 4 個 GP，
   各自從 N(μ, σ²) 抽 [Δs, Δd, Δe_ψ, Δv] 更新 s、d、航向誤差與車速。
   - 自車：第 0 步用實際觀測，之後取上一步計畫在對應時刻的狀態。
   - 歷史欄位（τ−0.2、τ−0.4 s 的 d）：不晚於現在用實際紀錄；晚於現在時，自車用計畫值，前車用該樣本自己的推演值。
3. 預測分佈：各節點的樣本平均縱向進度與 d 組成前車平均軌跡，供成本與避撞檢查；
   避撞責任的圓改成橢圓，沿賽道半軸 r_col+kσ_s、橫向半軸 r_col+kσ_d。
4. 候選與成本沿用 relative_overtake。所有候選共用同一份前車預測，終點間距成本等同比較自車進度
   （論文是最大化進度的 MPCC，這裡沿用候選＋Pure Pursuit）。

退回：沒有模型、沒有上一步計畫、歷史不足、非封閉賽道或第 0 步超出訓練範圍時，整段改用定速等 d 預測，
決策與 relative_overtake 相同；之後某個樣本超出範圍時，只有該樣本改為等速等 d 前進。
"""

from dataclasses import dataclass, field, replace
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from core.track import Track
from core.track_geometry import track_length, track_tangent_angle
from core.validation import require_nonnegative, require_positive
from strategy.base import CandidateSnapshot, GpPredictionSnapshot, StrategyDecision
from strategy.bicycle_prediction import BicyclePrediction, BicyclePredictionConfig, prediction_config_for
from strategy.interaction_features import LateralHistory, heading_error, paper_feature_rows
from strategy.interaction_gp import predict_profile_trajectory
from strategy.observations import VehicleObservation, create_vehicle_observation
from strategy.paper_gp import PaperGpModel
from strategy.relative_overtaking import (
    RelativeCandidate, RelativeOvertakingConfig, RelativeOvertakingStrategy,
)
from strategy.responsibility import PositionUncertainty, ResponsibilityConfig
from strategy.rollout_overtaking import PreviousPlan, plan_d_at, plan_state_at

STEP_TOLERANCE = 1e-9


def _default_relative() -> RelativeOvertakingConfig:
    # 橢圓放大的是避撞責任的 r_col，因此預設啟用責任約束（與 main 相同的 overlap_rule）。
    return RelativeOvertakingConfig(responsibility=ResponsibilityConfig(overlap_rule=True))


@dataclass(frozen=True)
class PaperRolloutOvertakingConfig:
    relative: RelativeOvertakingConfig = field(default_factory=_default_relative)
    samples: int = 25            # Q：每次決策的前車樣本數（論文 25）
    ellipse_sigma: float = 2.0   # k：避撞橢圓半軸＝r_col＋k·σ（沿賽道、橫向）；0 表示只用平均軌跡
    seed: int = 0                # 抽樣亂數種子；每台車的策略各自持有一個亂數產生器
    history_margin: float = 0.1  # 歷史多保留的秒數

    def __post_init__(self) -> None:
        if not isinstance(self.relative, RelativeOvertakingConfig):
            raise ValueError("PaperRolloutOvertakingConfig.relative must be a RelativeOvertakingConfig")
        for name, low, high in (("samples", 1, None), ("seed", 0, 2**32 - 1)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, int) or value < low
                    or (high is not None and value > high)):
                raise ValueError(f"PaperRolloutOvertakingConfig.{name} must be an integer >= {low}"
                                 + ("" if high is None else f" and <= {high}") + f" (got {value!r})")
        require_nonnegative(self, "ellipse_sigma")
        require_positive(self, "history_margin")

    @property
    def transition_length(self) -> float:
        """供參考路徑長度規則讀取。"""
        return self.relative.transition_length


def _sample_std(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1])


@dataclass(frozen=True)
class PaperRollout:
    """一次取樣推演。node_times 相對現在（含時域終點）；progress、offsets 的形狀為 (Q, 節點數)。

    progress 是各樣本相對現在的縱向進度（m）；exit_steps 是樣本改為等速等 d 的步（−1 表示全程用 GP），
    exit_reasons 是當時超出範圍的欄位（空字串表示沒有）。
    """

    node_times: tuple[float, ...]
    progress: NDArray[np.float64]
    offsets: NDArray[np.float64]
    exit_steps: tuple[int, ...]
    exit_reasons: tuple[str, ...]

    @property
    def mean_progress(self) -> NDArray[np.float64]:
        return self.progress.mean(axis=0)

    @property
    def mean_d(self) -> NDArray[np.float64]:
        return self.offsets.mean(axis=0)

    @property
    def std_s(self) -> NDArray[np.float64]:
        return _sample_std(self.progress)

    @property
    def std_d(self) -> NDArray[np.float64]:
        return _sample_std(self.offsets)

    @property
    def held_count(self) -> int:
        return sum(1 for reason in self.exit_reasons if reason)


class PaperRolloutOvertakingStrategy(RelativeOvertakingStrategy):
    """每台車各自保存觀測歷史、上一步計畫、亂數產生器與推演結果；GP 模型可共用。"""

    def __init__(
        self,
        config: PaperRolloutOvertakingConfig,
        prediction_config: BicyclePredictionConfig,
        other_prediction_configs: Mapping[str, BicyclePredictionConfig],
        model: PaperGpModel | None = None,
    ) -> None:
        # 在建立策略時才檢查（而非設定檔），關閉 main 的責任約束時，未指派此策略的場景仍可執行。
        if config.ellipse_sigma > 0.0 and config.relative.responsibility is None:
            raise ValueError("paper_rollout_overtake: ellipse_sigma > 0 widens the responsibility radius r_col; "
                             "enable relative.responsibility or set ellipse_sigma=0")
        super().__init__(config.relative, prediction_config, other_prediction_configs)
        self.paper_config = config
        self.model = model
        segment = 0.2
        if model is not None:
            segment = model.segment_seconds
            # 推演節點與歷史欄位都落在 GP 一步的整數倍上，不需要內插、也不縮放一步的預測。
            for name, value in (("history segment", model.segment_seconds),
                                ("prediction_horizon", config.relative.prediction_horizon)):
                ratio = value / model.horizon_seconds
                if round(ratio) < 1 or not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=STEP_TOLERANCE):
                    raise ValueError(f"{name} must be a whole number of GP steps ({model.horizon_seconds:g} s)")
        self.history = LateralHistory(2.0 * segment + config.history_margin)
        self.rng = np.random.default_rng(config.seed)
        self.previous_plan: PreviousPlan | None = None
        self.last_plan: PreviousPlan | None = None
        self.last_rollout: PaperRollout | None = None
        self.last_reason = ""   # 本次決策未推演時的原因
        self._origin_s = 0.0    # 推演起點的前車 s，供橢圓切線角使用
        self._track: Track | None = None
        self._uncertainty_cache: dict[tuple[float, ...], PositionUncertainty] = {}
        self._vehicle: VehicleObservation | None = None

    def update(
        self, vehicle: VehicleObservation, others: Sequence[VehicleObservation],
        track: Track, time_seconds: float,
    ) -> StrategyDecision:
        # 每一步都記錄（包含不需決策的步），才有 t−0.1～t−0.4 s 的 d。
        self.history.record(time_seconds, (vehicle, *others))
        self._vehicle = vehicle
        try:
            return super().update(vehicle, others, track, time_seconds)
        finally:
            self._vehicle = None

    def _select_target(self, vehicle, others, track):
        # 每次決策開始時清掉上一輪的推演。
        self.last_rollout, self.last_reason, self._uncertainty_cache = None, "", {}
        return super()._select_target(vehicle, others, track)

    def _predict_other(self, other: VehicleObservation, track: Track, time_seconds: float) -> BicyclePrediction:
        # 退回用的定速等 d 預測（relative_overtake 原本的作法）。
        fallback = super()._predict_other(other, track, time_seconds)
        target = self.target_state
        if target is None or other.vehicle_id != target.vehicle_id or self._vehicle is None:
            return fallback
        plan = self.previous_plan
        self.last_plan = plan
        if self.model is None:
            reason = "no_model"
        elif plan is None:
            reason = "no_previous_plan"
        elif not track.closed:
            reason = "open_track"
        else:
            rollout, reason = self._rollout(self._vehicle, other, track, time_seconds, plan)
            if rollout is not None:
                self.last_rollout, self._origin_s, self._track = rollout, other.track_position.s, track
                return self._mean_prediction(other, track, rollout)
        self.last_reason = reason
        return fallback

    def _rollout(
        self, vehicle: VehicleObservation, front: VehicleObservation, track: Track,
        now: float, plan: PreviousPlan,
    ) -> tuple[PaperRollout | None, str]:
        model, samples = self.model, self.paper_config.samples
        step = model.horizon_seconds
        segment_steps = round(model.segment_seconds / step)
        count = round(self.config.prediction_horizon / step)
        lag = now - plan.time_seconds   # 上一步計畫上對應「現在」的時間
        # 早於現在的歷史欄位用實際紀錄：兩車在 now−step … now−2·segment 的 d。
        recorded: dict[tuple[str, int], float] = {}
        for back in range(1, 2 * segment_steps + 1):
            for key in (vehicle.vehicle_id, front.vehicle_id):
                value = self.history.d_at(key, now - back * step)
                if value is None:
                    return None, "no_history"
                recorded[key, -back] = value
        length = track_length(track)
        s = np.full(samples, front.track_position.s)
        d = np.full(samples, front.track_position.d)
        psi = np.full(samples, heading_error(front, track))
        speed = np.full(samples, max(0.0, front.state.velocity))
        travelled = np.zeros(samples)
        progress, offsets = np.zeros((samples, count + 1)), np.zeros((samples, count + 1))
        offsets[:, 0] = d
        active = np.ones(samples, dtype=bool)
        exit_steps, exit_reasons = [-1] * samples, [""] * samples
        for k in range(count):
            ego = vehicle if k == 0 else create_vehicle_observation(
                vehicle.vehicle_id, plan_state_at(plan.prediction, lag + k * step, track), vehicle.geometry, track)
            ego_past, front_past = [], np.empty((samples, 2))
            for column, back in enumerate((2 * segment_steps, segment_steps)):   # t−2 段、t−1 段
                node = k - back
                if node < 0:
                    ego_past.append(recorded[vehicle.vehicle_id, node])
                    front_past[:, column] = recorded[front.vehicle_id, node]
                else:
                    ego_past.append(vehicle.track_position.d if node == 0
                                    else plan_d_at(plan.prediction, lag + node * step))
                    front_past[:, column] = offsets[:, node]
            rows = paper_feature_rows(ego, (ego_past[0], ego_past[1]), s, d, psi, speed,
                                      front_past, track, model.curvature_lookahead)
            reasons = model.check_rows(rows)
            if k == 0 and reasons[0] is not None:
                return None, f"out_of_range:{reasons[0]}"
            for index in np.flatnonzero(active):
                if reasons[index] is not None:
                    active[index] = False
                    exit_steps[index], exit_reasons[index] = k, reasons[index]
            # 每步都為全部樣本抽亂數，亂數序列不受哪些樣本退出影響。
            noise = self.rng.standard_normal((samples, 4))
            # 退出的樣本：等速、等 d、航向誤差與車速不變。
            delta = np.zeros((samples, 4))
            delta[:, 0] = speed * step
            if active.any():
                # 第 0 步所有樣本的輸入相同，只查一列。
                mean, std = model.predict(rows[:1] if k == 0 else rows[active])
                delta[active] = mean + std * noise[active]
            travelled = travelled + delta[:, 0]
            s = (s + delta[:, 0]) % length
            d = d + delta[:, 1]
            psi = np.remainder(psi + delta[:, 2] + math.pi, 2.0 * math.pi) - math.pi
            speed = np.maximum(speed + delta[:, 3], 0.0)
            progress[:, k + 1], offsets[:, k + 1] = travelled, d
        nodes = tuple(float(k * step) for k in range(count + 1))
        return PaperRollout(nodes, progress, offsets, tuple(exit_steps), tuple(exit_reasons)), ""

    def _mean_prediction(
        self, front: VehicleObservation, track: Track, rollout: PaperRollout,
    ) -> BicyclePrediction:
        """節點平均線性內插到預測格點：s 用平均縱向進度、d 用平均 d。"""
        config = prediction_config_for(self.other_prediction_configs, front.vehicle_id)
        horizon, dt = self.config.prediction_horizon, config.model.time_step
        times = np.minimum(np.arange(math.ceil(horizon / dt) + 1) * dt, horizon)
        nodes = np.asarray(rollout.node_times)
        return predict_profile_trajectory(
            front, track, np.interp(times, nodes, rollout.mean_d), max(0.0, front.state.velocity),
            horizon, dt, progress=np.interp(times, nodes, rollout.mean_progress))

    def _pair_uncertainty(self, vehicle_id: str, times: Sequence[float]) -> PositionUncertainty | None:
        """目標前車：平均位置的賽道切線角，以及 k·σ_s、k·σ_d（線性內插到檢查時間）。"""
        rollout, target, k = self.last_rollout, self.target_state, self.paper_config.ellipse_sigma
        if rollout is None or k == 0.0 or target is None or vehicle_id != target.vehicle_id:
            return None
        key = tuple(float(time) for time in times)
        cached = self._uncertainty_cache.get(key)
        if cached is None:
            at, nodes = np.asarray(key), np.asarray(rollout.node_times)
            progress = np.interp(at, nodes, rollout.mean_progress)
            cached = PositionUncertainty(
                np.array([track_tangent_angle(self._origin_s + value, self._track) for value in progress]),
                k * np.interp(at, nodes, rollout.std_s), k * np.interp(at, nodes, rollout.std_d))
            self._uncertainty_cache[key] = cached
        return cached

    def _snapshot(self, vehicle, time_seconds, best) -> CandidateSnapshot:
        snapshot = super()._snapshot(vehicle, time_seconds, best)
        # 保存本步選中候選的預測軌跡，下一步沿它推演前車。
        if best is not None:
            self.previous_plan = PreviousPlan(time_seconds, best.prediction, best.maneuver.target_d)
        rollout, target = self.last_rollout, self.target_state
        if rollout is None or target is None:
            return snapshot
        prediction = self.last_other_predictions[target.vehicle_id]
        gp = GpPredictionSnapshot(
            target.vehicle_id, prediction.states, prediction.times,
            delta_d_mean=float(rollout.mean_d[-1] - rollout.mean_d[0]), delta_d_std=float(rollout.std_d[-1]),
            horizon_seconds=rollout.node_times[-1],
        )
        return replace(snapshot, gp_predictions=(*snapshot.gp_predictions, gp))

    def _details(self, no_target: bool, best: RelativeCandidate) -> tuple[str, ...]:
        details = super()._details(no_target, best)
        if self.model is None or no_target:
            return details
        rollout = self.last_rollout
        if rollout is None:
            return (*details, f"gp=off({self.last_reason})")
        samples = len(rollout.exit_reasons)
        text = (f"paper_gp Q={samples} +{rollout.node_times[-1]:g}s: ds={rollout.mean_progress[-1]:.2f} "
                f"std_s={rollout.std_s[-1]:.3f} d={rollout.mean_d[-1]:+.2f} std_d={rollout.std_d[-1]:.3f}")
        if rollout.held_count:
            text += f" const_v={rollout.held_count}/{samples}"
        return (*details, text)
