"""以互動歷史 GP 預測前車橫移，其餘沿用 relative_overtake 的候選、成本與避撞檢查。

conditioning="candidate"：每個橫向候選以自己的目標 d 查 GP，與訓練時 d_target 的定義（當下採用的目標）一致。
conditioning="previous"：以上一次決策的目標查一次，所有候選共用同一份前車預測。
GP 未啟用（無模型、歷史不足或輸入超出訓練範圍）時退回定速、等 d 預測，行為與 relative_overtake 相同。
"""

from dataclasses import dataclass, field, replace
import math
from typing import Mapping, Sequence

import numpy as np

from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length
from core.validation import require_nonnegative, require_positive
from strategy.base import CandidateSnapshot, GpPredictionSnapshot, StrategyDecision
from strategy.bicycle_prediction import BicyclePrediction, BicyclePredictionConfig, prediction_config_for
from strategy.collision_check import (
    PredictedCollision, PredictedVehicle, first_predicted_collision, states_at_times,
)
from strategy.common import lateral_targets
from strategy.interaction_features import LateralHistory, history_state_features
from strategy.interaction_gp import InteractionGpModel, lateral_profile, predict_profile_trajectory
from strategy.observations import VehicleObservation
from strategy.relative_overtaking import (
    RelativeCandidate, RelativeOvertakingConfig, RelativeOvertakingStrategy,
)
from strategy.responsibility import PairPrediction, evaluate_responsibility

CONDITIONING_MODES = ("candidate", "previous")
# GP 訓練時出現過的 d_target；預設候選只用這些值。
TRAINED_OFFSETS = (-0.6, -0.45, -0.3, 0.0, 0.3, 0.45, 0.6)


@dataclass(frozen=True)
class InteractionOvertakingConfig:
    # transition_length 3.0 m 與訓練資料的 probe_follow 相同，同一個 d_target 才代表相同的橫移快慢。
    relative: RelativeOvertakingConfig = field(default_factory=lambda: RelativeOvertakingConfig(
        candidate_offsets=TRAINED_OFFSETS, transition_length=3.0))
    conditioning: str = "candidate"
    band_sigma: float = 2.0       # 邊界軌跡＝平均 ± band_sigma·σ；0 表示只用平均
    history_margin: float = 0.1   # 歷史多保留的秒數

    def __post_init__(self) -> None:
        if not isinstance(self.relative, RelativeOvertakingConfig):
            raise ValueError("InteractionOvertakingConfig.relative must be a RelativeOvertakingConfig")
        if self.conditioning not in CONDITIONING_MODES:
            raise ValueError(f"InteractionOvertakingConfig.conditioning must be one of {CONDITIONING_MODES}")
        require_nonnegative(self, "band_sigma")
        require_positive(self, "history_margin")

    @property
    def transition_length(self) -> float:
        """供參考路徑長度規則讀取。"""
        return self.relative.transition_length


@dataclass(frozen=True)
class GpQuery:
    """一個 d_target 的查詢結果；reason 為空字串表示 GP 已啟用。"""

    d_target: float
    reason: str = ""
    mean: float | None = None
    std: float | None = None
    prediction: BicyclePrediction | None = None
    bands: tuple[BicyclePrediction, ...] = ()

    @property
    def active(self) -> bool:
        return not self.reason


class InteractionOvertakingStrategy(RelativeOvertakingStrategy):
    """每台車持有自己的觀測歷史與查詢結果；GP 模型可共用。"""

    def __init__(
        self,
        config: InteractionOvertakingConfig,
        prediction_config: BicyclePredictionConfig,
        other_prediction_configs: Mapping[str, BicyclePredictionConfig],
        model: InteractionGpModel | None = None,
    ) -> None:
        super().__init__(config.relative, prediction_config, other_prediction_configs)
        self.interaction_config = config
        self.model = model
        segment = 0.2 if model is None else model.segment_seconds
        self.history = LateralHistory(2.0 * segment + config.history_margin)
        self.gp_queries: dict[float, GpQuery] = {}
        self.last_gp_query: GpQuery | None = None
        self._vehicle: VehicleObservation | None = None
        if model is not None:
            untrained = sorted(offset for offset in {0.0, *config.relative.candidate_offsets}
                               if not model.is_trained_target(offset))
            if untrained:
                raise ValueError(f"candidate offsets {untrained} never appear as d_target in GP training; "
                                 f"use values from {model.trained_targets}")

    def update(
        self, vehicle: VehicleObservation, others: Sequence[VehicleObservation],
        track: Track, time_seconds: float,
    ) -> StrategyDecision:
        # 每一步都記錄（包含不需決策的步），才有 t−0.2、t−0.4 的 d。
        self.history.record(time_seconds, (vehicle, *others))
        self._vehicle = vehicle
        try:
            return super().update(vehicle, others, track, time_seconds)
        finally:
            self._vehicle = None

    def _select_target(self, vehicle, others, track):
        # 每次決策開始時清掉上一輪的查詢。
        self.gp_queries = {}
        return super()._select_target(vehicle, others, track)

    def _predict_other(self, other: VehicleObservation, track: Track, time_seconds: float) -> BicyclePrediction:
        fallback = super()._predict_other(other, track, time_seconds)
        target = self.target_state
        if target is None or other.vehicle_id != target.vehicle_id or self._vehicle is None:
            return fallback
        vehicle = self._vehicle
        if self.interaction_config.conditioning == "candidate":
            targets = lateral_targets(self.config.candidate_offsets, track, vehicle.geometry,
                                      self.config.safety_margin)
        else:
            targets = (self.decision.target_d,)
        self.gp_queries = self._query(vehicle, other, track, time_seconds, targets)
        # 共用的前車預測：previous 模式即唯一的查詢；candidate 模式各候選評估時再換上自己的預測。
        shared = self.gp_queries.get(self.decision.target_d)
        return shared.prediction if shared is not None and shared.active else fallback

    def _query(
        self, vehicle: VehicleObservation, front: VehicleObservation, track: Track,
        time_seconds: float, targets: Sequence[float],
    ) -> dict[float, GpQuery]:
        if self.model is None:
            return {target: GpQuery(target, "no_model") for target in targets}
        segment = self.model.segment_seconds
        past = [self.history.d_at(key, time_seconds - k * segment)
                for key in (vehicle.vehicle_id, front.vehicle_id) for k in (2, 1)]
        if any(value is None for value in past):
            return {target: GpQuery(target, "no_history") for target in targets}
        state = history_state_features(vehicle, front, (past[0], past[1]), (past[2], past[3]), track)
        X = np.array([(*state, target) for target in targets], dtype=float)
        reasons = [self.model.check(row) for row in X]
        active = [index for index, reason in enumerate(reasons) if reason is None]
        queries = {target: GpQuery(target, f"out_of_range:{reason}")
                   for target, reason in zip(targets, reasons) if reason is not None}
        if active:
            means, stds = self.model.predict(X[active])
            for index, mean, std in zip(active, means, stds):
                queries[targets[index]] = self._trajectories(front, track, targets[index], float(mean), float(std))
        return {target: queries[target] for target in targets}

    def _trajectories(
        self, front: VehicleObservation, track: Track, d_target: float, mean: float, std: float,
    ) -> GpQuery:
        """平均軌跡與 ±kσ 邊界：s 定速，d 在 GP 時域內線性到預測值，之後維持。"""
        config = prediction_config_for(self.other_prediction_configs, front.vehicle_id)
        horizon, step = self.config.prediction_horizon, config.model.time_step
        times = np.minimum(np.arange(math.ceil(horizon / step) + 1) * step, horizon)
        d0, gp_horizon = front.track_position.d, self.model.horizon_seconds
        speed = max(0.0, front.state.velocity)

        def build(delta: float) -> BicyclePrediction:
            return predict_profile_trajectory(front, track, lateral_profile(times, d0, delta, gp_horizon),
                                              speed, horizon, step)

        band = self.interaction_config.band_sigma * std
        bands = () if band <= 0.0 else (build(mean + band), build(mean - band))
        return GpQuery(d_target, mean=mean, std=std, prediction=build(mean), bands=bands)

    def _query_for(self, target_d: float) -> GpQuery | None:
        if self.interaction_config.conditioning == "previous":
            return next(iter(self.gp_queries.values()), None)
        return self.gp_queries.get(target_d)

    def _evaluate_candidate(
        self, vehicle: VehicleObservation, track: Track, target_d: float, gap: float,
        obstacles: Sequence[PredictedVehicle], others: Sequence[VehicleObservation] = (),
        speed_limit: float | None = None,
    ) -> RelativeCandidate:
        query = self._query_for(target_d)
        target_id = None if self.target_state is None else self.target_state.vehicle_id
        if query is None or not query.active or target_id is None:
            return super()._evaluate_candidate(vehicle, track, target_d, gap, obstacles, others, speed_limit)
        # 換上此候選對應的前車預測，評估完還原，其他候選不受影響。
        saved = (self.last_other_predictions[target_id], self.last_target_prediction)
        self.last_other_predictions[target_id] = query.prediction
        self.last_target_prediction = query.prediction
        obstacles = tuple(replace(item, prediction=query.prediction) if item.vehicle_id == target_id else item
                          for item in obstacles)
        try:
            candidate = super()._evaluate_candidate(vehicle, track, target_d, gap, obstacles, others, speed_limit)
        finally:
            self.last_other_predictions[target_id], self.last_target_prediction = saved
        if not candidate.feasible:
            return candidate
        violation = self._band_violation(vehicle, candidate.prediction, query, target_id, others, track)
        if violation is None:
            return candidate
        return RelativeCandidate(candidate.maneuver, candidate.prediction, None, violation, speed_limit=speed_limit)

    def _band_violation(
        self, vehicle: VehicleObservation, prediction: BicyclePrediction, query: GpQuery,
        target_id: str, others: Sequence[VehicleObservation], track: Track,
    ) -> PredictedCollision | None:
        """前車位於 ±kσ 邊界時，自車 rollout 須通過與平均軌跡相同的檢查。"""
        if not query.bands:
            return None
        target = next(other for other in others if other.vehicle_id == target_id)
        k = self.interaction_config.band_sigma
        gap0 = signed_longitudinal_gap(vehicle.track_position, target.track_position, track_length(track))
        for sign, band in zip(("+", "-"), query.bands):
            label = f"{target_id} (mean{sign}{k:g}σ)"
            if self.config.responsibility is None:
                hit = first_predicted_collision(prediction, vehicle.geometry,
                                                (PredictedVehicle(label, target.geometry, band),))
            else:
                delta_s = (gap0 + np.interp(prediction.times, band.times, band.progress)
                           - np.asarray(prediction.progress))
                pair = PairPrediction(label, target.geometry, states_at_times(band, prediction.times), delta_s)
                hit = evaluate_responsibility(prediction.times, prediction.states, vehicle.geometry,
                                              (pair,), self.config.responsibility).violation
            if hit is not None:
                return hit
        return None

    def _snapshot(self, vehicle, time_seconds, best) -> CandidateSnapshot:
        snapshot = super()._snapshot(vehicle, time_seconds, best)
        self.last_gp_query = None if best is None else self._query_for(best.maneuver.target_d)
        query = self.last_gp_query
        if query is None or not query.active or self.target_state is None or self.model is None:
            return snapshot
        gp = GpPredictionSnapshot(
            self.target_state.vehicle_id, query.prediction.states, query.prediction.times,
            delta_d_mean=query.mean, delta_d_std=query.std,
            horizon_seconds=self.model.horizon_seconds, d_target=query.d_target,
        )
        return replace(snapshot, gp_predictions=(*snapshot.gp_predictions, gp))

    def _details(self, no_target: bool, best: RelativeCandidate) -> tuple[str, ...]:
        details = super()._details(no_target, best)
        query = self.last_gp_query
        if self.model is None or no_target or query is None:
            return details
        if query.active:
            return (*details, f"gp d_target={query.d_target:+.2f} mean={query.mean:+.3f} std={query.std:.3f}")
        return (*details, f"gp=off({query.reason})")
