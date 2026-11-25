"""先篩選預測碰撞，再比較預測終點間距與可選的安全距離成本。"""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from core.pure_pursuit import validate_speed_limit
from core.track import Track
from core.track_geometry import TrackPosition, signed_longitudinal_gap, track_length
from core.validation import (
    require_finite_values, require_nonnegative, require_nonnegative_values, require_positive,
)
from strategy.base import CandidatePath, CandidateSnapshot, GpPredictionSnapshot, StrategyDecision
from strategy.bicycle_prediction import (
    BicyclePrediction, BicyclePredictionConfig, constant_offset_maneuver,
    predict_bicycle_trajectory, prediction_config_for,
)
from strategy.collision_check import (
    PredictedCollision, PredictedVehicle, first_predicted_collision, states_at_times,
)
from strategy.common import (
    RelativeTargetState, accumulated_gap, check_other_ids, check_track_and_time,
    decision_due, lateral_targets, nearest_vehicle_ahead,
)
from strategy.gp_prediction import GpRoute, predict_gp_trajectory
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver, start_or_continue_maneuver
from strategy.responsibility import (
    PairPrediction, PositionUncertainty, ResponsibilityConfig, ResponsibilityResult, evaluate_responsibility,
)


@dataclass(frozen=True)
class RelativeOvertakingConfig:
    candidate_offsets: tuple[float, ...] = (
        -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6,
    )
    detection_range: float = 6.0
    transition_length: float = 3.0
    decision_period: float = 0.1
    prediction_horizon: float = 0.8
    safety_margin: float = 0.1
    speed_candidates_enabled: bool = False
    speed_candidate_delta: float = 1.0
    extra_speed_candidate_deltas: tuple[float, ...] = ()
    safety_weight: float = 0.0     # 1/m；0 表示不加安全距離成本
    safety_distance: float = 0.5   # m；矩形間隙小於此值才開始懲罰
    deviation_weight: float = 0.0  # 1/m；偏離中心線成本，0 表示不加
    # None：沿用矩形重疊剔除；設定後改用避撞責任約束剔除候選。
    responsibility: ResponsibilityConfig | None = None
    # 1/m；responsibility.leader_soft=True 時，自車在前的懲罰權重；0 表示前車不避讓。
    leader_penalty_weight: float = 0.0
    # m；候選的目標 d 與目前目標不同時，加上的固定成本；0 表示不加。
    switch_weight: float = 0.0
    # 無單位；乘在終點間距 gap_T（＝前車 s − 自車 s，即自車進度）上。越大越重視進度，1 為原本的成本。
    progress_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.responsibility is not None and not isinstance(
                self.responsibility, ResponsibilityConfig):
            raise ValueError("responsibility must be a ResponsibilityConfig or None")
        require_positive(self, "detection_range", "transition_length",
                         "decision_period", "prediction_horizon")
        require_nonnegative(self, "safety_margin", "speed_candidate_delta", "safety_weight",
                            "safety_distance", "deviation_weight", "leader_penalty_weight",
                            "switch_weight", "progress_weight")
        require_finite_values(self, "candidate_offsets")
        require_nonnegative_values(self, "extra_speed_candidate_deltas")


@dataclass(frozen=True)
class RelativeCandidate:
    maneuver: LateralManeuver
    prediction: BicyclePrediction
    predicted_gap: float | None
    collision: PredictedCollision | None = None
    speed_limit: float | None = None
    safety_cost: float = 0.0  # 已乘上 safety_weight
    deviation_cost: float = 0.0  # 已乘上 deviation_weight
    leader_cost: float = 0.0  # 已乘上 leader_penalty_weight；自車在前時的避碰懲罰
    switch_cost: float = 0.0  # 換目標 d 時為 switch_weight，否則 0
    progress_weight: float = 1.0  # 乘在 gap_T 上的權重

    def __post_init__(self) -> None:
        validate_speed_limit(self.speed_limit)

    @property
    def feasible(self) -> bool:
        return self.collision is None

    @property
    def relative_cost(self) -> float:
        # 沒有比較對象就沒有相對成本；不另加自車進度項。
        return 0.0 if self.predicted_gap is None else self.predicted_gap

    @property
    def total_cost(self) -> float:
        if not self.feasible:
            raise ValueError("collision-rejected candidates do not have a cost")
        return (self.weighted_gap + self.safety_cost + self.deviation_cost + self.leader_cost
                + self.switch_cost)

    @property
    def weighted_gap(self) -> float:
        """progress_weight × gap_T；權重 1 時與 relative_cost 相同。"""
        return self.progress_weight * self.relative_cost


class RelativeOvertakingStrategy:
    """每台車持有獨立狀態；其他車每輪只預測一次，供所有候選共用。"""

    def __init__(
        self,
        config: RelativeOvertakingConfig,
        prediction_config: BicyclePredictionConfig,
        other_prediction_configs: Mapping[str, BicyclePredictionConfig],
        gp_routes: Mapping[str, GpRoute] | None = None,
    ) -> None:
        self.config = config
        self.prediction_config = prediction_config
        self.other_prediction_configs = dict(other_prediction_configs)
        self.gp_routes = dict(gp_routes or {})
        # 觀測歷史屬於各進攻車自己；僅 GP 模型可共用。
        self.previous_gp_positions: dict[str, tuple[TrackPosition, float]] = {}
        self.decision = StrategyDecision("cruise")
        self.last_evaluation_time = -math.inf
        self.last_candidates: tuple[RelativeCandidate, ...] = ()
        self.last_other_predictions: dict[str, BicyclePrediction] = {}
        self.last_target_prediction: BicyclePrediction | None = None
        self.target_state: RelativeTargetState | None = None

    def update(
        self,
        vehicle: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> StrategyDecision:
        check_track_and_time("relative overtaking", track, time_seconds, self.last_evaluation_time)
        ids = check_other_ids(vehicle, others)
        if not decision_due(time_seconds, self.last_evaluation_time, self.config.decision_period):
            return self.decision

        target, gap = self._select_target(vehicle, others, track)
        self.previous_gp_positions = {
            key: value for key, value in self.previous_gp_positions.items() if key in ids
        }
        self.last_other_predictions = {
            other.vehicle_id: self._predict_other(other, track, time_seconds) for other in others
        }
        self.last_target_prediction = (
            None if target is None else self.last_other_predictions[target.vehicle_id]
        )
        obstacles = tuple(
            PredictedVehicle(other.vehicle_id, other.geometry,
                             self.last_other_predictions[other.vehicle_id])
            for other in others
        )
        # 每條橫向路徑都搭配巡航 (None) 與各個低速要求，一起比較成本。
        speed_limits = (None, *self._speed_limits(target, gap))
        self.last_candidates = tuple(
            self._evaluate_candidate(
                vehicle, track, target_d, gap, obstacles, others, speed_limit=speed_limit,
            )
            for target_d in lateral_targets(
                self.config.candidate_offsets, track, vehicle.geometry, self.config.safety_margin,
            )
            for speed_limit in speed_limits
        )
        feasible = [candidate for candidate in self.last_candidates if candidate.feasible]
        # 同分優先保留原目標，再保留原速度要求。
        best = min(feasible, key=lambda candidate: (
            candidate.total_cost,
            candidate.maneuver.target_d != self.decision.target_d,
            candidate.speed_limit != self.decision.speed_limit,
        )) if feasible else self._latest_collision(self.last_candidates)
        snapshot = self._snapshot(vehicle, time_seconds, best)
        self.last_evaluation_time = time_seconds
        if best.collision is not None:
            reason = f"latest predicted collision @ {best.collision.time_seconds:.2f}s"
            self.decision = StrategyDecision(
                mode="no_feasible_path",
                target_d=best.maneuver.target_d,
                target_vehicle_id=None if target is None else target.vehicle_id,
                maneuver=best.maneuver,
                speed_limit=best.speed_limit,
                reason="all candidates predict overlap; " + reason,
                details=(reason,),
                candidate_snapshot=snapshot,
            )
            return self.decision

        self.decision = StrategyDecision(
            mode="cruise" if target is None else "attack",
            target_d=best.maneuver.target_d,
            target_vehicle_id=None if target is None else target.vehicle_id,
            maneuver=best.maneuver,
            speed_limit=best.speed_limit,
            reason="no target vehicle; prefer previous feasible target" if target is None
            else "lowest predicted relative gap",
            details=self._details(target is None, best),
            candidate_snapshot=snapshot,
        )
        return self.decision

    def _select_target(
        self, vehicle: VehicleObservation, others: Sequence[VehicleObservation], track: Track,
    ) -> tuple[VehicleObservation | None, float]:
        """優先選最近的新前車；沒有前車時保留原對象，超車後仍繼續評分。"""
        target, gap = nearest_vehicle_ahead(vehicle, others, track, self.config.detection_range)
        previous = self.target_state
        if target is None and previous is not None:
            target = next((other for other in others if other.vehicle_id == previous.vehicle_id), None)
        if target is None:
            self.target_state = None
            return None, 0.0
        if previous is not None and previous.vehicle_id == target.vehicle_id:
            gap = accumulated_gap(previous, vehicle, target, track)
        self.target_state = RelativeTargetState(
            target.vehicle_id, vehicle.track_position, target.track_position, gap,
        )
        return target, gap

    def _predict_other(
        self, other: VehicleObservation, track: Track, time_seconds: float,
    ) -> BicyclePrediction:
        config = prediction_config_for(self.other_prediction_configs, other.vehicle_id)
        if other.vehicle_id not in self.gp_routes:
            # 未指定 GP 的車保留原本的定速、等 d 自行車預測。
            return predict_bicycle_trajectory(
                other, track, constant_offset_maneuver(other, config), config,
                self.config.prediction_horizon, constant_speed=True,
            )
        # 先估計沿 s 的速度；第一次沒有歷史時，以目前車速近似。
        speed = other.state.velocity
        previous = self.previous_gp_positions.get(other.vehicle_id)
        if previous is not None and time_seconds > previous[1]:
            distance = signed_longitudinal_gap(previous[0], other.track_position, track_length(track))
            speed = max(0.0, distance / (time_seconds - previous[1]))
        self.previous_gp_positions[other.vehicle_id] = (other.track_position, time_seconds)
        return predict_gp_trajectory(
            other, track, self.gp_routes[other.vehicle_id], speed,
            self.config.prediction_horizon, config.model.time_step,
        )

    def _latest_collision(
        self, candidates: Sequence[RelativeCandidate],
    ) -> RelativeCandidate:
        """全部不可行時：最晚碰撞優先；同時間選速度要求較低者，再選進度較多者。"""
        cruise_speed = self.prediction_config.controller.target_speed

        def key(candidate: RelativeCandidate) -> tuple[float, float, float]:
            if candidate.collision is None:
                raise ValueError("latest-collision fallback expects rejected candidates only")
            speed = cruise_speed if candidate.speed_limit is None else min(
                candidate.speed_limit, cruise_speed,
            )
            return (-candidate.collision.time_seconds, speed, -candidate.prediction.progress[-1])

        return min(candidates, key=key)

    def _safety_penalty(
        self, vehicle: VehicleObservation, prediction: BicyclePrediction,
        others: Sequence[VehicleObservation], track: Track,
    ) -> float:
        """Frenet 近似的矩形間隙：間隙小於 safety_distance 時，依不足量平方懲罰。"""
        if self.config.safety_weight == 0.0 or not others:
            return 0.0
        length = track_length(track)
        own_times = np.asarray(prediction.times)
        own_box = vehicle.geometry.collision or vehicle.geometry
        penalty = 0.0
        for other in others:
            other_prediction = self.last_other_predictions[other.vehicle_id]
            other_times = np.asarray(other_prediction.times)
            if (len(prediction.offsets) != len(own_times)
                    or len(other_prediction.offsets) != len(other_times)):
                raise ValueError("safety cost requires predicted offsets at every time")
            # 橫向距離＝兩車 d 差；縱向距離見 _relative_progress。
            delta_s = self._relative_progress(vehicle, other, prediction, length)
            delta_d = (np.interp(own_times, other_times, other_prediction.offsets)
                       - np.asarray(prediction.offsets))
            other_box = other.geometry.collision or other.geometry
            half_length = (own_box.length + other_box.length) / 2.0
            half_width = (own_box.width + other_box.width) / 2.0
            # 任一方向分開即不重疊，因此取兩方向中較大的間隙。
            clearance = np.maximum(np.abs(delta_s) - half_length, np.abs(delta_d) - half_width)
            shortage = np.maximum(self.config.safety_distance - clearance, 0.0)
            penalty += float(np.mean(shortage**2))
        return self.config.safety_weight * penalty

    def _relative_progress(
        self, vehicle: VehicleObservation, other: VehicleObservation,
        prediction: BicyclePrediction, length: float,
    ) -> np.ndarray:
        """各預測時間點的 Δs＝起始間距＋對手進度－自車進度；正值表示對手在前。"""
        other_prediction = self.last_other_predictions[other.vehicle_id]
        gap0 = signed_longitudinal_gap(vehicle.track_position, other.track_position, length)
        return (gap0 + np.interp(prediction.times, other_prediction.times, other_prediction.progress)
                - np.asarray(prediction.progress))

    def _responsibility_result(
        self, vehicle: VehicleObservation, prediction: BicyclePrediction,
        others: Sequence[VehicleObservation], track: Track,
    ) -> ResponsibilityResult:
        """逐對檢查避撞責任：自車在後時須保持 r_col；leader_soft 時自車在前只計懲罰。"""
        config = self.config.responsibility
        if config is None or not others:
            return ResponsibilityResult(None)
        length = track_length(track)
        pairs = tuple(
            PairPrediction(
                other.vehicle_id, other.geometry,
                states_at_times(self.last_other_predictions[other.vehicle_id], prediction.times),
                self._relative_progress(vehicle, other, prediction, length),
                self._pair_uncertainty(other.vehicle_id, prediction.times),
            )
            for other in others
        )
        return evaluate_responsibility(
            prediction.times, prediction.states, vehicle.geometry, pairs, config,
        )

    def _pair_uncertainty(
        self, vehicle_id: str, times: Sequence[float],
    ) -> PositionUncertainty | None:
        """對方位置的不確定性橢圓；預設沒有，維持圓形約束。子類別可依自己的預測提供。"""
        return None

    def _deviation_cost(self, prediction: BicyclePrediction) -> float:
        """偏離中心線成本：預測各點 d 的平方取平均；t=0 各候選相同，不列入。"""
        if self.config.deviation_weight == 0.0:
            return 0.0
        if len(prediction.offsets) != len(prediction.times):
            raise ValueError("deviation cost requires predicted offsets at every time")
        offsets = np.asarray(prediction.offsets[1:])
        return self.config.deviation_weight * float(np.mean(offsets**2))

    def _evaluate_candidate(
        self, vehicle: VehicleObservation, track: Track, target_d: float, gap: float,
        obstacles: Sequence[PredictedVehicle], others: Sequence[VehicleObservation] = (),
        speed_limit: float | None = None,
    ) -> RelativeCandidate:
        maneuver = start_or_continue_maneuver(
            self.decision.maneuver, vehicle.track_position, target_d,
            self.config.transition_length, track,
        )
        prediction = predict_bicycle_trajectory(
            vehicle, track, maneuver, self.prediction_config, self.config.prediction_horizon,
            speed_limit=speed_limit,
        )
        # 預設用矩形重疊剔除；設定 responsibility 時改用避撞責任約束。
        leader_penalty = 0.0
        if self.config.responsibility is None:
            collision = first_predicted_collision(prediction, vehicle.geometry, obstacles)
        else:
            result = self._responsibility_result(vehicle, prediction, others, track)
            collision, leader_penalty = result.violation, result.leader_penalty
        if collision is not None:
            return RelativeCandidate(maneuver, prediction, None, collision, speed_limit=speed_limit)
        opponent = self.last_target_prediction
        # 用累積進度算終點間距，跨起終點不跳變；保留負值表示自車領先。
        predicted_gap = None if opponent is None else gap + opponent.progress[-1] - prediction.progress[-1]
        return RelativeCandidate(
            maneuver, prediction, predicted_gap, speed_limit=speed_limit,
            progress_weight=self.config.progress_weight,
            safety_cost=self._safety_penalty(vehicle, prediction, others, track),
            deviation_cost=self._deviation_cost(prediction),
            leader_cost=self.config.leader_penalty_weight * leader_penalty,
            # 最單純的換目標成本：目標 d 與目前決策不同就加固定值，不看換多遠。
            switch_cost=(self.config.switch_weight
                         if maneuver.target_d != self.decision.target_d else 0.0),
        )

    def _speed_limits(
        self, target: VehicleObservation | None, gap: float,
    ) -> tuple[float, ...]:
        """前車速度用來建立候選，可行性由完整 rollout 的碰撞檢查判定。"""
        if not self.config.speed_candidates_enabled or target is None or gap <= 0.0:
            return ()
        front_speed = max(0.0, target.state.velocity)
        # 速度差使用正值；例如 4.0 代表前車速度減去 4 m/s。
        deltas = (0.0, self.config.speed_candidate_delta, *self.config.extra_speed_candidate_deltas)
        speeds = (max(0.0, front_speed - delta) for delta in deltas)
        cruise_speed = self.prediction_config.controller.target_speed
        return tuple(dict.fromkeys(
            speed for speed in speeds if speed < cruise_speed
        ))

    def _details(self, no_target: bool, best: RelativeCandidate) -> tuple[str, ...]:
        extra = ()
        if self.config.safety_weight > 0.0:
            extra += (f"safety={best.safety_cost:.2f}",)
        if self.config.deviation_weight > 0.0:
            extra += (f"dev={best.deviation_cost:.2f}",)
        if self._leader_cost_active():
            extra += (f"leader={best.leader_cost:.2f}",)
        if self.config.switch_weight > 0.0:
            extra += (f"switch={best.switch_cost:.2f}",)
        if no_target:
            if not extra:
                return ("no target vehicle; collision filtering only",)
            return ("no target vehicle; collision filtering and extra costs", *extra)
        return (f"gap_T={best.relative_cost:+.2f} m", *extra, f"J={best.total_cost:.2f}")

    def _cost_terms(self, candidate: RelativeCandidate) -> tuple[tuple[str, float], ...]:
        """只列出實際參與比較的成本項；安全項關閉時維持原本的顯示。"""
        if not candidate.feasible:
            return ()
        terms: tuple[tuple[str, float], ...] = ()
        if candidate.predicted_gap is not None:
            # 顯示加權後的值，候選標籤各項加總才等於 J。
            terms += (("relative_gap", candidate.weighted_gap),)
        if self.config.safety_weight > 0.0:
            terms += (("safety", candidate.safety_cost),)
        if self.config.deviation_weight > 0.0:
            terms += (("deviation", candidate.deviation_cost),)
        if self._leader_cost_active():
            terms += (("leader", candidate.leader_cost),)
        if self.config.switch_weight > 0.0:
            terms += (("switch", candidate.switch_cost),)
        # 不另放 total：候選標籤會把各項加總成 J。
        return terms

    def _leader_cost_active(self) -> bool:
        responsibility = self.config.responsibility
        return (responsibility is not None
                and (responsibility.leader_soft or responsibility.overlap_rule)
                and self.config.leader_penalty_weight > 0.0)

    def _snapshot(
        self, vehicle: VehicleObservation, time_seconds: float, best: RelativeCandidate | None,
    ) -> CandidateSnapshot:
        return CandidateSnapshot(
            evaluated_at=time_seconds, position=vehicle.track_position, distances=(),
            # 直接交出碰撞檢查用的狀態，不在顯示端重新預測。
            gp_predictions=tuple(
                GpPredictionSnapshot(key, prediction.states, prediction.times)
                for key, prediction in self.last_other_predictions.items() if key in self.gp_routes
            ),
            candidates=tuple(
                CandidatePath(
                    maneuver=candidate.maneuver, selected=candidate is best,
                    feasible=candidate.feasible,
                    reason="" if candidate.collision is None else candidate.collision.reason,
                    cost_terms=self._cost_terms(candidate),
                    predicted_states=candidate.prediction.states,
                    speed_limit=candidate.speed_limit,
                )
                for candidate in self.last_candidates
            ),
        )
