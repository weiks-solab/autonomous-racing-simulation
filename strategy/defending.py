"""依橫向對齊成本，從少量平滑路徑中選擇防守位置。"""

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Sequence

import numpy as np

from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length
from core.vehicle_state import VehicleState
from strategy.bicycle_prediction import BicyclePredictionConfig, predict_bicycle_trajectory
from strategy.collision_check import PredictedCollision, states_at_times
from strategy.defense_conditions import RearPrediction, has_predicted_collision, predict_rear
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver
from strategy.prediction import predict_path_trajectory
from strategy.responsibility import (
    PairPrediction, ResponsibilityConfig, ResponsibilityResult, evaluate_responsibility,
)


class DefenseMode(Enum):
    CRUISE = "cruise"
    DEFEND = "defend"
    RETURN = "return"
    NO_FEASIBLE_PATH = "no_feasible_path"


@dataclass(frozen=True)
class DefenseConfig:
    """距離單位為公尺，時間單位為秒；常用設定集中在 main.py。"""

    q_block: float = 6.0
    progress_weight: float = 1.0
    smooth_weight: float = 2.0
    detection_range: float = 8.0
    return_tolerance: float = 0.08
    transition_length: float = 3.0
    decision_period: float = 0.2
    candidate_offsets: tuple[float, ...] = (-0.6, -0.3, 0.0, 0.3, 0.6)
    prediction_horizon: float = 1.5
    prediction_step: float = 0.1
    path_sample_spacing: float = 0.2
    safety_margin: float = 0.1
    collision_check_enabled: bool = True
    collision_horizon: float = 1.0
    deviation_weight: float = 0.0  # 偏離中心線成本，0 表示不加
    # 設定後，以避撞責任約束逐對檢查所有其他車，取代 collision_check_enabled 的矩形檢查。
    responsibility: ResponsibilityConfig | None = None
    # responsibility.leader_soft=True 時，守方在前的避碰懲罰權重（與 q_block 同尺度），併入 J_race。
    leader_penalty_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.responsibility is not None and not isinstance(
                self.responsibility, ResponsibilityConfig):
            raise ValueError("responsibility must be a ResponsibilityConfig or None")
        nonnegative = (
            self.q_block, self.progress_weight, self.smooth_weight,
            self.return_tolerance, self.safety_margin, self.deviation_weight,
            self.leader_penalty_weight,
        )
        positive = (
            self.detection_range, self.transition_length,
            self.decision_period, self.prediction_horizon, self.prediction_step,
            self.collision_horizon, self.path_sample_spacing,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError(
                "defense weights, margins and tolerances must be non-negative and finite"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("defense distances and timing must be positive and finite")
        if not all(math.isfinite(value) for value in self.candidate_offsets):
            raise ValueError("candidate_offsets must be finite")
        if not isinstance(self.collision_check_enabled, bool):
            raise ValueError("collision_check_enabled must be a bool")


@dataclass(frozen=True)
class DefenseCandidate:
    target_d: float
    maneuver: LateralManeuver
    race_cost: float | None
    blocking_cost: float | None
    collision: PredictedCollision | None = None
    predicted_states: tuple[VehicleState, ...] = ()
    deviation_cost: float | None = None  # 已併入 race_cost，只供檢查
    leader_cost: float | None = None  # 守方在前的避碰懲罰，已乘權重並併入 race_cost

    @property
    def feasible(self) -> bool:
        return self.collision is None

    @property
    def total_cost(self) -> float:
        if self.race_cost is None or self.blocking_cost is None:
            raise ValueError("collision-rejected candidates do not have a cost")
        return self.race_cost + self.blocking_cost


@dataclass(frozen=True)
class DefenseDecision:
    mode: DefenseMode
    target_d: float
    target_vehicle_id: str | None
    maneuver: LateralManeuver | None
    reason: str
    race_cost: float | None = 0.0
    blocking_cost: float | None = 0.0

    @property
    def display_text(self) -> str:
        text = f"{self.mode.value.upper()}\ntarget_d={self.target_d:+.2f}"
        if self.race_cost is None or self.blocking_cost is None:
            return text + "\n" + self.reason
        return text + f"\nrace={self.race_cost:.2f} block={self.blocking_cost:.2f}"


class CostBasedDefenseStrategy:
    """只輸出橫向目標，不計算方向盤角度，也不改變目標速度。"""

    def __init__(
        self,
        config: DefenseConfig,
        ego_id: str,
        prediction_config: BicyclePredictionConfig | None = None,
    ) -> None:
        if not isinstance(ego_id, str) or not ego_id.strip():
            raise ValueError("ego_id must be a non-empty string")
        self.config = config
        self.ego_id = ego_id
        # 有設定時，自車候選改用 Pure Pursuit＋自行車模型 rollout；否則沿用幾何預測。
        self.prediction_config = prediction_config
        self.decision = DefenseDecision(
            DefenseMode.CRUISE, 0.0, None, None, "following centerline"
        )
        self.last_candidates: tuple[DefenseCandidate, ...] = ()
        self.last_evaluation_time = -math.inf
        self.last_rear_prediction: RearPrediction | None = None
        # 責任約束用：所有其他車的等 d、等速預測，以及當下的有號縱向間距（對方 s − 自車 s）。
        self.last_other_predictions: tuple[RearPrediction, ...] = ()
        self.last_other_gaps: tuple[float, ...] = ()

    def update(
        self,
        defender: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> DefenseDecision:
        # 只在固定間隔重新比較路徑。
        if not track.closed:
            raise ValueError("defending requires a closed base track")
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("time_seconds must be non-negative and finite")
        if time_seconds < self.last_evaluation_time:
            raise ValueError("time_seconds must not move backwards")
        ids = [other.vehicle_id for other in others]
        if len(set(ids)) != len(ids) or defender.vehicle_id in ids:
            raise ValueError("vehicle IDs must be unique; exclude defender from others")
        if defender.vehicle_id == self.ego_id:
            raise ValueError("defender and ego must be different vehicles")
        if time_seconds - self.last_evaluation_time < self.config.decision_period:
            return self.decision
        length = track_length(track)
        ego = next((other for other in others if other.vehicle_id == self.ego_id), None)
        gap = math.inf if ego is None else signed_longitudinal_gap(
            defender.track_position, ego.track_position, length
        )
        # 指定對象在後方且位於偵測範圍內，才計算對齊成本。
        defending = (
            ego is not None
            and -self.config.detection_range <= gap < 0
            and self.config.q_block > 0.0
        )
        ego_d = ego.track_position.d if defending and ego is not None else None
        self.last_rear_prediction = None
        self.last_other_predictions = ()
        self.last_other_gaps = ()
        if self.config.responsibility is not None:
            # 責任約束不限指定對象：每台車都檢查，領先越多的對象約束越寬。
            self.last_other_predictions = tuple(
                predict_rear(
                    other, track, self.config.collision_horizon, self.config.prediction_step,
                    self.config.path_sample_spacing,
                )
                for other in others
            )
            self.last_other_gaps = tuple(
                signed_longitudinal_gap(defender.track_position, other.track_position, length)
                for other in others
            )
        elif self.config.collision_check_enabled and ego is not None:
            self.last_rear_prediction = predict_rear(
                ego, track, self.config.collision_horizon, self.config.prediction_step,
                self.config.path_sample_spacing,
            )

        # 可關閉的碰撞條件先篩選，通過後才參與成本比較。
        targets = self._candidate_targets(defender, track)
        self.last_candidates = tuple(
            self._evaluate_candidate(defender, track, target, ego_d)
            for target in targets
        )
        feasible = [candidate for candidate in self.last_candidates if candidate.feasible]
        if not feasible:
            self.decision = replace(
                self.decision, mode=DefenseMode.NO_FEASIBLE_PATH,
                reason="all candidates predict overlap; continuing previous path",
                race_cost=None, blocking_cost=None,
            )
            self.last_evaluation_time = time_seconds
            return self.decision
        best = min(feasible, key=lambda candidate: candidate.total_cost)
        maneuver = best.maneuver
        centered = abs(defender.track_position.d) <= self.config.return_tolerance
        if abs(best.target_d) < 1e-9 and centered:
            maneuver = None

        # 模式只用於顯示，不覆蓋成本選出的目標。
        mode = DefenseMode.DEFEND if defending else DefenseMode.RETURN
        if not defending and maneuver is None:
            mode = DefenseMode.CRUISE

        self.decision = DefenseDecision(
            mode=mode,
            target_d=best.target_d,
            target_vehicle_id=self.ego_id if defending else None,
            maneuver=maneuver,
            reason="lowest cost",
            race_cost=best.race_cost,
            blocking_cost=best.blocking_cost,
        )
        self.last_evaluation_time = time_seconds
        return self.decision

    def _candidate_targets(
        self,
        defender: VehicleObservation,
        track: Track,
    ) -> tuple[float, ...]:
        maximum_d = (
            track.width / 2.0 - defender.geometry.width / 2.0 - self.config.safety_margin
        )
        if maximum_d < 0.0:
            raise ValueError("track is too narrow for defender width and safety_margin")
        # 中心線固定列入候選；不加入對方位置或臨時目標。
        return tuple(dict.fromkeys(
            target for target in (0.0, *self.config.candidate_offsets)
            if abs(target) <= maximum_d
        ))

    def _evaluate_candidate(
        self,
        defender: VehicleObservation,
        track: Track,
        target_d: float,
        ego_d: float | None,
    ) -> DefenseCandidate:
        # 保留未完成動作的起點，避免每次更新都重新開始過渡。
        maneuver = self.decision.maneuver
        if maneuver is None or not math.isclose(maneuver.target_d, target_d):
            maneuver = LateralManeuver(
                defender.track_position.s, defender.track_position.d,
                target_d, self.config.transition_length,
            )
        step_count = math.ceil(self.config.prediction_horizon / self.config.prediction_step)
        cost_times = np.linspace(0.0, self.config.prediction_horizon, step_count + 1)
        rear = self.last_rear_prediction
        check_times = self._check_times()
        if self.prediction_config is None:
            own = self._predict_geometric(defender, track, maneuver, cost_times, check_times)
        else:
            own = self._predict_bicycle(defender, track, maneuver, cost_times, check_times)
        offsets, progress, predicted_states = own.offsets, own.progress, own.predicted_states
        collision = None
        leader_penalty = 0.0
        if rear is not None and own.check_states is not None:
            collision = has_predicted_collision(defender, own.check_states, rear)
        elif self.last_other_predictions:
            result = self._responsibility_result(defender, own)
            collision, leader_penalty = result.violation, result.leader_penalty
        if collision is not None:
            return DefenseCandidate(target_d, maneuver, None, None, collision, predicted_states)
        # 偏離中心線成本：與對齊成本相同的取樣點，d 的平方加總。
        deviation_cost = float(self.config.deviation_weight * np.sum(offsets[1:] ** 2))
        # 守方在前時不剔除，改把預測中心距離不足 r_col 的懲罰乘權重加入。
        leader_cost = float(self.config.leader_penalty_weight * leader_penalty)
        # J_race：獎勵進度，保留原本的橫移平順成本，再加偏離中心線與前車避碰成本。
        race_cost = float(
            -self.config.progress_weight * progress
            + self.config.smooth_weight * np.sum(np.diff(offsets) ** 2)
            + deviation_cost
            + leader_cost
        )
        # 對齊指定對象當下的 d，整段評分使用同一個位置。
        blocking_cost = 0.0
        if ego_d is not None:
            blocking_cost = float(
                self.config.q_block * np.sum((offsets[1:] - ego_d) ** 2)
            )
        return DefenseCandidate(
            target_d, maneuver, race_cost, blocking_cost, predicted_states=predicted_states,
            deviation_cost=deviation_cost, leader_cost=leader_cost,
        )

    def _check_times(self) -> tuple[float, ...] | None:
        """碰撞或責任檢查的時間點；其他車的預測都使用同一組時間。"""
        if self.last_rear_prediction is not None:
            return self.last_rear_prediction.times
        if self.last_other_predictions:
            return self.last_other_predictions[0].times
        return None

    def _responsibility_result(
        self, defender: VehicleObservation, own: "_OwnPrediction",
    ) -> ResponsibilityResult:
        """逐對檢查避撞責任：對方在前或並排時自車要保持 r_col；leader_soft 時自車在前只計懲罰。"""
        config = self.config.responsibility
        if config is None or own.check_states is None or own.check_progress is None:
            return ResponsibilityResult(None)
        pairs = tuple(
            PairPrediction(
                other.vehicle_id, other.geometry, other.states,
                # Δs＝起始間距＋對方進度－自車進度；正值表示對方在前。
                gap + np.asarray(other.progress) - own.check_progress,
            )
            for other, gap in zip(self.last_other_predictions, self.last_other_gaps)
        )
        return evaluate_responsibility(
            self.last_other_predictions[0].times, own.check_states,
            defender.geometry, pairs, config,
        )

    def _predict_geometric(
        self,
        defender: VehicleObservation,
        track: Track,
        maneuver: LateralManeuver,
        cost_times: np.ndarray,
        check_times: tuple[float, ...] | None,
    ) -> "_OwnPrediction":
        """原本的幾何預測：沿候選折線以目前車速走滿 v*t。"""
        candidate = replace(defender, maneuver=maneuver)
        times = cost_times if check_times is None else np.union1d(cost_times, check_times)
        prediction = predict_path_trajectory(candidate, track, times, self.config.path_sample_spacing)
        cost_indices = np.searchsorted(times, cost_times)
        check_states = None
        check_progress = None
        predicted_states = prediction.states
        if check_times is not None:
            check_indices = np.searchsorted(times, check_times)
            check_states = tuple(prediction.states[index] for index in check_indices)
            check_progress = np.asarray(prediction.progress)[check_indices]
            predicted_states = check_states
        # 走滿 v*T 後的 s_end - s_start；跨越起終點仍連續累積。
        progress = float(prediction.progress[cost_indices[-1]])
        return _OwnPrediction(
            prediction.offsets[cost_indices], progress, check_states, check_progress,
            predicted_states,
        )

    def _predict_bicycle(
        self,
        defender: VehicleObservation,
        track: Track,
        maneuver: LateralManeuver,
        cost_times: np.ndarray,
        check_times: tuple[float, ...] | None,
    ) -> "_OwnPrediction":
        """以防守車自己的 Pure Pursuit 與自行車模型 rollout，再插值到成本與碰撞時間點。"""
        if self.prediction_config is None:
            raise ValueError("bicycle prediction requires prediction_config")
        horizon = self.config.prediction_horizon
        if check_times is not None:
            # 碰撞條件的時長可能較長，rollout 需同時涵蓋兩者。
            horizon = max(horizon, check_times[-1])
        prediction = predict_bicycle_trajectory(
            defender, track, maneuver, self.prediction_config, horizon,
        )
        times = np.asarray(prediction.times)
        offsets = np.interp(cost_times, times, prediction.offsets)
        # 防守車依巡航速度加減速，進度取成本時域終點的 s_end - s_start。
        progress = float(np.interp(self.config.prediction_horizon, times, prediction.progress))
        check_states = None if check_times is None else states_at_times(prediction, check_times)
        check_progress = None if check_times is None else np.interp(
            check_times, times, prediction.progress,
        )
        return _OwnPrediction(offsets, progress, check_states, check_progress, prediction.states)


@dataclass(frozen=True)
class _OwnPrediction:
    """防守車單一候選的預測：成本時間點的 d 與進度，以及檢查時間點的狀態與進度。"""

    offsets: np.ndarray
    progress: float
    check_states: tuple[VehicleState, ...] | None
    check_progress: np.ndarray | None
    predicted_states: tuple[VehicleState, ...]
