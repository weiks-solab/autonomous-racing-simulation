"""依橫向對齊成本，從少量平滑路徑中選擇防守位置。"""

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Sequence

import numpy as np

from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length
from core.vehicle_state import VehicleState
from strategy.collision_check import PredictedCollision
from strategy.defense_conditions import RearPrediction, has_predicted_collision, predict_rear
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver
from strategy.prediction import predict_path_trajectory


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

    def __post_init__(self) -> None:
        nonnegative = (
            self.q_block, self.progress_weight, self.smooth_weight,
            self.return_tolerance, self.safety_margin,
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

    def __init__(self, config: DefenseConfig, ego_id: str) -> None:
        if not isinstance(ego_id, str) or not ego_id.strip():
            raise ValueError("ego_id must be a non-empty string")
        self.config = config
        self.ego_id = ego_id
        self.decision = DefenseDecision(
            DefenseMode.CRUISE, 0.0, None, None, "following centerline"
        )
        self.last_candidates: tuple[DefenseCandidate, ...] = ()
        self.last_evaluation_time = -math.inf
        self.last_rear_prediction: RearPrediction | None = None

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
            and -self.config.detection_range <= gap < -2.0
            and self.config.q_block > 0.0
        )
        ego_d = ego.track_position.d if defending and ego is not None else None
        self.last_rear_prediction = None
        if self.config.collision_check_enabled and ego is not None:
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
        candidate = replace(defender, maneuver=maneuver)
        step_count = math.ceil(self.config.prediction_horizon / self.config.prediction_step)
        cost_times = np.linspace(0.0, self.config.prediction_horizon, step_count + 1)
        rear = self.last_rear_prediction
        times = cost_times if rear is None else np.union1d(cost_times, rear.times)
        prediction = predict_path_trajectory(candidate, track, times, self.config.path_sample_spacing)
        cost_indices = np.searchsorted(times, cost_times)
        predicted_states = prediction.states
        if rear is not None:
            predicted_states = tuple(
                prediction.states[index] for index in np.searchsorted(times, rear.times)
            )
            collision = has_predicted_collision(defender, predicted_states, rear)
            if collision is not None:
                return DefenseCandidate(target_d, maneuver, None, None, collision, predicted_states)
        offsets = prediction.offsets[cost_indices]
        # 走滿 v*T 後的 s_end - s_start；跨越起終點仍連續累積。
        progress = float(prediction.progress[cost_indices[-1]])
        # J_race：獎勵進度，保留原本的橫移平順成本。
        race_cost = float(
            -self.config.progress_weight * progress
            + self.config.smooth_weight * np.sum(np.diff(offsets) ** 2)
        )
        # 對齊指定對象當下的 d，整段評分使用同一個位置。
        blocking_cost = 0.0
        if ego_d is not None:
            blocking_cost = float(
                self.config.q_block * np.sum((offsets[1:] - ego_d) ** 2)
            )
        return DefenseCandidate(
            target_d, maneuver, race_cost, blocking_cost, predicted_states=predicted_states,
        )
