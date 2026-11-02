"""依橫向對齊成本，從少量平滑路徑中選擇防守位置。"""

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Sequence

import numpy as np

from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length, world_to_track
from core.vehicle_geometry import oriented_rectangle_corners, rectangles_overlap
from core.vehicle_state import VehicleState
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver
from strategy.prediction import predict_vehicle_state, predicted_offset


class DefenseMode(Enum):
    CRUISE = "cruise"
    DEFEND = "defend"
    HOLD = "hold"
    RETURN = "return"


@dataclass(frozen=True)
class DefenseConfig:
    """距離單位為公尺，時間單位為秒；常用設定集中在 main.py。"""

    q_block: float = 6.0
    center_weight: float = 1.0
    smooth_weight: float = 2.0
    detection_range: float = 8.0
    minimum_speed_advantage: float = 0.2
    minimum_gap: float = 1.2
    return_tolerance: float = 0.08
    transition_length: float = 3.0
    decision_period: float = 0.2
    hold_time: float = 0.6
    switch_improvement: float = 0.05
    candidate_offsets: tuple[float, ...] = (-0.6, -0.3, 0.0, 0.3, 0.6)
    prediction_horizon: float = 1.5
    prediction_step: float = 0.1
    safety_margin: float = 0.1

    def __post_init__(self) -> None:
        nonnegative = (
            self.q_block, self.center_weight, self.smooth_weight,
            self.minimum_speed_advantage, self.return_tolerance,
            self.hold_time, self.switch_improvement, self.safety_margin,
        )
        positive = (
            self.detection_range, self.minimum_gap, self.transition_length,
            self.decision_period, self.prediction_horizon, self.prediction_step,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError(
                "defense weights, margins and tolerances must be non-negative and finite"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("defense distances and timing must be positive and finite")
        if self.minimum_gap >= self.detection_range:
            raise ValueError("minimum_gap must be smaller than detection_range")
        if not all(math.isfinite(value) for value in self.candidate_offsets):
            raise ValueError("candidate_offsets must be finite")


@dataclass(frozen=True)
class DefenseCandidate:
    target_d: float
    maneuver: LateralManeuver
    race_cost: float
    blocking_cost: float
    feasible: bool
    reason: str

    @property
    def total_cost(self) -> float:
        return self.race_cost + self.blocking_cost


@dataclass(frozen=True)
class DefenseDecision:
    mode: DefenseMode
    target_d: float
    target_vehicle_id: str | None
    maneuver: LateralManeuver | None
    reason: str
    race_cost: float = 0.0
    blocking_cost: float = 0.0

    @property
    def display_text(self) -> str:
        return (
            f"{self.mode.value.upper()}\ntarget_d={self.target_d:+.2f}"
            f"\nrace={self.race_cost:.2f} block={self.blocking_cost:.2f}"
        )


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
        self.last_switch_time = -math.inf

    def update(
        self,
        defender: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> DefenseDecision:
        # 1. 檢查輸入；只在固定間隔重新比較路徑。
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
        self.last_evaluation_time = time_seconds

        length = track_length(track)
        ego = next((other for other in others if other.vehicle_id == self.ego_id), None)
        gap = math.inf if ego is None else signed_longitudinal_gap(
            defender.track_position, ego.track_position, length
        )
        # 2. 只有指定 ego 在後方追近時，才啟用防守成本。
        threat = (
            ego is not None
            and -self.config.detection_range <= gap < -self.config.minimum_gap
            and ego.state.velocity - defender.state.velocity
            >= self.config.minimum_speed_advantage
            and self.config.q_block > 0.0
        )
        nearby = any(
            abs(signed_longitudinal_gap(defender.track_position, other.track_position, length))
            <= self.config.minimum_gap
            for other in others
        )
        mode = DefenseMode.DEFEND if threat else DefenseMode.RETURN
        if nearby:
            mode = DefenseMode.HOLD

        if not threat and not nearby and self.decision.maneuver is None:
            self.last_candidates = ()
            self.decision = DefenseDecision(
                DefenseMode.CRUISE, 0.0, None, None, "following centerline"
            )
            return self.decision

        # 3. 評估各條候選，先排除越界或預測碰撞的路徑。
        targets = self._candidate_targets(defender, ego, track, threat, nearby)
        evaluations = tuple(
            self._evaluate_candidate(defender, others, track, target, ego, threat)
            for target in targets
        )
        self.last_candidates = evaluations
        feasible = [candidate for candidate in evaluations if candidate.feasible]
        if not feasible:
            # 幾何預測無解時保留原路徑，並明確回報；不假裝能保證避碰。
            self.decision = replace(
                self.decision, mode=DefenseMode.HOLD,
                reason="no feasible candidate; continuing previous path",
            )
            return self.decision

        # 4. 比較成本；原路徑安全時，避免頻繁換目標。
        best = min(feasible, key=lambda candidate: candidate.total_cost)
        current = next(
            (candidate for candidate in feasible
             if math.isclose(candidate.target_d, self.decision.target_d)),
            None,
        )
        # 原路徑仍可行時，短暫鎖定目標，且只接受明顯更好的新選擇。
        if threat and not nearby and current is not None:
            locked = time_seconds - self.last_switch_time < self.config.hold_time
            small_gain = (
                current.total_cost - best.total_cost <= self.config.switch_improvement
            )
            if locked or small_gain:
                best = current

        changed = not math.isclose(best.target_d, self.decision.target_d)
        if changed:
            self.last_switch_time = time_seconds
        maneuver = best.maneuver
        centered = abs(defender.track_position.d) <= self.config.return_tolerance
        if abs(best.target_d) < 1e-9 and centered:
            maneuver = None
            if not threat and not nearby:
                mode = DefenseMode.CRUISE
        elif not threat and not nearby and abs(best.target_d) > 1e-9:
            mode = DefenseMode.HOLD

        self.decision = DefenseDecision(
            mode=mode,
            target_d=best.target_d,
            target_vehicle_id=self.ego_id if threat or nearby else None,
            maneuver=maneuver,
            reason=("nearby vehicle; do not chase laterally"
                    if nearby else "lowest feasible cost"),
            race_cost=best.race_cost,
            blocking_cost=best.blocking_cost,
        )
        return self.decision

    def _candidate_targets(
        self,
        defender: VehicleObservation,
        ego: VehicleObservation | None,
        track: Track,
        threat: bool,
        nearby: bool,
    ) -> tuple[float, ...]:
        current_d = defender.track_position.d
        if nearby:
            # 近距離只比較繼續原路徑與停止橫移，不再追著 ego 換邊。
            targets = [self.decision.target_d, current_d]
        elif threat and ego is not None:
            maximum_d = max(
                0.0,
                track.width / 2.0 - defender.geometry.width / 2.0 - self.config.safety_margin,
            )
            ego_d = float(np.clip(ego.track_position.d, -maximum_d, maximum_d))
            targets = [
                self.decision.target_d, current_d, 0.0,
                *self.config.candidate_offsets, ego_d,
            ]
        else:
            targets = [0.0, self.decision.target_d, current_d]
        return tuple(dict.fromkeys(targets))

    def _evaluate_candidate(
        self,
        defender: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        target_d: float,
        ego: VehicleObservation | None,
        threat: bool,
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
        times = np.linspace(0.0, self.config.prediction_horizon, step_count + 1)
        offsets = np.array([
            predicted_offset(candidate, max(defender.state.velocity, 0.0) * float(time), track)
            for time in times
        ])
        offsets[0] = defender.track_position.d

        # J_race：偏好中心線，並減少突然的橫向變化。
        race_cost = float(
            self.config.center_weight * np.sum(offsets[1:] ** 2)
            + self.config.smooth_weight * np.sum(np.diff(offsets) ** 2)
        )
        # 原公式的 ego 位置固定為「現在的 d」，不是 ego 未來的位置。
        blocking_cost = 0.0
        if threat and ego is not None:
            blocking_cost = float(
                self.config.q_block * np.sum((offsets[1:] - ego.track_position.d) ** 2)
            )

        reason = self._check_path(candidate, others, track, target_d, times)
        return DefenseCandidate(
            target_d, maneuver, race_cost, blocking_cost,
            reason == "clear", reason,
        )

    def _check_path(
        self,
        candidate: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        target_d: float,
        times: np.ndarray,
    ) -> str:
        maximum_d = (
            track.width / 2.0 - candidate.geometry.width / 2.0 - self.config.safety_margin
        )
        if maximum_d < 0.0 or abs(target_d) > maximum_d:
            return "track width exceeded"
        for time in times:
            state = predict_vehicle_state(candidate, float(time), track)
            # 逐一檢查車身四個角，包含彎道與橫移時的車身旋轉。
            for x, y in oriented_rectangle_corners(state, candidate.geometry):
                corner = world_to_track(
                    VehicleState(float(x), float(y), state.yaw, 0.0), track
                )
                if abs(corner.d) > track.width / 2.0 - self.config.safety_margin:
                    return "body corner outside track margin"
            for other in others:
                other_state = predict_vehicle_state(other, float(time), track)
                if rectangles_overlap(state, candidate.geometry, other_state, other.geometry):
                    return f"predicted overlap with {other.vehicle_id}"
        return "clear"
