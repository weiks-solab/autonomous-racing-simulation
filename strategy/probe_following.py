"""以目標車距跟車，輪流左右試探，觀察前車的橫向防守反應。"""

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import numpy as np

from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length, world_to_track
from core.validation import require_finite_values, require_nonnegative, require_positive
from core.vehicle_geometry import VehicleGeometry, oriented_rectangle_corners
from strategy.base import CandidatePath, CandidateSnapshot, StrategyDecision
from strategy.bicycle_prediction import (
    BicyclePrediction, BicyclePredictionConfig, constant_offset_maneuver,
    predict_bicycle_trajectory, prediction_config_for,
)
from strategy.collision_check import PredictedVehicle, first_predicted_collision
from strategy.common import (
    RelativeTargetState, accumulated_gap, check_other_ids, check_track_and_time,
    decision_due, lateral_targets, nearest_vehicle_ahead,
)
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver, start_or_continue_maneuver


@dataclass(frozen=True)
class ProbeFollowingConfig:
    target_gap: float = 1.5          # 前車 s 減後車 s，後軸參考點的距離，m。
    detection_range: float = 10.0
    prediction_horizon: float = 1.0
    decision_period: float = 0.1
    transition_length: float = 3.0
    speed_step: float = 1.0         # 比較前車速度減此值、相同速度、加此值。
    probe_offset: float = 0.6
    probe_gap_tolerance: float = 1.0
    lateral_tolerance: float = 0.1
    hold_seconds: float = 2.0       # 首次到達試探位置後，繼續保持目標多久。
    clearance: float = 0.3          # 前車車尾與後車車頭的最小縱向投影留距，m。
    safety_margin: float = 0.1
    gap_scale: float = 1.0          # 成本正規化尺度，m。
    speed_scale: float = 1.0        # 成本正規化尺度，m/s。
    lateral_scale: float = 0.6      # 成本正規化尺度，m。
    gap_weight: float = 2.0
    speed_weight: float = 1.0
    probe_weight: float = 2.0
    switch_weight: float = 0.5
    # 依序輪流的橫向試探目標 d（m，左正）；None 時為左、回中、右、回中。
    probe_sequence: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        require_positive(
            self, "target_gap", "detection_range", "prediction_horizon", "decision_period",
            "transition_length", "speed_step", "probe_offset", "probe_gap_tolerance",
            "lateral_tolerance", "hold_seconds", "gap_scale", "speed_scale", "lateral_scale",
        )
        require_nonnegative(self, "clearance", "safety_margin", "gap_weight",
                            "speed_weight", "probe_weight", "switch_weight")
        if self.target_gap <= self.clearance:
            raise ValueError("target_gap must exceed clearance")
        if self.probe_sequence is not None:
            object.__setattr__(self, "probe_sequence", tuple(float(value) for value in self.probe_sequence))
            if not self.probe_sequence:
                raise ValueError("ProbeFollowingConfig.probe_sequence must not be empty")
            require_finite_values(self, "probe_sequence")

    @property
    def sequence(self) -> tuple[float, ...]:
        """實際使用的試探目標序列。"""
        if self.probe_sequence is not None:
            return self.probe_sequence
        return (self.probe_offset, 0.0, -self.probe_offset, 0.0)


@dataclass(frozen=True)
class ProbeCandidate:
    """保存一組路徑與速度的預測，gaps 為後軸車距，clearances 為車身留距。"""

    maneuver: LateralManeuver
    speed_limit: float
    prediction: BicyclePrediction
    gaps: tuple[float, ...]
    clearances: tuple[float, ...]
    cost_terms: tuple[tuple[str, float], ...]
    reason: str = ""

    @property
    def feasible(self) -> bool:
        return not self.reason

    @property
    def total_cost(self) -> float:
        if not self.feasible:
            raise ValueError("rejected candidates do not have a cost")
        return sum(value for _, value in self.cost_terms)


def body_progress_bounds(
    prediction: BicyclePrediction, geometry: VehicleGeometry, track: Track,
) -> np.ndarray:
    """投影四個實體車身角點，得到相對於當下後軸的縱向最小／最大距離。"""
    length = track_length(track)
    bounds = []
    for state in prediction.states:
        axle = world_to_track(state, track)
        offsets = [
            signed_longitudinal_gap(
                axle, world_to_track(replace(state, x=float(x), y=float(y)), track), length,
            )
            for x, y in oriented_rectangle_corners(state, geometry)
        ]
        bounds.append((min(offsets), max(offsets)))
    return np.asarray(bounds)


class ProbeFollowingStrategy:
    """每車獨立保存跟車對象、跨圈間距、試探階段與保持時間。"""

    def __init__(
        self, config: ProbeFollowingConfig, prediction_config: BicyclePredictionConfig,
        other_prediction_configs: Mapping[str, BicyclePredictionConfig],
    ) -> None:
        self.config = config
        self.prediction_config = prediction_config
        self.other_prediction_configs = dict(other_prediction_configs)
        self.decision = StrategyDecision("follow")
        self.target_state: RelativeTargetState | None = None
        self.last_evaluation_time = -math.inf
        self.last_candidates: tuple[ProbeCandidate, ...] = ()
        self.phase_index = 0
        self.phase_active = False
        self.hold_started_at: float | None = None

    def update(
        self, vehicle: VehicleObservation, others: Sequence[VehicleObservation],
        track: Track, time_seconds: float,
    ) -> StrategyDecision:
        check_track_and_time("probe following", track, time_seconds, self.last_evaluation_time)
        check_other_ids(vehicle, others)
        if not decision_due(time_seconds, self.last_evaluation_time, self.config.decision_period):
            return self.decision
        self.last_evaluation_time = time_seconds

        target, gap = self._select_target(vehicle, others, track)
        if target is None:
            self.last_candidates = ()
            self.decision = StrategyDecision("cruise", reason="no front vehicle")
            return self.decision
        desired_d, mode = self._probe_target(vehicle, gap, time_seconds)
        obstacles = tuple(self._predict_other(other, track) for other in others)
        front = next(other for other in obstacles if other.vehicle_id == target.vehicle_id)
        front_bounds = body_progress_bounds(front.prediction, target.geometry, track)
        speeds = self._speed_limits(target.state.velocity)
        self.last_candidates = tuple(
            self._evaluate_candidate(vehicle, track, target_d, speed, gap, desired_d,
                                     front.prediction, front_bounds, obstacles)
            # 僅比較回中、目前目標與本次試探目標，每條路徑最多比較四種速度。
            for target_d in lateral_targets(
                (self.decision.target_d, desired_d), track, vehicle.geometry, self.config.safety_margin,
            )
            for speed in speeds
        )
        feasible = [candidate for candidate in self.last_candidates if candidate.feasible]
        # 試探期間固定橫向目標，只在該目標的可行速度之間比較成本。
        preferred = [candidate for candidate in feasible
                     if self.phase_active and candidate.maneuver.target_d == desired_d]
        if self.phase_active and not preferred:
            # 該側全部不可行時中止保持，沿用其他可行候選的選取流程。
            self.phase_active, self.hold_started_at = False, None
            mode = "follow"
        eligible = preferred or feasible
        best = min(eligible, key=lambda candidate: candidate.total_cost) if eligible else None
        snapshot = CandidateSnapshot(
            time_seconds, vehicle.track_position, (), tuple(
                CandidatePath(
                    maneuver=candidate.maneuver, speed_limit=candidate.speed_limit,
                    selected=candidate is best, feasible=candidate.feasible,
                    reason=candidate.reason, cost_terms=candidate.cost_terms,
                    predicted_states=candidate.prediction.states,
                )
                for candidate in self.last_candidates
            ),
        )
        if best is None:
            self.hold_started_at = None
            self.decision = replace(
                self.decision, mode="no_feasible_path", candidate_snapshot=snapshot,
                reason="all candidates rejected; continuing previous decision",
            )
            return self.decision
        self.decision = StrategyDecision(
            mode, target_d=best.maneuver.target_d, target_vehicle_id=target.vehicle_id,
            maneuver=best.maneuver, speed_limit=best.speed_limit,
            candidate_snapshot=snapshot,
            reason="holding probe target; lowest cost at this target" if self.phase_active
                   else "lowest gap, speed, probe and switch cost",
            details=(f"gap={gap:.2f}/{self.config.target_gap:.2f} m",
                     f"probe_d={desired_d:+.2f} J={best.total_cost:.2f}"),
        )
        return self.decision

    def _select_target(
        self, vehicle: VehicleObservation, others: Sequence[VehicleObservation], track: Track,
    ) -> tuple[VehicleObservation | None, float]:
        """一次試探鎖定同一前車，沿用既有累積進度表示法處理跨起終點。"""
        previous = self.target_state
        target = None if previous is None else next(
            (other for other in others if other.vehicle_id == previous.vehicle_id), None)
        if target is None:
            # 換新對象時，試探循環從頭開始。
            self.phase_index, self.hold_started_at = 0, None
            self.phase_active = False
            target, gap = nearest_vehicle_ahead(vehicle, others, track, self.config.detection_range)
            if target is None:
                self.target_state = None
                return None, gap
        else:
            gap = accumulated_gap(previous, vehicle, target, track)
        self.target_state = RelativeTargetState(
            target.vehicle_id, vehicle.track_position, target.track_position, gap,
        )
        return target, gap

    def _probe_target(
        self, vehicle: VehicleObservation, gap: float, time_seconds: float,
    ) -> tuple[float, str]:
        # 車距只控制是否啟動新試探；已啟動的目標不因車距波動立刻回中。
        in_range = abs(gap - self.config.target_gap) <= self.config.probe_gap_tolerance
        if not self.phase_active and not in_range:
            self.hold_started_at = None
            return 0.0, "follow"
        self.phase_active = True
        sequence = self.config.sequence
        desired_d = sequence[self.phase_index]
        reached = abs(vehicle.track_position.d - desired_d) <= self.config.lateral_tolerance
        # 首次實際到達後才計時；正常追蹤誤差不重置計時。
        if self.hold_started_at is None and reached:
            self.hold_started_at = time_seconds
        elif (self.hold_started_at is not None
              and time_seconds - self.hold_started_at >= self.config.hold_seconds - 1e-9):
            self.phase_index = (self.phase_index + 1) % len(sequence)
            self.hold_started_at = None
            desired_d = sequence[self.phase_index]
            # 返回中心線可接續執行；下一次向側邊試探仍須車距合適。
            self.phase_active = desired_d == 0.0 or in_range
            if not self.phase_active:
                return 0.0, "follow"
        mode = "return" if desired_d == 0.0 else "probe_left" if desired_d > 0.0 else "probe_right"
        return desired_d, mode

    def _predict_other(self, other: VehicleObservation, track: Track) -> PredictedVehicle:
        config = prediction_config_for(self.other_prediction_configs, other.vehicle_id)
        # 沿用定速、等 d 的對手假設，實際防守仍由對手自己的策略執行。
        prediction = predict_bicycle_trajectory(
            other, track, constant_offset_maneuver(other, config), config,
            self.config.prediction_horizon, constant_speed=True,
        )
        return PredictedVehicle(other.vehicle_id, other.geometry, prediction)

    def _speed_limits(self, front_speed: float) -> tuple[float, ...]:
        cruise = self.prediction_config.controller.target_speed
        speeds = [
            min(cruise, max(0.0, front_speed + offset))
            for offset in (-self.config.speed_step, 0.0, self.config.speed_step)
        ]
        # 加入原巡航速度供遠距追近；仍須通過相同的 rollout 與碰撞檢查。
        speeds.append(cruise)
        return tuple(dict.fromkeys(speeds))

    def _evaluate_candidate(
        self, vehicle: VehicleObservation, track: Track, target_d: float, speed_limit: float,
        gap: float, desired_d: float, front: BicyclePrediction, front_bounds: np.ndarray,
        obstacles: Sequence[PredictedVehicle],
    ) -> ProbeCandidate:
        maneuver = start_or_continue_maneuver(
            self.decision.maneuver, vehicle.track_position, target_d,
            self.config.transition_length, track,
        )
        prediction = predict_bicycle_trajectory(
            vehicle, track, maneuver, self.prediction_config, self.config.prediction_horizon,
            speed_limit=speed_limit,
        )
        own_progress = np.asarray(prediction.progress)
        front_progress = np.interp(prediction.times, front.times, front.progress)
        gaps = gap + front_progress - own_progress
        own_bounds = body_progress_bounds(prediction, vehicle.geometry, track)
        clearances = gaps + np.interp(prediction.times, front.times, front_bounds[:, 0]) - own_bounds[:, 1]
        collision = first_predicted_collision(prediction, vehicle.geometry, obstacles)
        reason = "" if collision is None else collision.reason
        if np.any(gaps <= 0.0) or np.any(clearances < self.config.clearance):
            reason = "front vehicle longitudinal clearance violated"
        terms = ()
        if not reason:
            dt = prediction.times[-1] - prediction.times[-2]
            relative_speed = ((own_progress[-1] - own_progress[-2])
                              - (front_progress[-1] - front_progress[-2])) / dt
            end_d = world_to_track(prediction.states[-1], track).d
            config = self.config
            # 四項成本均先除以尺度，取總和最小的候選。
            terms = (
                ("gap", float(config.gap_weight * np.mean(((gaps[1:] - config.target_gap) / config.gap_scale)**2))),
                ("speed", config.speed_weight * (relative_speed / config.speed_scale)**2),
                ("probe", config.probe_weight * ((end_d - desired_d) / config.lateral_scale)**2),
                ("switch", config.switch_weight * ((target_d - self.decision.target_d) / config.lateral_scale)**2),
            )
        return ProbeCandidate(maneuver, speed_limit, prediction, tuple(gaps), tuple(clearances), terms, reason)
