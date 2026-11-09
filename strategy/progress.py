"""只比較自車五條候選的預測進度，不預測其他車或檢查碰撞。"""

from dataclasses import dataclass
import math
from typing import Sequence

from core.track import Track
from core.track_geometry import track_length
from strategy.base import CandidatePath, CandidateSnapshot, StrategyDecision
from strategy.bicycle_prediction import (
    BicyclePrediction, BicyclePredictionConfig, predict_bicycle_trajectory,
)
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver, continue_lateral_maneuver


@dataclass(frozen=True)
class ProgressConfig:
    candidate_offsets: tuple[float, ...] = (-0.6, -0.3, 0.0, 0.3, 0.6)
    transition_length: float = 3.0
    decision_period: float = 0.1
    prediction_horizon: float = 1.5
    safety_margin: float = 0.1

    def __post_init__(self) -> None:
        positive = (self.transition_length, self.decision_period, self.prediction_horizon)
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("progress distances and timing must be positive and finite")
        if not math.isfinite(self.safety_margin) or self.safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative and finite")
        if not all(math.isfinite(value) for value in self.candidate_offsets):
            raise ValueError("candidate_offsets must be finite")


@dataclass(frozen=True)
class ProgressCandidate:
    maneuver: LateralManeuver
    prediction: BicyclePrediction

    @property
    def progress(self) -> float:
        return self.prediction.progress[-1]


class ProgressStrategy:
    """以 Pure Pursuit 與自行車模型預測自車，只輸出最大進度的橫向目標。"""

    def __init__(self, config: ProgressConfig, prediction_config: BicyclePredictionConfig) -> None:
        self.config = config
        self.prediction_config = prediction_config
        self.decision = StrategyDecision("progress")
        self.last_evaluation_time = -math.inf
        self.last_candidates: tuple[ProgressCandidate, ...] = ()

    def update(
        self,
        vehicle: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> StrategyDecision:
        """others 僅保留共用介面；其他車的位置、速度與路徑不影響選擇。"""
        if not track.closed:
            raise ValueError("progress strategy requires a closed track")
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("time_seconds must be non-negative and finite")
        if time_seconds < self.last_evaluation_time:
            raise ValueError("time_seconds must not move backwards")
        if time_seconds - self.last_evaluation_time < self.config.decision_period:
            return self.decision

        self.last_candidates = tuple(
            self._evaluate_candidate(vehicle, track, target_d)
            for target_d in self._candidate_targets(vehicle, track)
        )
        best = max(self.last_candidates, key=lambda candidate: candidate.progress)
        snapshot = CandidateSnapshot(
            evaluated_at=time_seconds,
            position=vehicle.track_position,
            distances=(),
            candidates=tuple(
                CandidatePath(
                    maneuver=candidate.maneuver,
                    selected=candidate is best,
                    cost_terms=(("progress", -candidate.progress),),
                    predicted_states=candidate.prediction.states,
                )
                for candidate in self.last_candidates
            ),
        )
        self.decision = StrategyDecision(
            mode="progress",
            target_d=best.maneuver.target_d,
            maneuver=best.maneuver,
            reason="maximum predicted progress; other vehicles ignored",
            details=(f"ds={best.progress:.2f}", f"J={-best.progress:.2f}"),
            candidate_snapshot=snapshot,
        )
        self.last_evaluation_time = time_seconds
        return self.decision

    def _candidate_targets(self, vehicle: VehicleObservation, track: Track) -> tuple[float, ...]:
        maximum_d = track.width / 2.0 - vehicle.geometry.width / 2.0 - self.config.safety_margin
        if maximum_d < 0.0:
            raise ValueError("track is too narrow for vehicle width and safety_margin")
        return tuple(dict.fromkeys(
            target for target in (0.0, *self.config.candidate_offsets)
            if abs(target) <= maximum_d
        ))

    def _evaluate_candidate(
        self, vehicle: VehicleObservation, track: Track, target_d: float,
    ) -> ProgressCandidate:
        maneuver = self.decision.maneuver
        if maneuver is None or maneuver.target_d != target_d:
            maneuver = LateralManeuver(
                vehicle.track_position.s, vehicle.track_position.d,
                target_d, self.config.transition_length,
            )
        else:
            maneuver = continue_lateral_maneuver(
                maneuver, vehicle.track_position.s, track_length(track),
            )
        prediction = predict_bicycle_trajectory(
            vehicle, track, maneuver, self.prediction_config, self.config.prediction_horizon,
        )
        return ProgressCandidate(maneuver, prediction)
