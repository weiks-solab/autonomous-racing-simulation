"""只比較自車五條候選的預測進度，不預測其他車或檢查碰撞。"""

from dataclasses import dataclass
import math
from typing import Sequence

from core.track import Track
from core.validation import require_finite_values, require_nonnegative, require_positive
from strategy.base import CandidatePath, CandidateSnapshot, StrategyDecision
from strategy.bicycle_prediction import (
    BicyclePrediction, BicyclePredictionConfig, predict_bicycle_trajectory,
)
from strategy.common import check_track_and_time, decision_due, lateral_targets
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver, start_or_continue_maneuver


@dataclass(frozen=True)
class ProgressConfig:
    candidate_offsets: tuple[float, ...] = (-0.6, -0.3, 0.0, 0.3, 0.6)
    transition_length: float = 3.0
    decision_period: float = 0.1
    prediction_horizon: float = 1.5
    safety_margin: float = 0.1

    def __post_init__(self) -> None:
        require_positive(self, "transition_length", "decision_period", "prediction_horizon")
        require_nonnegative(self, "safety_margin")
        require_finite_values(self, "candidate_offsets")


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
        check_track_and_time("progress strategy", track, time_seconds, self.last_evaluation_time)
        if not decision_due(time_seconds, self.last_evaluation_time, self.config.decision_period):
            return self.decision

        self.last_candidates = tuple(
            self._evaluate_candidate(vehicle, track, target_d)
            for target_d in lateral_targets(
                self.config.candidate_offsets, track, vehicle.geometry, self.config.safety_margin,
            )
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

    def _evaluate_candidate(
        self, vehicle: VehicleObservation, track: Track, target_d: float,
    ) -> ProgressCandidate:
        maneuver = start_or_continue_maneuver(
            self.decision.maneuver, vehicle.track_position, target_d,
            self.config.transition_length, track,
        )
        prediction = predict_bicycle_trajectory(
            vehicle, track, maneuver, self.prediction_config, self.config.prediction_horizon,
        )
        return ProgressCandidate(maneuver, prediction)
