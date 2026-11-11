"""先篩選預測碰撞，再只比較預測終點的對手 s 減自車 s。"""

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from core.track import Track
from core.track_geometry import TrackPosition, signed_longitudinal_gap, track_length
from strategy.base import CandidatePath, CandidateSnapshot, StrategyDecision
from strategy.bicycle_prediction import (
    BicyclePrediction, BicyclePredictionConfig, predict_bicycle_trajectory,
)
from strategy.collision_check import PredictedCollision, PredictedVehicle, first_predicted_collision
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver, continue_lateral_maneuver


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

    def __post_init__(self) -> None:
        positive = (self.detection_range, self.transition_length,
                    self.decision_period, self.prediction_horizon)
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("relative overtaking distances and timing must be positive and finite")
        if not math.isfinite(self.safety_margin) or self.safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative and finite")
        if not all(math.isfinite(value) for value in self.candidate_offsets):
            raise ValueError("candidate_offsets must be finite")


@dataclass(frozen=True)
class RelativeCandidate:
    maneuver: LateralManeuver
    prediction: BicyclePrediction
    predicted_gap: float | None
    collision: PredictedCollision | None = None

    @property
    def feasible(self) -> bool:
        return self.collision is None

    @property
    def total_cost(self) -> float:
        if not self.feasible:
            raise ValueError("collision-rejected candidates do not have a cost")
        # 沒有比較對象就沒有相對成本；不另加自車進度項。
        return 0.0 if self.predicted_gap is None else self.predicted_gap


@dataclass(frozen=True)
class RelativeTargetState:
    """保存同一對象的累積間距，避免領先半圈或一圈時跳回最短距離。"""

    vehicle_id: str
    vehicle_position: TrackPosition
    target_position: TrackPosition
    gap: float


class RelativeOvertakingStrategy:
    """每台車持有獨立狀態；其他車每輪只預測一次，供所有候選共用。"""

    def __init__(
        self,
        config: RelativeOvertakingConfig,
        prediction_config: BicyclePredictionConfig,
        other_prediction_configs: Mapping[str, BicyclePredictionConfig],
    ) -> None:
        self.config = config
        self.prediction_config = prediction_config
        self.other_prediction_configs = dict(other_prediction_configs)
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
        if not track.closed:
            raise ValueError("relative overtaking requires a closed track")
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("time_seconds must be non-negative and finite")
        if time_seconds < self.last_evaluation_time:
            raise ValueError("time_seconds must not move backwards")
        ids = [other.vehicle_id for other in others]
        if len(set(ids)) != len(ids) or vehicle.vehicle_id in ids:
            raise ValueError("vehicle IDs must be unique; exclude self from others")
        if time_seconds - self.last_evaluation_time < self.config.decision_period:
            return self.decision

        target, gap = self._select_target(vehicle, others, track)
        self.last_other_predictions = {
            other.vehicle_id: self._predict_other(other, track) for other in others
        }
        self.last_target_prediction = (
            None if target is None else self.last_other_predictions[target.vehicle_id]
        )
        obstacles = tuple(
            PredictedVehicle(other.vehicle_id, other.geometry,
                             self.last_other_predictions[other.vehicle_id])
            for other in others
        )
        self.last_candidates = tuple(
            self._evaluate_candidate(vehicle, track, target_d, gap, obstacles)
            for target_d in self._candidate_targets(vehicle, track)
        )
        feasible = [candidate for candidate in self.last_candidates if candidate.feasible]
        # 同分優先保留原目標；沒有比較對象時也不偷偷加入自車進度項。
        best = min(feasible, key=lambda candidate: (
            candidate.total_cost, candidate.maneuver.target_d != self.decision.target_d,
        )) if feasible else None
        snapshot = self._snapshot(vehicle, time_seconds, best)
        self.last_evaluation_time = time_seconds
        if best is None:
            self.decision = replace(
                self.decision, mode="no_feasible_path",
                reason="all candidates predict overlap; continuing previous path",
                details=("continuing previous path",), candidate_snapshot=snapshot,
            )
            return self.decision

        self.decision = StrategyDecision(
            mode="cruise" if target is None else "attack",
            target_d=best.maneuver.target_d,
            target_vehicle_id=None if target is None else target.vehicle_id,
            maneuver=best.maneuver,
            reason="no target vehicle; prefer previous feasible target" if target is None
            else "lowest predicted relative gap",
            details=("no target vehicle; collision filtering only",) if target is None
            else (f"gap_T={best.total_cost:+.2f} m", f"J={best.total_cost:.2f}"),
            candidate_snapshot=snapshot,
        )
        return self.decision

    def _select_target(
        self, vehicle: VehicleObservation, others: Sequence[VehicleObservation], track: Track,
    ) -> tuple[VehicleObservation | None, float]:
        """優先選最近的新前車；沒有前車時保留原對象，超車後仍繼續評分。"""
        target, gap = self._front_vehicle(vehicle, others, track)
        previous = self.target_state
        if target is None and previous is not None:
            target = next((other for other in others if other.vehicle_id == previous.vehicle_id), None)
        if target is None:
            self.target_state = None
            return None, 0.0
        if previous is not None and previous.vehicle_id == target.vehicle_id:
            length = track_length(track)
            own_progress = signed_longitudinal_gap(previous.vehicle_position,
                                                   vehicle.track_position, length)
            target_progress = signed_longitudinal_gap(previous.target_position,
                                                      target.track_position, length)
            gap = previous.gap + target_progress - own_progress
        self.target_state = RelativeTargetState(
            target.vehicle_id, vehicle.track_position, target.track_position, gap,
        )
        return target, gap

    def _front_vehicle(
        self, vehicle: VehicleObservation, others: Sequence[VehicleObservation], track: Track,
    ) -> tuple[VehicleObservation | None, float]:
        length = track_length(track)
        ahead = [
            (signed_longitudinal_gap(vehicle.track_position, other.track_position, length),
             other.vehicle_id, other)
            for other in others
        ]
        ahead = [item for item in ahead if 0.0 < item[0] <= self.config.detection_range]
        if not ahead:
            return None, 0.0
        gap, _, target = min(ahead, key=lambda item: item[:2])
        return target, gap

    def _predict_other(self, other: VehicleObservation, track: Track) -> BicyclePrediction:
        config = self.other_prediction_configs.get(other.vehicle_id)
        if config is None:
            raise ValueError(f"missing prediction configuration for {other.vehicle_id}")
        maneuver = LateralManeuver(
            other.track_position.s, other.track_position.d,
            other.track_position.d, config.path.path_length,
        )
        return predict_bicycle_trajectory(
            other, track, maneuver, config, self.config.prediction_horizon, constant_speed=True,
        )

    def _candidate_targets(self, vehicle: VehicleObservation, track: Track) -> tuple[float, ...]:
        maximum_d = track.width / 2.0 - vehicle.geometry.width / 2.0 - self.config.safety_margin
        if maximum_d < 0.0:
            raise ValueError("track is too narrow for vehicle width and safety_margin")
        return tuple(dict.fromkeys(
            target for target in (0.0, *self.config.candidate_offsets)
            if abs(target) <= maximum_d
        ))

    def _evaluate_candidate(
        self, vehicle: VehicleObservation, track: Track, target_d: float, gap: float,
        obstacles: Sequence[PredictedVehicle],
    ) -> RelativeCandidate:
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
        collision = first_predicted_collision(prediction, vehicle.geometry, obstacles)
        if collision is not None:
            return RelativeCandidate(maneuver, prediction, None, collision)
        opponent = self.last_target_prediction
        # 用累積進度算終點間距，跨起終點不跳變；保留負值表示自車領先。
        predicted_gap = None if opponent is None else gap + opponent.progress[-1] - prediction.progress[-1]
        return RelativeCandidate(maneuver, prediction, predicted_gap)

    def _snapshot(
        self, vehicle: VehicleObservation, time_seconds: float, best: RelativeCandidate | None,
    ) -> CandidateSnapshot:
        return CandidateSnapshot(
            evaluated_at=time_seconds, position=vehicle.track_position, distances=(),
            candidates=tuple(
                CandidatePath(
                    maneuver=candidate.maneuver, selected=candidate is best,
                    feasible=candidate.feasible,
                    reason="" if candidate.collision is None else candidate.collision.reason,
                    cost_terms=(("relative_gap", candidate.total_cost),)
                    if candidate.feasible and candidate.predicted_gap is not None else (),
                    predicted_states=candidate.prediction.states,
                )
                for candidate in self.last_candidates
            ),
        )
