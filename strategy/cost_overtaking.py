"""五條候選先剔除預測碰撞，再依進度與超車獎勵選擇。"""

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length
from strategy.base import CandidatePath, CandidateSnapshot, StrategyDecision
from strategy.bicycle_prediction import (
    BicyclePrediction,
    BicyclePredictionConfig,
    predict_bicycle_trajectory,
)
from strategy.observations import VehicleObservation
from strategy.collision_check import PredictedCollision, PredictedVehicle, first_predicted_collision
from strategy.offset_path import LateralManeuver, continue_lateral_maneuver


@dataclass(frozen=True)
class CostOvertakingConfig:
    progress_weight: float = 1.0
    overtaking_weight: float = 10.0
    candidate_offsets: tuple[float, ...] = (
    -0.6, -0.45, -0.3, -0.15,
     0.0,
     0.15, 0.3, 0.45, 0.6,
)
    detection_range: float = 6.0
    transition_length: float = 6.0
    decision_period: float = 0.2
    prediction_horizon: float = 5.0
    safety_margin: float = 0.1

    def __post_init__(self) -> None:
        nonnegative = (self.progress_weight, self.overtaking_weight, self.safety_margin)
        positive = (self.detection_range, self.transition_length,
                    self.decision_period, self.prediction_horizon)
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("attack weights and margin must be non-negative and finite")
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("attack distances and timing must be positive and finite")
        if not all(math.isfinite(value) for value in self.candidate_offsets):
            raise ValueError("candidate_offsets must be finite")


@dataclass(frozen=True)
class AttackCandidate:
    maneuver: LateralManeuver
    prediction: BicyclePrediction
    progress_cost: float | None
    overtaking_cost: float | None
    pass_reward: float | None
    collision: PredictedCollision | None = None

    @property
    def feasible(self) -> bool:
        return self.collision is None

    @property
    def total_cost(self) -> float:
        if self.progress_cost is None or self.overtaking_cost is None:
            raise ValueError("collision-rejected candidates do not have a cost")
        return self.progress_cost + self.overtaking_cost


class CostBasedOvertakingStrategy:
    """實作共用 update 介面；每台車獨立持有決策與預測快照。"""

    def __init__(
        self,
        config: CostOvertakingConfig,
        prediction_config: BicyclePredictionConfig,
        other_prediction_configs: Mapping[str, BicyclePredictionConfig],
    ) -> None:
        self.config = config
        self.prediction_config = prediction_config
        self.other_prediction_configs = dict(other_prediction_configs)
        self.decision = StrategyDecision("progress")
        self.last_evaluation_time = -math.inf
        self.last_candidates: tuple[AttackCandidate, ...] = ()
        self.last_target_prediction: BicyclePrediction | None = None
        self.last_other_predictions: dict[str, BicyclePrediction] = {}

    def update(
        self,
        vehicle: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
        time_seconds: float,
    ) -> StrategyDecision:
        if not track.closed:
            raise ValueError("cost overtaking requires a closed track")
        if not math.isfinite(time_seconds) or time_seconds < 0.0:
            raise ValueError("time_seconds must be non-negative and finite")
        if time_seconds < self.last_evaluation_time:
            raise ValueError("time_seconds must not move backwards")
        ids = [other.vehicle_id for other in others]
        if len(set(ids)) != len(ids) or vehicle.vehicle_id in ids:
            raise ValueError("vehicle IDs must be unique; exclude self from others")
        if time_seconds - self.last_evaluation_time < self.config.decision_period:
            return self.decision

        target, gap = self._front_vehicle(vehicle, others, track)
        # 所有其他車每輪只預測一次；偵測距離只影響超車獎勵對象。
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
        best = min(feasible, key=lambda candidate: candidate.total_cost) if feasible else None
        snapshot = self._snapshot(vehicle, time_seconds, best)
        if best is None:
            # 全部被剔除時沿用原路徑，包含原動作起點；只更新診斷資訊。
            self.decision = replace(
                self.decision,
                mode="no_feasible_path",
                reason="all candidates predict overlap; continuing previous path",
                details=("continuing previous path",),
                candidate_snapshot=snapshot,
            )
            self.last_evaluation_time = time_seconds
            return self.decision
        self.decision = StrategyDecision(
            mode="attack" if target is not None else "progress",
            target_d=best.maneuver.target_d,
            target_vehicle_id=None if target is None else target.vehicle_id,
            maneuver=best.maneuver,
            reason="lowest progress and overtaking cost",
            details=(f"ds={best.prediction.progress[-1]:.2f} pass={best.pass_reward:.0f}",
                     f"J={best.total_cost:.2f}"),
            candidate_snapshot=snapshot,
        )
        self.last_evaluation_time = time_seconds
        return self.decision

    def _front_vehicle(
        self,
        vehicle: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
    ) -> tuple[VehicleObservation | None, float]:
        length = track_length(track)
        candidates = [
            (signed_longitudinal_gap(vehicle.track_position, other.track_position, length),
             other.vehicle_id, other)
            for other in others
        ]
        ahead = [item for item in candidates if 0.0 < item[0] <= self.config.detection_range]
        if not ahead:
            return None, 0.0
        gap, _, target = min(ahead, key=lambda item: item[:2])
        return target, gap

    def _predict_other(
        self, target: VehicleObservation, track: Track,
    ) -> BicyclePrediction:
        config = self.other_prediction_configs.get(target.vehicle_id)
        if config is None:
            raise ValueError(f"missing prediction configuration for {target.vehicle_id}")
        # 只保留其他車當下的 d 與速度，不讀取其已選路徑或防守決策。
        maneuver = LateralManeuver(
            target.track_position.s, target.track_position.d,
            target.track_position.d, config.path.path_length,
        )
        return predict_bicycle_trajectory(
            target, track, maneuver, config, self.config.prediction_horizon,
            constant_speed=True,
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
    ) -> AttackCandidate:
        maneuver = self.decision.maneuver
        if maneuver is None or maneuver.target_d != target_d:
            maneuver = LateralManeuver(
                vehicle.track_position.s, vehicle.track_position.d,
                target_d, self.config.transition_length,
            )
        else:
            maneuver = continue_lateral_maneuver(maneuver, vehicle.track_position.s, track_length(track))
        prediction = predict_bicycle_trajectory(
            vehicle, track, maneuver, self.prediction_config, self.config.prediction_horizon,
        )
        collision = first_predicted_collision(prediction, vehicle.geometry, obstacles)
        if collision is not None:
            return AttackCandidate(maneuver, prediction, None, None, None, collision)
        # 進度跨越起終點仍連續累積；終點超過前車時給一次獎勵。
        progress = prediction.progress[-1]
        front = self.last_target_prediction
        reward = float(front is not None and progress > gap + front.progress[-1])
        return AttackCandidate(
            maneuver, prediction, -self.config.progress_weight * progress,
            -self.config.overtaking_weight * reward, reward,
        )

    def _snapshot(
        self, vehicle: VehicleObservation, time_seconds: float, best: AttackCandidate | None,
    ) -> CandidateSnapshot:
        return CandidateSnapshot(
            evaluated_at=time_seconds,
            position=vehicle.track_position,
            distances=(),
            candidates=tuple(
                CandidatePath(
                    maneuver=candidate.maneuver,
                    selected=candidate is best,
                    feasible=candidate.feasible,
                    reason="" if candidate.collision is None else candidate.collision.reason,
                    cost_terms=(("progress", candidate.progress_cost),
                                ("pass", candidate.overtaking_cost))
                    if candidate.progress_cost is not None and candidate.overtaking_cost is not None
                    else (),
                    predicted_states=candidate.prediction.states,
                )
                for candidate in self.last_candidates
            ),
        )
