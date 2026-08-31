"""Deterministic rule-based overtaking decisions and candidate checks."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from core.track import Track
from core.track_geometry import (
    signed_longitudinal_gap,
    track_length,
    track_to_world,
)
from core.vehicle_geometry import rectangles_overlap
from core.vehicle_state import VehicleState
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver, lateral_offset_at_distance


class OvertakingMode(Enum):
    FOLLOW = "follow"
    PREPARE_OVERTAKE = "prepare_overtake"
    OVERTAKE = "overtake"
    RETURN = "return"


class OvertakingSide(Enum):
    LEFT = 1
    RIGHT = -1


@dataclass(frozen=True)
class OvertakingConfig:
    detection_range: float = 6.0
    minimum_speed_advantage: float = 0.5
    lateral_offset: float = 0.6
    safety_margin: float = 0.1
    pass_clearance: float = 1.0
    return_tolerance: float = 0.08
    transition_length: float = 6.0
    prediction_horizon: float = 5.0
    prediction_step: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.detection_range,
            self.minimum_speed_advantage,
            self.lateral_offset,
            self.safety_margin,
            self.pass_clearance,
            self.return_tolerance,
            self.transition_length,
            self.prediction_horizon,
            self.prediction_step,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("overtaking configuration must be finite")
        if self.detection_range <= 0.0:
            raise ValueError("detection_range must be positive")
        if self.minimum_speed_advantage < 0.0:
            raise ValueError("minimum_speed_advantage must be non-negative")
        if self.lateral_offset <= 0.0:
            raise ValueError("lateral_offset must be positive")
        if self.safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative")
        if self.pass_clearance <= 0.0:
            raise ValueError("pass_clearance must be positive")
        if self.return_tolerance < 0.0:
            raise ValueError("return_tolerance must be non-negative")
        if self.transition_length <= 0.0:
            raise ValueError("transition_length must be positive")
        if self.prediction_horizon <= 0.0 or self.prediction_step <= 0.0:
            raise ValueError("prediction timing must be positive")


@dataclass(frozen=True)
class OvertakingDecision:
    mode: OvertakingMode
    target_d: float
    target_vehicle_id: str | None
    side: OvertakingSide | None
    reason: str
    maneuver: LateralManeuver | None = None

    @property
    def display_text(self) -> str:
        lines = [self.mode.value.upper(), f"target_d={self.target_d:+.2f}"]
        if self.side is not None:
            lines.append(f"side={self.side.name}")
        if self.target_vehicle_id is not None:
            lines.append(f"target={self.target_vehicle_id}")
        return "\n".join(lines)


@dataclass(frozen=True)
class CandidateEvaluation:
    side: OvertakingSide
    target_d: float
    feasible: bool
    reason: str


def evaluate_candidate(
    ego: VehicleObservation,
    others: Sequence[VehicleObservation],
    track: Track,
    side: OvertakingSide,
    config: OvertakingConfig,
) -> CandidateEvaluation:
    """Check an approximate lateral bound and short-horizon SAT overlap.

    The width formula is only a lateral candidate filter. On a curved track it
    does not prove that every corner of the oriented body remains in bounds.

    Prediction is geometric in track coordinates, not a vehicle-model rollout:
    opponents keep constant speed and d, while ego follows the prescribed
    smooth d(s). No bicycle dynamics, steering, or physics are integrated.
    """
    target_d = side.value * config.lateral_offset
    maximum_d = (
        track.width / 2.0
        - ego.geometry.width / 2.0
        - config.safety_margin
    )
    if maximum_d <= 0.0:
        return CandidateEvaluation(side, target_d, False, "no lateral space")
    if abs(ego.track_position.d) > maximum_d or abs(target_d) > maximum_d:
        return CandidateEvaluation(side, target_d, False, "track width exceeded")
    if _candidate_has_predicted_collision(ego, others, track, target_d, config):
        return CandidateEvaluation(side, target_d, False, "predicted overlap")
    return CandidateEvaluation(side, target_d, True, "candidate is clear")


class RuleBasedOvertakingStrategy:
    """One-ego state machine whose selected pass side is locked in OVERTAKE."""

    def __init__(self, config: OvertakingConfig) -> None:
        self.config = config
        self.mode = OvertakingMode.FOLLOW
        self.target_vehicle_id: str | None = None
        self.selected_side: OvertakingSide | None = None
        self.locked_target_d = 0.0
        self.maneuver: LateralManeuver | None = None
        self.last_candidate_evaluations: tuple[CandidateEvaluation, ...] = ()

    def update(
        self,
        ego: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
    ) -> OvertakingDecision:
        other_by_id = {observation.vehicle_id: observation for observation in others}
        if len(other_by_id) != len(others):
            raise ValueError("other vehicle IDs must be unique")
        if ego.vehicle_id in other_by_id:
            raise ValueError("ego must not also appear in others")

        if self.mode is OvertakingMode.FOLLOW:
            return self._update_follow(ego, others, track)
        if self.mode is OvertakingMode.PREPARE_OVERTAKE:
            return self._update_prepare(ego, others, other_by_id, track)
        if self.mode is OvertakingMode.OVERTAKE:
            return self._update_overtake(ego, other_by_id, track)
        return self._update_return(ego)

    def _update_follow(
        self,
        ego: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
    ) -> OvertakingDecision:
        target = self._nearest_slower_vehicle_ahead(ego, others, track)
        if target is None:
            return self._decision("no slower vehicle in detection range")
        self.mode = OvertakingMode.PREPARE_OVERTAKE
        self.target_vehicle_id = target.vehicle_id
        return self._decision("slower vehicle detected ahead")

    def _update_prepare(
        self,
        ego: VehicleObservation,
        others: Sequence[VehicleObservation],
        other_by_id: dict[str, VehicleObservation],
        track: Track,
    ) -> OvertakingDecision:
        target = other_by_id.get(self.target_vehicle_id or "")
        if target is None or not self._is_trigger_target(ego, target, track):
            self._reset()
            return self._decision("target is no longer suitable")

        evaluations = tuple(
            evaluate_candidate(ego, others, track, side, self.config)
            for side in (OvertakingSide.LEFT, OvertakingSide.RIGHT)
        )
        self.last_candidate_evaluations = evaluations
        feasible = [evaluation for evaluation in evaluations if evaluation.feasible]
        if not feasible:
            return self._decision("no feasible overtaking side")

        selected = min(
            feasible,
            key=lambda evaluation: (
                abs(evaluation.target_d - ego.track_position.d),
                0 if evaluation.side is OvertakingSide.LEFT else 1,
            ),
        )
        self.selected_side = selected.side
        self.locked_target_d = selected.target_d
        self._start_maneuver(ego, self.locked_target_d)
        self.mode = OvertakingMode.OVERTAKE
        return self._decision(f"locked {selected.side.name.lower()} side")

    def _update_overtake(
        self,
        ego: VehicleObservation,
        other_by_id: dict[str, VehicleObservation],
        track: Track,
    ) -> OvertakingDecision:
        target = other_by_id.get(self.target_vehicle_id or "")
        if target is None:
            self._start_maneuver(ego, 0.0)
            self.mode = OvertakingMode.RETURN
            return self._decision("target unavailable; returning to center")
        gap = signed_longitudinal_gap(
            ego.track_position,
            target.track_position,
            track_length(track),
        )
        if gap < -self.config.pass_clearance:
            self._start_maneuver(ego, 0.0)
            self.mode = OvertakingMode.RETURN
            return self._decision("target is safely behind")
        return self._decision("overtaking side remains locked")

    def _update_return(self, ego: VehicleObservation) -> OvertakingDecision:
        if abs(ego.track_position.d) <= self.config.return_tolerance:
            self._reset()
            return self._decision("returned to centerline")
        return self._decision("returning to centerline")

    def _nearest_slower_vehicle_ahead(
        self,
        ego: VehicleObservation,
        others: Sequence[VehicleObservation],
        track: Track,
    ) -> VehicleObservation | None:
        candidates: list[tuple[float, str, VehicleObservation]] = []
        length = track_length(track)
        for other in others:
            gap = signed_longitudinal_gap(
                ego.track_position, other.track_position, length
            )
            if self._is_trigger_target(ego, other, track, gap=gap):
                candidates.append((gap, other.vehicle_id, other))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[:2])[2]

    def _is_trigger_target(
        self,
        ego: VehicleObservation,
        other: VehicleObservation,
        track: Track,
        *,
        gap: float | None = None,
    ) -> bool:
        if gap is None:
            gap = signed_longitudinal_gap(
                ego.track_position,
                other.track_position,
                track_length(track),
            )
        speed_advantage = ego.state.velocity - other.state.velocity
        return (
            0.0 < gap <= self.config.detection_range
            and speed_advantage >= self.config.minimum_speed_advantage
        )

    def _decision(self, reason: str) -> OvertakingDecision:
        target_d = (
            self.locked_target_d
            if self.mode is OvertakingMode.OVERTAKE
            else 0.0
        )
        return OvertakingDecision(
            mode=self.mode,
            target_d=target_d,
            target_vehicle_id=self.target_vehicle_id,
            side=self.selected_side,
            reason=reason,
            maneuver=self.maneuver,
        )

    def _start_maneuver(
        self,
        ego: VehicleObservation,
        target_d: float,
    ) -> None:
        self.maneuver = LateralManeuver(
            start_s=ego.track_position.s,
            start_d=ego.track_position.d,
            target_d=target_d,
            transition_length=self.config.transition_length,
        )

    def _reset(self) -> None:
        self.mode = OvertakingMode.FOLLOW
        self.target_vehicle_id = None
        self.selected_side = None
        self.locked_target_d = 0.0
        self.maneuver = None
        self.last_candidate_evaluations = ()


def _candidate_has_predicted_collision(
    ego: VehicleObservation,
    others: Sequence[VehicleObservation],
    track: Track,
    target_d: float,
    config: OvertakingConfig,
) -> bool:
    step_count = math.ceil(config.prediction_horizon / config.prediction_step)
    for step in range(step_count + 1):
        time = min(step * config.prediction_step, config.prediction_horizon)
        ego_distance = max(ego.state.velocity, 0.0) * time
        ego_d = lateral_offset_at_distance(
            ego_distance,
            ego.track_position.d,
            target_d,
            config.transition_length,
        )
        ego_d_ahead = lateral_offset_at_distance(
            ego_distance + 0.05,
            ego.track_position.d,
            target_d,
            config.transition_length,
        )
        predicted_ego = _state_from_track_coordinates(
            ego.track_position.s + ego_distance,
            ego_d,
            ego.state.velocity,
            track,
            d_ahead=ego_d_ahead,
        )
        for other in others:
            other_distance = max(other.state.velocity, 0.0) * time
            predicted_other = _state_from_track_coordinates(
                other.track_position.s + other_distance,
                other.track_position.d,
                other.state.velocity,
                track,
            )
            if rectangles_overlap(
                predicted_ego,
                ego.geometry,
                predicted_other,
                other.geometry,
            ):
                return True
    return False


def _state_from_track_coordinates(
    s: float,
    d: float,
    velocity: float,
    track: Track,
    *,
    d_ahead: float | None = None,
) -> VehicleState:
    point = track_to_world(s, d, track)
    point_ahead = track_to_world(
        s + 0.05,
        d if d_ahead is None else d_ahead,
        track,
    )
    yaw = math.atan2(point_ahead[1] - point[1], point_ahead[0] - point[0])
    return VehicleState(float(point[0]), float(point[1]), yaw, velocity)
