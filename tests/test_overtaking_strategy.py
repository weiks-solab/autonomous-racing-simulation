from dataclasses import replace
import unittest

import numpy as np

from core.track import Track
from core.track_geometry import TrackPosition, track_to_world
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver
from strategy.overtaking import (
    OvertakingConfig,
    OvertakingMode,
    OvertakingSide,
    RuleBasedOvertakingStrategy,
    evaluate_candidate,
)


def _observation(
    vehicle_id: str,
    s: float,
    d: float,
    velocity: float,
    track: Track,
) -> VehicleObservation:
    point = track_to_world(s, d, track)
    point_ahead = track_to_world(s + 0.05, d, track)
    yaw = float(np.arctan2(point_ahead[1] - point[1], point_ahead[0] - point[0]))
    return VehicleObservation(
        vehicle_id,
        VehicleState(float(point[0]), float(point[1]), yaw, velocity),
        TrackPosition(s % 40.0, d, 0),
        VehicleGeometry(0.5, 0.3, 0.1),
    )


class OvertakingStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(
            np.array(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))),
            width=4.0,
        )
        self.config = OvertakingConfig(
            detection_range=6.0,
            minimum_speed_advantage=0.5,
            lateral_offset=0.8,
            safety_margin=0.1,
            pass_clearance=1.0,
            return_tolerance=0.1,
            transition_length=4.0,
            prediction_horizon=3.0,
            prediction_step=0.25,
        )

    def test_slower_vehicle_ahead_triggers_prepare(self) -> None:
        strategy = RuleBasedOvertakingStrategy(self.config)
        decision = strategy.update(
            _observation("ego", 5.0, 0.0, 5.0, self.track),
            (_observation("target", 8.0, 0.0, 3.0, self.track),),
            self.track,
        )
        self.assertIs(decision.mode, OvertakingMode.PREPARE_OVERTAKE)
        self.assertEqual(decision.target_vehicle_id, "target")

    def test_vehicle_behind_does_not_trigger(self) -> None:
        strategy = RuleBasedOvertakingStrategy(self.config)
        decision = strategy.update(
            _observation("ego", 5.0, 0.0, 5.0, self.track),
            (_observation("behind", 3.0, 0.0, 3.0, self.track),),
            self.track,
        )
        self.assertIs(decision.mode, OvertakingMode.FOLLOW)

    def test_small_speed_advantage_does_not_trigger(self) -> None:
        strategy = RuleBasedOvertakingStrategy(self.config)
        decision = strategy.update(
            _observation("ego", 5.0, 0.0, 3.2, self.track),
            (_observation("target", 8.0, 0.0, 3.0, self.track),),
            self.track,
        )
        self.assertIs(decision.mode, OvertakingMode.FOLLOW)

    def test_left_and_right_candidates_can_be_feasible(self) -> None:
        ego = _observation("ego", 5.0, 0.0, 5.0, self.track)
        for side in (OvertakingSide.LEFT, OvertakingSide.RIGHT):
            with self.subTest(side=side):
                result = evaluate_candidate(ego, (), self.track, side, self.config)
                self.assertTrue(result.feasible)

    def test_boundary_filter_rejects_candidate(self) -> None:
        narrow_track = Track(self.track.centerline, width=0.6)
        result = evaluate_candidate(
            _observation("ego", 5.0, 0.0, 5.0, narrow_track),
            (),
            narrow_track,
            OvertakingSide.LEFT,
            self.config,
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason, "track width exceeded")

    def test_candidate_collision_rejects_occupied_side(self) -> None:
        ego = _observation("ego", 5.0, 0.0, 5.0, self.track)
        blocker = _observation("blocker", 5.0, 0.8, 5.0, self.track)
        left = evaluate_candidate(
            ego, (blocker,), self.track, OvertakingSide.LEFT, self.config
        )
        right = evaluate_candidate(
            ego, (blocker,), self.track, OvertakingSide.RIGHT, self.config
        )
        self.assertFalse(left.feasible)
        self.assertTrue(right.feasible)

    def test_state_machine_locks_side_until_return(self) -> None:
        strategy = RuleBasedOvertakingStrategy(self.config)
        ego = _observation("ego", 5.0, 0.0, 5.0, self.track)
        target = _observation("target", 8.0, 0.0, 3.0, self.track)

        self.assertIs(
            strategy.update(ego, (target,), self.track).mode,
            OvertakingMode.PREPARE_OVERTAKE,
        )
        overtake = strategy.update(ego, (target,), self.track)
        self.assertIs(overtake.mode, OvertakingMode.OVERTAKE)
        self.assertIs(overtake.side, OvertakingSide.LEFT)
        self.assertIsNotNone(overtake.maneuver)
        self.assertEqual(overtake.maneuver.start_s, ego.track_position.s)
        self.assertEqual(overtake.maneuver.start_d, ego.track_position.d)
        locked_d = overtake.target_d
        locked_maneuver = overtake.maneuver

        obstacle_on_left = _observation("blocker", 5.0, 0.8, 5.0, self.track)
        still_overtaking = strategy.update(
            ego, (target, obstacle_on_left), self.track
        )
        self.assertIs(still_overtaking.mode, OvertakingMode.OVERTAKE)
        self.assertIs(still_overtaking.side, OvertakingSide.LEFT)
        self.assertEqual(still_overtaking.target_d, locked_d)
        self.assertEqual(still_overtaking.target_vehicle_id, "target")
        self.assertEqual(still_overtaking.maneuver, locked_maneuver)

        passed_ego = _observation("ego", 10.0, 0.8, 5.0, self.track)
        passed_target = _observation("target", 8.0, 0.0, 3.0, self.track)
        returning = strategy.update(passed_ego, (passed_target,), self.track)
        self.assertIs(returning.mode, OvertakingMode.RETURN)
        self.assertEqual(returning.target_d, 0.0)
        self.assertIsNotNone(returning.maneuver)
        self.assertEqual(returning.maneuver.start_s, passed_ego.track_position.s)
        self.assertEqual(returning.maneuver.start_d, passed_ego.track_position.d)
        return_maneuver = returning.maneuver

        not_centered = strategy.update(
            _observation("ego", 11.0, 0.3, 5.0, self.track),
            (passed_target,),
            self.track,
        )
        self.assertIs(not_centered.mode, OvertakingMode.RETURN)
        self.assertEqual(not_centered.maneuver, return_maneuver)
        centered = strategy.update(
            _observation("ego", 12.0, 0.05, 5.0, self.track),
            (passed_target,),
            self.track,
        )
        self.assertIs(centered.mode, OvertakingMode.FOLLOW)

    def test_candidate_check_uses_defenders_supplied_maneuver(self) -> None:
        ego = _observation("ego", 5.0, 0.0, 5.0, self.track)
        defender = _observation("defender", 8.0, 0.0, 3.0, self.track)
        passive = evaluate_candidate(
            ego, (defender,), self.track, OvertakingSide.LEFT, self.config
        )
        defending = replace(defender, maneuver=LateralManeuver(8.0, 0.0, 0.8, 2.0))
        active = evaluate_candidate(
            ego, (defending,), self.track, OvertakingSide.LEFT, self.config
        )
        self.assertTrue(passive.feasible)
        self.assertFalse(active.feasible)

    def test_deterministic_selection_prefers_left_on_tie(self) -> None:
        strategy = RuleBasedOvertakingStrategy(self.config)
        ego = _observation("ego", 5.0, 0.0, 5.0, self.track)
        target = _observation("target", 8.0, 0.0, 3.0, self.track)
        strategy.update(ego, (target,), self.track)
        decision = strategy.update(ego, (target,), self.track)
        self.assertIs(decision.side, OvertakingSide.LEFT)

    def test_overtake_origin_stays_fixed_across_updates(self) -> None:
        strategy = RuleBasedOvertakingStrategy(self.config)
        ego = _observation("ego", 5.0, 0.0, 5.0, self.track)
        target = _observation("target", 8.0, 0.0, 3.0, self.track)
        strategy.update(ego, (target,), self.track)
        started = strategy.update(ego, (target,), self.track)

        moved_ego = _observation("ego", 6.0, 0.3, 5.0, self.track)
        moved_target = _observation("target", 9.0, 0.0, 3.0, self.track)
        continued = strategy.update(moved_ego, (moved_target,), self.track)

        self.assertIs(continued.mode, OvertakingMode.OVERTAKE)
        self.assertEqual(continued.maneuver, started.maneuver)
        self.assertEqual(continued.target_vehicle_id, "target")
        self.assertEqual(continued.target_d, started.target_d)

    def test_return_origin_stays_fixed_across_updates(self) -> None:
        strategy = RuleBasedOvertakingStrategy(self.config)
        ego = _observation("ego", 5.0, 0.0, 5.0, self.track)
        target = _observation("target", 8.0, 0.0, 3.0, self.track)
        strategy.update(ego, (target,), self.track)
        strategy.update(ego, (target,), self.track)
        returning = strategy.update(
            _observation("ego", 10.0, 0.8, 5.0, self.track),
            (_observation("target", 8.0, 0.0, 3.0, self.track),),
            self.track,
        )
        continued = strategy.update(
            _observation("ego", 11.0, 0.4, 5.0, self.track),
            (_observation("target", 9.0, 0.0, 3.0, self.track),),
            self.track,
        )

        self.assertIs(continued.mode, OvertakingMode.RETURN)
        self.assertEqual(continued.maneuver, returning.maneuver)
        self.assertEqual(returning.maneuver.start_s, 10.0)
        self.assertEqual(returning.maneuver.start_d, 0.8)

    def test_return_transition_replaces_overtake_origin(self) -> None:
        strategy = RuleBasedOvertakingStrategy(self.config)
        ego = _observation("ego", 38.0, 0.0, 5.0, self.track)
        target = _observation("target", 1.0, 0.0, 3.0, self.track)
        strategy.update(ego, (target,), self.track)
        overtake = strategy.update(ego, (target,), self.track)
        self.assertEqual(overtake.maneuver.start_s, 38.0)

        returning = strategy.update(
            _observation("ego", 3.0, 0.7, 5.0, self.track),
            (_observation("target", 1.0, 0.0, 3.0, self.track),),
            self.track,
        )
        self.assertIs(returning.mode, OvertakingMode.RETURN)
        self.assertEqual(returning.maneuver.start_s, 3.0)
        self.assertEqual(returning.maneuver.start_d, 0.7)
        self.assertEqual(returning.maneuver.target_d, 0.0)


if __name__ == "__main__":
    unittest.main()
