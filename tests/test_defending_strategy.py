"""檢查防守成本、觸發條件與多車路徑篩選。"""

from dataclasses import replace
import math
import unittest

import numpy as np

from core.track import Track
from core.track_geometry import track_to_world
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.defending import CostBasedDefenseStrategy, DefenseConfig, DefenseMode
from strategy.observations import VehicleObservation, create_vehicle_observation
from strategy.offset_path import LateralManeuver
from strategy.prediction import predict_vehicle_state


def observation(
    vehicle_id: str,
    s: float,
    d: float,
    speed: float,
    track: Track,
    maneuver: LateralManeuver | None = None,
) -> VehicleObservation:
    point = track_to_world(s, d, track)
    ahead = track_to_world(s + 0.05, d, track)
    yaw = math.atan2(ahead[1] - point[1], ahead[0] - point[0])
    return create_vehicle_observation(
        vehicle_id,
        VehicleState(float(point[0]), float(point[1]), yaw, speed),
        VehicleGeometry(0.55, 0.3, 0.12),
        track,
        maneuver,
    )


class DefendingStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        # 長直線讓成本與左右選擇的測試不受彎道影響。
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))), width=2.0)
        self.config = DefenseConfig()
        self.defender = observation("defender", 20.0, 0.0, 3.0, self.track)
        self.ego = observation("ego", 14.0, 0.6, 5.0, self.track)

    def test_approaching_ego_causes_same_side_defense(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        decision = strategy.update(self.defender, (self.ego,), self.track, 0.0)
        self.assertIs(decision.mode, DefenseMode.DEFEND)
        self.assertGreater(decision.target_d, 0.0)
        self.assertIsNotNone(decision.maneuver)

    def test_zero_block_weight_keeps_centerline(self) -> None:
        strategy = CostBasedDefenseStrategy(replace(self.config, q_block=0.0), "ego")
        decision = strategy.update(self.defender, (self.ego,), self.track, 0.0)
        self.assertIs(decision.mode, DefenseMode.CRUISE)
        self.assertEqual(decision.target_d, 0.0)
        self.assertIsNone(decision.maneuver)

    def test_larger_block_weight_gives_stronger_alignment(self) -> None:
        offsets = []
        for weight in (0.0, 1.0, 6.0):
            strategy = CostBasedDefenseStrategy(replace(self.config, q_block=weight), "ego")
            offsets.append(strategy.update(self.defender, (self.ego,), self.track, 0.0).target_d)
        self.assertEqual(offsets, sorted(offsets))
        self.assertGreater(offsets[-1], offsets[0])

    def test_cost_uses_current_ego_offset_even_when_ego_has_a_future_path(self) -> None:
        ego = replace(self.ego, maneuver=LateralManeuver(14.0, 0.6, -0.6, 10.0))
        config = replace(self.config, prediction_horizon=0.5, prediction_step=0.25)
        strategy = CostBasedDefenseStrategy(config, "ego")
        strategy.update(self.defender, (ego,), self.track, 0.0)
        candidate = next(item for item in strategy.last_candidates if item.target_d == 0.6)

        # v=3、過渡長度=3，所以兩個時間點的 smoothstep 輸入為 .25 與 .5。
        offsets = np.array((0.0, 0.6 * (3 * 0.25**2 - 2 * 0.25**3), 0.3))
        expected_race = np.sum(offsets[1:] ** 2) + 2.0 * np.sum(np.diff(offsets) ** 2)
        expected_block = 6.0 * np.sum((offsets[1:] - 0.6) ** 2)
        self.assertAlmostEqual(candidate.race_cost, expected_race)
        self.assertAlmostEqual(candidate.blocking_cost, expected_block)

    def test_no_defense_for_ahead_slow_far_or_unspecified_vehicle(self) -> None:
        scenarios = (
            (observation("ego", 25.0, 0.6, 5.0, self.track),),
            (observation("ego", 14.0, 0.6, 3.0, self.track),),
            (observation("ego", 5.0, 0.6, 5.0, self.track),),
            (observation("other", 14.0, 0.6, 5.0, self.track),),
            (),
        )
        for others in scenarios:
            with self.subTest(others=others):
                strategy = CostBasedDefenseStrategy(self.config, "ego")
                decision = strategy.update(self.defender, others, self.track, 0.0)
                self.assertIs(decision.mode, DefenseMode.CRUISE)
                self.assertEqual(decision.blocking_cost, 0.0)

    def test_nearby_ego_does_not_start_lateral_chasing(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        ego = observation("ego", 19.5, 0.6, 5.0, self.track)
        decision = strategy.update(self.defender, (ego,), self.track, 0.0)
        self.assertIs(decision.mode, DefenseMode.HOLD)
        self.assertEqual(decision.target_d, 0.0)
        self.assertEqual(decision.blocking_cost, 0.0)

    def test_hold_time_prevents_immediate_side_switch(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        first = strategy.update(self.defender, (self.ego,), self.track, 0.0)
        right_ego = observation("ego", 14.0, -0.6, 5.0, self.track)
        held = strategy.update(self.defender, (right_ego,), self.track, 0.3)
        switched = strategy.update(self.defender, (right_ego,), self.track, 0.8)
        self.assertEqual(held.target_d, first.target_d)
        self.assertEqual(held.maneuver, first.maneuver)
        self.assertLess(switched.target_d, 0.0)

    def test_maneuver_origin_is_preserved_while_car_moves(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        first = strategy.update(self.defender, (self.ego,), self.track, 0.0)
        moved = observation("defender", 21.0, 0.2, 3.0, self.track)
        ego = observation("ego", 16.0, 0.6, 5.0, self.track)
        continued = strategy.update(moved, (ego,), self.track, 0.3)
        self.assertEqual(first.maneuver, continued.maneuver)

    def test_returns_to_center_after_ego_has_passed(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        strategy.update(self.defender, (self.ego,), self.track, 0.0)
        shifted = observation("defender", 22.0, 0.5, 3.0, self.track)
        passed_ego = observation("ego", 28.0, 0.6, 5.0, self.track)
        returning = strategy.update(shifted, (passed_ego,), self.track, 1.0)
        self.assertIs(returning.mode, DefenseMode.RETURN)
        self.assertEqual(returning.target_d, 0.0)
        self.assertAlmostEqual(returning.maneuver.start_d, 0.5)
        centered = observation("defender", 26.0, 0.03, 3.0, self.track)
        finished = strategy.update(centered, (passed_ego,), self.track, 2.0)
        self.assertIs(finished.mode, DefenseMode.CRUISE)
        self.assertIsNone(finished.maneuver)

    def test_all_other_vehicles_are_checked_before_choosing_a_side(self) -> None:
        # 第三車占據左側，且相距 2 m，避免只觸發近距離 HOLD。
        blocker = observation("third", 22.0, 0.6, 1.5, self.track)
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        decision = strategy.update(self.defender, (self.ego, blocker), self.track, 0.0)
        left = next(item for item in strategy.last_candidates if item.target_d == 0.6)
        self.assertFalse(left.feasible)
        self.assertIn("third", left.reason)
        self.assertLess(decision.target_d, 0.6)

    def test_out_of_bounds_target_is_rejected(self) -> None:
        config = replace(self.config, candidate_offsets=(-2.0, 2.0))
        strategy = CostBasedDefenseStrategy(config, "ego")
        strategy.update(self.defender, (self.ego,), self.track, 0.0)
        for item in strategy.last_candidates:
            if abs(item.target_d) == 2.0:
                self.assertFalse(item.feasible)
                self.assertEqual(item.reason, "track width exceeded")

    def test_unsafe_path_can_be_abandoned_during_hold_time(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        first = strategy.update(self.defender, (self.ego,), self.track, 0.0)
        blocker = observation("third", 22.0, 0.6, 1.5, self.track)
        changed = strategy.update(self.defender, (self.ego, blocker), self.track, 0.3)
        self.assertLess(changed.target_d, first.target_d)
        previous_path = next(item for item in strategy.last_candidates
                             if item.target_d == first.target_d)
        self.assertFalse(previous_path.feasible)

    def test_body_corners_are_checked_even_when_reference_point_is_inside(self) -> None:
        shifted = observation("defender", 20.0, 0.6, 3.0, self.track)
        rotated = replace(shifted, state=replace(shifted.state, yaw=math.pi / 4))
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        strategy.update(rotated, (self.ego,), self.track, 0.0)
        self.assertTrue(all(not item.feasible for item in strategy.last_candidates))
        self.assertTrue(any(item.reason == "body corner outside track margin"
                            for item in strategy.last_candidates))

    def test_no_feasible_candidate_is_reported(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        overlap = observation("third", 20.0, 0.0, 3.0, self.track)
        decision = strategy.update(self.defender, (self.ego, overlap), self.track, 0.0)
        self.assertIs(decision.mode, DefenseMode.HOLD)
        self.assertIn("no feasible candidate", decision.reason)
        self.assertTrue(all(not item.feasible for item in strategy.last_candidates))

    def test_behind_detection_wraps_at_start_finish_line(self) -> None:
        track = Track.create_oval(10.0, 5.0, 200, width=2.0)
        defender = observation("defender", 1.0, 0.0, 3.0, track)
        ego = observation("ego", -5.0, 0.5, 5.0, track)
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        decision = strategy.update(defender, (ego,), track, 0.0)
        self.assertIs(decision.mode, DefenseMode.DEFEND)
        self.assertGreater(decision.target_d, 0.0)

    def test_prediction_follows_supplied_maneuver_across_finish_line(self) -> None:
        vehicle = observation("ego", 399.0, 0.0, 2.0, self.track,
                              LateralManeuver(399.0, 0.0, 0.6, 4.0))
        prediction = predict_vehicle_state(vehicle, 1.0, self.track)
        # 往前 2 m 後通過起點，橫移完成一半。
        expected = track_to_world(401.0, 0.3, self.track)
        np.testing.assert_allclose((prediction.x, prediction.y), expected)
        self.assertIs(predict_vehicle_state(vehicle, 0.0, self.track), vehicle.state)

    def test_invalid_inputs_are_rejected(self) -> None:
        for kwargs in ({"q_block": -1}, {"prediction_step": 0},
                       {"hold_time": float("nan")}, {"candidate_offsets": (float("inf"),)},
                       {"minimum_gap": 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DefenseConfig(**kwargs)
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        with self.assertRaises(ValueError):
            strategy.update(self.defender, (self.ego, self.ego), self.track, 0.0)
        with self.assertRaises(ValueError):
            strategy.update(self.defender, (self.defender,), self.track, 0.0)
        with self.assertRaises(ValueError):
            strategy.update(self.defender, (self.ego,), self.track, -1.0)


if __name__ == "__main__":
    unittest.main()
