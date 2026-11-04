"""可停用、可替換的防守幾何碰撞條件與原路徑保留。"""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length, world_to_track
from sim.candidate_paths import candidate_path_label, candidate_path_points
from strategy.assignment import DefendingStrategy
from strategy.defense_conditions import has_predicted_collision, predict_rear
from strategy.defending import CostBasedDefenseStrategy, DefenseConfig, DefenseMode
from strategy.offset_path import LateralManeuver
from tests.test_defending_strategy import observation


class DefenseCollisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))), width=2.0)
        self.config = DefenseConfig(collision_horizon=1.0)
        self.defender = observation("front", 20.0, 0.0, 3.0, self.track)
        # 落後超過 2 m，讓對齊成本啟用，同時能在 1 秒內追上。
        self.rear = observation("rear", 17.9, 0.6, 5.2, self.track)

    def test_collision_rejects_alignment_target_before_cost_comparison(self) -> None:
        strategy = CostBasedDefenseStrategy(replace(self.config, q_block=1e8), "rear")
        decision = strategy.update(self.defender, (self.rear,), self.track, 0.0)
        aligned = next(item for item in strategy.last_candidates if item.target_d == 0.6)
        self.assertFalse(aligned.feasible)
        self.assertEqual(aligned.collision.vehicle_id, "rear")
        self.assertIsNone(aligned.race_cost)
        self.assertIsNone(aligned.blocking_cost)
        with self.assertRaisesRegex(ValueError, "do not have a cost"):
            _ = aligned.total_cost
        self.assertLess(decision.target_d, 0.6)
        self.assertNotEqual(decision.mode, DefenseMode.NO_FEASIBLE_PATH)
        self.assertAlmostEqual(
            decision.race_cost + decision.blocking_cost,
            min(item.total_cost for item in strategy.last_candidates if item.feasible),
        )

    def test_disabling_condition_restores_original_cost_only_selection(self) -> None:
        strategy = CostBasedDefenseStrategy(replace(self.config, collision_check_enabled=False), "rear")
        with patch("strategy.defending.has_predicted_collision") as check:
            decision = strategy.update(self.defender, (self.rear,), self.track, 0.0)
        check.assert_not_called()
        self.assertEqual(decision.target_d, 0.6)
        self.assertIsNone(strategy.last_rear_prediction)
        self.assertTrue(all(item.feasible for item in strategy.last_candidates))
        self.assertTrue(all(len(item.predicted_states) == 16 for item in strategy.last_candidates))

    def test_collision_condition_can_be_replaced_without_changing_costs(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "rear")
        with patch("strategy.defending.has_predicted_collision", return_value=None) as check:
            decision = strategy.update(self.defender, (self.rear,), self.track, 0.0)
        self.assertEqual(check.call_count, 5)
        self.assertEqual(decision.target_d, 0.6)

    def test_collision_horizon_changes_filter_without_changing_cost_horizon(self) -> None:
        rear = observation("rear", 16.0, 0.6, 5.0, self.track)
        short_config = replace(self.config, collision_horizon=0.5, prediction_horizon=0.5)
        short = CostBasedDefenseStrategy(short_config, "rear")
        long = CostBasedDefenseStrategy(replace(short_config, collision_horizon=2.0), "rear")
        short_result = short.update(self.defender, (rear,), self.track, 0.0)
        long_result = long.update(self.defender, (rear,), self.track, 0.0)
        self.assertEqual(short_result.target_d, 0.6)
        self.assertLess(long_result.target_d, 0.6)
        short_center = next(item for item in short.last_candidates if item.target_d == 0.0)
        long_center = next(item for item in long.last_candidates if item.target_d == 0.0)
        self.assertEqual(short_center.race_cost, long_center.race_cost)
        self.assertEqual(short_center.blocking_cost, long_center.blocking_cost)
        self.assertEqual(len(short.last_rear_prediction.times), 6)
        self.assertEqual(len(long.last_rear_prediction.times), 21)

    def test_rear_is_predicted_once_without_bicycle_rollout_or_future_plan(self) -> None:
        with_plan = replace(self.rear, maneuver=LateralManeuver(17.9, 0.6, -0.6, 1.0))
        plain, planned = (CostBasedDefenseStrategy(self.config, "rear") for _ in range(2))
        expected = plain.update(self.defender, (self.rear,), self.track, 0.0)
        with patch("strategy.bicycle_prediction.predict_bicycle_trajectory") as bicycle, \
             patch("strategy.defending.predict_rear", wraps=predict_rear) as geometric:
            actual = planned.update(self.defender, (with_plan,), self.track, 0.0)
        bicycle.assert_not_called()
        self.assertEqual(geometric.call_count, 1)
        self.assertEqual(actual, expected)
        self.assertEqual(planned.last_candidates, plain.last_candidates)
        prediction = planned.last_rear_prediction
        self.assertIs(prediction.states[0], self.rear.state)
        for time, state in zip(prediction.times, prediction.states):
            self.assertAlmostEqual(state.x, 17.9 + 5.2 * time)
            self.assertAlmostEqual(state.y, 0.6)
            self.assertEqual(state.velocity, 5.2)

    def test_only_assigned_rear_is_checked_and_missing_rear_skips_condition(self) -> None:
        far_rear = observation("rear", 14.0, 0.6, 5.0, self.track)
        third = replace(self.defender, vehicle_id="third")
        plain, extra = (CostBasedDefenseStrategy(self.config, "rear") for _ in range(2))
        expected = plain.update(self.defender, (far_rear,), self.track, 0.0)
        actual = extra.update(self.defender, (far_rear, third), self.track, 0.0)
        self.assertEqual(actual, expected)
        self.assertEqual(extra.last_candidates, plain.last_candidates)
        absent = CostBasedDefenseStrategy(self.config, "rear")
        with patch("strategy.defending.has_predicted_collision") as check:
            absent.update(self.defender, (third,), self.track, 0.0)
        check.assert_not_called()

    def test_collision_check_is_independent_of_alignment_activation(self) -> None:
        cases = (
            (replace(self.config, q_block=0.0), 20.0, 5.0),
            (replace(self.config, detection_range=1.0), 17.0, 8.0),
            (self.config, 20.6, 0.0),
        )
        for config, s, speed in cases:
            with self.subTest(s=s, speed=speed):
                rear = observation("rear", s, 0.0, speed, self.track)
                strategy = CostBasedDefenseStrategy(config, "rear")
                strategy.update(self.defender, (rear,), self.track, 0.0)
                center = next(item for item in strategy.last_candidates if item.target_d == 0.0)
                self.assertFalse(center.feasible)
                self.assertIsNotNone(strategy.last_rear_prediction)

    def test_all_rejected_keeps_previous_path_and_recovers_when_clear(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "rear")
        far_rear = observation("rear", 14.0, 0.6, 5.0, self.track)
        previous = strategy.update(self.defender, (far_rear,), self.track, 0.0)
        self.assertEqual(previous.target_d, 0.6)
        moved = observation("front", 21.0, 0.2, 3.0, self.track)
        overlapping = replace(moved, vehicle_id="rear")
        retained = strategy.update(moved, (overlapping,), self.track, 0.3)
        self.assertIs(retained.mode, DefenseMode.NO_FEASIBLE_PATH)
        self.assertEqual(retained.target_d, previous.target_d)
        self.assertIs(retained.maneuver, previous.maneuver)
        self.assertIsNone(retained.race_cost)
        self.assertIn("continuing previous path", retained.display_text)
        self.assertTrue(all(item.collision.time_seconds == 0.0 for item in strategy.last_candidates))
        self.assertIs(strategy.update(moved, (overlapping,), self.track, 0.4), retained)
        recovered = strategy.update(moved, (far_rear,), self.track, 0.6)
        self.assertIsNot(recovered.mode, DefenseMode.NO_FEASIBLE_PATH)
        self.assertIsNotNone(recovered.race_cost)

    def test_snapshot_draws_collision_horizon_and_rejected_candidates(self) -> None:
        config = replace(self.config, prediction_horizon=0.5, collision_horizon=2.0)
        adapter = DefendingStrategy(config, "rear")
        rear = observation("rear", 16.0, 0.6, 5.0, self.track)
        result = adapter.update(self.defender, (rear,), self.track, 0.0)
        snapshot = result.candidate_snapshot
        center = next(item for item in snapshot.candidates if item.maneuver.target_d == 0.0)
        points = candidate_path_points(snapshot, center, self.track)
        np.testing.assert_allclose(points[-1], (26.0, 0.0))
        self.assertEqual(len(points), 21)
        rejected = next(item for item in snapshot.candidates if item.maneuver.target_d == 0.6)
        self.assertFalse(rejected.feasible)
        self.assertFalse(rejected.selected)
        label = candidate_path_label(rejected)
        self.assertIn("predicted overlap with rear", label)
        self.assertNotIn("J=", label)

    def test_initial_all_rejected_keeps_centerline_and_has_no_selected_candidate(self) -> None:
        adapter = DefendingStrategy(self.config, "rear")
        touching = observation("rear", 19.45, 0.0, 5.0, self.track)
        result = adapter.update(self.defender, (touching,), self.track, 0.0)
        self.assertEqual(result.mode, "no_feasible_path")
        self.assertIsNone(result.maneuver)
        self.assertEqual(result.target_d, 0.0)
        self.assertEqual(result.details, ("continuing previous path",))
        self.assertTrue(all(not item.selected for item in result.candidate_snapshot.candidates))
        self.assertIn("continuing previous path", result.display_text)

    def test_rear_prediction_wraps_at_finish_line(self) -> None:
        track = Track.create_oval(10.0, 10.0, 500, width=2.0)
        rear = observation("rear", -1.0, 0.6, 5.0, track)
        result = predict_rear(rear, track, 1.0, 0.1)
        end = world_to_track(result.states[-1], track)
        progress = signed_longitudinal_gap(rear.track_position, end, track_length(track))
        self.assertAlmostEqual(progress, 5.0 / (1.0 - 0.6 / 10.0), delta=0.03)
        self.assertAlmostEqual(end.d, 0.6, delta=0.01)

    def test_invalid_collision_settings_and_mismatched_samples_are_rejected(self) -> None:
        for kwargs in ({"collision_horizon": 0.0}, {"collision_horizon": float("nan")},
                       {"collision_check_enabled": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DefenseConfig(**kwargs)
        rear = predict_rear(self.rear, self.track, 1.0, 0.1)
        with self.assertRaisesRegex(ValueError, "same sample times"):
            has_predicted_collision(self.defender, (self.defender.state,), rear)


if __name__ == "__main__":
    unittest.main()
