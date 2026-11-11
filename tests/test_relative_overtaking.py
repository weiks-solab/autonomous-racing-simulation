"""相對進度只算一次；保留負間距、跨圈連續性與多車碰撞篩選。"""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length
from sim.candidate_paths import candidate_path_label, candidate_path_points
from strategy.bicycle_prediction import predict_bicycle_trajectory
from strategy.offset_path import LateralManeuver
from strategy.relative_overtaking import RelativeOvertakingConfig, RelativeOvertakingStrategy
from tests.test_bicycle_prediction import observation, prediction_config


class RelativeOvertakingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.config = RelativeOvertakingConfig(prediction_horizon=1.0)
        self.prediction = prediction_config()
        self.ego = observation(self.track, "ego", 20.0, 0.0, 6.0)
        self.front = observation(self.track, "front", 22.0, -0.6, 3.0)

    def strategy(self, config: RelativeOvertakingConfig | None = None) -> RelativeOvertakingStrategy:
        return RelativeOvertakingStrategy(
            self.config if config is None else config, self.prediction,
            {name: prediction_config(5.0) for name in ("front", "third", "behind", "blocker")},
        )

    def test_only_relative_gap_cost_is_used_and_a_negative_gap_is_preferred(self) -> None:
        strategy = self.strategy()
        decision = strategy.update(self.ego, (self.front,), self.track, 0.0)
        self.assertEqual(len(strategy.last_candidates), 9)
        self.assertEqual(decision.mode, "attack")
        self.assertEqual(decision.target_vehicle_id, "front")
        selected = next(item for item in strategy.last_candidates
                        if item.maneuver.target_d == decision.target_d)
        self.assertEqual(decision.target_d, 0.0)
        self.assertAlmostEqual(selected.total_cost, -1.0)
        for candidate, evaluation in zip(decision.candidate_snapshot.candidates,
                                         strategy.last_candidates):
            if candidate.feasible:
                # 直線上自車前進約 6 m、前車前進 3 m，初始相距 2 m。
                expected = 2.0 + 3.0 - evaluation.prediction.progress[-1]
                self.assertAlmostEqual(evaluation.total_cost, expected)
                self.assertEqual(candidate.cost_terms, (("relative_gap", evaluation.total_cost),))
                self.assertNotIn("pass=", candidate_path_label(candidate))
                self.assertNotIn("progress=", candidate_path_label(candidate))
            else:
                self.assertEqual(candidate.cost_terms, ())
                self.assertIsNone(evaluation.predicted_gap)

    def test_gap_changes_continuously_before_and_after_passing_without_a_bonus(self) -> None:
        costs = []
        for front_s in (22.0, 23.0, 24.0):
            strategy = self.strategy()
            front = observation(self.track, "front", front_s, -0.6, 3.0)
            strategy.update(self.ego, (front,), self.track, 0.0)
            center = next(item for item in strategy.last_candidates if item.maneuver.target_d == 0.0)
            costs.append(center.total_cost)
        np.testing.assert_allclose(costs, (-1.0, 0.0, 1.0), atol=1e-9)

    def test_opponent_speed_changes_the_cost_without_using_its_target_speed(self) -> None:
        for speed in (3.0, 5.0):
            with self.subTest(speed=speed):
                strategy = self.strategy()
                front = observation(self.track, "front", 22.0, -0.6, speed)
                strategy.update(self.ego, (front,), self.track, 0.0)
                center = next(item for item in strategy.last_candidates if item.maneuver.target_d == 0.0)
                self.assertAlmostEqual(center.total_cost, 2.0 + speed - 6.0)
                self.assertTrue(all(state.velocity == speed
                                    for state in strategy.last_target_prediction.states))

    def test_closed_track_gap_stays_continuous_across_the_finish(self) -> None:
        track = Track.create_oval(10.0, 10.0, 500)
        ego = observation(track, "ego", track_length(track) - 0.5, 0.0, 6.0)
        front = observation(track, "front", 0.5, -0.6, 3.0)
        strategy = self.strategy()
        decision = strategy.update(ego, (front,), track, 0.0)
        self.assertEqual(decision.target_vehicle_id, "front")
        initial_gap = signed_longitudinal_gap(ego.track_position, front.track_position,
                                              track_length(track))
        self.assertAlmostEqual(initial_gap, 1.0, delta=0.01)
        selected = next(item for item in strategy.last_candidates
                        if item.maneuver.target_d == decision.target_d)
        self.assertAlmostEqual(selected.total_cost, initial_gap
                               + strategy.last_target_prediction.progress[-1]
                               - selected.prediction.progress[-1])
        self.assertLess(selected.total_cost, 0.0)
        self.assertGreater(selected.total_cost, -5.0)

    def test_no_front_target_does_not_fall_back_to_optimizing_own_progress(self) -> None:
        track = Track.create_oval(10.0, 10.0, 500)
        ego = observation(track, "ego", 1.0, 0.0, 6.0)
        strategy = self.strategy()
        decision = strategy.update(ego, (), track, 0.0)
        self.assertEqual(decision.mode, "cruise")
        self.assertIsNone(decision.target_vehicle_id)
        self.assertEqual(decision.target_d, 0.0)
        center = next(item for item in strategy.last_candidates if item.maneuver.target_d == 0.0)
        self.assertGreater(max(item.prediction.progress[-1] for item in strategy.last_candidates),
                           center.prediction.progress[-1])
        self.assertTrue(all(item.total_cost == 0.0 and item.predicted_gap is None
                            for item in strategy.last_candidates))
        self.assertTrue(all(item.cost_terms == () for item in decision.candidate_snapshot.candidates))

    def test_missing_target_keeps_the_previous_feasible_maneuver(self) -> None:
        strategy = self.strategy()
        ego = observation(self.track, "ego", 20.0, 0.6, 6.0)
        first = strategy.update(ego, (self.front,), self.track, 0.0)
        self.assertEqual(first.target_d, 0.6)
        moved = observation(self.track, "ego", 21.0, 0.6, 6.0)
        held = strategy.update(moved, (), self.track, 0.05)
        self.assertIs(held, first)
        continued = strategy.update(moved, (), self.track, 0.2)
        self.assertEqual(continued.mode, "cruise")
        self.assertEqual(continued.target_d, first.target_d)
        self.assertEqual(continued.maneuver, first.maneuver)
        self.assertEqual(continued.candidate_snapshot.evaluated_at, 0.2)

    def test_passed_opponent_remains_the_target_even_beyond_detection_range(self) -> None:
        strategy = self.strategy()
        strategy.update(self.ego, (self.front,), self.track, 0.0)
        for time, ego_s, front_s, expected_gap in ((1.0, 26.0, 25.0, -1.0),
                                                  (3.0, 38.0, 31.0, -7.0)):
            with self.subTest(time=time):
                ego = observation(self.track, "ego", ego_s, 0.0, 6.0)
                front = observation(self.track, "front", front_s, -0.6, 3.0)
                decision = strategy.update(ego, (front,), self.track, time)
                self.assertEqual(decision.mode, "attack")
                self.assertEqual(decision.target_vehicle_id, "front")
                self.assertAlmostEqual(strategy.target_state.gap, expected_gap)
                center = next(item for item in strategy.last_candidates if item.maneuver.target_d == 0.0)
                self.assertAlmostEqual(center.total_cost, expected_gap + 3.0 - 6.0)
                selected = next(item for item in strategy.last_candidates
                                if item.maneuver.target_d == decision.target_d)
                self.assertEqual(selected.total_cost, min(item.total_cost for item in strategy.last_candidates
                                                         if item.feasible))
                self.assertTrue(all(item.cost_terms for item in decision.candidate_snapshot.candidates
                                    if item.feasible))

    def test_retained_opponent_gap_does_not_wrap_after_half_a_lap_or_a_full_lap(self) -> None:
        track = Track.create_oval(10.0, 10.0, 500)
        front = observation(track, "front", 3.0, -0.6, 0.0)
        strategy = self.strategy()
        length = track_length(track)
        for distance in range(0, 71, 10):
            with self.subTest(distance=distance):
                ego = observation(track, "ego", (1.0 + distance) % length, 0.0, 6.0)
                decision = strategy.update(ego, (front,), track, distance / 6.0)
                self.assertEqual(decision.target_vehicle_id, "front")
                self.assertAlmostEqual(strategy.target_state.gap, 2.0 - distance, delta=0.01)
                for candidate in strategy.last_candidates:
                    if candidate.feasible:
                        self.assertAlmostEqual(candidate.total_cost,
                                               strategy.target_state.gap - candidate.prediction.progress[-1])

    def test_new_front_target_replaces_passed_opponent_and_missing_target_clears_history(self) -> None:
        strategy = self.strategy()
        strategy.update(self.ego, (self.front,), self.track, 0.0)
        ego = observation(self.track, "ego", 26.0, 0.0, 6.0)
        passed = observation(self.track, "front", 25.0, -0.6, 3.0)
        third = observation(self.track, "third", 29.0, -0.6, 3.0)
        decision = strategy.update(ego, (passed, third), self.track, 1.0)
        self.assertEqual(decision.target_vehicle_id, "third")
        self.assertAlmostEqual(strategy.target_state.gap, 3.0)
        moved = observation(self.track, "ego", 32.0, 0.0, 6.0)
        behind = observation(self.track, "front", 28.0, -0.6, 3.0)
        lost = strategy.update(moved, (behind,), self.track, 2.0)
        self.assertEqual(lost.mode, "cruise")
        self.assertIsNone(lost.target_vehicle_id)
        self.assertIsNone(strategy.target_state)

    def test_each_other_is_predicted_once_and_its_selected_path_is_ignored(self) -> None:
        third = observation(self.track, "third", 30.0, 0.3, 4.0)
        strategy = self.strategy()
        with patch("strategy.relative_overtaking.predict_bicycle_trajectory",
                   wraps=predict_bicycle_trajectory) as predictor:
            expected = strategy.update(self.ego, (self.front, third), self.track, 0.0)
        ids = [call.args[0].vehicle_id for call in predictor.call_args_list]
        self.assertEqual(ids.count("ego"), 9)
        self.assertEqual(ids.count("front"), 1)
        self.assertEqual(ids.count("third"), 1)
        self.assertIs(strategy.last_target_prediction, strategy.last_other_predictions["front"])
        planned = replace(self.front, maneuver=LateralManeuver(22.0, -0.6, 0.6, 1.0))
        self.assertEqual(self.strategy().update(self.ego, (planned, third), self.track, 0.0), expected)

    def test_nearest_front_is_the_target_but_third_vehicle_collision_is_filtered(self) -> None:
        third = observation(self.track, "third", 24.0, 0.0, 0.0)
        behind = observation(self.track, "behind", 18.0, 0.6, 1.0)
        strategy = self.strategy()
        decision = strategy.update(self.ego, (third, behind, self.front), self.track, 0.0)
        self.assertEqual(decision.target_vehicle_id, "front")
        center = next(item for item in strategy.last_candidates if item.maneuver.target_d == 0.0)
        self.assertEqual(center.collision.vehicle_id, "third")
        with self.assertRaisesRegex(ValueError, "do not have a cost"):
            _ = center.total_cost
        self.assertNotEqual(decision.target_d, 0.0)

    def test_collision_filter_is_still_active_without_a_reward_target(self) -> None:
        strategy = self.strategy(replace(self.config, prediction_horizon=2.0))
        front = observation(self.track, "front", 28.0, 0.0, 0.0)
        decision = strategy.update(self.ego, (front,), self.track, 0.0)
        self.assertIsNone(decision.target_vehicle_id)
        self.assertIsNone(strategy.last_target_prediction)
        center = next(item for item in strategy.last_candidates if item.maneuver.target_d == 0.0)
        self.assertFalse(center.feasible)
        self.assertEqual(center.collision.vehicle_id, "front")
        self.assertNotEqual(decision.target_d, 0.0)

    def test_all_rejected_keeps_previous_path_and_recovers(self) -> None:
        strategy = self.strategy()
        ego = observation(self.track, "ego", 20.0, 0.6, 6.0)
        previous = strategy.update(ego, (self.front,), self.track, 0.0)
        moved = observation(self.track, "ego", 21.0, 0.5, 6.0)
        blocker = replace(moved, vehicle_id="blocker")
        decision = strategy.update(moved, (blocker,), self.track, 0.2)
        self.assertEqual(decision.mode, "no_feasible_path")
        self.assertIs(decision.maneuver, previous.maneuver)
        self.assertEqual(decision.target_d, previous.target_d)
        self.assertTrue(all(not item.feasible and not item.selected
                            for item in decision.candidate_snapshot.candidates))
        self.assertTrue(all(item.collision.time_seconds == 0.0 for item in strategy.last_candidates))
        recovered = strategy.update(moved, (), self.track, 0.4)
        self.assertEqual(recovered.mode, "cruise")
        self.assertEqual(recovered.target_d, previous.target_d)

    def test_visualization_uses_rollout_and_width_filter_accepts_only_in_bounds_targets(self) -> None:
        strategy = self.strategy()
        result = strategy.update(self.ego, (self.front,), self.track, 0.0)
        self.assertEqual(sum(item.selected for item in result.candidate_snapshot.candidates), 1)
        for candidate, evaluation in zip(result.candidate_snapshot.candidates, strategy.last_candidates):
            np.testing.assert_array_equal(
                candidate_path_points(result.candidate_snapshot, candidate, self.track),
                [(state.x, state.y) for state in evaluation.prediction.states],
            )
        narrow = replace(self.track, width=0.9)
        strategy = self.strategy()
        result = strategy.update(self.ego, (), narrow, 0.0)
        self.assertEqual({item.maneuver.target_d for item in result.candidate_snapshot.candidates},
                         {-0.15, 0.0, 0.15})
        with self.assertRaisesRegex(ValueError, "too narrow"):
            self.strategy().update(self.ego, (), replace(self.track, width=0.4), 0.0)

    def test_invalid_configuration_ids_track_or_time_are_rejected(self) -> None:
        for kwargs in ({"prediction_horizon": 0.0}, {"decision_period": -1.0},
                       {"transition_length": float("nan")}, {"detection_range": 0.0},
                       {"safety_margin": -0.1}, {"candidate_offsets": (float("inf"),)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RelativeOvertakingConfig(**kwargs)
        strategy = self.strategy()
        for others in ((self.front, self.front), (self.ego,)):
            with self.subTest(others=others), self.assertRaises(ValueError):
                strategy.update(self.ego, others, self.track, 0.0)
        for time in (-1.0, float("nan")):
            with self.subTest(time=time), self.assertRaises(ValueError):
                strategy.update(self.ego, (), self.track, time)
        with self.assertRaisesRegex(ValueError, "closed track"):
            strategy.update(self.ego, (), replace(self.track, closed=False), 0.0)
        with self.assertRaisesRegex(ValueError, "missing prediction"):
            strategy.update(self.ego, (replace(self.front, vehicle_id="unknown"),), self.track, 0.0)
        strategy.update(self.ego, (), self.track, 1.0)
        with self.assertRaisesRegex(ValueError, "backwards"):
            strategy.update(self.ego, (), self.track, 0.5)


if __name__ == "__main__":
    unittest.main()
