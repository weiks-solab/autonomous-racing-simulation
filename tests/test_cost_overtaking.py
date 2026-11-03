"""五候選、兩項成本與前車定速預測的行為測試。"""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from core.track import Track
from sim.candidate_paths import candidate_path_label, candidate_path_points
from strategy.cost_overtaking import CostBasedOvertakingStrategy, CostOvertakingConfig
from strategy.bicycle_prediction import predict_bicycle_trajectory
from strategy.offset_path import LateralManeuver
from tests.test_bicycle_prediction import observation, prediction_config


class CostOvertakingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))), width=2.0)
        self.config = CostOvertakingConfig(prediction_horizon=1.0, transition_length=3.0)
        self.prediction = prediction_config()
        self.ego = observation(self.track, "ego", 20.0, 0.0, 6.0)
        self.front = observation(self.track, "front", 22.0, 0.0, 3.0)

    def strategy(self, config: CostOvertakingConfig | None = None) -> CostBasedOvertakingStrategy:
        return CostBasedOvertakingStrategy(
            self.config if config is None else config, self.prediction,
            {"front": prediction_config(5.0)},
        )

    def test_only_clear_candidates_use_progress_and_pass_reward_costs(self) -> None:
        strategy = self.strategy(replace(self.config, progress_weight=2.0, overtaking_weight=7.0))
        front = observation(self.track, "front", 22.0, -0.6, 3.0)
        decision = strategy.update(self.ego, (front,), self.track, 0.0)
        self.assertEqual(len(strategy.last_candidates), 5)
        self.assertEqual({item.maneuver.target_d for item in strategy.last_candidates},
                         {-0.6, -0.3, 0.0, 0.3, 0.6})
        for candidate in strategy.last_candidates:
            if not candidate.feasible:
                self.assertIsNone(candidate.progress_cost)
                self.assertIsNone(candidate.overtaking_cost)
                continue
            self.assertAlmostEqual(candidate.progress_cost, -2.0 * candidate.prediction.progress[-1])
            self.assertEqual(candidate.pass_reward, 1.0)
            self.assertEqual(candidate.overtaking_cost, -7.0)
        chosen = next(item for item in strategy.last_candidates
                      if item.maneuver.target_d == decision.target_d)
        self.assertTrue(chosen.feasible)
        self.assertEqual(chosen.total_cost, min(item.total_cost for item in strategy.last_candidates
                                              if item.feasible))
        self.assertEqual(decision.target_vehicle_id, "front")
        self.assertEqual(decision.target_d, 0.0)

    def test_no_pass_reward_when_horizon_ends_behind_front_vehicle(self) -> None:
        strategy = self.strategy()
        front = observation(self.track, "front", 24.0, 0.0, 3.0)
        strategy.update(self.ego, (front,), self.track, 0.0)
        self.assertTrue(all(item.pass_reward == 0.0 for item in strategy.last_candidates))

    def test_no_target_uses_progress_without_forcing_a_return_state(self) -> None:
        strategy = self.strategy()
        decision = strategy.update(self.ego, (), self.track, 0.0)
        self.assertEqual(decision.mode, "progress")
        self.assertIsNone(decision.target_vehicle_id)
        self.assertIsNone(strategy.last_target_prediction)
        self.assertTrue(all(item.overtaking_cost == 0.0 for item in strategy.last_candidates))
        self.assertIsNotNone(decision.maneuver)

    def test_nearest_front_target_does_not_require_a_speed_advantage(self) -> None:
        strategy = CostBasedOvertakingStrategy(
            self.config, self.prediction,
            {name: self.prediction for name in ("near", "far", "behind")},
        )
        near = observation(self.track, "near", 22.0, 0.0, 8.0)
        far = observation(self.track, "far", 24.0, 0.0, 1.0)
        behind = observation(self.track, "behind", 19.0, 0.0, 1.0)
        result = strategy.update(self.ego, (far, behind, near), self.track, 0.0)
        self.assertEqual(result.target_vehicle_id, "near")

    def test_front_plan_does_not_change_prediction_or_decision(self) -> None:
        baseline = self.strategy()
        expected = baseline.update(self.ego, (self.front,), self.track, 0.0)
        front = replace(self.front, maneuver=LateralManeuver(22.0, 0.0, -0.6, 1.0))
        actual_strategy = self.strategy()
        actual = actual_strategy.update(self.ego, (front,), self.track, 0.0)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_strategy.last_target_prediction, baseline.last_target_prediction)
        self.assertTrue(all(state.velocity == self.front.state.velocity
                            for state in actual_strategy.last_target_prediction.states))

    def test_updates_hold_snapshot_between_evaluations_and_recompute_next_period(self) -> None:
        strategy = self.strategy()
        first = strategy.update(self.ego, (self.front,), self.track, 0.0)
        moved = observation(self.track, "ego", 21.0, 0.2, 6.0)
        held = strategy.update(moved, (self.front,), self.track, 0.1)
        self.assertIs(held, first)
        next_decision = strategy.update(moved, (self.front,), self.track, 0.3)
        self.assertEqual(next_decision.candidate_snapshot.evaluated_at, 0.3)
        for candidate in next_decision.candidate_snapshot.candidates:
            self.assertEqual(candidate.predicted_states[0], moved.state)

    def test_snapshot_draws_bicycle_states_and_displays_each_cost(self) -> None:
        strategy = self.strategy()
        result = strategy.update(self.ego, (self.front,), self.track, 0.0)
        snapshot = result.candidate_snapshot
        self.assertEqual(len(snapshot.candidates), 5)
        for candidate, evaluation in zip(snapshot.candidates, strategy.last_candidates):
            points = candidate_path_points(snapshot, candidate, self.track)
            expected = [(state.x, state.y) for state in evaluation.prediction.states]
            np.testing.assert_array_equal(points, expected)
            label = candidate_path_label(candidate)
            if candidate.feasible:
                self.assertIn("progress=", label)
                self.assertIn("pass=", label)
                self.assertIn("J=", label)
            else:
                self.assertIn("predicted overlap", label)
                self.assertFalse(candidate.selected)
                self.assertEqual(candidate.cost_terms, ())

    def test_collision_is_a_hard_filter_even_with_large_pass_reward(self) -> None:
        strategy = self.strategy(replace(self.config, overtaking_weight=1e9))
        front = observation(self.track, "front", 23.0, 0.0, 3.0)
        decision = strategy.update(self.ego, (front,), self.track, 0.0)
        center = next(item for item in strategy.last_candidates if item.maneuver.target_d == 0.0)
        self.assertFalse(center.feasible)
        self.assertIsNone(center.pass_reward)
        with self.assertRaisesRegex(ValueError, "do not have a cost"):
            _ = center.total_cost
        self.assertNotEqual(decision.mode, "no_feasible_path")
        self.assertNotEqual(decision.target_d, 0.0)
        chosen = next(item for item in strategy.last_candidates
                      if item.maneuver.target_d == decision.target_d)
        self.assertTrue(chosen.feasible)
        self.assertEqual(chosen.total_cost, min(item.total_cost for item in strategy.last_candidates
                                              if item.feasible))

    def test_collision_check_includes_a_third_vehicle(self) -> None:
        strategy = CostBasedOvertakingStrategy(
            self.config, self.prediction, {"front": self.prediction, "third": self.prediction},
        )
        front = observation(self.track, "front", 23.0, -0.6, 3.0)
        third = observation(self.track, "third", 24.0, 0.0, 0.0)
        decision = strategy.update(self.ego, (front, third), self.track, 0.0)
        center = next(item for item in strategy.last_candidates if item.maneuver.target_d == 0.0)
        self.assertEqual(center.collision.vehicle_id, "third")
        self.assertEqual(decision.target_vehicle_id, "front")
        self.assertGreater(decision.target_d, 0.0)

    def test_collision_check_is_not_limited_to_reward_detection_range(self) -> None:
        strategy = self.strategy(replace(self.config, prediction_horizon=2.0))
        front = observation(self.track, "front", 28.0, 0.0, 0.0)
        decision = strategy.update(self.ego, (front,), self.track, 0.0)
        self.assertIsNone(decision.target_vehicle_id)
        self.assertIsNone(strategy.last_target_prediction)
        center = next(item for item in strategy.last_candidates if item.maneuver.target_d == 0.0)
        self.assertEqual(center.collision.vehicle_id, "front")
        self.assertNotEqual(decision.target_d, 0.0)

    def test_all_rejected_keeps_previous_maneuver_and_recovers_when_clear(self) -> None:
        strategy = CostBasedOvertakingStrategy(
            self.config, self.prediction, {"blocker": self.prediction},
        )
        initial = observation(self.track, "ego", 20.0, 0.6, 6.0)
        previous = strategy.update(initial, (), self.track, 0.0)
        self.assertEqual(previous.target_d, 0.6)
        moved = observation(self.track, "ego", 21.0, 0.5, 6.0)
        blocker = replace(moved, vehicle_id="blocker")
        retained = strategy.update(moved, (blocker,), self.track, 0.3)
        self.assertEqual(retained.mode, "no_feasible_path")
        self.assertEqual(retained.target_d, previous.target_d)
        self.assertIs(retained.maneuver, previous.maneuver)
        self.assertEqual(retained.candidate_snapshot.evaluated_at, 0.3)
        self.assertTrue(all(not item.feasible and not item.selected
                            for item in retained.candidate_snapshot.candidates))
        self.assertTrue(all(item.collision.time_seconds == 0.0 for item in strategy.last_candidates))
        self.assertIs(strategy.update(moved, (blocker,), self.track, 0.4), retained)
        recovered = strategy.update(moved, (), self.track, 0.6)
        self.assertEqual(recovered.mode, "progress")
        self.assertEqual(sum(item.selected for item in recovered.candidate_snapshot.candidates), 1)

    def test_first_all_rejected_evaluation_keeps_initial_centerline(self) -> None:
        strategy = self.strategy()
        touching = observation(self.track, "front", 20.55, 0.0, 6.0)
        decision = strategy.update(self.ego, (touching,), self.track, 0.0)
        self.assertEqual(decision.mode, "no_feasible_path")
        self.assertEqual(decision.target_d, 0.0)
        self.assertIsNone(decision.maneuver)
        self.assertTrue(all(item.collision.time_seconds == 0.0 for item in strategy.last_candidates))

    def test_other_vehicles_are_predicted_once_and_target_prediction_is_reused(self) -> None:
        strategy = CostBasedOvertakingStrategy(
            self.config, self.prediction, {"front": self.prediction, "third": self.prediction},
        )
        third = observation(self.track, "third", 10.0, 0.0, 1.0)
        with patch("strategy.cost_overtaking.predict_bicycle_trajectory",
                   wraps=predict_bicycle_trajectory) as predictor:
            strategy.update(self.ego, (self.front, third), self.track, 0.0)
        ids = [call.args[0].vehicle_id for call in predictor.call_args_list]
        self.assertEqual(ids.count("front"), 1)
        self.assertEqual(ids.count("third"), 1)
        self.assertEqual(ids.count("ego"), 5)
        self.assertIs(strategy.last_target_prediction, strategy.last_other_predictions["front"])

    def test_front_uses_its_own_model_and_short_reference_path(self) -> None:
        track = Track.create_oval(10.0, 5.0, 200, width=2.0)
        ego = observation(track, "ego", 1.0, 0.0, 6.0)
        front = observation(track, "front", 3.0, 0.2, 4.0)
        normal = self.strategy()
        normal.update(ego, (front,), track, 0.0)
        limited = replace(
            self.prediction,
            model=replace(self.prediction.model, max_steering_angle=0.03),
            path=replace(self.prediction.path, path_length=1.0),
        )
        strategy = CostBasedOvertakingStrategy(self.config, self.prediction, {"front": limited})
        strategy.update(ego, (front,), track, 0.0)
        first, second = normal.last_target_prediction.states[-1], strategy.last_target_prediction.states[-1]
        self.assertNotEqual(first, second)
        expected_turn = (
            front.state.velocity / limited.model.wheelbase
            * np.tan(limited.model.max_steering_angle) * limited.model.time_step
        )
        limited_turn = strategy.last_target_prediction.states[1].yaw - front.state.yaw
        self.assertAlmostEqual(limited_turn, expected_turn)
        normal_turn = normal.last_target_prediction.states[1].yaw - front.state.yaw
        self.assertGreater(normal_turn, expected_turn)
        self.assertTrue(all(state.velocity == 4.0 for state in strategy.last_target_prediction.states))

    def test_width_filter_only_keeps_available_fixed_targets(self) -> None:
        strategy = self.strategy()
        narrow = replace(self.track, width=0.9)
        decision = strategy.update(self.ego, (self.front,), narrow, 0.0)
        self.assertEqual(len(strategy.last_candidates), 1)
        self.assertEqual(decision.target_d, 0.0)

    def test_invalid_configuration_or_observations_are_rejected(self) -> None:
        for kwargs in ({"progress_weight": -1.0}, {"overtaking_weight": float("nan")},
                       {"decision_period": 0.0}, {"candidate_offsets": (float("inf"),)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CostOvertakingConfig(**kwargs)
        strategy = self.strategy()
        with self.assertRaises(ValueError):
            strategy.update(self.ego, (self.front, self.front), self.track, 0.0)
        with self.assertRaises(ValueError):
            strategy.update(self.ego, (self.ego,), self.track, 0.0)
        with self.assertRaises(ValueError):
            strategy.update(self.ego, (), self.track, -1.0)
        with self.assertRaisesRegex(ValueError, "missing prediction"):
            strategy.update(self.ego, (replace(self.front, vehicle_id="unconfigured"),), self.track, 0.0)
        strategy.update(self.ego, (self.front,), self.track, 1.0)
        with self.assertRaisesRegex(ValueError, "backwards"):
            strategy.update(self.ego, (), self.track, 0.5)


if __name__ == "__main__":
    unittest.main()
