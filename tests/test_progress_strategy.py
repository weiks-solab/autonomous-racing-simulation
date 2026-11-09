"""確認純進度策略使用自車預測，且不受其他車位置與路徑影響。"""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from core.track import Track
from core.track_geometry import track_length, world_to_track
from sim.candidate_paths import candidate_path_label, candidate_path_points
from strategy.bicycle_prediction import predict_bicycle_trajectory
from strategy.offset_path import LateralManeuver
from strategy.progress import ProgressConfig, ProgressStrategy
from tests.test_bicycle_prediction import observation, prediction_config


class ProgressStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.config = ProgressConfig(prediction_horizon=1.0)
        self.prediction = prediction_config()
        self.ego = observation(self.track, "ego", 20.0, 0.0, 6.0)

    def strategy(self) -> ProgressStrategy:
        return ProgressStrategy(self.config, self.prediction)

    def test_five_own_rollouts_ignore_even_an_overlapping_vehicle(self) -> None:
        baseline = self.strategy().update(self.ego, (), self.track, 0.0)
        blocker = replace(self.ego, vehicle_id="blocker")
        strategy = self.strategy()
        with patch("strategy.progress.predict_bicycle_trajectory",
                   wraps=predict_bicycle_trajectory) as predictor, \
             patch("strategy.collision_check.first_predicted_collision",
                   side_effect=AssertionError("progress must not check collisions")):
            decision = strategy.update(self.ego, (blocker,), self.track, 0.0)
        self.assertEqual(predictor.call_count, 5)
        self.assertTrue(all(call.args[0] is self.ego for call in predictor.call_args_list))
        self.assertEqual(decision, baseline)
        self.assertIsNone(decision.target_vehicle_id)
        self.assertEqual(decision.mode, "progress")
        self.assertEqual({item.maneuver.target_d for item in strategy.last_candidates},
                         {-0.6, -0.3, 0.0, 0.3, 0.6})
        self.assertEqual(decision.target_d, 0.0)
        self.assertTrue(all(item.feasible for item in decision.candidate_snapshot.candidates))

    def test_other_speed_position_and_planned_path_do_not_change_the_decision(self) -> None:
        expected = self.strategy().update(self.ego, (), self.track, 0.0)
        for s, d, speed in ((19.0, -0.6, 10.0), (22.0, 0.6, 0.0), (60.0, 0.0, 3.0)):
            with self.subTest(s=s, d=d, speed=speed):
                other = observation(self.track, "other", s, d, speed)
                other = replace(other, maneuver=LateralManeuver(s, d, -d, 3.0))
                self.assertEqual(self.strategy().update(self.ego, (other,), self.track, 0.0),
                                 expected)

    def test_progress_only_cost_and_visualization_use_the_bicycle_rollout(self) -> None:
        strategy = self.strategy()
        decision = strategy.update(self.ego, (), self.track, 0.0)
        selected = []
        for candidate, evaluation in zip(decision.candidate_snapshot.candidates,
                                         strategy.last_candidates):
            np.testing.assert_array_equal(
                candidate_path_points(decision.candidate_snapshot, candidate, self.track),
                [(state.x, state.y) for state in evaluation.prediction.states],
            )
            self.assertEqual(candidate.cost_terms, (("progress", -evaluation.progress),))
            self.assertNotIn("pass=", candidate_path_label(candidate))
            self.assertIs(candidate.predicted_states[0], self.ego.state)
            if candidate.selected:
                selected.append(evaluation)
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected[0].progress, 6.0)

    def test_inner_path_is_preferred_and_progress_crosses_the_finish(self) -> None:
        track = Track.create_oval(10.0, 10.0, 500)
        ego = observation(track, "ego", track_length(track) - 0.5, 0.0, 6.0)
        strategy = self.strategy()
        decision = strategy.update(ego, (), track, 0.0)
        self.assertGreater(decision.target_d, 0.0)
        best = next(item for item in strategy.last_candidates
                    if item.maneuver.target_d == decision.target_d)
        center = next(item for item in strategy.last_candidates if item.maneuver.target_d == 0.0)
        self.assertGreater(best.progress, center.progress)
        self.assertGreater(best.progress, 5.0)
        self.assertLess(world_to_track(best.prediction.states[-1], track).s, ego.track_position.s)

    def test_holds_decision_then_reuses_the_unfinished_maneuver_origin(self) -> None:
        strategy = self.strategy()
        ego = observation(self.track, "ego", 20.0, 0.6, 6.0)
        first = strategy.update(ego, (), self.track, 0.0)
        self.assertEqual(first.target_d, 0.6)
        moved = observation(self.track, "ego", 21.0, 0.6, 6.0)
        self.assertIs(strategy.update(moved, (), self.track, 0.05), first)
        updated = strategy.update(moved, (), self.track, 0.2)
        self.assertIsNot(updated.candidate_snapshot, first.candidate_snapshot)
        self.assertEqual(updated.candidate_snapshot.evaluated_at, 0.2)
        self.assertEqual(updated.maneuver, first.maneuver)
        self.assertIs(updated.candidate_snapshot.candidates[0].predicted_states[0], moved.state)

    def test_road_width_still_limits_candidate_targets(self) -> None:
        strategy = self.strategy()
        result = strategy.update(self.ego, (), replace(self.track, width=0.9), 0.0)
        self.assertEqual(len(result.candidate_snapshot.candidates), 1)
        self.assertEqual(result.target_d, 0.0)
        with self.assertRaisesRegex(ValueError, "too narrow"):
            self.strategy().update(self.ego, (), replace(self.track, width=0.4), 0.0)

    def test_invalid_configuration_track_and_time_are_rejected(self) -> None:
        for kwargs in ({"transition_length": 0.0}, {"decision_period": 0.0},
                       {"prediction_horizon": float("nan")}, {"safety_margin": -0.1},
                       {"candidate_offsets": (float("inf"),)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ProgressConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "closed track"):
            self.strategy().update(self.ego, (), replace(self.track, closed=False), 0.0)
        for time in (-1.0, float("nan")):
            with self.subTest(time=time), self.assertRaises(ValueError):
                self.strategy().update(self.ego, (), self.track, time)
        strategy = self.strategy()
        strategy.update(self.ego, (), self.track, 1.0)
        with self.assertRaisesRegex(ValueError, "backwards"):
            strategy.update(self.ego, (), self.track, 0.5)


if __name__ == "__main__":
    unittest.main()
