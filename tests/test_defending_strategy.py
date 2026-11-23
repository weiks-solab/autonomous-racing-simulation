"""檢查防守成本、固定候選與只使用當下位置的決策。"""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from core.track import Track
from core.track_geometry import track_to_world
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.bicycle_prediction import predict_bicycle_trajectory
from strategy.defending import CostBasedDefenseStrategy, DefenseConfig, DefenseMode
from strategy.observations import VehicleObservation, create_vehicle_observation
from strategy.offset_path import LateralManeuver
from strategy.prediction import predict_path_trajectory, predict_vehicle_state
from tests.test_bicycle_prediction import prediction_config


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
        # 原成本行為的回歸測試；碰撞條件另外以獨立場景覆蓋。
        self.config = DefenseConfig(collision_check_enabled=False)
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

        # 直線上 x 的變化就是賽道進度；d 取自走滿 v*t 後的取樣位置。
        offsets = np.array([state.y for state in candidate.predicted_states])
        progress = candidate.predicted_states[-1].x - self.defender.state.x
        self.assertLess(progress, 3.0 * 0.5)
        expected_race = -config.progress_weight * progress
        expected_race += config.smooth_weight * np.sum(np.diff(offsets) ** 2)
        expected_block = 6.0 * np.sum((offsets[1:] - 0.6) ** 2)
        self.assertAlmostEqual(candidate.race_cost, expected_race)
        self.assertAlmostEqual(candidate.blocking_cost, expected_block)

    def test_other_vehicles_plans_do_not_change_costs_or_decision(self) -> None:
        results = []
        for maneuver in (None, LateralManeuver(14.0, 0.6, -0.6, 3.0),
                         LateralManeuver(14.0, 0.6, 0.6, 3.0)):
            strategy = CostBasedDefenseStrategy(self.config, "ego")
            ego = replace(self.ego, maneuver=maneuver)
            decision = strategy.update(self.defender, (ego,), self.track, 0.0)
            results.append((decision, strategy.last_candidates))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])

    def test_no_defense_for_ahead_alongside_far_or_unspecified_vehicle(self) -> None:
        scenarios = (
            (observation("ego", 25.0, 0.6, 5.0, self.track),),
            (observation("ego", 20.0, 0.6, 5.0, self.track),),
            (observation("ego", 19.95, 0.6, 5.0, self.track),),
            (observation("ego", 19.0, 0.6, 5.0, self.track),),
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

    def test_rear_speed_does_not_gate_alignment_beyond_one_metre(self) -> None:
        for s, speed in ((14.0, 1.0), (14.0, 3.0), (17.9, 5.0), (18.0, 5.0), (18.9, 1.0)):
            with self.subTest(s=s, speed=speed):
                strategy = CostBasedDefenseStrategy(self.config, "ego")
                ego = observation("ego", s, 0.6, speed, self.track)
                decision = strategy.update(self.defender, (ego,), self.track, 0.0)
                self.assertIs(decision.mode, DefenseMode.DEFEND)
                self.assertGreater(decision.target_d, 0.0)
                self.assertGreater(decision.blocking_cost, 0.0)

    def test_lowest_cost_can_switch_side_at_next_evaluation(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        first = strategy.update(self.defender, (self.ego,), self.track, 0.0)
        right_ego = observation("ego", 14.0, -0.6, 5.0, self.track)
        switched = strategy.update(self.defender, (right_ego,), self.track, 0.3)
        self.assertGreater(first.target_d, 0.0)
        self.assertLess(switched.target_d, 0.0)
        self.assertAlmostEqual(
            switched.race_cost + switched.blocking_cost,
            min(candidate.total_cost for candidate in strategy.last_candidates),
        )

    def test_decision_period_is_respected(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        first = strategy.update(self.defender, (self.ego,), self.track, 0.0)
        right_ego = observation("ego", 14.0, -0.6, 5.0, self.track)
        self.assertEqual(
            strategy.update(self.defender, (right_ego,), self.track, 0.1), first,
        )

    def test_maneuver_origin_is_preserved_while_car_moves(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        first = strategy.update(self.defender, (self.ego,), self.track, 0.0)
        moved = observation("defender", 21.0, 0.2, 3.0, self.track)
        ego = observation("ego", 16.0, 0.6, 5.0, self.track)
        continued = strategy.update(moved, (ego,), self.track, 0.3)
        self.assertEqual(first.maneuver, continued.maneuver)

    def test_no_centerline_preference_after_ego_has_passed(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        strategy.update(self.defender, (self.ego,), self.track, 0.0)
        shifted = observation("defender", 24.0, 0.6, 3.0, self.track)
        passed_ego = observation("ego", 28.0, 0.6, 5.0, self.track)
        continued = strategy.update(shifted, (passed_ego,), self.track, 1.0)
        self.assertIs(continued.mode, DefenseMode.RETURN)
        self.assertEqual(continued.target_d, 0.6)
        self.assertEqual(continued.blocking_cost, 0.0)
        centered = observation("defender", 26.0, 0.03, 3.0, self.track)
        finished = strategy.update(centered, (passed_ego,), self.track, 2.0)
        self.assertIs(finished.mode, DefenseMode.CRUISE)
        self.assertIsNone(finished.maneuver)

    def test_progress_reward_alone_prefers_less_lateral_travel_on_a_straight(self) -> None:
        defender = observation("defender", 20.0, 0.6, 3.0, self.track)
        for weight in (0.0, 2.0):
            with self.subTest(weight=weight):
                config = replace(self.config, q_block=0.0, smooth_weight=0.0,
                                 progress_weight=weight)
                strategy = CostBasedDefenseStrategy(config, "ego")
                decision = strategy.update(defender, (), self.track, 0.0)
                if weight == 0.0:
                    self.assertTrue(all(item.total_cost == 0.0 for item in strategy.last_candidates))
                else:
                    self.assertEqual(decision.target_d, 0.6)
                    self.assertAlmostEqual(decision.race_cost, -weight * 3.0 * 1.5)

    def test_constant_lateral_offsets_have_equal_progress_on_a_straight(self) -> None:
        costs = []
        for d in (0.0, 0.6):
            strategy = CostBasedDefenseStrategy(replace(self.config, q_block=0.0), "ego")
            defender = observation("defender", 20.0, d, 3.0, self.track)
            decision = strategy.update(defender, (), self.track, 0.0)
            self.assertEqual(decision.target_d, d)
            costs.append(decision.race_cost)
        self.assertAlmostEqual(costs[0], costs[1])

    def test_progress_reward_prefers_inner_path_and_wraps_at_finish(self) -> None:
        track = Track.create_oval(10.0, 10.0, 500, width=2.0)
        config = replace(self.config, q_block=0.0, smooth_weight=0.0, prediction_horizon=2.0)
        decisions = []
        for s in (0.5, -0.5):
            strategy = CostBasedDefenseStrategy(config, "ego")
            defender = observation("defender", s, 0.0, 3.0, track)
            decision = strategy.update(defender, (), track, 0.0)
            self.assertEqual(decision.target_d, 0.6)
            costs = {item.target_d: item.race_cost for item in strategy.last_candidates}
            self.assertLess(costs[0.6], costs[0.0])
            self.assertLess(costs[0.0], costs[-0.6])
            decisions.append(decision)
        self.assertAlmostEqual(decisions[0].race_cost, decisions[1].race_cost, delta=0.01)

    def test_stationary_vehicle_has_zero_progress_reward(self) -> None:
        config = replace(self.config, q_block=0.0)
        strategy = CostBasedDefenseStrategy(config, "ego")
        defender = observation("defender", 20.0, 0.6, 0.0, self.track)
        strategy.update(defender, (), self.track, 0.0)
        self.assertTrue(all(item.race_cost == 0.0 for item in strategy.last_candidates))

    def test_collisions_and_other_vehicles_do_not_override_cost_selection(self) -> None:
        baseline = CostBasedDefenseStrategy(self.config, "ego")
        expected = baseline.update(self.defender, (self.ego,), self.track, 0.0)
        for s, d, speed in ((22.0, 0.6, 1.5), (20.0, 0.0, 3.0)):
            with self.subTest(s=s, d=d):
                blocker = observation("third", s, d, speed, self.track)
                strategy = CostBasedDefenseStrategy(self.config, "ego")
                decision = strategy.update(
                    self.defender, (self.ego, blocker), self.track, 0.0,
                )
                self.assertEqual(decision, expected)
                self.assertEqual(strategy.last_candidates, baseline.last_candidates)

    def test_only_fixed_targets_within_available_width_are_scored(self) -> None:
        config = replace(self.config, candidate_offsets=(-2.0, -0.8, -0.6, 0.6, 0.8, 2.0))
        strategy = CostBasedDefenseStrategy(config, "ego")
        defender = observation("defender", 20.0, 0.13, 3.0, self.track)
        ego = observation("ego", 14.0, 0.47, 5.0, self.track)
        decision = strategy.update(defender, (ego,), self.track, 0.0)
        # 路寬 2、車寬 .3、留距 .1，目標限制為 |d| <= .75。
        self.assertEqual(
            {item.target_d for item in strategy.last_candidates}, {0.0, -0.6, 0.6},
        )
        self.assertAlmostEqual(
            decision.race_cost + decision.blocking_cost,
            min(item.total_cost for item in strategy.last_candidates),
        )

    def test_narrower_track_removes_wide_targets(self) -> None:
        track = replace(self.track, width=0.9)
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        strategy.update(self.defender, (self.ego,), track, 0.0)
        self.assertEqual([item.target_d for item in strategy.last_candidates], [0.0])

    def test_body_rotation_does_not_filter_candidates(self) -> None:
        shifted = observation("defender", 20.0, 0.6, 3.0, self.track)
        rotated = replace(shifted, state=replace(shifted.state, yaw=math.pi / 4))
        baseline = CostBasedDefenseStrategy(self.config, "ego")
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        expected = baseline.update(shifted, (self.ego,), self.track, 0.0)
        actual = strategy.update(rotated, (self.ego,), self.track, 0.0)
        self.assertEqual(actual, expected)
        for candidate, reference in zip(strategy.last_candidates, baseline.last_candidates):
            self.assertEqual(candidate.target_d, reference.target_d)
            self.assertEqual(candidate.total_cost, reference.total_cost)
            self.assertEqual(candidate.feasible, reference.feasible)
            self.assertIs(candidate.predicted_states[0], rotated.state)
            self.assertEqual(candidate.predicted_states[1:], reference.predicted_states[1:])

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
        for kwargs in ({"q_block": -1}, {"progress_weight": -1},
                       {"progress_weight": float("nan")}, {"prediction_step": 0},
                       {"path_sample_spacing": 0},
                       {"detection_range": float("nan")},
                       {"candidate_offsets": (float("inf"),)},
                       {"return_tolerance": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DefenseConfig(**kwargs)
        strategy = CostBasedDefenseStrategy(self.config, "ego")
        with self.assertRaises(ValueError):
            strategy.update(self.defender, (self.ego, self.ego), self.track, 0.0)
        with self.assertRaises(ValueError):
            strategy.update(self.defender, (self.defender,), self.track, 0.0)
        with self.assertRaises(ValueError):
            strategy.update(self.defender, (self.ego,), self.track, -1.0)
        with self.assertRaisesRegex(ValueError, "too narrow"):
            strategy.update(
                self.defender, (self.ego,), replace(self.track, width=0.4), 0.0,
            )
        with self.assertRaisesRegex(ValueError, "closed"):
            strategy.update(
                self.defender, (self.ego,), replace(self.track, closed=False), 0.0,
            )
        strategy.update(self.defender, (self.ego,), self.track, 1.0)
        with self.assertRaisesRegex(ValueError, "backwards"):
            strategy.update(self.defender, (self.ego,), self.track, 0.5)


class DefenseDeviationCostTests(unittest.TestCase):
    """偏離中心線成本併入 J_race：w_dev × Σ d_k²，取樣點與對齊成本相同。"""

    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))), width=2.0)
        self.config = DefenseConfig(collision_check_enabled=False, prediction_horizon=1.0)
        self.defender = observation("defender", 20.0, 0.0, 3.0, self.track)
        self.ego = observation("ego", 14.0, 0.6, 5.0, self.track)

    def evaluate(self, weight: float, prediction=None) -> CostBasedDefenseStrategy:
        strategy = CostBasedDefenseStrategy(
            replace(self.config, deviation_weight=weight), "ego", prediction,
        )
        strategy.update(self.defender, (self.ego,), self.track, 0.0)
        return strategy

    def test_deviation_term_is_added_to_race_cost_for_both_predictions(self) -> None:
        for prediction in (None, prediction_config(3.0)):
            with self.subTest(bicycle=prediction is not None):
                plain, weighted = self.evaluate(0.0, prediction), self.evaluate(2.0, prediction)
                for before, after in zip(plain.last_candidates, weighted.last_candidates):
                    self.assertEqual(before.deviation_cost, 0.0)
                    self.assertGreaterEqual(after.deviation_cost, 0.0)
                    self.assertAlmostEqual(after.race_cost, before.race_cost + after.deviation_cost)
                    self.assertEqual(after.blocking_cost, before.blocking_cost)
                center = next(item for item in weighted.last_candidates if item.target_d == 0.0)
                self.assertAlmostEqual(center.deviation_cost, 0.0)
                wide = next(item for item in weighted.last_candidates if item.target_d == 0.6)
                self.assertGreater(wide.deviation_cost, 0.0)

    def test_deviation_value_matches_sampled_offsets(self) -> None:
        strategy = self.evaluate(2.0)
        candidate = next(item for item in strategy.last_candidates if item.target_d == 0.6)
        vehicle = replace(self.defender, maneuver=candidate.maneuver)
        prediction = predict_path_trajectory(vehicle, self.track, np.linspace(0.0, 1.0, 11), 0.2)
        self.assertAlmostEqual(candidate.deviation_cost,
                               2.0 * float(np.sum(prediction.offsets[1:] ** 2)))

    def test_large_weight_overrides_alignment_and_invalid_weight_is_rejected(self) -> None:
        self.assertGreater(self.evaluate(0.0).decision.target_d, 0.0)
        self.assertEqual(self.evaluate(100.0).decision.target_d, 0.0)
        for value in (-1.0, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DefenseConfig(deviation_weight=value)


class BicycleDefensePredictionTests(unittest.TestCase):
    """提供 prediction_config 時，自車候選改用 Pure Pursuit＋自行車模型 rollout。"""

    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))), width=2.0)
        self.config = DefenseConfig(collision_check_enabled=False, prediction_horizon=1.0)
        self.prediction = prediction_config(3.0)
        self.defender = observation("defender", 20.0, 0.0, 3.0, self.track)
        self.ego = observation("ego", 14.0, 0.6, 5.0, self.track)

    def test_costs_use_interpolated_bicycle_rollout(self) -> None:
        strategy = CostBasedDefenseStrategy(self.config, "ego", self.prediction)
        with patch("strategy.defending.predict_bicycle_trajectory",
                   wraps=predict_bicycle_trajectory) as rollout, \
             patch("strategy.defending.predict_path_trajectory") as geometric:
            decision = strategy.update(self.defender, (self.ego,), self.track, 0.0)
        geometric.assert_not_called()
        self.assertEqual(rollout.call_count, len(strategy.last_candidates))
        self.assertTrue(all(call.args[4] == 1.0 for call in rollout.call_args_list))
        self.assertIs(decision.mode, DefenseMode.DEFEND)
        self.assertGreater(decision.target_d, 0.0)
        for candidate in strategy.last_candidates:
            with self.subTest(target=candidate.target_d):
                result = predict_bicycle_trajectory(
                    self.defender, self.track, candidate.maneuver, self.prediction, 1.0,
                )
                cost_times = np.linspace(0.0, 1.0, 11)
                offsets = np.interp(cost_times, result.times, result.offsets)
                expected_race = (-self.config.progress_weight * result.progress[-1]
                                 + self.config.smooth_weight * np.sum(np.diff(offsets) ** 2))
                expected_block = self.config.q_block * np.sum((offsets[1:] - 0.6) ** 2)
                self.assertAlmostEqual(candidate.race_cost, expected_race)
                self.assertAlmostEqual(candidate.blocking_cost, expected_block)
                self.assertEqual(candidate.predicted_states, result.states)

    def test_collision_condition_extends_rollout_and_uses_rear_sample_times(self) -> None:
        config = replace(self.config, collision_check_enabled=True, collision_horizon=1.5)
        strategy = CostBasedDefenseStrategy(config, "ego", self.prediction)
        with patch("strategy.defending.predict_bicycle_trajectory",
                   wraps=predict_bicycle_trajectory) as rollout, \
             patch("strategy.defending.has_predicted_collision", return_value=None) as check:
            strategy.update(self.defender, (self.ego,), self.track, 0.0)
        self.assertTrue(all(call.args[4] == 1.5 for call in rollout.call_args_list))
        rear_times = strategy.last_rear_prediction.times
        self.assertTrue(all(len(call.args[1]) == len(rear_times) for call in check.call_args_list))
        self.assertTrue(all(call.args[1][0] == self.defender.state for call in check.call_args_list))

    def test_without_prediction_config_geometric_prediction_is_kept(self) -> None:
        with patch("strategy.defending.predict_bicycle_trajectory") as rollout:
            geometric = CostBasedDefenseStrategy(self.config, "ego")
            decision = geometric.update(self.defender, (self.ego,), self.track, 0.0)
        rollout.assert_not_called()
        self.assertIsNone(geometric.prediction_config)
        self.assertIs(decision.mode, DefenseMode.DEFEND)


if __name__ == "__main__":
    unittest.main()
