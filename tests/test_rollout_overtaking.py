"""沿上一步軌跡推演對手：時間對齊、特徵來源、增量換算、退回與維持、工廠註冊。"""

from dataclasses import replace
import math
import unittest

import numpy as np

from core.track import Track
from core.track_geometry import world_to_track
from strategy.assignment import StrategyAssignment, StrategyConfigs, create_strategy_factories
from strategy.bicycle_prediction import constant_offset_maneuver, predict_bicycle_trajectory
from strategy.interaction_features import INTERACTION_FEATURES
from strategy.interaction_gp import InteractionGpModel
from strategy.interaction_overtaking import (
    InteractionOvertakingConfig, InteractionOvertakingStrategy, TRAINED_OFFSETS,
)
from strategy.offset_path import OffsetPathConfig
from strategy.relative_overtaking import RelativeOvertakingConfig, RelativeOvertakingStrategy
from strategy.rollout_overtaking import (
    RolloutOvertakingConfig, RolloutOvertakingStrategy, plan_d_at, plan_state_at,
)
from tests.test_bicycle_prediction import observation, prediction_config
from tests.test_interaction_overtaking import FakePredictor, fake_model

COLUMN = {name: index for index, name in enumerate(INTERACTION_FEATURES)}


def bounded_model(predictor, **limits):
    """訓練輸入範圍很寬，只依 limits 限制指定欄位，例如 front_d=(-1, 0.1)。"""
    low = np.full(len(INTERACTION_FEATURES), -10.0)
    high = np.full(len(INTERACTION_FEATURES), 10.0)
    for name, (lower, upper) in limits.items():
        low[COLUMN[name]], high[COLUMN[name]] = lower, upper
    rows = [np.where(np.arange(len(low)) % 2 == 0, low, high), np.where(np.arange(len(low)) % 2 == 0, high, low)]
    rows += [np.r_[low[:-1], target] for target in TRAINED_OFFSETS]
    return InteractionGpModel.from_training_inputs(predictor, np.array(rows))


class RolloutOvertakingTests(unittest.TestCase):
    def setUp(self):
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.relative = RelativeOvertakingConfig(
            candidate_offsets=TRAINED_OFFSETS, transition_length=3.0, decision_period=0.1,
            prediction_horizon=1.0, detection_range=6.0)
        self.configs = {"ego": prediction_config(4.0), "front": prediction_config(3.0)}

    def strategy(self, model=None, relative=None, **kwargs):
        config = RolloutOvertakingConfig(relative=relative or self.relative, **kwargs)
        return RolloutOvertakingStrategy(config, self.configs["ego"], self.configs, model)

    def run_steps(self, strategy, until=0.4, ego_d=0.0, front_gap=2.0, start=0):
        decision = None
        for step in range(start, round(until / 0.02) + 1):
            ego = observation(self.track, "ego", 20.0, ego_d, 4.0)
            front = observation(self.track, "front", 20.0 + front_gap, 0.0, 3.0)
            decision = strategy.update(ego, (front,), self.track, step * 0.02)
        return decision

    def test_constant_mean_accumulates_over_the_horizon(self):
        strategy = self.strategy(fake_model(FakePredictor(mean=0.1, std=0.02)), band_sigma=2.0)
        decision = self.run_steps(strategy)
        rollout = strategy.last_rollout
        self.assertEqual(len(rollout.node_times), 11)
        np.testing.assert_allclose(rollout.offsets[5], 0.1, atol=1e-9)     # d(0.5) = d0 + μ
        np.testing.assert_allclose(rollout.offsets[10], 0.2, atol=1e-9)    # d(1.0) = d0 + 2μ
        np.testing.assert_allclose(rollout.sigmas[5], 0.02, atol=1e-9)     # 0.5 s 時與單次查詢的 σ 相同
        query = strategy.last_gp_query
        offsets = np.array(query.prediction.offsets)
        np.testing.assert_allclose(offsets[-1], 0.2, atol=1e-9)
        np.testing.assert_allclose(np.array(query.bands[0].offsets)[-1], 0.2 + 2 * 0.04, atol=1e-9)
        gp = decision.candidate_snapshot.gp_predictions[-1]
        self.assertAlmostEqual(gp.delta_d_mean, 0.1)
        self.assertTrue(any(text.startswith("rollout d_target=") for text in decision.details))

    def test_queries_follow_previous_plan_and_history(self):
        predictor = FakePredictor()
        strategy = self.strategy(fake_model(predictor))
        self.run_steps(strategy, until=0.38, ego_d=0.3)
        plan = strategy.previous_plan
        calls = len(predictor.calls)
        self.run_steps(strategy, until=0.4, ego_d=0.3, start=20)
        rows = np.vstack(predictor.calls[calls:])
        self.assertEqual(rows.shape[0], 10)
        lag = 0.4 - plan.time_seconds
        self.assertAlmostEqual(lag, 0.1)
        nodes = np.arange(10) * 0.1
        for index, tau in enumerate(nodes):
            with self.subTest(tau=tau):
                self.assertAlmostEqual(rows[index, COLUMN["rear_d"]], plan_d_at(plan.prediction, lag + tau), places=9)
                self.assertEqual(rows[index, COLUMN["d_target"]], plan.target_d)
        # τ=0：早於現在用實際歷史（d=0.3），現在用上一步軌跡。
        self.assertAlmostEqual(rows[0, COLUMN["rear_dd_late"]], plan_d_at(plan.prediction, lag) - 0.3, places=9)
        # τ=0.3：t−0.1 用實際歷史，t+0.1 起用軌跡。
        self.assertAlmostEqual(rows[3, COLUMN["rear_dd_early"]], plan_d_at(plan.prediction, lag + 0.1) - 0.3, places=9)
        self.assertAlmostEqual(rows[3, COLUMN["rear_dd_late"]],
                               plan_d_at(plan.prediction, lag + 0.3) - plan_d_at(plan.prediction, lag + 0.1), places=9)

    def test_plan_is_extended_beyond_its_end(self):
        vehicle = observation(self.track, "ego", 20.0, 0.2, 3.0)
        config = self.configs["front"]
        plan = predict_bicycle_trajectory(vehicle, self.track, constant_offset_maneuver(vehicle, config), config,
                                          1.0, constant_speed=True)
        end = world_to_track(plan.states[-1], self.track)
        extended = world_to_track(plan_state_at(plan, 1.2, self.track), self.track)
        self.assertAlmostEqual(extended.s, end.s + plan.states[-1].velocity * 0.2, places=6)
        self.assertAlmostEqual(extended.d, end.d, places=6)
        self.assertEqual(plan_d_at(plan, 1.5), plan.offsets[-1])

    def test_first_decision_without_previous_plan_falls_back(self):
        strategy = self.strategy(fake_model(FakePredictor()))
        decision = self.run_steps(strategy, until=0.0)
        self.assertIn("gp=off(no_previous_plan)", decision.details)
        self.assertEqual(decision.candidate_snapshot.gp_predictions, ())
        self.assertIsNotNone(strategy.previous_plan)

    def test_first_node_out_of_range_falls_back(self):
        strategy = self.strategy(fake_model(FakePredictor(), gap_range=(3.0, 6.0)))
        decision = self.run_steps(strategy)
        self.assertIn("gp=off(out_of_range:gap)", decision.details)
        self.assertIsNone(strategy.last_rollout)

    def test_later_node_out_of_range_holds_last_offset(self):
        # 每步 +0.2×0.2=0.04 m；前車 d 超過 0.1 m 的節點起維持。
        strategy = self.strategy(bounded_model(FakePredictor(mean=0.2, std=0.0), front_d=(-1.0, 0.1)))
        decision = self.run_steps(strategy)
        rollout = strategy.last_rollout
        self.assertEqual(rollout.reasons[:3], ("", "", ""))
        self.assertEqual(rollout.reasons[3], "front_d")
        np.testing.assert_allclose(rollout.offsets[:4], (0.0, 0.04, 0.08, 0.12), atol=1e-9)
        np.testing.assert_allclose(rollout.offsets[4:], 0.12, atol=1e-9)
        self.assertTrue(any("hold@0.3s(front_d)" in text for text in decision.details))

    def test_single_half_second_step_matches_single_query(self):
        relative = replace(self.relative, prediction_horizon=0.5)
        rollout = self.strategy(fake_model(FakePredictor(mean=0.3, std=0.0)), relative=relative,
                                rollout_step=0.5, band_sigma=0.0)
        single = InteractionOvertakingStrategy(
            InteractionOvertakingConfig(relative=relative, conditioning="previous", band_sigma=0.0),
            self.configs["ego"], self.configs, fake_model(FakePredictor(mean=0.3, std=0.0)))
        self.run_steps(rollout)
        self.run_steps(single)
        np.testing.assert_allclose(rollout.last_gp_query.prediction.offsets,
                                   single.last_gp_query.prediction.offsets, atol=1e-9)

    def test_without_model_matches_relative_overtake(self):
        baseline = RelativeOvertakingStrategy(self.relative, self.configs["ego"], self.configs)
        strategy = self.strategy()
        for step in range(21):
            ego = observation(self.track, "ego", 20.0 + 0.08 * step, 0.0, 4.0)
            front = observation(self.track, "front", 22.0 + 0.06 * step, 0.02 * step, 3.0)
            expected = baseline.update(ego, (front,), self.track, step * 0.02)
            actual = strategy.update(ego, (front,), self.track, step * 0.02)
            self.assertEqual((actual.mode, actual.target_d, actual.speed_limit),
                             (expected.mode, expected.target_d, expected.speed_limit))

    def test_config_validation_factory_and_path_length(self):
        for kwargs in ({"rollout_step": 0.0}, {"rollout_step": math.inf}, {"band_sigma": -1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RolloutOvertakingConfig(**kwargs)
        with self.assertRaises(ValueError):
            self.strategy(fake_model(FakePredictor()), rollout_step=0.6)
        configs = StrategyConfigs()
        strategy = create_strategy_factories(configs, self.configs)["rollout_overtake"]("ego")
        self.assertIsInstance(strategy, RolloutOvertakingStrategy)
        path = configs.path_config_for(OffsetPathConfig(path_length=1.0), (StrategyAssignment("rollout_overtake"),))
        self.assertEqual(path.path_length, 3.0)


if __name__ == "__main__":
    unittest.main()
