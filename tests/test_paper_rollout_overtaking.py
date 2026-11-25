"""論文方式的超車決策：取樣推演、特徵來源、退回與維持、不確定性橢圓的接入、工廠註冊。"""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from core.track import Track
from core.track_geometry import track_length
from strategy.assignment import StrategyAssignment, StrategyConfigs, create_strategy_factories
from strategy.interaction_features import PAPER_FEATURES, heading_error, paper_state_features
from strategy.interaction_gp import predict_profile_trajectory
from strategy.observations import create_vehicle_observation
from strategy.offset_path import OffsetPathConfig
from strategy.paper_rollout_overtaking import PaperRolloutOvertakingConfig, PaperRolloutOvertakingStrategy
from strategy.relative_overtaking import RelativeOvertakingConfig, RelativeOvertakingStrategy
from strategy.responsibility import PositionUncertainty, ResponsibilityConfig, ResponsibilityResult
from strategy.rollout_overtaking import plan_d_at, plan_state_at
from tests.test_bicycle_prediction import observation, prediction_config
from tests.test_paper_gp import fake_paper_model

COLUMN = {name: index for index, name in enumerate(PAPER_FEATURES)}
DT = 0.02


class PaperRolloutOvertakingTests(unittest.TestCase):
    def setUp(self):
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.relative = RelativeOvertakingConfig(
            candidate_offsets=(-0.6, 0.0, 0.6), transition_length=3.0, decision_period=0.1,
            prediction_horizon=1.0, detection_range=6.0,
            responsibility=ResponsibilityConfig(overlap_rule=True))
        self.configs = {"ego": prediction_config(4.0), "front": prediction_config(3.0)}

    def strategy(self, model=None, relative=None, **kwargs):
        config = PaperRolloutOvertakingConfig(relative=relative or self.relative, **kwargs)
        return PaperRolloutOvertakingStrategy(config, self.configs["ego"], self.configs, model)

    def observations(self, step, ego_d=0.0, front_gap=2.0, front_d=0.0):
        return (observation(self.track, "ego", 20.0, ego_d, 4.0),
                observation(self.track, "front", 20.0 + front_gap, front_d, 3.0))

    def run_steps(self, strategy, until=0.4, start=0, **kwargs):
        decision = None
        for step in range(start, round(until / DT) + 1):
            ego, front = self.observations(step, **kwargs)
            decision = strategy.update(ego, (front,), self.track, step * DT)
        return decision

    def test_deterministic_rollout_follows_the_one_step_recursion(self):
        # σ=0：每步 Δs=0.25、Δd=0.01，10 步後進度 2.5 m、d 增加 0.1 m。
        strategy = self.strategy(fake_paper_model(means=(0.25, 0.01, 0.0, 0.0)), samples=5)
        decision = self.run_steps(strategy)
        rollout = strategy.last_rollout
        self.assertEqual(len(rollout.node_times), 11)
        np.testing.assert_allclose(rollout.mean_progress, np.arange(11) * 0.25, atol=1e-12)
        np.testing.assert_allclose(rollout.mean_d, np.arange(11) * 0.01, atol=1e-12)
        np.testing.assert_allclose(rollout.std_s, 0.0, atol=1e-12)
        self.assertEqual(rollout.held_count, 0)
        target = strategy.last_target_prediction
        self.assertAlmostEqual(target.progress[-1], 2.5)
        self.assertAlmostEqual(target.offsets[-1], 0.1)
        self.assertAlmostEqual(target.progress[25], 1.25)   # 0.5 s：節點間線性內插
        gp = decision.candidate_snapshot.gp_predictions[-1]
        self.assertAlmostEqual(gp.delta_d_mean, 0.1)
        self.assertEqual(gp.horizon_seconds, 1.0)
        self.assertTrue(any(text.startswith("paper_gp Q=5") for text in decision.details))

    def test_sampling_is_seeded_and_spreads(self):
        model = fake_paper_model(means=(0.3, 0.0, 0.0, 0.0), stds=(0.02, 0.01, 0.0, 0.0))
        first, second = self.strategy(model, seed=3), self.strategy(model, seed=3)
        other = self.strategy(model, seed=4)
        for strategy in (first, second, other):
            self.run_steps(strategy)
        np.testing.assert_array_equal(first.last_rollout.progress, second.last_rollout.progress)
        self.assertFalse(np.array_equal(first.last_rollout.progress, other.last_rollout.progress))
        std_s, std_d = first.last_rollout.std_s, first.last_rollout.std_d
        self.assertEqual((std_s[0], std_d[0]), (0.0, 0.0))
        self.assertGreater(std_s[-1], std_s[1])
        # 10 步獨立抽樣：σ_s 約 0.02·√10，25 個樣本的估計誤差寬鬆容許。
        self.assertLess(abs(std_s[-1] - 0.02 * math.sqrt(10)), 0.03)

    def test_features_use_actual_state_first_then_previous_plan(self):
        model = fake_paper_model(means=(0.25, 0.01, 0.0, 0.0))
        strategy = self.strategy(model, samples=2)
        self.run_steps(strategy, until=0.38)
        plan = strategy.previous_plan
        calls = model.predictors[0].calls
        calls.clear()
        self.run_steps(strategy, until=0.4, start=20)
        self.assertEqual(len(calls), 10)
        ego, front = self.observations(20)
        lag = 0.4 - plan.time_seconds
        # 第 0 步：兩車都是實際觀測，歷史是實際紀錄（d 一直為 0），與訓練用的特徵相同；只查一列。
        expected = paper_state_features(ego, front, (0.0, 0.0), (0.0, 0.0), self.track)
        self.assertEqual(calls[0].shape, (1, len(PAPER_FEATURES)))
        np.testing.assert_allclose(calls[0][0], expected, atol=1e-12)
        # 第 1 步：自車取上一步計畫在 lag+0.1 s 的狀態。
        planned = create_vehicle_observation(
            "ego", plan_state_at(plan.prediction, lag + 0.1, self.track), ego.geometry, self.track)
        row = calls[1][0]
        np.testing.assert_allclose(row[[COLUMN["rear_d"], COLUMN["rear_heading_error"], COLUMN["rear_speed"]]],
                                   (planned.track_position.d, heading_error(planned, self.track),
                                    planned.state.velocity), atol=1e-12)
        self.assertAlmostEqual(row[COLUMN["front_d"]], 0.01)
        self.assertAlmostEqual(row[COLUMN["gap"]], 22.25 - planned.track_position.s, places=9)
        # 第 2 步：τ−0.2 是現在（實際 d），τ−0.4 早於現在（實際紀錄）；τ 本身用計畫（自車）與樣本（前車）。
        row = calls[2][0]
        planned = create_vehicle_observation(
            "ego", plan_state_at(plan.prediction, lag + 0.2, self.track), ego.geometry, self.track)
        self.assertAlmostEqual(row[COLUMN["front_dd_late"]], 0.02)
        self.assertAlmostEqual(row[COLUMN["front_dd_early"]], 0.0)
        self.assertAlmostEqual(row[COLUMN["rear_dd_late"]], planned.track_position.d - ego.track_position.d)
        # 第 3 步：τ−0.2=0.1 s 晚於現在，自車用計畫值。
        row = calls[3][0]
        self.assertAlmostEqual(row[COLUMN["rear_dd_early"]], plan_d_at(plan.prediction, lag + 0.1))
        self.assertAlmostEqual(row[COLUMN["front_dd_early"]], 0.01)

    def test_fallbacks_keep_relative_overtake_prediction(self):
        model = fake_paper_model(means=(0.25, 0.01, 0.0, 0.0))
        strategy = self.strategy(model)
        decision = self.run_steps(strategy, until=0.0)
        self.assertIn("gp=off(no_previous_plan)", decision.details)
        decision = self.run_steps(strategy, until=0.3, start=1)
        self.assertIn("gp=off(no_history)", decision.details)
        self.assertIsNone(strategy.last_rollout)
        narrow = self.strategy(fake_paper_model(means=(0.25, 0.01, 0.0, 0.0), gap=(-10.0, 1.0)))
        decision = self.run_steps(narrow)
        self.assertIn("gp=off(out_of_range:gap)", decision.details)
        reference = RelativeOvertakingStrategy(self.relative, self.configs["ego"], self.configs)
        self.run_steps(reference)
        self.assertEqual(narrow.last_target_prediction, reference.last_target_prediction)
        self.assertEqual(narrow.decision.target_d, reference.decision.target_d)
        self.assertFalse(narrow.decision.candidate_snapshot.gp_predictions)

    def test_sample_leaving_the_range_continues_at_constant_speed(self):
        # front_d 上限 0.025：第 3 步（d=0.03）起超出，改為等速（3 m/s → 每步 0.3 m）、d 不變。
        model = fake_paper_model(means=(0.25, 0.01, 0.0, 0.0), front_d=(-10.0, 0.025))
        strategy = self.strategy(model, samples=3)
        decision = self.run_steps(strategy)
        rollout = strategy.last_rollout
        self.assertEqual(rollout.exit_steps, (3, 3, 3))
        self.assertEqual(rollout.exit_reasons, ("front_d",) * 3)
        np.testing.assert_allclose(np.diff(rollout.mean_progress), [0.25] * 3 + [0.3] * 7, atol=1e-12)
        np.testing.assert_allclose(rollout.mean_d[3:], 0.03, atol=1e-12)
        self.assertTrue(any("const_v=3/3" in text for text in decision.details))

    def test_uncertainty_reaches_the_responsibility_check(self):
        model = fake_paper_model(means=(0.3, 0.0, 0.0, 0.0), stds=(0.02, 0.01, 0.0, 0.0))
        strategy = self.strategy(model, ellipse_sigma=2.0)
        captured = []

        def capture(times, states, geometry, pairs, config):
            captured.append(pairs)
            return ResponsibilityResult(None)

        with patch("strategy.relative_overtaking.evaluate_responsibility", side_effect=capture):
            self.run_steps(strategy)
        rollout = strategy.last_rollout
        times = strategy.last_target_prediction.times
        uncertainty = captured[-1][0].uncertainty
        self.assertIsInstance(uncertainty, PositionUncertainty)
        nodes = np.asarray(rollout.node_times)
        np.testing.assert_allclose(uncertainty.along, 2.0 * np.interp(times, nodes, rollout.std_s))
        np.testing.assert_allclose(uncertainty.across, 2.0 * np.interp(times, nodes, rollout.std_d))
        np.testing.assert_allclose(uncertainty.tangent, 0.0)   # 前車在第一段直線上
        self.assertIsNone(strategy._pair_uncertainty("someone-else", times))
        flat = self.strategy(model, ellipse_sigma=0.0)
        self.run_steps(flat)
        self.assertIsNotNone(flat.last_rollout)
        self.assertIsNone(flat._pair_uncertainty("front", times))

    def test_wider_ellipse_rejects_the_close_follow_candidate(self):
        # 前車正前方 0.75 m、σ_s 大：圓形可行的貼近跟車，在橢圓下被剔除。
        model = fake_paper_model(means=(0.4, 0.0, 0.0, 0.0), stds=(0.15, 0.0, 0.0, 0.0))
        relative = replace(self.relative, candidate_offsets=(0.0,), speed_candidates_enabled=True,
                           extra_speed_candidate_deltas=(3.0,))
        results = {}
        for k in (0.0, 3.0):
            strategy = self.strategy(model, relative=relative, ellipse_sigma=k)
            self.run_steps(strategy, front_gap=0.9)
            results[k] = [candidate.feasible for candidate in strategy.last_candidates]
        self.assertGreater(sum(results[0.0]), sum(results[3.0]))

    def test_without_model_matches_relative_overtake(self):
        paper = self.strategy(None)
        reference = RelativeOvertakingStrategy(self.relative, self.configs["ego"], self.configs)
        for step in range(round(1.0 / DT) + 1):
            ego, front = self.observations(step, ego_d=0.1 * math.sin(step / 7.0))
            a = paper.update(ego, (front,), self.track, step * DT)
            b = reference.update(ego, (front,), self.track, step * DT)
            self.assertEqual((a.mode, a.target_d, a.speed_limit, a.reason, a.details, a.maneuver),
                             (b.mode, b.target_d, b.speed_limit, b.reason, b.details, b.maneuver))

    def test_config_validation_factory_and_path_length(self):
        for kwargs in ({"samples": 0}, {"samples": 2.5}, {"samples": True}, {"seed": -1},
                       {"ellipse_sigma": -1.0}, {"ellipse_sigma": math.nan}, {"history_margin": 0.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PaperRolloutOvertakingConfig(**{"relative": self.relative, **kwargs})
        # 橢圓放大的是責任約束的 r_col：關閉責任約束時，設定檔可建立（main 仍可載入），建立策略才報錯。
        without = replace(self.relative, responsibility=None)
        PaperRolloutOvertakingConfig(relative=without)
        with self.assertRaises(ValueError):
            self.strategy(relative=without)
        self.strategy(relative=without, ellipse_sigma=0.0)
        with self.assertRaises(ValueError):   # 預測時域須是 GP 一步（0.1 s）的整數倍
            self.strategy(fake_paper_model(), relative=replace(self.relative, prediction_horizon=0.95))
        configs = StrategyConfigs()
        strategy = create_strategy_factories(configs, self.configs)["paper_rollout_overtake"]("ego")
        self.assertIsInstance(strategy, PaperRolloutOvertakingStrategy)
        self.assertIsNone(strategy.model)
        path = configs.path_config_for(OffsetPathConfig(path_length=1.0),
                                       (StrategyAssignment("paper_rollout_overtake"),))
        self.assertEqual(path.path_length, 3.0)


class ProfileProgressTests(unittest.TestCase):
    def setUp(self):
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))

    def test_progress_array_sets_positions_and_heading(self):
        front = observation(self.track, "front", 398.0, 0.2, 3.0)
        times = np.arange(6) * 0.2
        progress = np.array((0.0, 0.5, 1.0, 1.0, 2.5, 4.0))   # 第 2～3 點停住；跨過起終點
        offsets = np.full(6, 0.2)
        prediction = predict_profile_trajectory(front, self.track, offsets, 3.0, 1.0, 0.2, progress=progress)
        self.assertEqual(prediction.progress, tuple(progress))
        self.assertEqual(prediction.times, tuple(times))
        length = track_length(self.track)
        for state, value in zip(prediction.states[1:], progress[1:]):
            expected_s = (398.0 + value) % length
            projected = create_vehicle_observation("x", state, front.geometry, self.track).track_position
            self.assertAlmostEqual(projected.s, expected_s, places=6)
        yaws = [state.yaw for state in prediction.states]
        self.assertAlmostEqual(yaws[2], yaws[1])                      # 停住的段落沿用前一點的航向
        self.assertTrue(all(math.isfinite(yaw) for yaw in yaws))
        # 未給 progress 時維持定速：與原本的呼叫方式相同。
        constant = predict_profile_trajectory(front, self.track, offsets, 3.0, 1.0, 0.2)
        self.assertEqual(constant.progress, tuple(3.0 * times))
        for bad in (np.zeros(5), np.r_[np.zeros(5), np.nan]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                predict_profile_trajectory(front, self.track, offsets, 3.0, 1.0, 0.2, progress=bad)


if __name__ == "__main__":
    unittest.main()
