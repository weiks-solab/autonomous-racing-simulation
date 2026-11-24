"""互動 GP 超車：查詢條件、退回規則、前車軌跡、±kσ 邊界檢查與工廠註冊。"""

from dataclasses import replace
import pickle
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from core.track import Track
from strategy.assignment import StrategyConfigs, create_strategy_factories
from strategy.interaction_features import INTERACTION_FEATURES
from strategy.interaction_gp import InteractionGpModel, compact_arrays
from strategy.interaction_overtaking import (
    InteractionOvertakingConfig, InteractionOvertakingStrategy, TRAINED_OFFSETS,
)
from strategy.offset_path import OffsetPathConfig
from strategy.relative_overtaking import RelativeOvertakingConfig, RelativeOvertakingStrategy
from tests.test_bicycle_prediction import observation, prediction_config
from train_gp import TrainingConfig, create_model

GAP = INTERACTION_FEATURES.index("gap")


class FakePredictor:
    """平均 = mean_per_target(d_target)，標準差固定；記錄每次收到的輸入。"""

    def __init__(self, mean=0.0, std=0.01):
        self.mean, self.std, self.calls = mean, std, []

    def predict(self, X, return_std=False):
        X = np.asarray(X)
        self.calls.append(X.copy())
        mean = np.array([self.mean(row[-1]) if callable(self.mean) else self.mean for row in X])
        return mean, np.full(len(X), self.std)


def fake_model(predictor, gap_range=(-10.0, 10.0)):
    """訓練輸入涵蓋很寬的範圍，只讓 gap 依參數限制。"""
    low = np.full(len(INTERACTION_FEATURES), -10.0)
    high = np.full(len(INTERACTION_FEATURES), 10.0)
    low[GAP], high[GAP] = gap_range
    rows = [np.where(np.arange(len(low)) % 2 == 0, low, high), np.where(np.arange(len(low)) % 2 == 0, high, low)]
    rows += [np.r_[low[:-1], target] for target in TRAINED_OFFSETS]
    return InteractionGpModel.from_training_inputs(predictor, np.array(rows))


class InteractionOvertakingTests(unittest.TestCase):
    def setUp(self):
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.relative = RelativeOvertakingConfig(
            candidate_offsets=TRAINED_OFFSETS, transition_length=3.0, decision_period=0.1,
            prediction_horizon=1.0, detection_range=6.0)
        self.configs = {"ego": prediction_config(4.0), "front": prediction_config(3.0)}

    def strategy(self, model=None, **kwargs):
        config = InteractionOvertakingConfig(relative=self.relative, **kwargs)
        return InteractionOvertakingStrategy(config, self.configs["ego"], self.configs, model)

    def run_steps(self, strategy, ego_d=0.0, front_gap=2.0, until=0.4):
        """以固定位置更新到 until（每 0.02 s 一步），回傳最後一次決策。"""
        decision = None
        for step in range(round(until / 0.02) + 1):
            ego = observation(self.track, "ego", 20.0, ego_d, 4.0)
            front = observation(self.track, "front", 20.0 + front_gap, 0.0, 3.0)
            decision = strategy.update(ego, (front,), self.track, step * 0.02)
        return decision

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

    def test_first_decision_falls_back_without_history(self):
        strategy = self.strategy(fake_model(FakePredictor()))
        decision = self.run_steps(strategy, until=0.0)
        self.assertEqual(strategy.last_gp_query.reason, "no_history")
        self.assertIn("gp=off(no_history)", decision.details)
        self.assertEqual(decision.candidate_snapshot.gp_predictions, ())

    def test_out_of_range_gap_falls_back(self):
        strategy = self.strategy(fake_model(FakePredictor(), gap_range=(3.0, 6.0)))
        decision = self.run_steps(strategy, front_gap=2.0)
        self.assertEqual(strategy.last_gp_query.reason, "out_of_range:gap")
        self.assertIn("gp=off(out_of_range:gap)", decision.details)

    def test_candidate_mode_queries_every_target_with_current_state(self):
        predictor = FakePredictor()
        strategy = self.strategy(fake_model(predictor), conditioning="candidate")
        self.run_steps(strategy)
        queried = predictor.calls[-1]
        self.assertEqual(sorted(queried[:, -1]), sorted(TRAINED_OFFSETS))
        # 除了 d_target，所有列的狀態輸入相同（都屬於當下時刻）。
        np.testing.assert_array_equal(queried[:, :-1], np.repeat(queried[:1, :-1], len(queried), axis=0))

    def test_previous_mode_queries_previous_decision_target(self):
        # 前車預測會往右移，讓自車換到非 0 的目標，再確認下一次查詢用的是這個上一次目標。
        predictor = FakePredictor(mean=-0.4)
        strategy = self.strategy(fake_model(predictor), conditioning="previous")
        queried = []
        for step in range(41):
            ego = observation(self.track, "ego", 20.0, 0.0, 4.0)
            front = observation(self.track, "front", 21.2, 0.0, 3.0)
            before, calls = strategy.decision.target_d, len(predictor.calls)
            strategy.update(ego, (front,), self.track, step * 0.02)
            if len(predictor.calls) > calls:
                self.assertEqual(predictor.calls[-1].shape[0], 1)   # 每次決策只查一次
                queried.append((before, predictor.calls[-1][0, -1]))
        self.assertGreaterEqual(len(queried), 2)
        for before, used in queried:
            self.assertEqual(used, before)

    def test_front_trajectory_reaches_predicted_offset_at_gp_horizon(self):
        strategy = self.strategy(fake_model(FakePredictor(mean=0.3, std=0.0)), band_sigma=0.0)
        decision = self.run_steps(strategy)
        query = strategy.last_gp_query
        self.assertTrue(query.active)
        times, offsets = np.array(query.prediction.times), np.array(query.prediction.offsets)
        self.assertAlmostEqual(offsets[0], 0.0)
        self.assertAlmostEqual(float(np.interp(0.25, times, offsets)), 0.15, places=6)
        np.testing.assert_allclose(offsets[times >= 0.5], 0.3, atol=1e-9)
        gp = decision.candidate_snapshot.gp_predictions[-1]
        self.assertEqual((gp.vehicle_id, gp.delta_d_mean, gp.horizon_seconds), ("front", 0.3, 0.5))

    def test_band_rejects_candidate_that_only_meets_the_band(self):
        # 自車已在左側 +0.6，前車在中心；平均不會碰到，+2σ 邊界（+0.5 m）會。
        rejected = {}
        for band in (0.0, 2.0):
            strategy = self.strategy(fake_model(FakePredictor(mean=0.0, std=0.25)), band_sigma=band)
            self.run_steps(strategy, ego_d=0.6, front_gap=0.9)
            stay = next(candidate for candidate in strategy.last_candidates
                        if candidate.maneuver.target_d == 0.6 and candidate.speed_limit is None)
            rejected[band] = stay
        self.assertTrue(rejected[0.0].feasible)
        self.assertFalse(rejected[2.0].feasible)
        self.assertIn("mean+2σ", rejected[2.0].collision.reason)

    def test_untrained_candidate_offsets_are_rejected(self):
        relative = replace(self.relative, candidate_offsets=(-0.4, 0.0, 0.4))
        with self.assertRaises(ValueError):
            InteractionOvertakingStrategy(InteractionOvertakingConfig(relative=relative),
                                          self.configs["ego"], self.configs, fake_model(FakePredictor()))
        for kwargs in ({"conditioning": "future"}, {"band_sigma": -1.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InteractionOvertakingConfig(**kwargs)

    def test_factory_and_reference_path_length(self):
        configs = StrategyConfigs()
        factories = create_strategy_factories(configs, self.configs)
        self.assertIsInstance(factories["interaction_overtake"]("ego"), InteractionOvertakingStrategy)
        self.assertEqual(configs.interaction_overtake.transition_length, 3.0)
        from strategy.assignment import StrategyAssignment
        path = configs.path_config_for(OffsetPathConfig(path_length=1.0),
                                       (StrategyAssignment("interaction_overtake"),))
        self.assertEqual(path.path_length, 3.0)


class InteractionModelLoadTests(unittest.TestCase):
    def test_load_checks_bundle_and_recovers_training_ranges(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, len(INTERACTION_FEATURES)))
        X[:, -1] = rng.choice(TRAINED_OFFSETS, 30)
        pipeline = create_model(X.shape[1], TrainingConfig(optimizer_restarts=0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # 隨機資料只為建立可載入的模型，收斂提示不影響檢查
            pipeline.fit(X, rng.normal(size=30))
        bundle = {"pipeline": pipeline, "feature_names": INTERACTION_FEATURES, "target": "opponent_delta_d",
                  "label_horizon_seconds": 0.5, "history_segment_seconds": 0.2}
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            good, bad = Path(temporary) / "good.pkl", Path(temporary) / "bad.pkl"
            good.write_bytes(pickle.dumps(bundle))
            bad.write_bytes(pickle.dumps({**bundle, "feature_names": INTERACTION_FEATURES[:-1]}))
            model = InteractionGpModel.load(good)
            np.testing.assert_allclose(model.lower, X.min(axis=0), atol=1e-9)
            np.testing.assert_allclose(model.upper, X.max(axis=0), atol=1e-9)
            self.assertEqual(set(model.trained_targets), set(np.round(X[:, -1], 6)))
            mean, std = model.predict(X[:2])
            self.assertEqual((mean.shape, std.shape), ((2,), (2,)))
            self.assertIsNone(model.check(X[0]))
            outside = X[0].copy()
            outside[GAP] = X[:, GAP].max() + 1.0
            self.assertEqual(model.check(outside), "gap")
            with self.assertRaises(ValueError):
                InteractionGpModel.load(bad)
            # 精簡檔：不存分解矩陣，載入時以固定超參數重建，預測與原模型一致。
            compact = Path(temporary) / "model.npz"
            np.savez_compressed(compact, **compact_arrays(bundle))
            rebuilt = InteractionGpModel.load(compact)
            mean_b, std_b = rebuilt.predict(X[:5])
            mean_a, std_a = model.predict(X[:5])
            np.testing.assert_allclose(mean_b, mean_a, atol=1e-9)
            np.testing.assert_allclose(std_b, std_a, atol=1e-9)
            np.testing.assert_array_equal(rebuilt.lower, model.lower)
            arrays = compact_arrays(bundle)
            arrays["metadata"] = np.array('{"format": "other"}')
            wrong = Path(temporary) / "wrong.npz"
            np.savez_compressed(wrong, **arrays)
            with self.assertRaises(ValueError):
                InteractionGpModel.load(wrong)


if __name__ == "__main__":
    unittest.main()
