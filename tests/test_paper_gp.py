"""論文格式一步 GP 的載入：欄位與設定核對、訓練範圍、4 個輸出的預測形狀，以及精簡檔來回。"""

import pickle
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from strategy.interaction_features import CURVATURE_LOOKAHEAD, PAPER_FEATURES, PAPER_TARGETS
from strategy.paper_gp import PaperGpModel, paper_compact_arrays
from train_gp import TrainingConfig, create_model

COLUMN = {name: index for index, name in enumerate(PAPER_FEATURES)}


class ConstantPredictor:
    """平均與標準差固定；記錄每次收到的輸入。"""

    def __init__(self, mean=0.0, std=0.0):
        self.mean, self.std, self.calls = mean, std, []

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        self.calls.append(X.copy())
        return np.full(len(X), self.mean), np.full(len(X), self.std)


def fake_paper_model(means=(0.0, 0.0, 0.0, 0.0), stds=(0.0, 0.0, 0.0, 0.0), **limits) -> PaperGpModel:
    """訓練輸入範圍 ±10（後車−前車速度 ±20），只依 limits 限制指定欄位，例如 gap=(-10, 1.0)。"""
    low, high = np.full(len(PAPER_FEATURES), -10.0), np.full(len(PAPER_FEATURES), 10.0)
    for name, (lower, upper) in limits.items():
        low[COLUMN[name]], high[COLUMN[name]] = lower, upper
    even = np.arange(len(PAPER_FEATURES)) % 2 == 0
    rows = np.array((np.where(even, low, high), np.where(even, high, low)))
    predictors = [ConstantPredictor(mean, std) for mean, std in zip(means, stds)]
    return PaperGpModel.from_training_inputs(predictors, rows)


def fitted_bundle(seed: int = 0, samples: int = 30) -> tuple[dict, np.ndarray]:
    """4 個小 GP 以同一批隨機輸入擬合，只為建立可載入的模型。"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(samples, len(PAPER_FEATURES)))
    pipelines = {}
    for target in PAPER_TARGETS:
        pipeline = create_model(X.shape[1], TrainingConfig(optimizer_restarts=0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # 隨機資料的收斂提示不影響檢查
            pipeline.fit(X, rng.normal(size=samples))
        pipelines[target] = pipeline
    bundle = {"pipelines": pipelines, "feature_names": PAPER_FEATURES, "target_names": PAPER_TARGETS,
              "label_horizon_seconds": 0.1, "history_segment_seconds": 0.2, "history_seconds": 0.4,
              "curvature_lookahead_m": CURVATURE_LOOKAHEAD, "curvature_spacing_m": 0.5,
              "model_kind": "paper_format_one_step"}
    return bundle, X


class PaperGpModelTests(unittest.TestCase):
    def test_ranges_reasons_and_prediction_shape(self):
        model = fake_paper_model(means=(0.3, 0.01, 0.0, -0.1), stds=(0.02, 0.01, 0.005, 0.03), gap=(-2.0, 3.0))
        self.assertEqual(model.relative_speed_range, (-20.0, 20.0))
        inside = np.zeros(len(PAPER_FEATURES))
        gap_out, speed_out, missing = inside.copy(), inside.copy(), inside.copy()
        gap_out[COLUMN["gap"]] = 3.5
        missing[COLUMN["front_d"]] = np.nan
        self.assertEqual(model.check_rows(np.array((inside, gap_out, missing))), [None, "gap", "non_finite"])
        # 每欄都在範圍內，兩車速度差 20 m/s 恰好在訓練範圍的邊界上；範圍較窄的模型則判為超出。
        speed_out[COLUMN["rear_speed"]], speed_out[COLUMN["front_speed"]] = 10.0, -10.0
        self.assertIsNone(model.check_rows(speed_out)[0])
        narrow = PaperGpModel(model.predictors, model.lower, model.upper, (-1.0, 1.0))
        self.assertEqual(narrow.check_rows(speed_out), ["relative_speed"])
        mean, std = model.predict(np.array((inside, inside)))
        self.assertEqual((mean.shape, std.shape), ((2, 4), (2, 4)))
        np.testing.assert_allclose(mean[0], (0.3, 0.01, 0.0, -0.1))
        np.testing.assert_allclose(std[1], (0.02, 0.01, 0.005, 0.03))
        with self.assertRaises(ValueError):
            model.check_rows(np.zeros((1, 13)))
        with self.assertRaises(ValueError):
            PaperGpModel.from_training_inputs(model.predictors[:3], np.zeros((2, len(PAPER_FEATURES))))
        broken = PaperGpModel.from_training_inputs(
            [ConstantPredictor(np.nan)] + list(model.predictors[1:]), np.zeros((2, len(PAPER_FEATURES))))
        with self.assertRaises(ValueError):
            broken.predict(inside)

    def test_load_checks_bundle_and_compact_round_trip(self):
        bundle, X = fitted_bundle()
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            good = Path(temporary) / "good.pkl"
            good.write_bytes(pickle.dumps(bundle))
            model = PaperGpModel.load(good)
            np.testing.assert_allclose(model.lower, X.min(axis=0), atol=1e-9)
            np.testing.assert_allclose(model.upper, X.max(axis=0), atol=1e-9)
            self.assertEqual((model.horizon_seconds, model.segment_seconds), (0.1, 0.2))
            mean, std = model.predict(X[:3])
            self.assertEqual(mean.shape, (3, 4))
            for column, target in enumerate(PAPER_TARGETS):
                expected_mean, expected_std = bundle["pipelines"][target].predict(X[:3], return_std=True)
                np.testing.assert_allclose(mean[:, column], expected_mean)
                np.testing.assert_allclose(std[:, column], expected_std)
            # 任何一項與線上特徵不一致都拒絕載入。
            wrong = {
                "feature order": {"feature_names": PAPER_FEATURES[::-1]},
                "targets": {"target_names": PAPER_TARGETS[:3]},
                "horizon": {"label_horizon_seconds": 0.5},
                "segment": {"history_segment_seconds": 0.1},
                "zeta": {"curvature_lookahead_m": 2.0},
                "kind": {"model_kind": "other"},
                "missing pipeline": {"pipelines": {t: bundle["pipelines"][t] for t in PAPER_TARGETS[:3]}},
            }
            for label, change in wrong.items():
                with self.subTest(label=label):
                    path = Path(temporary) / f"{label.replace(' ', '_')}.pkl"
                    path.write_bytes(pickle.dumps({**bundle, **change}))
                    with self.assertRaises(ValueError):
                        PaperGpModel.load(path)
            # 精簡檔：不存分解矩陣，載入時重建，4 個輸出的預測與原模型一致。
            compact = Path(temporary) / "model.npz"
            np.savez_compressed(compact, **paper_compact_arrays(bundle, "abc"))
            rebuilt = PaperGpModel.load(compact)
            mean_b, std_b = rebuilt.predict(X[:5])
            mean_a, std_a = model.predict(X[:5])
            np.testing.assert_allclose(mean_b, mean_a, rtol=0.0, atol=1e-9)
            np.testing.assert_allclose(std_b, std_a, rtol=0.0, atol=1e-9)
            np.testing.assert_array_equal(rebuilt.lower, model.lower)
            self.assertEqual(rebuilt.relative_speed_range, model.relative_speed_range)
            arrays = paper_compact_arrays(bundle)
            arrays["metadata"] = np.array('{"format": "interaction_gp_compact_v1"}')
            other = Path(temporary) / "other.npz"
            np.savez_compressed(other, **arrays)
            with self.assertRaises(ValueError):
                PaperGpModel.load(other)
        with self.assertRaises(ValueError):
            PaperGpModel.load(Path(__file__).resolve().parents[2] / "outside.pkl")


if __name__ == "__main__":
    unittest.main()
