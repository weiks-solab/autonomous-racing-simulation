from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from core.track import Track
from core.track_geometry import TrackPosition
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.observations import VehicleObservation
from training_data import (
    DatasetSchema, TrainingDataCollector, TrainingDataset, VehiclePair,
    build_dataset, export_dataset,
)
from training_data.io import PROJECT_ROOT
from training_data.schema import DEFAULT_FEATURES

try:
    import train_gp
except ModuleNotFoundError as error:
    if error.name != "sklearn":
        raise
    train_gp = None


def training_fixture(extra_feature: bool = False) -> tuple:
    track = Track(np.array(((0.0, 0.0), (200.0, 0.0))), closed=False)
    collector = TrainingDataCollector(0.02)
    geometry = VehicleGeometry(0.55, 0.3, 0.12)
    for episode in range(5):
        for index in range(7):
            observations = {}
            positions = (
                ("ego", 7.0 + episode * 5 + index * 0.25, 0.2 * np.sin(index + 0.3 * episode),
                 6.0 + episode * 0.1 + index * 0.02),
                ("opponent", 10.0 + episode * 5 + index * 0.2, -0.1 + episode * 0.02 + 0.004 * index ** 2,
                 4.0 + episode * 0.2 + 0.03 * index ** 2),
            )
            for vehicle_id, s, d, speed in positions:
                observations[vehicle_id] = VehicleObservation(
                    vehicle_id, VehicleState(s, float(d), 0.0, speed),
                    TrackPosition(s, float(d), 0), geometry,
                )
            collector.record(f"episode-{episode}", index, index * 0.02, observations)
    schema = DatasetSchema(
        DEFAULT_FEATURES + (("ego_speed",) if extra_feature else ()),
        ("opponent_delta_d", "opponent_delta_speed"),
    )
    dataset = build_dataset(
        collector.episodes, track,
        (VehiclePair("ego", "opponent"), VehiclePair("opponent", "ego")), schema=schema,
    )
    return track, collector.episodes, dataset


@unittest.skipIf(train_gp is None, "GP training tests require scikit-learn (requirements-gp.txt)")
class GPTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        _, _, self.dataset = training_fixture()
        self.config = train_gp.TrainingConfig(optimizer_restarts=0)

    def test_split_keeps_entire_episodes_and_vehicle_pairs_together(self) -> None:
        train, test = train_gp.split_episodes(self.dataset, self.config)
        train_groups = {self.dataset.samples[index].episode_id for index in train}
        test_groups = {self.dataset.samples[index].episode_id for index in test}
        self.assertFalse(train_groups & test_groups)
        self.assertEqual(len(train_groups), 4)
        self.assertEqual(len(test_groups), 1)
        self.assertEqual(set(train) | set(test), set(range(len(self.dataset.samples))))
        self.assertEqual({self.dataset.samples[index].pair for index in test},
                         {VehiclePair("ego", "opponent"), VehiclePair("opponent", "ego")})
        repeated = train_gp.split_episodes(self.dataset, self.config)
        np.testing.assert_array_equal(train, repeated[0])
        np.testing.assert_array_equal(test, repeated[1])

    def test_scaler_does_not_learn_from_large_held_out_values(self) -> None:
        train, test = train_gp.split_episodes(self.dataset, self.config)
        held_out = set(test)
        samples = tuple(
            replace(sample, features=(1e6, *sample.features[1:])) if index in held_out else sample
            for index, sample in enumerate(self.dataset.samples)
        )
        dataset = replace(self.dataset, samples=samples)
        result = train_gp.train_model(dataset, self.config)
        actual_mean = result.model.named_steps["scaler"].mean_
        np.testing.assert_allclose(actual_mean, dataset.X[train].mean(axis=0))
        self.assertNotAlmostEqual(actual_mean[0], dataset.X.mean(axis=0)[0])

    def test_extra_features_and_selecting_another_target(self) -> None:
        _, _, dataset = training_fixture(extra_feature=True)
        config = replace(self.config, target="opponent_delta_speed")
        result = train_gp.train_model(dataset, config)
        self.assertEqual(result.model.n_features_in_, 5)
        self.assertEqual(result.target_spec.name, "opponent_delta_speed")
        self.assertEqual(result.target_spec.unit, "m/s")
        self.assertEqual(result.mean.shape, (len(result.test_indices),))
        self.assertTrue(np.all(np.isfinite(result.mean)))
        self.assertTrue(np.all(result.std >= 0.0))
        expected = dataset.Y[result.test_indices, 1]
        errors = result.mean - expected
        self.assertAlmostEqual(result.metrics["gp"]["mae"], np.abs(errors).mean())

    def test_metrics_use_original_units_and_include_zero_baseline(self) -> None:
        actual = np.array([0.01, -0.02])
        predicted = np.array([0.02, -0.02])
        metrics = train_gp.prediction_errors(actual, predicted)
        self.assertAlmostEqual(metrics["mae"], 0.005)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.00005))
        result = train_gp.train_model(self.dataset, self.config)
        actual = self.dataset.Y[result.test_indices, 0]
        self.assertAlmostEqual(result.metrics["always_zero"]["mae"], np.abs(actual).mean())

    def test_empty_single_episode_and_unknown_target_are_clear_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "沒有有效樣本"):
            train_gp.train_model(replace(self.dataset, samples=()), self.config)
        single = tuple(sample for sample in self.dataset.samples if sample.episode_id == "episode-0")
        with self.assertRaisesRegex(ValueError, "至少需要 2"):
            train_gp.train_model(replace(self.dataset, samples=single), self.config)
        with self.assertRaisesRegex(ValueError, "可用欄位"):
            train_gp.train_model(self.dataset, replace(self.config, target="unknown"))

    def test_invalid_settings_and_empty_training_split_are_rejected(self) -> None:
        for kwargs in (
            {"test_fraction": 0.0}, {"test_fraction": float("nan")}, {"seed": -1},
            {"seed": 2 ** 32}, {"length_scale": 0.0}, {"noise_level": float("inf")},
            {"optimizer_restarts": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                train_gp.TrainingConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "測試比例太高"):
            train_gp.split_episodes(self.dataset, replace(self.config, test_fraction=0.99))


@unittest.skipIf(train_gp is None, "GP training tests require scikit-learn (requirements-gp.txt)")
class GPTrainingIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.track, cls.episodes, cls.dataset = training_fixture()
        cls.config = train_gp.TrainingConfig(optimizer_restarts=0)
        cls.result = train_gp.train_model(cls.dataset, cls.config)

    def setUp(self) -> None:
        cache = PROJECT_ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="gp-training-test-", dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.data_directory = self.directory / "data"
        self.output = self.directory / "model"
        export_dataset(self.dataset, self.episodes, self.track, self.data_directory)

    def test_saved_model_preserves_predictions_schema_units_and_horizon(self) -> None:
        train_gp.save_results(self.output, self.data_directory, self.dataset, self.result, self.config)
        with (self.output / "model.pkl").open("rb") as stream:
            bundle = pickle.load(stream)
        self.assertEqual(bundle["feature_names"], self.dataset.schema.feature_names)
        self.assertEqual(bundle["target_spec"]["name"], "opponent_delta_d")
        self.assertEqual(bundle["target_spec"]["unit"], "m")
        self.assertEqual(bundle["label_horizon_seconds"], 0.02)
        mean, std = bundle["pipeline"].predict(self.dataset.X[self.result.test_indices], return_std=True)
        np.testing.assert_array_equal(mean, self.result.mean)
        np.testing.assert_array_equal(std, self.result.std)

    def test_report_and_predictions_identify_the_held_out_samples(self) -> None:
        train_gp.save_results(self.output, self.data_directory, self.dataset, self.result, self.config)
        report = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertFalse(set(report["train_episode_ids"]) & set(report["test_episode_ids"]))
        self.assertEqual(report["metrics"], self.result.metrics)
        self.assertEqual(report["test_sample_count"], len(self.result.test_indices))
        with (self.output / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row, index, std in zip(rows, self.result.test_indices, self.result.std):
            sample = self.dataset.samples[index]
            self.assertEqual(row["episode_id"], sample.episode_id)
            self.assertEqual(row["ego_id"], sample.pair.ego_id)
            self.assertEqual(row["opponent_id"], sample.pair.opponent_id)
            self.assertEqual(float(row["actual"]), sample.targets[0])
            self.assertEqual(float(row["predicted_std"]), std)
            self.assertEqual(float(row["feature_relative_s"]), sample.features[0])
        self.assertEqual((self.output / "evaluation.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_existing_output_is_not_overwritten(self) -> None:
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            train_gp.save_results(self.output, self.data_directory, self.dataset, self.result, self.config)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertFalse((self.output / "model.pkl").exists())

    def test_cli_reads_collected_folder_and_writes_all_outputs(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = train_gp.main(["--data", str(self.data_directory), "--output", str(self.output)])
        self.assertEqual(code, 0)
        self.assertIn("GP 測試 MAE", output.getvalue())
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {"model.pkl", "report.json", "predictions.csv", "evaluation.png"})
        report = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["target_spec"]["name"], "opponent_delta_d")
        self.assertEqual(report["label_horizon_seconds"], 0.02)

    def test_cli_reports_invalid_target_without_creating_output(self) -> None:
        errors = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
            train_gp.main(["--data", str(self.data_directory), "--target", "unknown", "--output", str(self.output)])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("可用欄位", errors.getvalue())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
