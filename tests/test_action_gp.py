"""八輸入使用當下已執行候選，舊七輸入與 episode 切分保持相容。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from generate_training_data import _demo_settings
from interaction_gp import split_groups, train
from strategy.base import StrategyDecision
from core.vehicle_state import ControlCommand
from tests.test_bicycle_prediction import observation
from tests import test_interaction_gp
from train_action_gp import collect_current
from training_data.interaction import ACTION_FEATURES, FEATURES, build_interaction_dataset
from training_data.io import PROJECT_ROOT, load_dataset
from training_data.schema import SamplingConfig, VehiclePair


class ActionGpTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = test_interaction_gp.InteractionTests()
        fixture.setUp()
        self.track, self.episode, self.pair = fixture.track, fixture.episode, fixture.pair
        self.sampling = SamplingConfig(0.02, 25, 5)

    def test_eighth_feature_is_current_executed_target_not_future_action(self) -> None:
        targets = {("episode", 5): 0.6, ("episode", 10): -0.3, ("episode", 30): -0.6}
        dataset = build_interaction_dataset((self.episode,), self.track, self.pair, self.sampling, targets)
        self.assertEqual(dataset.schema.feature_names, ACTION_FEATURES)
        np.testing.assert_allclose(dataset.X[:, -1], [0.6, -0.3])
        np.testing.assert_allclose(dataset.Y[:, 0], 0.05)
        changed = {**targets, ("episode", 30): 0.3}
        np.testing.assert_array_equal(dataset.X, build_interaction_dataset(
            (self.episode,), self.track, self.pair, self.sampling, changed).X)

    def test_missing_action_is_an_error_and_legacy_features_remain_seven(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing executed d_target"):
            build_interaction_dataset((self.episode,), self.track, self.pair, self.sampling, {})
        legacy = build_interaction_dataset((self.episode,), self.track, self.pair, self.sampling)
        self.assertEqual(legacy.schema.feature_names, FEATURES)
        self.assertEqual(legacy.X.shape[1], 7)

    def test_current_scene_collection_and_all_training_samples_use_12_4_4(self) -> None:
        settings, specs, assignments = _demo_settings(False)
        pair = VehiclePair("vehicle-1", "vehicle-2")
        seen = []

        def simulate(scenario, active_settings, collector, seconds, step_observer):
            seen.append(scenario)
            self.assertIs(active_settings, settings)
            for index in range(round(seconds / collector.time_step) + 1):
                observations = {vehicle.vehicle_id: observation(
                    settings.track, vehicle.vehicle_id, vehicle.start_s + index * 0.06,
                    index * 0.002 if vehicle.vehicle_id == pair.opponent_id else 0.0, 3.0,
                ) for vehicle in scenario.vehicles}
                collector.record(scenario.episode_id, index, index * collector.time_step, observations)
                if index < round(seconds / collector.time_step):
                    step_observer(index, observations[pair.ego_id],
                                  StrategyDecision("attack", target_d=0.6 if index < 10 else -0.3),
                                  ControlCommand(0.0, 0.0))

        cache = PROJECT_ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            data, output = Path(temporary) / "data", Path(temporary) / "model"
            with patch("train_action_gp._demo_settings", return_value=(settings, specs, assignments)), \
                 patch("train_action_gp.run_episode", side_effect=simulate), redirect_stdout(io.StringIO()):
                collect_current(data, pair, seconds=0.8)
                train(data, output, seed=42, max_train=0)
            self.assertEqual(len(seen), 20)
            self.assertEqual(dict(seen[0].assignments), assignments)
            self.assertEqual([vehicle.target_speed for vehicle in seen[0].vehicles], [spec[2] for spec in specs])
            self.assertTrue(all(vehicle.initial_speed == 0.0 for scenario in seen for vehicle in scenario.vehicles))
            dataset = load_dataset(data)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual({name: len(info["episodes"]) for name, info in report["splits"].items()},
                             {"train": 12, "validation": 4, "test": 4})
            self.assertEqual(report["fit_count"], report["splits"]["train"]["count"])
            with (output / "model.pkl").open("rb") as stream:
                bundle = pickle.load(stream)
            self.assertEqual(bundle["feature_names"], ACTION_FEATURES)
            self.assertEqual(bundle["model_kind"], "interaction_action")
            groups = split_groups(np.array([sample.episode_id for sample in dataset.samples]), 42)
            np.testing.assert_allclose(bundle["pipeline"].named_steps["scaler"].mean_,
                                       dataset.X[groups["train"]].mean(axis=0))
            self.assertTrue((output / "events.csv").exists())


if __name__ == "__main__":
    unittest.main()
