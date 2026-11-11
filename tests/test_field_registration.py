"""Verify editable field lists, same-time labels and the real collector/trainer flow."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.pure_pursuit import PurePursuitConfig
from core.track import Track
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_model import VehicleModelConfig
import generate_training_data
from sim.data_generation import SimulationSettings
from strategy.assignment import StrategyAssignment
from tests.test_training_data import PAIR, record_frames
from training_data import (
    DatasetSchema, InputField, OutputField, SamplingConfig, TrainingDataCollector, VehiclePair,
    build_dataset, create_default_registry, load_dataset, load_recordings,
)
from training_data import fields
from training_data.io import PROJECT_ROOT

try:
    import train_gp
except ModuleNotFoundError as error:
    if error.name != "sklearn":
        raise
    train_gp = None


class FieldRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track.create_oval(10.0, 5.0, 80)
        self.collector = TrainingDataCollector(0.02)
        record_frames(self.collector, "mapping", 4)
        self.schema = DatasetSchema(("opponent_s",), ("opponent_current_d",))
        self.sampling = SamplingConfig(label_horizon_steps=0)

    def test_s_and_d_use_the_same_frame_including_the_terminal_frame(self) -> None:
        dataset = build_dataset(self.collector.episodes, self.track, (PAIR,), self.sampling, self.schema)
        np.testing.assert_allclose(dataset.X[:, 0], [4.0, 5.0, 6.0, 7.0])
        np.testing.assert_allclose(dataset.Y[:, 0], [0.1, 0.11, 0.14, 0.19])
        self.assertEqual(dataset.sampling.label_horizon_seconds, 0.0)
        self.assertEqual(dataset.sampling.target_timing, "current")
        self.assertNotIn("insufficient_future", dataset.skipped)

    def test_future_d_and_delta_still_use_future_frames(self) -> None:
        schema = DatasetSchema(("opponent_s",), ("opponent_future_d", "opponent_delta_d"))
        dataset = build_dataset(self.collector.episodes, self.track, (PAIR,), SamplingConfig(), schema)
        np.testing.assert_allclose(dataset.X[:, 0], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(dataset.Y[:, 0], [0.11, 0.14, 0.19])
        np.testing.assert_allclose(dataset.Y[:, 1], [0.01, 0.03, 0.05])
        self.assertEqual(dataset.sampling.target_timing, "future")

    def test_wrong_horizons_and_mixed_target_times_are_rejected(self) -> None:
        future = DatasetSchema(("opponent_s",), ("opponent_future_d",))
        mixed = DatasetSchema(("opponent_s",), ("opponent_current_d", "opponent_future_d"))
        cases = ((self.schema, SamplingConfig(), "當下答案"),
                 (future, self.sampling, "未來答案"), (mixed, self.sampling, "分開收集"))
        for schema, sampling, message in cases:
            with self.subTest(schema=schema), self.assertRaisesRegex(ValueError, message):
                build_dataset(self.collector.episodes, self.track, (PAIR,), sampling, schema)

    def test_editing_the_lists_registers_new_input_and_current_target(self) -> None:
        new_input = InputField("ego_x", "m", "Current ego world x.", lambda c: c.ego.state.x)
        new_output = OutputField("ego_current_speed", "m/s", "Current ego speed.",
                                 lambda c: c.ego.state.velocity, timing="current")
        with patch.object(fields, "INPUT_FIELDS", (*fields.INPUT_FIELDS, new_input)), \
             patch.object(fields, "OUTPUT_FIELDS", (*fields.OUTPUT_FIELDS, new_output)):
            schema = DatasetSchema(("ego_x",), ("ego_current_speed",))
            dataset = build_dataset(self.collector.episodes, self.track, (PAIR,), self.sampling, schema)
        np.testing.assert_allclose(dataset.X[:, 0], [2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(dataset.Y[:, 0], [8.0, 8.0, 8.0, 8.0])
        with self.assertRaisesRegex(ValueError, "unregistered"):
            create_default_registry().resolve_features(schema)

    def test_custom_future_target_can_read_future_ego(self) -> None:
        output = OutputField("ego_future_x", "m", "Future ego x.",
                             lambda c: c.future_ego.state.x, timing="future")
        with patch.object(fields, "OUTPUT_FIELDS", (*fields.OUTPUT_FIELDS, output)):
            schema = DatasetSchema(("opponent_s",), ("ego_future_x",))
            dataset = build_dataset(self.collector.episodes, self.track, (PAIR,), SamplingConfig(), schema)
        np.testing.assert_allclose(dataset.Y[:, 0], [3.0, 4.0, 5.0])

    def test_duplicate_fields_and_invalid_timing_are_clear_errors(self) -> None:
        with patch.object(fields, "INPUT_FIELDS", (*fields.INPUT_FIELDS, fields.INPUT_FIELDS[0])):
            with self.assertRaisesRegex(ValueError, "unique name"):
                create_default_registry()
        bad = replace(fields.OUTPUT_FIELDS[0], timing="unknown")
        with patch.object(fields, "OUTPUT_FIELDS", (bad,)):
            with self.assertRaisesRegex(ValueError, "timing"):
                create_default_registry()

    def test_progress_is_the_default_opponent_in_a_mixed_scene(self) -> None:
        assignments = {"attacker": StrategyAssignment("cost_overtake"),
                       "opponent": StrategyAssignment("progress")}
        pairs = generate_training_data._select_pairs(None, ("attacker", "opponent"), assignments)
        self.assertEqual(pairs, (VehiclePair("attacker", "opponent"),))
        explicit = generate_training_data._select_pairs(
            [["opponent", "attacker"]], ("attacker", "opponent"), assignments,
        )
        self.assertEqual(explicit, (VehiclePair("opponent", "attacker"),))

    def test_list_fields_does_not_start_simulation(self) -> None:
        output = io.StringIO()
        with patch("generate_training_data._demo_settings", side_effect=AssertionError("must not simulate")), \
             redirect_stdout(output):
            code = generate_training_data.main(["--list-fields"])
        self.assertEqual(code, 0)
        self.assertIn("opponent_s", output.getvalue())
        self.assertIn("opponent_current_d [m, 當下]", output.getvalue())
        self.assertIn("opponent_delta_d [m, 未來]", output.getvalue())

    def test_cli_rejects_bad_time_before_starting_simulation(self) -> None:
        errors = io.StringIO()
        with patch("generate_training_data._demo_settings", side_effect=AssertionError("must not simulate")), \
             redirect_stderr(errors), self.assertRaises(SystemExit):
            generate_training_data.main(["--features", "opponent_s", "--targets", "opponent_current_d",
                                         "--horizon-steps", "1"])
        self.assertIn("當下答案", errors.getvalue())


class FieldRegistrationIOTests(unittest.TestCase):
    def setUp(self) -> None:
        cache = PROJECT_ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="field-registration-test-", dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "data"

    def collect(self, inputs: tuple[str, ...], outputs: tuple[str, ...], *extra_args: str) -> int:
        settings = SimulationSettings(
            Track.create_oval(10.0, 5.0, 80),
            VehicleModelConfig(0.33, 0.02, math.radians(50.0), 0.0, 10.0),
            VehicleGeometry(0.55, 0.3, 0.12),
            PurePursuitConfig(2.0, 0.0, 1.5, 3.0, math.radians(50.0)),
        )
        specs = (("ego", 0.0, 5.0), ("opponent", 5.0, 4.0))
        assignments = {"opponent": StrategyAssignment("progress")}
        with patch("generate_training_data._demo_settings", return_value=(settings, specs, assignments)), \
             patch.object(fields, "SELECTED_INPUTS", inputs), patch.object(fields, "SELECTED_OUTPUTS", outputs), \
             redirect_stdout(io.StringIO()):
            return generate_training_data.main([
                "--episodes", "3", "--seconds", "0.1", "--output", str(self.output), *extra_args,
            ])

    def test_cli_uses_selected_lists_and_infers_zero_horizon(self) -> None:
        self.assertEqual(self.collect(("opponent_s",), ("opponent_current_d",)), 0)
        dataset = load_dataset(self.output)
        self.assertEqual(dataset.X.shape, (18, 1))
        self.assertEqual(dataset.Y.shape, (18, 1))
        self.assertEqual(dataset.sampling.label_horizon_steps, 0)
        self.assertTrue(all(sample.pair == VehiclePair("ego", "opponent") for sample in dataset.samples))
        track, sampling, episodes = load_recordings(self.output)
        frames = {episode.episode_id: episode.frames for episode in episodes}
        for sample in dataset.samples:
            opponent = frames[sample.episode_id][sample.step_index].observations["opponent"]
            self.assertEqual(sample.features[0], opponent.track_position.s)
            self.assertEqual(sample.targets[0], opponent.track_position.d)
        rebuilt = build_dataset(episodes, track, (VehiclePair("ego", "opponent"),), sampling, dataset.schema)
        np.testing.assert_array_equal(rebuilt.Y, dataset.Y)
        metadata = json.loads((self.output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["target_timing"], "current")

    def test_legacy_future_selection_still_infers_one_step(self) -> None:
        self.collect(("relative_s", "relative_d", "relative_speed", "opponent_d"), ("opponent_delta_d",))
        dataset = load_dataset(self.output)
        self.assertEqual(dataset.X.shape, (15, 4))
        self.assertEqual(dataset.sampling.label_horizon_steps, 1)

    def test_metadata_timing_is_checked_and_old_metadata_is_readable(self) -> None:
        self.collect(("opponent_s",), ("opponent_current_d",))
        path = self.output / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["target_timing"] = "future"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "答案時間"):
            load_dataset(self.output)
        del metadata["target_timing"]
        path.write_text(json.dumps(metadata), encoding="utf-8")
        dataset = load_dataset(self.output)
        self.assertEqual(dataset.sampling.target_timing, "current")

    @unittest.skipIf(train_gp is None, "GP training requires scikit-learn")
    def test_same_time_collector_output_trains_without_target_argument(self) -> None:
        self.collect(("opponent_s",), ("opponent_current_d",))
        model_dir = self.output.parent / "model"
        output = io.StringIO()
        with redirect_stdout(output):
            code = train_gp.main(["--data", str(self.output), "--output", str(model_dir)])
        self.assertEqual(code, 0)
        self.assertIn("同時刻對應", output.getvalue())
        report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["target_spec"]["name"], "opponent_current_d")
        self.assertEqual(report["target_timing"], "current")
        self.assertEqual(report["label_horizon_seconds"], 0.0)
        self.assertIn("training_mean", report["metrics"])
        self.assertEqual((model_dir / "evaluation.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    @unittest.skipIf(train_gp is None, "GP training requires scikit-learn")
    def test_multiple_new_targets_require_an_explicit_training_choice(self) -> None:
        extra = OutputField("opponent_current_speed", "m/s", "Current speed.",
                            lambda c: c.opponent.state.velocity, timing="current")
        with patch.object(fields, "OUTPUT_FIELDS", (*fields.OUTPUT_FIELDS, extra)):
            self.collect(("opponent_s",), ("opponent_current_d", "opponent_current_speed"))
        dataset = load_dataset(self.output)
        with self.assertRaisesRegex(ValueError, "--target"):
            train_gp.resolve_target_name(dataset)
        self.assertEqual(train_gp.resolve_target_name(dataset, "opponent_current_d"), "opponent_current_d")


if __name__ == "__main__":
    unittest.main()
