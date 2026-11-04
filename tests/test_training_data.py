import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.track import Track
from core.track_geometry import TrackPosition
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver
from training_data import (
    DatasetSchema,
    FeatureContext,
    FieldSpec,
    SamplingConfig,
    TrainingDataCollector,
    TrainingDataset,
    VehiclePair,
    build_dataset,
    create_default_registry,
    export_dataset,
    load_dataset,
    load_recordings,
)
from training_data.io import PROJECT_ROOT, project_path


GEOMETRY = VehicleGeometry(0.5, 0.3, 0.1)
PAIR = VehiclePair("ego", "opponent")


def observation(vehicle_id: str, s: float, d: float, speed: float) -> VehicleObservation:
    return VehicleObservation(
        vehicle_id, VehicleState(s, d, 0.0, speed), TrackPosition(s, d, 0), GEOMETRY,
    )


def record_frames(
    collector: TrainingDataCollector, episode_id: str, count: int,
    collisions: dict | None = None, missing: set[int] | None = None,
) -> None:
    for index in range(count):
        observations = {
            "ego": observation("ego", 2.0 + index, 0.3, 8.0),
            "opponent": observation("opponent", 4.0 + index, 0.1 + 0.01 * index ** 2, 6.0 + 0.1 * index),
            "third": observation("third", 15.0 + index, -0.3, 3.0),
        }
        if missing and index in missing:
            del observations["opponent"]
        collector.record(
            episode_id, index, index * collector.time_step, observations,
            () if collisions is None else collisions.get(index, ()),
        )


class TrainingDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0))))

    def test_feature_signs_order_and_actual_future_displacement(self) -> None:
        collector = TrainingDataCollector(0.02)
        record_frames(collector, "episode-a", 3)
        dataset = build_dataset(collector.episodes, self.track, (PAIR,))
        np.testing.assert_allclose(dataset.X[0], [-2.0, 0.2, 2.0, 0.1])
        np.testing.assert_allclose(dataset.Y[:, 0], [0.01, 0.03])
        self.assertEqual(dataset.X.shape, (2, 4))
        self.assertEqual(dataset.Y.shape, (2, 1))
        self.assertEqual(dataset.skipped, {"insufficient_future": 1})

    def test_closed_s_wraps_but_open_s_does_not(self) -> None:
        collector = TrainingDataCollector(0.02)
        observations = {"ego": observation("ego", 1.0, 0.0, 4.0),
                        "opponent": observation("opponent", 79.0, 0.0, 4.0)}
        for index in range(2):
            collector.record("wrap", index, index * 0.02, observations)
        wrapped = build_dataset(collector.episodes, self.track, (PAIR,))
        self.assertAlmostEqual(wrapped.X[0, 0], 2.0)
        open_track = Track(np.array(((0.0, 0.0), (80.0, 0.0))), closed=False)
        unwrapped = build_dataset(collector.episodes, open_track, (PAIR,))
        self.assertAlmostEqual(unwrapped.X[0, 0], -78.0)

    def test_horizon_stride_and_episode_boundaries_are_independent(self) -> None:
        collector = TrainingDataCollector(0.02)
        record_frames(collector, "first", 7)
        record_frames(collector, "second", 2)
        config = SamplingConfig(label_horizon_steps=5, sample_stride_steps=2)
        dataset = build_dataset(collector.episodes, self.track, (PAIR,), config)
        self.assertEqual(len(dataset.samples), 1)
        self.assertEqual(dataset.samples[0].episode_id, "first")
        self.assertAlmostEqual(dataset.Y[0, 0], 0.25)
        self.assertAlmostEqual(config.label_horizon_seconds, 0.1)
        self.assertEqual(dataset.skipped["insufficient_future"], 4)

    def test_multiple_directed_pairs_keep_their_vehicle_identity(self) -> None:
        collector = TrainingDataCollector(0.02)
        record_frames(collector, "many", 2)
        reverse = VehiclePair("opponent", "ego")
        third_pair = VehiclePair("third", "opponent")
        dataset = build_dataset(collector.episodes, self.track, (PAIR, reverse, third_pair))
        np.testing.assert_allclose(dataset.X[1, :3], -dataset.X[0, :3])
        np.testing.assert_allclose(dataset.Y[:, 0], [0.01, 0.0, 0.01])
        self.assertEqual([sample.pair for sample in dataset.samples], [PAIR, reverse, third_pair])

    def test_collision_inside_horizon_including_third_vehicle_rejects_sample(self) -> None:
        collector = TrainingDataCollector(0.02)
        record_frames(collector, "collision", 4, {1: (("third", "opponent"),)})
        config = SamplingConfig(label_horizon_steps=2)
        rejected = build_dataset(collector.episodes, self.track, (PAIR,), config)
        self.assertEqual(len(rejected.samples), 0)
        self.assertEqual(rejected.skipped["collision"], 2)
        retained = build_dataset(collector.episodes, self.track, (PAIR,), replace(config, reject_collisions=False))
        self.assertEqual(len(retained.samples), 2)

    def test_collision_on_terminal_frame_is_checked(self) -> None:
        collector = TrainingDataCollector(0.02)
        record_frames(collector, "terminal", 2, {1: (("ego", "opponent"),)})
        dataset = build_dataset(collector.episodes, self.track, (PAIR,))
        self.assertEqual(dataset.skipped["collision"], 1)

    def test_missing_vehicle_inside_horizon_is_not_replaced(self) -> None:
        collector = TrainingDataCollector(0.02)
        record_frames(collector, "missing", 3, missing={1})
        dataset = build_dataset(
            collector.episodes, self.track, (PAIR,), SamplingConfig(label_horizon_steps=2),
        )
        self.assertEqual(dataset.skipped["missing_vehicle"], 1)

    def test_custom_feature_and_multiple_targets_do_not_require_collector_changes(self) -> None:
        collector = TrainingDataCollector(0.02)
        record_frames(collector, "custom", 2)
        registry = create_default_registry()

        def ego_heading(context: FeatureContext) -> float:
            self.assertFalse(hasattr(context, "future"))
            return context.ego.state.yaw

        registry.register_feature(FieldSpec("ego_heading", "rad", "Current ego yaw."), ego_heading)
        schema = DatasetSchema(
            ("opponent_speed", "relative_s", "ego_heading"),
            ("opponent_delta_speed", "opponent_delta_d", "opponent_future_d"),
        )
        dataset = build_dataset(collector.episodes, self.track, (PAIR,), schema=schema, registry=registry)
        np.testing.assert_allclose(dataset.X[0], [6.0, -2.0, 0.0])
        np.testing.assert_allclose(dataset.Y[0], [0.1, 0.01, 0.11])
        self.assertEqual(dataset.Y.shape, (1, 3))
        with self.assertRaisesRegex(ValueError, "unregistered"):
            create_default_registry().resolve_features(schema)

    def test_nonfinite_extractor_result_is_counted_and_skipped(self) -> None:
        collector = TrainingDataCollector(0.02)
        record_frames(collector, "bad", 2)
        registry = create_default_registry()
        registry.register_feature(FieldSpec("invalid", "m", "Invalid feature for validation."), lambda ctx: math.nan)
        schema = DatasetSchema(("invalid",), ("opponent_delta_d",))
        dataset = build_dataset(collector.episodes, self.track, (PAIR,), schema=schema, registry=registry)
        self.assertEqual(dataset.skipped["non_finite"], 1)
        self.assertEqual(dataset.X.shape, (0, 1))
        self.assertEqual(dataset.Y.shape, (0, 1))

    def test_collector_rejects_gaps_wrong_times_and_mutable_observation_maps(self) -> None:
        collector = TrainingDataCollector(0.02)
        observations = {"ego": observation("ego", 0.0, 0.0, 1.0)}
        collector.record("episode", 0, 0.0, observations)
        observations.clear()
        self.assertIn("ego", collector.episodes[0].frames[0].observations)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            collector.record("episode", 2, 0.04, {"ego": observation("ego", 0.0, 0.0, 1.0)})
        with self.assertRaisesRegex(ValueError, "frame time"):
            collector.record("episode", 1, 0.03, {"ego": observation("ego", 0.0, 0.0, 1.0)})

    def test_invalid_schemas_pairs_and_time_settings_are_rejected(self) -> None:
        for kwargs in ({"label_horizon_steps": 0}, {"sample_stride_steps": True}, {"time_step": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SamplingConfig(**kwargs)
        with self.assertRaises(ValueError):
            DatasetSchema(("relative_s", "relative_s"))
        with self.assertRaises(ValueError):
            DatasetSchema(("time_s",))
        with self.assertRaises(ValueError):
            VehiclePair("ego", "ego")
        collector = TrainingDataCollector(0.02)
        record_frames(collector, "episode", 2)
        with self.assertRaisesRegex(ValueError, "time_step"):
            build_dataset(collector.episodes, self.track, (PAIR,), SamplingConfig(time_step=0.1))
        with self.assertRaisesRegex(ValueError, "unique"):
            build_dataset(collector.episodes, self.track, (PAIR, PAIR))
        with self.assertRaisesRegex(ValueError, "episode IDs"):
            build_dataset(collector.episodes * 2, self.track, (PAIR,))


class TrainingDataIOTests(unittest.TestCase):
    def setUp(self) -> None:
        cache = PROJECT_ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="training-data-test-", dir=cache)
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "bundle"
        self.track = Track.create_oval(10.0, 5.0, 40)
        self.collector = TrainingDataCollector(0.02)
        record_frames(self.collector, "roundtrip", 4, {2: (("ego", "third"),)})
        self.sampling = SamplingConfig(reject_collisions=False)

    def export(self) -> TrainingDataset:
        dataset = build_dataset(self.collector.episodes, self.track, (PAIR,), self.sampling)
        export_dataset(dataset, self.collector.episodes, self.track, self.output, {"seed": 42})
        return dataset

    def test_csv_metadata_and_raw_roundtrip_rebuild_new_schema(self) -> None:
        original = self.export()
        loaded = load_dataset(self.output)
        np.testing.assert_array_equal(original.X, loaded.X)
        np.testing.assert_array_equal(original.Y, loaded.Y)
        self.assertEqual(original.samples, loaded.samples)
        track, sampling, episodes = load_recordings(self.output)
        self.assertEqual(episodes, self.collector.episodes)
        np.testing.assert_array_equal(track.centerline, self.track.centerline)
        schema = DatasetSchema(("ego_speed", "opponent_d"), ("opponent_delta_d", "opponent_delta_speed"))
        rebuilt = build_dataset(episodes, track, (PAIR,), sampling, schema)
        self.assertEqual(rebuilt.X.shape, (3, 2))
        self.assertEqual(rebuilt.Y.shape, (3, 2))
        np.testing.assert_array_equal(rebuilt.Y[:, 0], original.Y[:, 0])
        metadata = json.loads((self.output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["run"]["seed"], 42)
        self.assertEqual(metadata["label_horizon_seconds"], 0.02)

    def test_raw_roundtrip_preserves_maneuvers_and_geometries(self) -> None:
        collector = TrainingDataCollector(0.02)
        observations = {
            "ego": replace(observation("ego", 1.0, 0.2, 3.0), maneuver=LateralManeuver(1.0, 0.2, 0.5, 4.0)),
            "opponent": observation("opponent", 4.0, -0.1, 2.0),
        }
        for index in range(2):
            collector.record("maneuver", index, index * 0.02, observations)
        dataset = build_dataset(collector.episodes, self.track, (PAIR,))
        export_dataset(dataset, collector.episodes, self.track, self.output)
        _, _, episodes = load_recordings(self.output)
        self.assertEqual(episodes, collector.episodes)

    def test_existing_output_is_not_overwritten(self) -> None:
        original = self.export()
        before = (self.output / "samples.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            export_dataset(original, self.collector.episodes, self.track, self.output)
        self.assertEqual((self.output / "samples.csv").read_bytes(), before)

    def test_paths_cannot_escape_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "inside this repository"):
            project_path(PROJECT_ROOT / ".." / "escaped-training-output")

    def test_loader_rejects_changed_column_order(self) -> None:
        self.export()
        path = self.output / "samples.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        rows[0][5], rows[0][6] = rows[0][6], rows[0][5]
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        with self.assertRaisesRegex(ValueError, "schema order"):
            load_dataset(self.output)

    def test_loader_rejects_truncated_raw_recording(self) -> None:
        self.export()
        path = self.output / "raw_states.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows[:-3])
        with self.assertRaisesRegex(ValueError, "episode lengths"):
            load_recordings(self.output)


if __name__ == "__main__":
    unittest.main()
