"""驗證特徵不看未來、標籤對齊與整場資料切分。"""

from dataclasses import replace
from contextlib import redirect_stdout
import io
import unittest
import tempfile
import pickle
from pathlib import Path
import numpy as np

import math

from core.track import Track
from interaction_gp import (
    PROBE_AMPLITUDES, PROBE_HOLD_RANGE, random_probe_config, select_features, speed_strata,
    split_groups, stratified_design, train,
)
from strategy.probe_following import ProbeFollowingConfig
from training_data.io import export_dataset
from tests.test_bicycle_prediction import observation
from training_data.interaction import (
    HISTORY_ACTION_FEATURES, HISTORY_FEATURES, build_interaction_dataset, curvature, features,
    heading_error, history_features,
)
from training_data.schema import FrameSnapshot, RecordedEpisode, SamplingConfig, VehiclePair


def ramp(index: int, start: int, end: int, before: float, after: float) -> float:
    """index 在 start 之前為 before，end 之後為 after，中間線性變化。"""
    if index <= start:
        return before
    if index >= end:
        return after
    return before + (after - before) * (index - start) / (end - start)


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.pair = VehiclePair("rear", "front")
        self.episode = RecordedEpisode("episode", tuple(
            FrameSnapshot("episode", i, i * 0.02, {
                "rear": observation(self.track, "rear", 20 + i * 0.06, 0.0, 3.0),
                "front": observation(self.track, "front", 23 + i * 0.06, i * 0.002, 3.0),
            }) for i in range(40)))

    def test_features_use_only_current_and_past(self):
        x = features(self.episode, 5, self.pair, self.track)
        shortened = replace(self.episode, frames=self.episode.frames[:6])
        self.assertEqual(x, features(shortened, 5, self.pair, self.track))
        self.assertAlmostEqual(x[0], 3.0)
        self.assertAlmostEqual(x[5], 0.1)
        self.assertAlmostEqual(x[6], 0.0)

    def test_half_second_labels_and_collision_interval(self):
        sampling = SamplingConfig(0.02, 25, 5)
        dataset = build_interaction_dataset((self.episode,), self.track, self.pair, sampling)
        self.assertEqual([sample.step_index for sample in dataset.samples], [5, 10])
        for sample in dataset.samples:
            self.assertAlmostEqual(sample.targets[0], 0.05)
        frames = list(self.episode.frames)
        frames[20] = replace(frames[20], collision_pairs=frozenset({("rear", "front")}))
        rejected = build_interaction_dataset((replace(self.episode, frames=tuple(frames)),),
                                             self.track, self.pair, sampling)
        self.assertEqual(len(rejected.samples), 0)
        self.assertEqual(rejected.skipped["collision"], 2)

    def test_episode_split_is_disjoint_and_repeatable(self):
        ids = np.repeat(np.arange(10).astype(str), 5)
        split = split_groups(ids, 42)
        again = split_groups(ids, 42)
        sets = [set(ids[index]) for index in split.values()]
        self.assertEqual(sorted(map(len, sets)), [2, 2, 6])
        for a in range(3):
            for b in range(a):
                self.assertFalse(sets[a] & sets[b])
        for name in split:
            np.testing.assert_array_equal(split[name], again[name])

    def test_curvature_is_signed_and_wraps(self):
        track = Track.create_oval(10.0, 10.0, 500)
        self.assertAlmostEqual(abs(curvature(track, 0.0)), 0.1, delta=0.005)

    def test_training_bundle_reload_and_train_only_scaler(self):
        episodes = tuple(RecordedEpisode(f"ep-{n}", tuple(
            replace(frame, episode_id=f"ep-{n}", observations={
                **frame.observations,
                "rear": replace(frame.observations["rear"],
                                state=replace(frame.observations["rear"].state, velocity=3.0 + n)),
            }) for frame in self.episode.frames)) for n in range(5))
        dataset = build_interaction_dataset(episodes, self.track, self.pair, SamplingConfig(0.02, 25, 5))
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            data, output = Path(temporary) / "data", Path(temporary) / "model"
            export_dataset(dataset, episodes, self.track, data)
            (data / "actions.csv").write_text("episode_id,step_index,time_s,mode,target_d\n" +
                "".join(f"ep-{n},5,0.1,follow,0.0\n" for n in range(5)), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                train(data, output, 42, 100)
            with (output / "model.pkl").open("rb") as stream:
                bundle = pickle.load(stream)
            indices = split_groups(np.array([sample.episode_id for sample in dataset.samples]), 42)["train"]
            np.testing.assert_allclose(bundle["pipeline"].named_steps["scaler"].mean_, dataset.X[indices].mean(axis=0))
            mean, std = bundle["pipeline"].predict(dataset.X[:1], return_std=True)
            self.assertTrue(np.isfinite(mean).all() and np.isfinite(std).all())
            self.assertTrue((output / "predictions.csv").exists())
            self.assertIn("first_arrival", (output / "events.csv").read_text(encoding="utf-8"))


class HistoryFeatureTests(unittest.TestCase):
    """互動歷史：t-0.4、t-0.2、t 三幀切兩段，d 左正。"""

    def setUp(self):
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.pair = VehiclePair("rear", "front")
        self.sampling = SamplingConfig(0.02, 25, 5)

    def episode(self, name: str = "episode", frames: int = 60, speed: float = 3.0) -> RecordedEpisode:
        # 使用者範例：後車前段 -0.02→+0.10，後段不動；前車前段不動，後段 -0.07→+0.03。
        return RecordedEpisode(name, tuple(
            FrameSnapshot(name, i, i * 0.02, {
                "rear": observation(self.track, "rear", 20 + i * 0.06, ramp(i, 20, 30, -0.02, 0.10), speed),
                "front": observation(self.track, "front", 23 + i * 0.06, ramp(i, 30, 40, -0.07, 0.03), speed),
            }) for i in range(frames)))

    def test_history_terms_match_user_example_and_ignore_future(self):
        episode = self.episode()
        x = history_features(episode, 40, self.pair, self.track, 10)
        np.testing.assert_allclose(x[8:], (0.12, 0.0, 0.0, 0.10), atol=1e-9)
        self.assertAlmostEqual(x[0], 3.0)
        np.testing.assert_allclose((x[1], x[4]), (0.10, 0.03), atol=1e-9)
        np.testing.assert_allclose((x[2], x[5], x[7]), (0.0, 0.0, 0.0), atol=1e-9)
        np.testing.assert_allclose((x[3], x[6]), (3.0, 3.0))
        shortened = replace(episode, frames=episode.frames[:41])
        self.assertEqual(x, history_features(shortened, 40, self.pair, self.track, 10))
        with self.assertRaises(ValueError):
            history_features(episode, 19, self.pair, self.track, 10)

    def test_heading_error_wraps_across_pi(self):
        # 第三段 (100,100)→(0,100) 的切線角為 π；車頭 -π+0.1 只差 +0.1 rad。
        vehicle = observation(self.track, "rear", 250.0, 0.0, 3.0)
        turned = replace(vehicle, state=replace(vehicle.state, yaw=-math.pi + 0.1))
        self.assertAlmostEqual(heading_error(turned, self.track), 0.1)

    def test_history_dataset_skips_short_history_and_collisions_in_history_window(self):
        dataset = build_interaction_dataset((self.episode(),), self.track, self.pair, self.sampling,
                                            feature_set="history")
        self.assertEqual(dataset.schema.feature_names, HISTORY_FEATURES)
        self.assertEqual([sample.step_index for sample in dataset.samples], [20, 25, 30])
        self.assertEqual(dataset.skipped["insufficient_history"], 4)
        self.assertEqual(dataset.skipped["insufficient_future"], 5)
        frames = list(self.episode().frames)
        frames[8] = replace(frames[8], collision_pairs=frozenset({("rear", "front")}))
        rejected = build_interaction_dataset((replace(self.episode(), frames=tuple(frames)),),
                                             self.track, self.pair, self.sampling, feature_set="history")
        self.assertEqual([sample.step_index for sample in rejected.samples], [30])
        self.assertEqual(rejected.skipped["collision"], 2)
        with self.assertRaises(ValueError):
            build_interaction_dataset((self.episode(),), self.track, self.pair, self.sampling,
                                      feature_set="history", history_segment_seconds=0.03)

    def test_feature_subset_trains_only_named_columns(self):
        episodes = tuple(self.episode(f"ep-{n}", speed=3.0 + n) for n in range(5))
        targets = {(episode.episode_id, frame.step_index): 0.3 * (n % 2)
                   for n, episode in enumerate(episodes) for frame in episode.frames}
        dataset = build_interaction_dataset(episodes, self.track, self.pair, self.sampling, targets,
                                            feature_set="history")
        self.assertEqual(dataset.schema.feature_names, HISTORY_ACTION_FEATURES)
        subset = ("gap", "rear_d", "front_d", "front_curvature", "rear_dd_late")
        columns = [HISTORY_ACTION_FEATURES.index(name) for name in subset]
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            data, output = Path(temporary) / "data", Path(temporary) / "model"
            export_dataset(dataset, episodes, self.track, data, {"history_segment_seconds": 0.2})
            (data / "actions.csv").write_text("episode_id,step_index,time_s,mode,target_d\n" +
                "".join(f"ep-{n},20,0.4,probe_left,0.3\n" for n in range(5)), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                train(data, output, 42, 0, subset)
            with (output / "model.pkl").open("rb") as stream:
                bundle = pickle.load(stream)
            self.assertEqual(bundle["feature_names"], subset)
            self.assertEqual(bundle["model_kind"], "interaction_history")
            self.assertEqual((bundle["history_seconds"], bundle["history_segment_seconds"]), (0.4, 0.2))
            indices = split_groups(np.array([sample.episode_id for sample in dataset.samples]), 42)["train"]
            np.testing.assert_allclose(bundle["pipeline"].named_steps["scaler"].mean_,
                                       dataset.X[indices][:, columns].mean(axis=0))
            report = (output / "report.json").read_text(encoding="utf-8")
            self.assertIn("rear_holding", report)
            self.assertIn("d_target", report)  # dataset_features 仍列出全部欄位

    def test_select_features_rejects_unknown_or_duplicate_names(self):
        self.assertEqual(select_features(("a", "b"), None), (("a", "b"), [0, 1]))
        self.assertEqual(select_features(("a", "b"), ["b"]), (("b",), [1]))
        for subset in ([], ["a", "a"], ["c"]):
            with self.subTest(subset=subset), self.assertRaises(ValueError):
                select_features(("a", "b"), subset)

    def test_random_probe_config_alternates_sides_and_is_repeatable(self):
        base = ProbeFollowingConfig(target_gap=2.5)
        config = random_probe_config(base, np.random.default_rng([42, 1]))
        again = random_probe_config(base, np.random.default_rng([42, 1]))
        self.assertEqual(config, again)
        sides, centers = config.sequence[0::2], config.sequence[1::2]
        self.assertEqual(set(centers), {0.0})
        self.assertTrue(all(abs(value) in PROBE_AMPLITUDES for value in sides))
        signs = np.sign(sides)
        self.assertTrue(np.all(signs[1:] == -signs[:-1]))
        self.assertTrue(PROBE_HOLD_RANGE[0] <= config.hold_seconds <= PROBE_HOLD_RANGE[1])
        self.assertEqual(config.target_gap, base.target_gap)

    def test_explicit_default_amplitudes_keep_original_draws(self):
        base = ProbeFollowingConfig(target_gap=2.5)
        self.assertEqual(random_probe_config(base, np.random.default_rng([42, 1])),
                         random_probe_config(base, np.random.default_rng([42, 1]), PROBE_AMPLITUDES, None))

    def test_fixed_amplitude_and_first_side(self):
        base = ProbeFollowingConfig(target_gap=2.5)
        config = random_probe_config(base, np.random.default_rng(1), (0.6,), -1.0)
        self.assertEqual(config.sequence, (-0.6, 0.0, 0.6, 0.0, -0.6, 0.0, 0.6, 0.0))
        for kwargs in ({"amplitudes": ()}, {"amplitudes": (0.0,)}, {"first_side": 0.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                random_probe_config(base, np.random.default_rng(1), **kwargs)

    def test_stratified_design_balances_side_and_start_within_each_stratum(self):
        cells = {}
        for number in range(16):
            stratum, side, moving = stratified_design(number, 4)
            cells.setdefault(stratum, set()).add((side, moving))
        expected = {(1.0, False), (-1.0, False), (1.0, True), (-1.0, True)}
        self.assertEqual(cells, {stratum: expected for stratum in range(4)})
        self.assertEqual(speed_strata([3, 4, 5]), ((3.0, 4.0), (4.0, 5.0)))
        for edges in ([3], [4, 3], [0, 1], [3, float("nan")]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                speed_strata(edges)


if __name__ == "__main__":
    unittest.main()
