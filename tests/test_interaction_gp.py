"""驗證特徵不看未來、標籤對齊與整場資料切分。"""

from dataclasses import replace
from contextlib import redirect_stdout
import io
import unittest
import tempfile
import pickle
from pathlib import Path
import numpy as np

from core.track import Track
from interaction_gp import split_groups, train
from training_data.io import export_dataset
from tests.test_bicycle_prediction import observation
from training_data.interaction import build_interaction_dataset, curvature, features
from training_data.schema import FrameSnapshot, RecordedEpisode, SamplingConfig, VehiclePair


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


if __name__ == "__main__":
    unittest.main()
