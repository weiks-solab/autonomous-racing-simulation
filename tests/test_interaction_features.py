"""共用互動特徵與歷史緩衝區：與訓練資料的計算一致，查詢不讀未來。"""

import math
import unittest

import numpy as np

from core.track import Track
from strategy.interaction_features import INTERACTION_FEATURES, LateralHistory, history_state_features
from tests.test_bicycle_prediction import observation
from tests.test_interaction_gp import ramp
from training_data.interaction import HISTORY_ACTION_FEATURES, history_features
from training_data.schema import FrameSnapshot, RecordedEpisode, VehiclePair


class InteractionFeatureTests(unittest.TestCase):
    def setUp(self):
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))

    def test_feature_order_matches_training_data(self):
        self.assertEqual(INTERACTION_FEATURES, HISTORY_ACTION_FEATURES)

    def test_shared_features_equal_training_features(self):
        episode = RecordedEpisode("ep", tuple(
            FrameSnapshot("ep", i, i * 0.02, {
                "rear": observation(self.track, "rear", 20 + i * 0.06, ramp(i, 20, 30, -0.02, 0.10), 3.0),
                "front": observation(self.track, "front", 23 + i * 0.06, ramp(i, 30, 40, -0.07, 0.03), 3.0),
            }) for i in range(41)))
        frames = episode.frames
        shared = history_state_features(
            frames[40].observations["rear"], frames[40].observations["front"],
            (frames[20].observations["rear"].track_position.d, frames[30].observations["rear"].track_position.d),
            (frames[20].observations["front"].track_position.d, frames[30].observations["front"].track_position.d),
            self.track)
        self.assertEqual(shared, history_features(episode, 40, VehiclePair("rear", "front"), self.track, 10))


class LateralHistoryTests(unittest.TestCase):
    def setUp(self):
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))

    def record(self, history, time, d):
        history.record(time, (observation(self.track, "car", 10.0, d, 3.0),))

    def test_exact_interpolated_and_missing_lookups(self):
        history = LateralHistory(0.5)
        for step in range(11):
            self.record(history, step * 0.02, 0.01 * step)
        self.assertAlmostEqual(history.d_at("car", 0.04), 0.02)
        self.assertAlmostEqual(history.d_at("car", 0.05), 0.025)
        self.assertIsNone(history.d_at("car", 0.3))     # 晚於最新紀錄
        self.assertIsNone(history.d_at("car", -0.1))    # 早於最舊紀錄
        self.assertIsNone(history.d_at("other", 0.0))

    def test_old_records_are_pruned_and_time_cannot_go_back(self):
        history = LateralHistory(0.4)
        for step in range(51):
            self.record(history, step * 0.02, 0.0)
        self.assertIsNotNone(history.d_at("car", 1.0 - 0.4))
        self.assertIsNone(history.d_at("car", 0.5))
        self.record(history, 1.0, 0.3)   # 同一時刻以最新值為準
        self.assertAlmostEqual(history.d_at("car", 1.0), 0.3)
        with self.assertRaises(ValueError):
            self.record(history, 0.9, 0.0)
        for window in (0.0, math.inf):
            with self.subTest(window=window), self.assertRaises(ValueError):
                LateralHistory(window)


if __name__ == "__main__":
    unittest.main()
