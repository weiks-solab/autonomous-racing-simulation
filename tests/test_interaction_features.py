"""共用互動特徵與歷史緩衝區：與訓練資料的計算一致，查詢不讀未來。"""

from dataclasses import replace
import math
import unittest

import numpy as np

from core.track import Track
from strategy.interaction_features import (
    INTERACTION_FEATURES, PAPER_FEATURES, PAPER_TARGET_UNITS, PAPER_TARGETS, LateralHistory,
    centerline_curvature, front_state_delta, heading_error, history_state_features, paper_feature_rows,
    paper_state_features,
)
from tests.test_bicycle_prediction import observation
from tests.test_interaction_gp import ramp
from training_data.interaction import HISTORY_ACTION_FEATURES, build_paper_dataset, history_features
from training_data.schema import FrameSnapshot, RecordedEpisode, SamplingConfig, VehiclePair


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


class PaperFeatureTests(unittest.TestCase):
    """論文格式：z_k 10 欄＋v2 歷史 4 欄，答案為前車一步的 [Δs, Δd, Δe_ψ, Δv]。"""

    def setUp(self):
        # 逆時針方形，周長 400 m；只有轉角附近曲率不為 0（左彎為正）。
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.pair = VehiclePair("rear", "front")

    def episode(self, frames: int = 60) -> RecordedEpisode:
        # 同 v2 歷史測試的例子：後車前段 -0.02→+0.10，前車後段 -0.07→+0.03；兩車每步前進 0.06 m。
        return RecordedEpisode("ep", tuple(
            FrameSnapshot("ep", i, i * 0.02, {
                "rear": observation(self.track, "rear", 20 + i * 0.06, ramp(i, 20, 30, -0.02, 0.10), 3.0),
                "front": observation(self.track, "front", 23 + i * 0.06, ramp(i, 30, 40, -0.07, 0.03), 3.0),
            }) for i in range(frames)))

    def test_order_values_and_history_match_v2(self):
        frames = self.episode().frames
        rear, front = frames[40].observations["rear"], frames[40].observations["front"]
        rear_past = tuple(frames[k].observations["rear"].track_position.d for k in (20, 30))
        front_past = tuple(frames[k].observations["front"].track_position.d for k in (20, 30))
        x = paper_state_features(rear, front, rear_past, front_past, self.track)
        self.assertEqual(len(x), len(PAPER_FEATURES))
        self.assertAlmostEqual(x[0], 3.0)                                   # gap＝前車 s − 後車 s
        np.testing.assert_allclose(x[1:7], (0.03, 0.0, 3.0, 0.10, 0.0, 3.0), atol=1e-9)
        np.testing.assert_allclose(x[7:10], (0.0, 0.0, 0.0), atol=1e-9)    # 直線段
        v2 = history_state_features(rear, front, rear_past, front_past, self.track)
        self.assertEqual(x[10:], v2[8:])                                    # 歷史 4 欄與 v2 完全相同
        np.testing.assert_allclose(x[10:], (0.12, 0.0, 0.0, 0.10), atol=1e-9)

    def test_curvature_lookahead_uses_zeta(self):
        rear = observation(self.track, "rear", 95.0, 0.0, 3.0)
        front = observation(self.track, "front", 98.0, 0.0, 3.0)
        for zeta, corner in ((1.0, 9), (2.0, 8)):
            with self.subTest(zeta=zeta):
                x = paper_state_features(rear, front, (0.0, 0.0), (0.0, 0.0), self.track, zeta)
                expected = [centerline_curvature(self.track, 98.0 + k * zeta) for k in range(3)]
                np.testing.assert_allclose(x[7:10], expected)
                self.assertGreater(x[corner], 0.0)                          # 轉角落在 s=100
                self.assertEqual(sum(value != 0.0 for value in x[7:10]), 1)
        for zeta in (0.0, -1.0, math.nan):
            with self.subTest(zeta=zeta), self.assertRaises(ValueError):
                paper_state_features(rear, front, (0.0, 0.0), (0.0, 0.0), self.track, zeta)

    def test_front_delta_wraps_lap_and_heading(self):
        now = observation(self.track, "front", 399.8, 0.1, 3.0)
        later = observation(self.track, "front", 0.3, 0.2, 3.4)
        np.testing.assert_allclose(front_state_delta(now, later, self.track), (0.5, 0.1, 0.0, 0.4), atol=1e-9)
        # 第一段切線角 0：航向誤差 π−0.05 → −π+0.05，實際只轉 +0.1 rad。
        base = observation(self.track, "front", 50.0, 0.0, 3.0)
        turned = replace(base, state=replace(base.state, yaw=math.pi - 0.05))
        crossed = replace(base, state=replace(base.state, yaw=-math.pi + 0.05))
        self.assertAlmostEqual(front_state_delta(turned, crossed, self.track)[2], 0.1)

    def test_dataset_labels_skips_and_schema(self):
        sampling = SamplingConfig(0.02, 5, 5)   # 每 0.1 s 取樣，答案為 0.1 s 後
        dataset = build_paper_dataset((self.episode(),), self.track, self.pair, sampling)
        self.assertEqual(dataset.schema.feature_names, PAPER_FEATURES)
        self.assertEqual(dataset.schema.target_names, PAPER_TARGETS)
        self.assertEqual(tuple(spec.unit for spec in dataset.target_specs), PAPER_TARGET_UNITS)
        self.assertEqual([sample.step_index for sample in dataset.samples], [20, 25, 30, 35, 40, 45, 50])
        self.assertEqual(dataset.skipped["insufficient_history"], 4)
        self.assertEqual(dataset.skipped["insufficient_future"], 1)
        by_step = {sample.step_index: sample for sample in dataset.samples}
        # step 30→35：前車 d 由 -0.07 線性到 -0.02，s 前進 5×0.06 m。
        np.testing.assert_allclose(by_step[30].targets, (0.3, 0.05, 0.0, 0.0), atol=1e-9)
        frames = self.episode().frames
        expected = paper_state_features(
            frames[40].observations["rear"], frames[40].observations["front"],
            tuple(frames[k].observations["rear"].track_position.d for k in (20, 30)),
            tuple(frames[k].observations["front"].track_position.d for k in (20, 30)), self.track)
        self.assertEqual(by_step[40].features, expected)
        # 碰撞在第 22 幀：歷史窗（前 0.4 s）到標籤窗涵蓋它的樣本都略過。
        collided = list(frames)
        collided[22] = replace(collided[22], collision_pairs=frozenset({("rear", "front")}))
        rejected = build_paper_dataset((replace(self.episode(), frames=tuple(collided)),),
                                       self.track, self.pair, sampling)
        self.assertEqual([sample.step_index for sample in rejected.samples], [45, 50])
        self.assertEqual(rejected.skipped["collision"], 5)

    def test_batch_rows_equal_single_features_for_the_same_states(self):
        # 線上推演一次算 Q 列；同一組狀態必須與訓練用的 paper_state_features 逐欄相同。
        rear = observation(self.track, "rear", 1.0, 0.2, 4.0)
        rear = replace(rear, state=replace(rear.state, yaw=rear.state.yaw - 0.05))
        rear_past = (0.05, 0.12)
        fronts = []
        for s, d, turn, speed in ((3.0, -0.3, 0.0, 3.5), (98.6, 0.4, 0.08, 2.0),
                                  (399.7, 0.1, -0.12, 5.0), (250.0, -0.5, 0.02, 0.0)):
            front = observation(self.track, "front", s, d, speed)   # 第 3 列在後車之後（跨圈，gap 為負）
            fronts.append(replace(front, state=replace(front.state, yaw=front.state.yaw + turn)))
        past = np.array(((0.0, -0.1), (0.3, 0.35), (0.2, 0.15), (-0.4, -0.45)))
        rows = paper_feature_rows(
            rear, rear_past, [front.track_position.s for front in fronts],
            [front.track_position.d for front in fronts], [heading_error(front, self.track) for front in fronts],
            [front.state.velocity for front in fronts], past, self.track)
        self.assertEqual(rows.shape, (4, len(PAPER_FEATURES)))
        for index, front in enumerate(fronts):
            with self.subTest(sample=index):
                expected = paper_state_features(rear, front, rear_past, tuple(past[index]), self.track)
                np.testing.assert_allclose(rows[index], expected, rtol=0.0, atol=1e-12)
        self.assertLess(rows[2, 0], 0.0)
        self.assertGreater(rows[1, 8], 0.0)   # s+ζ 落在轉角
        with self.assertRaises(ValueError):
            paper_feature_rows(rear, rear_past, [1.0, 2.0], [0.0], [0.0], [1.0], np.zeros((1, 2)), self.track)
        with self.assertRaises(ValueError):
            paper_feature_rows(rear, rear_past, [1.0], [0.0], [0.0], [1.0], np.zeros((1, 2)), self.track, 0.0)


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
