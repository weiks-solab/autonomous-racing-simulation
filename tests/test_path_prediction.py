"""以行駛距離求終點，包含橫移、內外側路徑與起終點跨越。"""

from dataclasses import replace
import unittest

import numpy as np

from core.track import Track
from core.track_geometry import track_length, world_to_track
from strategy.offset_path import LateralManeuver
from strategy.prediction import predict_path_trajectory
from tests.test_defending_strategy import observation


class PathPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))), width=2.0)

    def test_straight_fixed_offset_preserves_speed_and_initial_pose(self) -> None:
        vehicle = observation("car", 20.0, 0.6, 3.0, self.track)
        times = (0.0, 0.2, 0.5, 1.0)
        prediction = predict_path_trajectory(vehicle, self.track, times)
        self.assertIs(prediction.states[0], vehicle.state)
        np.testing.assert_allclose(prediction.progress, np.array(times) * 3.0)
        np.testing.assert_allclose(prediction.offsets, 0.6)
        for time, state in zip(times, prediction.states):
            self.assertAlmostEqual(state.x, 20.0 + 3.0 * time)
            self.assertAlmostEqual(state.y, 0.6)
            self.assertEqual(state.velocity, 3.0)

    def test_lateral_transition_uses_distance_to_endpoint_instead_of_average_ratio(self) -> None:
        vehicle = observation("car", 20.0, 0.0, 2.0, self.track,
                              LateralManeuver(20.0, 0.0, 0.6, 1.0))
        prediction = predict_path_trajectory(vehicle, self.track, (0.0, 0.5, 1.0), 0.02)
        # 橫移在前 1 m 完成，其後為直線；扣掉橫移額外走的距離即可求終點。
        x = np.linspace(0.0, 1.0, 10001)
        y = 0.6 * (3.0 * x**2 - 2.0 * x**3)
        transition_length = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        expected_progress = 2.0 - (transition_length - 1.0)
        self.assertAlmostEqual(prediction.progress[-1], expected_progress, delta=0.001)
        self.assertAlmostEqual(prediction.states[-1].x, 20.0 + expected_progress, delta=0.001)
        old_estimate = 2.0**2 / (transition_length + 1.0)
        self.assertGreater(abs(prediction.progress[-1] - old_estimate), 0.01)

    def test_inner_and_outer_paths_have_different_end_s_at_equal_travel(self) -> None:
        track = Track.create_oval(10.0, 10.0, 1000, width=3.0)
        for d in (-1.0, 0.0, 1.0):
            with self.subTest(d=d):
                vehicle = observation("car", 2.0, d, 4.0, track)
                prediction = predict_path_trajectory(vehicle, track, (0.0, 3.0), 0.05)
                expected = 12.0 / (1.0 - d / 10.0)
                self.assertAlmostEqual(prediction.progress[-1], expected, delta=0.03)

    def test_short_inner_path_is_extended_until_travel_is_covered(self) -> None:
        track = Track.create_oval(3.0, 3.0, 1000, width=6.0)
        vehicle = observation("car", 1.0, 2.0, 1.0, track)
        # 與圓形賽道的線段等距取樣，讓解析圓的預期值不受分段法線跳變影響。
        spacing = track_length(track) / 1000
        prediction = predict_path_trajectory(vehicle, track, (0.0, 1.0), spacing)
        self.assertAlmostEqual(prediction.progress[-1], 3.0, delta=0.03)
        self.assertAlmostEqual(world_to_track(prediction.states[-1], track).d, 2.0, delta=0.01)

    def test_progress_does_not_wrap_at_finish_or_after_a_lap(self) -> None:
        track = Track.create_oval(3.0, 3.0, 500, width=2.0)
        length = track_length(track)
        vehicle = observation("car", -0.5, 0.0, 2.0, track)
        prediction = predict_path_trajectory(vehicle, track, (0.0, 1.0, length), 0.02)
        np.testing.assert_allclose(prediction.progress, (0.0, 2.0, 2.0 * length), atol=0.01)

    def test_stationary_vehicle_does_not_move_toward_its_lateral_target(self) -> None:
        vehicle = observation("car", 20.0, 0.2, 0.0, self.track,
                              LateralManeuver(19.0, 0.0, 0.6, 3.0))
        prediction = predict_path_trajectory(vehicle, self.track, (0.0, 0.5, 1.0))
        self.assertTrue(all(state is vehicle.state for state in prediction.states))
        np.testing.assert_array_equal(prediction.progress, 0.0)
        np.testing.assert_array_equal(prediction.offsets, 0.2)

    def test_cost_samples_are_unchanged_by_a_longer_collision_horizon(self) -> None:
        vehicle = observation("car", 20.0, 0.0, 3.0, self.track,
                              LateralManeuver(20.0, 0.0, 0.6, 1.0))
        short = predict_path_trajectory(vehicle, self.track, (0.0, 0.5, 1.0))
        long = predict_path_trajectory(vehicle, self.track, (0.0, 0.5, 1.0, 2.0))
        np.testing.assert_allclose(short.progress, long.progress[:3])
        np.testing.assert_allclose(short.offsets, long.offsets[:3])
        self.assertEqual(short.states, long.states[:3])

    def test_invalid_times_spacing_and_open_track_are_rejected(self) -> None:
        vehicle = observation("car", 20.0, 0.0, 3.0, self.track)
        for times in ((), (1.0,), (0.0, -1.0), (0.0, float("nan")), ((0.0, 1.0),)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                predict_path_trajectory(vehicle, self.track, times)
        for spacing in (0.0, -1.0, float("nan")):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                predict_path_trajectory(vehicle, self.track, (0.0, 1.0), spacing)
        with self.assertRaisesRegex(ValueError, "closed"):
            predict_path_trajectory(vehicle, replace(self.track, closed=False), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
