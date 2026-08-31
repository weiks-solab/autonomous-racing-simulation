import unittest

import numpy as np

from core.track import Track
from core.track_geometry import (
    TrackPosition,
    track_length,
    track_to_world,
    world_to_track,
)
from core.vehicle_state import VehicleState
from strategy.offset_path import (
    LateralManeuver,
    OffsetPathConfig,
    generate_offset_track,
    lateral_offset_at_distance,
    smoothstep,
    wrapped_forward_distance,
)


class OffsetPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(
            np.array(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))),
            width=4.0,
        )

    def test_smoothstep_values_and_zero_endpoint_slopes(self) -> None:
        epsilon = 1e-6
        self.assertEqual(smoothstep(0.0), 0.0)
        self.assertEqual(smoothstep(1.0), 1.0)
        start_slope = (smoothstep(epsilon) - smoothstep(0.0)) / epsilon
        end_slope = (smoothstep(1.0) - smoothstep(1.0 - epsilon)) / epsilon
        self.assertAlmostEqual(start_slope, 0.0, places=5)
        self.assertAlmostEqual(end_slope, 0.0, places=5)

    def test_lateral_profile_reaches_and_holds_target(self) -> None:
        self.assertEqual(lateral_offset_at_distance(0.0, 0.2, 1.0, 4.0), 0.2)
        self.assertEqual(lateral_offset_at_distance(4.0, 0.2, 1.0, 4.0), 1.0)
        self.assertEqual(lateral_offset_at_distance(8.0, 0.2, 1.0, 4.0), 1.0)

    def test_track_to_world_matches_left_positive_convention(self) -> None:
        np.testing.assert_allclose(track_to_world(2.0, 1.0, self.track), (2.0, 1.0))
        np.testing.assert_allclose(track_to_world(2.0, -1.0, self.track), (2.0, -1.0))

    def test_track_to_world_wraps_closed_s(self) -> None:
        np.testing.assert_allclose(
            track_to_world(track_length(self.track) + 2.0, 0.0, self.track),
            track_to_world(2.0, 0.0, self.track),
        )

    def test_generated_path_is_open_and_wraps_base_s(self) -> None:
        start = TrackPosition(38.0, 0.0, 3)
        temporary = generate_offset_track(
            self.track,
            start,
            LateralManeuver(38.0, 0.0, 1.0, 4.0),
            config=OffsetPathConfig(sample_spacing=0.5, path_length=8.0),
        )
        self.assertFalse(temporary.closed)
        np.testing.assert_allclose(temporary.centerline[0], track_to_world(38.0, 0.0, self.track))
        final_position = world_to_track(
            VehicleState(
                float(temporary.centerline[-1, 0]),
                float(temporary.centerline[-1, 1]),
                0.0,
                0.0,
            ),
            self.track,
        )
        self.assertAlmostEqual(final_position.s, 6.0, places=6)
        self.assertAlmostEqual(final_position.d, 1.0, places=6)

    def test_generated_path_stays_within_configured_lateral_target(self) -> None:
        temporary = generate_offset_track(
            self.track,
            TrackPosition(2.0, 0.0, 0),
            LateralManeuver(2.0, 0.0, 1.0, 4.0),
            config=OffsetPathConfig(sample_spacing=0.5, path_length=8.0),
        )
        offsets = [
            world_to_track(
                VehicleState(float(point[0]), float(point[1]), 0.0, 0.0),
                self.track,
            ).d
            for point in temporary.centerline
        ]
        self.assertTrue(all(-1e-9 <= offset <= 1.0 + 1e-9 for offset in offsets))

    def test_regenerated_path_keeps_d_at_same_absolute_s(self) -> None:
        maneuver = LateralManeuver(2.0, 0.0, 1.0, 6.0)
        config = OffsetPathConfig(sample_spacing=0.5, path_length=8.0)
        earlier = generate_offset_track(
            self.track, TrackPosition(3.0, 0.1, 0), maneuver, config
        )
        later = generate_offset_track(
            self.track, TrackPosition(4.0, 0.3, 0), maneuver, config
        )

        # earlier[6] and later[4] both represent absolute s=6.0.
        np.testing.assert_allclose(earlier.centerline[6], later.centerline[4])
        expected_d = lateral_offset_at_distance(4.0, 0.0, 1.0, 6.0)
        self.assertAlmostEqual(float(earlier.centerline[6, 1]), expected_d)

    def test_progress_is_measured_from_original_origin(self) -> None:
        self.assertAlmostEqual(wrapped_forward_distance(2.0, 3.5, 40.0), 1.5)
        self.assertAlmostEqual(wrapped_forward_distance(2.0, 5.0, 40.0), 3.0)

    def test_generated_path_holds_exact_target_after_transition(self) -> None:
        maneuver = LateralManeuver(2.0, 0.2, 1.0, 4.0)
        temporary = generate_offset_track(
            self.track,
            TrackPosition(7.0, 0.7, 0),
            maneuver,
            OffsetPathConfig(sample_spacing=0.5, path_length=8.0),
        )
        offsets = [
            world_to_track(
                VehicleState(float(point[0]), float(point[1]), 0.0, 0.0),
                self.track,
            ).d
            for point in temporary.centerline[:5]
        ]
        for offset in offsets:
            self.assertAlmostEqual(offset, 1.0)

    def test_overtake_progress_wraps_across_start_finish(self) -> None:
        maneuver = LateralManeuver(38.0, 0.0, 1.0, 4.0)
        temporary = generate_offset_track(
            self.track,
            TrackPosition(1.0, 0.5, 0),
            maneuver,
            OffsetPathConfig(sample_spacing=0.5, path_length=8.0),
        )
        expected = lateral_offset_at_distance(3.0, 0.0, 1.0, 4.0)
        actual = world_to_track(
            VehicleState(*temporary.centerline[0], 0.0, 0.0), self.track
        ).d
        self.assertAlmostEqual(actual, expected)

    def test_return_progress_wraps_across_start_finish(self) -> None:
        maneuver = LateralManeuver(39.0, -0.8, 0.0, 4.0)
        temporary = generate_offset_track(
            self.track,
            TrackPosition(2.0, -0.4, 0),
            maneuver,
            OffsetPathConfig(sample_spacing=0.5, path_length=8.0),
        )
        expected = lateral_offset_at_distance(3.0, -0.8, 0.0, 4.0)
        actual = world_to_track(
            VehicleState(*temporary.centerline[0], 0.0, 0.0), self.track
        ).d
        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
