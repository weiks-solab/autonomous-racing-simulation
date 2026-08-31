import math
import unittest

import numpy as np

from core.track import Track
from core.track_geometry import (
    TrackBoundaryStatus,
    TrackPosition,
    classify_reference_point_boundary,
    relative_track_position,
    signed_longitudinal_gap,
    track_boundaries,
    track_length,
    track_to_world,
    world_to_track,
)
from core.vehicle_state import VehicleState


class TrackGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.square = Track(
            np.array(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))),
            width=4.0,
        )

    def test_world_to_track_on_known_ellipse_points(self) -> None:
        ellipse = Track.create_oval(4.0, 2.0, 400)
        start = world_to_track(VehicleState(4.0, 0.0, 0.0, 0.0), ellipse)
        opposite = world_to_track(VehicleState(-4.0, 0.0, 0.0, 0.0), ellipse)
        self.assertAlmostEqual(start.s, 0.0, places=10)
        self.assertAlmostEqual(start.d, 0.0, places=10)
        self.assertAlmostEqual(opposite.s, track_length(ellipse) / 2.0, places=10)
        self.assertAlmostEqual(opposite.d, 0.0, places=10)

    def test_signed_lateral_offset_uses_left_positive(self) -> None:
        left = world_to_track(VehicleState(2.0, 1.0, 0.0, 0.0), self.square)
        right = world_to_track(VehicleState(2.0, -1.0, 0.0, 0.0), self.square)
        self.assertAlmostEqual(left.s, 2.0)
        self.assertAlmostEqual(left.d, 1.0)
        self.assertEqual(left.centerline_index, 0)
        self.assertAlmostEqual(right.d, -1.0)

    def test_s_wraps_near_start_finish(self) -> None:
        before = world_to_track(VehicleState(0.0, 0.5, 0.0, 0.0), self.square)
        after = world_to_track(VehicleState(0.5, 0.0, 0.0, 0.0), self.square)
        self.assertAlmostEqual(before.s, track_length(self.square) - 0.5)
        self.assertAlmostEqual(after.s, 0.5)
        self.assertAlmostEqual(
            signed_longitudinal_gap(before, after, track_length(self.square)),
            1.0,
        )

    def test_relative_position_reports_ahead_behind_and_lateral_gap(self) -> None:
        reference = TrackPosition(s=39.5, d=-0.5, centerline_index=3)
        other = TrackPosition(s=0.5, d=0.75, centerline_index=0)
        relative = relative_track_position(reference, other, 40.0)
        self.assertAlmostEqual(relative.longitudinal_gap, 1.0)
        self.assertAlmostEqual(relative.lateral_gap, 1.25)
        self.assertTrue(relative.is_ahead)
        self.assertFalse(relative.is_behind)

        reverse = relative_track_position(other, reference, 40.0)
        self.assertAlmostEqual(reverse.longitudinal_gap, -1.0)
        self.assertTrue(reverse.is_behind)

    def test_half_lap_uses_negative_interval_endpoint(self) -> None:
        reference = TrackPosition(0.0, 0.0, 0)
        opposite = TrackPosition(20.0, 0.0, 2)
        self.assertEqual(signed_longitudinal_gap(reference, opposite, 40.0), -20.0)

    def test_reference_point_boundary_classification(self) -> None:
        self.assertIs(
            classify_reference_point_boundary(TrackPosition(0.0, 2.0, 0), self.square),
            TrackBoundaryStatus.INSIDE,
        )
        self.assertIs(
            classify_reference_point_boundary(TrackPosition(0.0, 2.1, 0), self.square),
            TrackBoundaryStatus.OUTSIDE_LEFT,
        )
        self.assertIs(
            classify_reference_point_boundary(TrackPosition(0.0, -2.1, 0), self.square),
            TrackBoundaryStatus.OUTSIDE_RIGHT,
        )

    def test_boundaries_have_expected_shape_and_offset(self) -> None:
        left, right = track_boundaries(self.square)
        self.assertEqual(left.shape, self.square.centerline.shape)
        self.assertEqual(right.shape, self.square.centerline.shape)
        self.assertAlmostEqual(
            float(np.linalg.norm(left[0] - self.square.centerline[0])),
            self.square.width / 2.0,
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(right[0] - self.square.centerline[0])),
            self.square.width / 2.0,
        )

    def test_invalid_track_width_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Track.create_oval(4.0, 2.0, 100, width=0.0)
        with self.assertRaises(ValueError):
            Track.create_oval(4.0, 2.0, 100, width=math.inf)

    def test_track_to_world_round_trip(self) -> None:
        point = track_to_world(3.0, 1.25, self.square)
        position = world_to_track(
            VehicleState(float(point[0]), float(point[1]), 0.0, 0.0),
            self.square,
        )
        self.assertAlmostEqual(position.s, 3.0)
        self.assertAlmostEqual(position.d, 1.25)


if __name__ == "__main__":
    unittest.main()
