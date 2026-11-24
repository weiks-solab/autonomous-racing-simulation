import math
import unittest

import numpy as np

from core.track import Track
from core.track_geometry import track_tangent_angle


class TrackTests(unittest.TestCase):
    def test_create_oval_is_true_ellipse(self) -> None:
        track = Track.create_oval(4.0, 2.0, 4)
        np.testing.assert_allclose(
            track.centerline,
            np.array(((4.0, 0.0), (0.0, 2.0), (-4.0, 0.0), (0.0, -2.0))),
            atol=1e-12,
        )

    def test_forward_search_wraps_across_closed_path_end(self) -> None:
        track = Track(np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))))
        self.assertEqual(track.forward_point_index(3, 0.5), 0)
        self.assertEqual(track.forward_point_index(3, 1.5), 1)

    def test_tangent_angle_follows_segments_and_wraps(self) -> None:
        track = Track(np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))))
        self.assertAlmostEqual(track_tangent_angle(0.5, track), 0.0)
        self.assertAlmostEqual(track_tangent_angle(1.5, track), math.pi / 2)
        self.assertAlmostEqual(abs(track_tangent_angle(2.5, track)), math.pi)
        self.assertAlmostEqual(track_tangent_angle(3.5, track), -math.pi / 2)
        self.assertAlmostEqual(track_tangent_angle(4.5, track), 0.0)
        with self.assertRaises(ValueError):
            track_tangent_angle(float("nan"), track)
        with self.assertRaises(ValueError):
            track_tangent_angle(5.0, Track(track.centerline, closed=False))

    def test_invalid_ellipse_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Track.create_oval(0.0, 2.0, 100)
        with self.assertRaises(ValueError):
            Track.create_oval(4.0, 2.0, 2)


if __name__ == "__main__":
    unittest.main()
