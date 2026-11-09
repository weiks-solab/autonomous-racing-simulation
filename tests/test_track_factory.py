"""Check the actual shape and coordinate queries of interchangeable tracks."""

from dataclasses import replace
import math
import unittest

import numpy as np

from core.track import Track
from core.track_factory import TrackConfig, create_track
from core.track_geometry import track_length, track_to_world, world_to_track
from core.vehicle_state import VehicleState


class TrackFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TrackConfig()
        self.track = create_track(self.config)

    def test_four_straights_and_four_circular_corners_form_one_closed_loop(self) -> None:
        points = self.track.centerline
        self.assertTrue(self.track.closed)
        self.assertEqual(points.shape, (500, 2))
        np.testing.assert_allclose(points.min(axis=0), (-10.0, -6.0))
        np.testing.assert_allclose(points.max(axis=0), (10.0, 6.0))
        np.testing.assert_allclose(points[0], (-7.0, -6.0))
        self.assertFalse(np.array_equal(points[0], points[-1]))

        # 每點到內側 14 x 6 矩形的距離為 3 m；也涵蓋四個圓弧。
        corner_offsets = np.maximum(np.abs(points) - (7.0, 3.0), 0.0)
        np.testing.assert_allclose(np.linalg.norm(corner_offsets, axis=1), 3.0, atol=1e-12)
        for axis, value in ((0, -10.0), (0, 10.0), (1, -6.0), (1, 6.0)):
            self.assertGreater(np.count_nonzero(np.isclose(points[:, axis], value)), 20)

    def test_uniform_sampling_closes_without_a_heading_jump(self) -> None:
        segments = np.roll(self.track.centerline, -1, axis=0) - self.track.centerline
        lengths = np.linalg.norm(segments, axis=1)
        expected_length = 2 * (20.0 + 12.0 - 4 * 3.0) + 2 * math.pi * 3.0
        self.assertAlmostEqual(track_length(self.track), expected_length, delta=0.003)
        self.assertGreater(float(lengths.min() / lengths.max()), 0.999)
        headings = np.arctan2(segments[:, 1], segments[:, 0])
        turns = (np.roll(headings, -1) - headings + math.pi) % (2 * math.pi) - math.pi
        self.assertGreaterEqual(float(turns.min()), -1e-12)
        self.assertLess(float(turns.max()), 0.05)

    def test_track_coordinates_round_trip_on_straights_and_corners(self) -> None:
        lengths = np.linalg.norm(np.roll(self.track.centerline, -1, axis=0)
                                 - self.track.centerline, axis=1)
        starts = np.concatenate(([0.0], np.cumsum(lengths)))
        for index in range(0, len(lengths), 17):
            s = float(starts[index] + lengths[index] / 2.0)
            for d in (-0.6, 0.0, 0.6):
                with self.subTest(index=index, d=d):
                    point = track_to_world(s, d, self.track)
                    position = world_to_track(VehicleState(*point, 0.0, 0.0), self.track)
                    self.assertAlmostEqual(position.s, s, places=7)
                    self.assertAlmostEqual(position.d, d, places=7)

    def test_oval_selection_preserves_the_existing_generator(self) -> None:
        config = replace(self.config, name="oval", semi_major_axis=8.0,
                         semi_minor_axis=4.0, point_count=120, width=1.5)
        expected = Track.create_oval(8.0, 4.0, 120, width=1.5)
        actual = create_track(config)
        np.testing.assert_array_equal(actual.centerline, expected.centerline)
        self.assertEqual(actual.width, expected.width)

    def test_custom_dimensions_change_the_geometry(self) -> None:
        config = replace(self.config, rectangle_length=30.0, rectangle_height=14.0,
                         corner_radius=4.0, point_count=600, width=1.5)
        track = create_track(config)
        self.assertEqual(track.centerline.shape, (600, 2))
        self.assertEqual(track.width, 1.5)
        np.testing.assert_allclose(track.centerline.min(axis=0), (-15.0, -7.0))
        np.testing.assert_allclose(track.centerline.max(axis=0), (15.0, 7.0))
        offsets = np.maximum(np.abs(track.centerline) - (11.0, 3.0), 0.0)
        np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), 4.0, atol=1e-12)

    def test_invalid_shapes_are_rejected(self) -> None:
        changes = (
            {"name": "unknown"}, {"rectangle_length": 0.0},
            {"rectangle_height": float("nan")}, {"corner_radius": float("inf")},
            {"corner_radius": 0.0}, {"corner_radius": 6.0},
            {"rectangle_length": 5.0}, {"width": 6.0}, {"width": -1.0},
            {"point_count": 15},
        )
        for values in changes:
            with self.subTest(values=values), self.assertRaises(ValueError):
                create_track(replace(self.config, **values))
        for count in (True, 40.5):
            with self.subTest(count=count), self.assertRaises(TypeError):
                create_track(replace(self.config, point_count=count))


if __name__ == "__main__":
    unittest.main()
