import math
from dataclasses import replace
import unittest

import numpy as np

from core.vehicle_geometry import (
    CollisionConfig,
    VehicleGeometry,
    oriented_rectangle_corners,
    rectangles_overlap,
)
from core.vehicle_state import VehicleState


class VehicleGeometryTests(unittest.TestCase):
    def test_collision_size_changes_overlap_without_changing_physical_corners(self) -> None:
        body = VehicleGeometry(0.55, 0.30, 0.12)
        first = VehicleState(0.0, 0.0, 0.0, 0.0)
        for config, position, expected in (
            (CollisionConfig(0.45, 0.24), (0.5, 0.0), False),
            (CollisionConfig(0.65, 0.40), (0.6, 0.0), True),
            (CollisionConfig(0.45, 0.24), (0.0, 0.27), False),
        ):
            with self.subTest(config=config, position=position):
                geometry = replace(body, collision=config)
                other = VehicleState(*position, 0.0, 0.0)
                self.assertEqual(rectangles_overlap(first, geometry, other, geometry), expected)
                self.assertNotEqual(rectangles_overlap(first, body, other, body), expected)
                np.testing.assert_array_equal(oriented_rectangle_corners(first, geometry),
                                              oriented_rectangle_corners(first, body))
                self.assertEqual(geometry.width, 0.30)

    def test_small_collision_rectangle_keeps_the_rotated_body_center(self) -> None:
        geometry = VehicleGeometry(0.55, 0.30, 0.12, collision=CollisionConfig(0.10, 0.08))
        state = VehicleState(1.0, 2.0, math.pi / 3.0, 0.0)
        body = oriented_rectangle_corners(state, geometry)
        collision = oriented_rectangle_corners(state, geometry, collision=True)
        np.testing.assert_allclose(collision.mean(axis=0), body.mean(axis=0))
        self.assertAlmostEqual(float(np.linalg.norm(collision[0] - collision[1])), 0.08)
        self.assertAlmostEqual(float(np.linalg.norm(collision[1] - collision[2])), 0.10)

    def test_invalid_collision_dimensions_are_rejected(self) -> None:
        for length, width in ((0.0, 0.3), (0.5, -0.1), (math.inf, 0.3), (0.5, math.nan)):
            with self.subTest(length=length, width=width), self.assertRaises(ValueError):
                CollisionConfig(length, width)

    def test_corners_use_rear_axle_to_center_offset(self) -> None:
        geometry = VehicleGeometry(length=4.0, width=2.0, rear_axle_to_center=1.0)
        state = VehicleState(x=1.0, y=2.0, yaw=0.0, velocity=0.0)
        np.testing.assert_allclose(
            oriented_rectangle_corners(state, geometry),
            np.array(((4.0, 3.0), (4.0, 1.0), (0.0, 1.0), (0.0, 3.0))),
        )

    def test_offset_rotates_with_vehicle_yaw(self) -> None:
        geometry = VehicleGeometry(length=2.0, width=1.0, rear_axle_to_center=0.5)
        state = VehicleState(x=1.0, y=2.0, yaw=math.pi / 2.0, velocity=0.0)
        corners = oriented_rectangle_corners(state, geometry)
        self.assertAlmostEqual(float(np.mean(corners[:, 0])), 1.0)
        self.assertAlmostEqual(float(np.mean(corners[:, 1])), 2.5)

    def test_overlapping_rectangles_collide(self) -> None:
        geometry = VehicleGeometry(2.0, 1.0, 0.5)
        self.assertTrue(
            rectangles_overlap(
                VehicleState(0.0, 0.0, 0.0, 0.0),
                geometry,
                VehicleState(1.0, 0.0, 0.0, 0.0),
                geometry,
            )
        )

    def test_separated_rectangles_do_not_collide(self) -> None:
        geometry = VehicleGeometry(2.0, 1.0, 0.5)
        self.assertFalse(
            rectangles_overlap(
                VehicleState(0.0, 0.0, 0.0, 0.0),
                geometry,
                VehicleState(4.0, 0.0, 0.0, 0.0),
                geometry,
            )
        )

    def test_rotated_rectangles_collide(self) -> None:
        geometry = VehicleGeometry(2.0, 0.8, 0.4)
        self.assertTrue(
            rectangles_overlap(
                VehicleState(0.0, 0.0, math.pi / 4.0, 0.0),
                geometry,
                VehicleState(0.5, 0.0, -math.pi / 4.0, 0.0),
                geometry,
            )
        )

    def test_touching_rectangles_count_as_collision(self) -> None:
        geometry = VehicleGeometry(2.0, 1.0, 0.5)
        self.assertTrue(
            rectangles_overlap(
                VehicleState(0.0, 0.0, 0.0, 0.0),
                geometry,
                VehicleState(2.0, 0.0, 0.0, 0.0),
                geometry,
            )
        )

    def test_invalid_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VehicleGeometry(0.0, 1.0, 0.0)
        with self.assertRaises(ValueError):
            VehicleGeometry(2.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            VehicleGeometry(2.0, 1.0, -0.1)
        with self.assertRaises(ValueError):
            VehicleGeometry(2.0, 1.0, 1.1)
        with self.assertRaises(ValueError):
            VehicleGeometry(2.0, 1.0, math.inf)


if __name__ == "__main__":
    unittest.main()
