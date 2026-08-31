import math
import unittest

import numpy as np

from core.vehicle_geometry import (
    VehicleGeometry,
    oriented_rectangle_corners,
    rectangles_overlap,
)
from core.vehicle_state import VehicleState


class VehicleGeometryTests(unittest.TestCase):
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
