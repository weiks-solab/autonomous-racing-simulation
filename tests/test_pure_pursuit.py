import unittest

import numpy as np

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.vehicle_state import VehicleState


class PurePursuitControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = PurePursuitController(
            PurePursuitConfig(1.0, 2.0, 1.0, 2.0, 0.6)
        )

    def test_target_to_left_produces_positive_steering(self) -> None:
        track = Track(np.array(((0.0, 0.0), (1.0, 1.0), (0.0, 2.0), (-1.0, 1.0))))
        command, target = self.controller.compute_command(
            VehicleState(0.0, 0.0, 0.0, 1.0), track, wheelbase=0.33
        )
        self.assertEqual(target.lookahead_index, 1)
        self.assertGreater(command.steering_angle, 0.0)

    def test_target_to_right_produces_negative_steering(self) -> None:
        track = Track(np.array(((0.0, 0.0), (1.0, -1.0), (0.0, -2.0), (-1.0, -1.0))))
        command, target = self.controller.compute_command(
            VehicleState(0.0, 0.0, 0.0, 1.0), track, wheelbase=0.33
        )
        self.assertEqual(target.lookahead_index, 1)
        self.assertLess(command.steering_angle, 0.0)

    def test_command_returns_the_selected_target(self) -> None:
        track = Track.create_oval(4.0, 2.0, 100)
        command, target = self.controller.compute_command(
            VehicleState(4.0, 0.0, 1.57, 0.0), track, wheelbase=0.33
        )
        self.assertIsNotNone(command)
        self.assertNotEqual(target.nearest_index, target.lookahead_index)


if __name__ == "__main__":
    unittest.main()
