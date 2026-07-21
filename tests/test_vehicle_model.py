import math
import unittest

from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import ControlCommand, VehicleState


class KinematicBicycleModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = VehicleModelConfig(2.0, 0.1, 0.5, 0.0, 10.0)
        self.model = KinematicBicycleModel(self.config)

    def test_stationary_vehicle_remains_stationary(self) -> None:
        state = VehicleState(1.0, 2.0, 0.3, 0.0)
        result = self.model.step(state, ControlCommand(0.0, 0.0))
        self.assertEqual(result, state)

    def test_straight_line_motion(self) -> None:
        result = self.model.step(
            VehicleState(0.0, 0.0, math.pi / 2.0, 4.0),
            ControlCommand(0.0, 0.0),
        )
        self.assertAlmostEqual(result.x, 0.0)
        self.assertAlmostEqual(result.y, 0.4)
        self.assertAlmostEqual(result.yaw, math.pi / 2.0)

    def test_pose_uses_current_velocity_before_velocity_update(self) -> None:
        result = self.model.step(
            VehicleState(0.0, 0.0, 0.0, 1.0), ControlCommand(0.0, 10.0)
        )
        self.assertAlmostEqual(result.x, 0.1)
        self.assertAlmostEqual(result.velocity, 2.0)

    def test_steering_and_velocity_are_limited(self) -> None:
        state = VehicleState(0.0, 0.0, 0.0, 9.8)
        result = self.model.step(state, ControlCommand(10.0, 10.0))
        expected_yaw = 9.8 / 2.0 * math.tan(0.5) * 0.1
        self.assertAlmostEqual(result.yaw, expected_yaw)
        self.assertEqual(result.velocity, 10.0)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VehicleModelConfig(0.0, 0.1, 0.5, 0.0, 1.0)
        with self.assertRaises(ValueError):
            VehicleModelConfig(1.0, -0.1, 0.5, 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
