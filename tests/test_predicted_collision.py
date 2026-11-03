"""預測碰撞必須比較同一時間的完整車身，包含不同步長與角度繞回。"""

import math
import unittest

from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.bicycle_prediction import BicyclePrediction
from strategy.collision_check import PredictedVehicle, first_predicted_collision


def prediction(times: tuple[float, ...], states: tuple[VehicleState, ...]) -> BicyclePrediction:
    return BicyclePrediction(states, tuple(0.0 for _ in states), times)


class PredictedCollisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = VehicleGeometry(0.2, 0.2, 0.0)

    def test_paths_crossing_at_different_times_do_not_collide(self) -> None:
        times = (0.0, 0.1, 0.2)
        ego = prediction(times, (VehicleState(-1, 0, 0, 1), VehicleState(0, 0, 0, 1),
                                 VehicleState(1, 0, 0, 1)))
        other = prediction(times, (VehicleState(0, -2, math.pi / 2, 1),
                                   VehicleState(0, -1, math.pi / 2, 1),
                                   VehicleState(0, 0, math.pi / 2, 1)))
        result = first_predicted_collision(ego, self.geometry,
                                           (PredictedVehicle("other", self.geometry, other),))
        self.assertIsNone(result)

    def test_different_time_steps_are_aligned_before_comparing_states(self) -> None:
        ego = prediction((0.0, 0.2), (VehicleState(-1, 0, 0, 1), VehicleState(1, 0, 0, 1)))
        stationary = VehicleState(0, 0, 0, 0)
        other = prediction((0.0, 0.1, 0.2), (stationary, stationary, stationary))
        result = first_predicted_collision(ego, self.geometry,
                                           (PredictedVehicle("other", self.geometry, other),))
        self.assertEqual(result.vehicle_id, "other")
        self.assertEqual(result.time_seconds, 0.1)

    def test_body_contact_at_time_zero_is_not_ignored(self) -> None:
        geometry = VehicleGeometry(0.55, 0.3, 0.12)
        ego_state, other_state = VehicleState(0, 0, 0, 0), VehicleState(0.55, 0, 0, 0)
        ego = prediction((0.0, 0.1), (ego_state, ego_state))
        other = prediction((0.0, 0.1), (other_state, other_state))
        result = first_predicted_collision(ego, geometry,
                                           (PredictedVehicle("touching", geometry, other),))
        self.assertEqual(result.time_seconds, 0.0)

    def test_earliest_collision_is_found_across_all_vehicles(self) -> None:
        times = (0.0, 0.1, 0.2)
        ego = prediction(times, (VehicleState(0, 0, 0, 1), VehicleState(1, 0, 0, 1),
                                 VehicleState(2, 0, 0, 1)))
        later_state, earlier_state = VehicleState(2, 0, 0, 0), VehicleState(1, 0, 0, 0)
        later = prediction(times, (later_state,) * 3)
        earlier = prediction(times, (earlier_state,) * 3)
        result = first_predicted_collision(ego, self.geometry, (
            PredictedVehicle("a-later", self.geometry, later),
            PredictedVehicle("z-earlier", self.geometry, earlier),
        ))
        self.assertEqual((result.vehicle_id, result.time_seconds), ("z-earlier", 0.1))

    def test_yaw_interpolation_crosses_pi_without_rotating_through_zero(self) -> None:
        geometry = VehicleGeometry(1.0, 0.2, 0.4)
        ego = prediction((0.0, 0.2), (
            VehicleState(0, 0, math.radians(179), 0),
            VehicleState(0, 0, math.radians(-179), 0),
        ))
        state = VehicleState(0.6, 0, 0, 0)
        other = prediction((0.0, 0.1, 0.2), (state,) * 3)
        result = first_predicted_collision(ego, geometry,
                                           (PredictedVehicle("other", self.geometry, other),))
        self.assertIsNone(result)

    def test_different_prediction_horizons_are_rejected(self) -> None:
        state = VehicleState(0, 0, 0, 0)
        ego = prediction((0.0, 0.2), (state, state))
        other = prediction((0.0, 0.1), (state, state))
        with self.assertRaisesRegex(ValueError, "same time interval"):
            first_predicted_collision(ego, self.geometry,
                                      (PredictedVehicle("other", self.geometry, other),))


if __name__ == "__main__":
    unittest.main()
