"""確認前向預測重用實際控制器與模型，且定速前車仍會轉彎。"""

from dataclasses import replace
import math
import unittest

import numpy as np

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.track_geometry import track_length, track_to_world
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import VehicleState
from strategy.bicycle_prediction import BicyclePredictionConfig, predict_bicycle_trajectory
from strategy.observations import VehicleObservation, create_vehicle_observation
from strategy.offset_path import (
    LateralManeuver, OffsetPathConfig, continue_lateral_maneuver, generate_offset_track,
)


def observation(
    track: Track, vehicle_id: str, s: float, d: float, velocity: float,
) -> VehicleObservation:
    point = track_to_world(s, d, track)
    ahead = track_to_world(s + 0.01, d, track)
    yaw = math.atan2(ahead[1] - point[1], ahead[0] - point[0])
    return create_vehicle_observation(
        vehicle_id, VehicleState(float(point[0]), float(point[1]), yaw, velocity),
        VehicleGeometry(0.55, 0.3, 0.12), track,
    )


def prediction_config(target_speed: float = 6.0) -> BicyclePredictionConfig:
    return BicyclePredictionConfig(
        VehicleModelConfig(0.33, 0.02, math.radians(50.0), 0.0, 10.0),
        PurePursuitConfig(2.0, target_speed, 1.5, 3.0, math.radians(50.0)),
        OffsetPathConfig(0.2, 7.0),
    )


class BicyclePredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))), width=2.0)
        self.config = prediction_config()

    def test_constant_speed_straight_prediction_and_partial_last_step(self) -> None:
        vehicle = observation(self.track, "front", 20.0, 0.0, 3.0)
        result = predict_bicycle_trajectory(
            vehicle, self.track, LateralManeuver(20.0, 0.0, 0.0, 3.0),
            self.config, 0.105, constant_speed=True,
        )
        self.assertEqual(len(result.states), 7)
        self.assertEqual(result.times[-1], 0.105)
        self.assertTrue(all(state.velocity == 3.0 for state in result.states))
        self.assertAlmostEqual(result.states[-1].x, 20.315)
        self.assertAlmostEqual(result.progress[-1], 0.315)
        self.assertEqual(vehicle.state.x, 20.0)

    def test_first_step_matches_existing_controller_and_bicycle_model(self) -> None:
        vehicle = observation(self.track, "ego", 20.0, 0.0, 1.0)
        maneuver = LateralManeuver(20.0, 0.0, 0.6, 3.0)
        reference = generate_offset_track(self.track, vehicle.track_position, maneuver, self.config.path)
        command, _ = PurePursuitController(self.config.controller).compute_command(
            vehicle.state, reference, self.config.model.wheelbase,
        )
        expected = KinematicBicycleModel(self.config.model).step(vehicle.state, command)
        result = predict_bicycle_trajectory(vehicle, self.track, maneuver, self.config, 0.1)
        self.assertEqual(result.states[1], expected)
        self.assertGreater(result.states[-1].velocity, vehicle.state.velocity)
        self.assertGreater(result.states[-1].yaw, 0.0)

    def test_fixed_speed_vehicle_turns_along_track_instead_of_a_straight_line(self) -> None:
        track = Track.create_oval(10.0, 5.0, 300)
        vehicle = observation(track, "front", 1.0, 0.2, 4.0)
        result = predict_bicycle_trajectory(
            vehicle, track, LateralManeuver(vehicle.track_position.s, 0.2, 0.2, 3.0),
            self.config, 1.0, constant_speed=True,
        )
        self.assertTrue(all(state.velocity == 4.0 for state in result.states))
        self.assertGreater(abs(result.states[-1].yaw - vehicle.state.yaw), 0.1)
        linear_end = np.array((vehicle.state.x + 4.0 * math.cos(vehicle.state.yaw),
                               vehicle.state.y + 4.0 * math.sin(vehicle.state.yaw)))
        actual_end = np.array((result.states[-1].x, result.states[-1].y))
        self.assertGreater(float(np.linalg.norm(actual_end - linear_end)), 0.2)

    def test_progress_is_continuous_across_start_finish(self) -> None:
        track = Track.create_oval(10.0, 5.0, 300)
        vehicle = observation(track, "ego", track_length(track) - 0.1, 0.0, 3.0)
        result = predict_bicycle_trajectory(
            vehicle, track, LateralManeuver(vehicle.track_position.s, 0.0, 0.0, 3.0),
            self.config, 0.3, constant_speed=True,
        )
        self.assertGreater(result.progress[-1], 0.7)
        self.assertLess(result.progress[-1], 1.1)

    def test_stationary_fixed_speed_vehicle_stays_stationary(self) -> None:
        vehicle = observation(self.track, "front", 20.0, 0.0, 0.0)
        result = predict_bicycle_trajectory(
            vehicle, self.track, LateralManeuver(20.0, 0.0, 0.0, 3.0),
            self.config, 0.1, constant_speed=True,
        )
        self.assertTrue(all(state == vehicle.state for state in result.states))

    def test_completed_reference_stays_at_target_after_another_lap(self) -> None:
        track = Track(np.array(((0, 0), (10, 0), (10, 10), (0, 10))), width=4.0)
        maneuver = LateralManeuver(39.0, 0.0, 0.6, 3.0)
        completed = continue_lateral_maneuver(maneuver, 5.0, track_length(track))
        self.assertEqual(completed.start_d, 0.6)
        next_lap = observation(track, "ego", 1.5, 0.6, 3.0)
        reference = generate_offset_track(track, next_lap.track_position, completed, self.config.path)
        np.testing.assert_allclose(reference.centerline[:, 1], 0.6, atol=1e-10)

    def test_invalid_horizon_track_and_initial_speed_are_rejected(self) -> None:
        vehicle = observation(self.track, "ego", 20.0, 0.0, 3.0)
        maneuver = LateralManeuver(20.0, 0.0, 0.0, 3.0)
        for horizon in (0.0, -1.0, float("nan")):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                predict_bicycle_trajectory(vehicle, self.track, maneuver, self.config, horizon)
        with self.assertRaisesRegex(ValueError, "closed"):
            predict_bicycle_trajectory(
                vehicle, replace(self.track, closed=False), maneuver, self.config, 0.1,
            )
        with self.assertRaisesRegex(ValueError, "velocity"):
            predict_bicycle_trajectory(
                replace(vehicle, state=replace(vehicle.state, velocity=11.0)),
                self.track, maneuver, self.config, 0.1,
            )


if __name__ == "__main__":
    unittest.main()
