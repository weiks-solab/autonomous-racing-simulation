import unittest

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import ControlCommand, VehicleState
from sim.simulated_vehicle import SimulatedVehicle, validate_unique_vehicle_ids


def _make_vehicle(index: int, target_speed: float | None = None) -> SimulatedVehicle:
    model = KinematicBicycleModel(
        VehicleModelConfig(0.33, 0.02, 0.5, 0.0, 5.0)
    )
    controller = PurePursuitController(
        PurePursuitConfig(
            lookahead_distance=1.0,
            target_speed=target_speed if target_speed is not None else 2.0 + index,
            speed_gain=1.0,
            max_acceleration=3.0,
            max_steering_angle=0.5,
        )
    )
    return SimulatedVehicle(
        vehicle_id=f"vehicle-{index}",
        state=VehicleState(4.0, index * 0.1, 1.57, 0.0),
        model=model,
        controller=controller,
    )


class SimulatedVehicleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track.create_oval(4.0, 2.0, 100)

    def test_initial_trajectory_contains_initial_state(self) -> None:
        vehicle = _make_vehicle(0)
        self.assertEqual(vehicle.trajectory, [vehicle.state])

    def test_step_returns_command_and_target_and_updates_vehicle(self) -> None:
        vehicle = _make_vehicle(0)
        initial_state = vehicle.state
        command, target = vehicle.step(self.track)
        self.assertIsInstance(command, ControlCommand)
        self.assertIs(vehicle.current_target, target)
        self.assertEqual(len(vehicle.trajectory), 2)
        self.assertIs(vehicle.trajectory[-1], vehicle.state)
        self.assertNotEqual(vehicle.state, initial_state)

    def test_empty_vehicle_id_is_rejected(self) -> None:
        vehicle = _make_vehicle(0)
        with self.assertRaises(ValueError):
            SimulatedVehicle("  ", vehicle.state, vehicle.model, vehicle.controller)

    def test_duplicate_vehicle_ids_are_rejected_at_collection_level(self) -> None:
        vehicles = [_make_vehicle(0), _make_vehicle(0)]
        with self.assertRaisesRegex(ValueError, "vehicle-0"):
            validate_unique_vehicle_ids(vehicles)

    def test_different_target_speeds_evolve_independently(self) -> None:
        slower = _make_vehicle(0, target_speed=1.0)
        faster = _make_vehicle(1, target_speed=4.0)
        slower.step(self.track)
        faster.step(self.track)
        self.assertLess(slower.state.velocity, faster.state.velocity)

    def test_supported_collection_sizes_have_independent_components(self) -> None:
        for vehicle_count in (1, 2, 4, 5):
            with self.subTest(vehicle_count=vehicle_count):
                vehicles = [_make_vehicle(index) for index in range(vehicle_count)]
                validate_unique_vehicle_ids(vehicles)
                self.assertEqual(len({id(v.model) for v in vehicles}), vehicle_count)
                self.assertEqual(
                    len({id(v.controller) for v in vehicles}), vehicle_count
                )
                self.assertEqual(
                    len({id(v.trajectory) for v in vehicles}), vehicle_count
                )
                self.assertEqual(len({id(v.state) for v in vehicles}), vehicle_count)

                for vehicle in vehicles:
                    vehicle.step(self.track)
                self.assertTrue(all(len(v.trajectory) == 2 for v in vehicles))


if __name__ == "__main__":
    unittest.main()
