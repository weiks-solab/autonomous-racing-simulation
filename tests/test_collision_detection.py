import unittest
from dataclasses import replace

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.vehicle_geometry import CollisionConfig, VehicleGeometry
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import VehicleState
from sim.simulated_vehicle import SimulatedVehicle, find_vehicle_collisions
from strategy.collision_check import PredictedVehicle, first_predicted_collision
from tests.test_predicted_collision import prediction


def _vehicle(vehicle_id: str, x: float) -> SimulatedVehicle:
    return SimulatedVehicle(
        vehicle_id=vehicle_id,
        state=VehicleState(x=x, y=0.0, yaw=0.0, velocity=1.0),
        model=KinematicBicycleModel(
            VehicleModelConfig(0.33, 0.02, 0.5, 0.0, 5.0)
        ),
        controller=PurePursuitController(
            PurePursuitConfig(1.0, 2.0, 1.0, 2.0, 0.5)
        ),
        geometry=VehicleGeometry(0.5, 0.3, 0.1),
    )


class CollisionCollectionTests(unittest.TestCase):
    def test_actual_and_predicted_checks_share_the_adjusted_collision_dimensions(self) -> None:
        vehicles = [_vehicle("a", 0.0), _vehicle("b", 0.3)]
        self.assertEqual(find_vehicle_collisions(vehicles), {("a", "b")})
        for vehicle in vehicles:
            vehicle.geometry = replace(vehicle.geometry, collision=CollisionConfig(0.2, 0.15))
        self.assertEqual(find_vehicle_collisions(vehicles), set())
        first, second = vehicles
        times = (0.0, 0.1)
        own = prediction(times, (first.state, first.state))
        other = prediction(times, (second.state, second.state))
        self.assertIsNone(first_predicted_collision(
            own, first.geometry, (PredictedVehicle(second.vehicle_id, second.geometry, other),),
        ))

    def test_arbitrary_vehicle_counts_are_supported(self) -> None:
        for count in (1, 2, 4, 5):
            with self.subTest(count=count):
                vehicles = [_vehicle(f"vehicle-{index}", index * 2.0) for index in range(count)]
                self.assertEqual(find_vehicle_collisions(vehicles), set())

    def test_only_overlapping_pairs_are_returned(self) -> None:
        vehicles = (
            _vehicle("a", 0.0),
            _vehicle("b", 0.2),
            _vehicle("c", 3.0),
            _vehicle("d", 3.2),
        )
        self.assertEqual(
            find_vehicle_collisions(vehicles),
            {("a", "b"), ("c", "d")},
        )

    def test_collision_check_does_not_modify_vehicle_behavior(self) -> None:
        vehicles = [_vehicle("a", 0.0), _vehicle("b", 0.2)]
        states_before = [vehicle.state for vehicle in vehicles]
        trajectories_before = [list(vehicle.trajectory) for vehicle in vehicles]
        velocities_before = [vehicle.state.velocity for vehicle in vehicles]

        self.assertEqual(find_vehicle_collisions(vehicles), {("a", "b")})

        self.assertEqual([vehicle.state for vehicle in vehicles], states_before)
        self.assertEqual(
            [vehicle.trajectory for vehicle in vehicles], trajectories_before
        )
        self.assertEqual(
            [vehicle.state.velocity for vehicle in vehicles], velocities_before
        )

    def test_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            find_vehicle_collisions([_vehicle("same", 0.0), _vehicle("same", 2.0)])


if __name__ == "__main__":
    unittest.main()
