import unittest

from core.track import Track
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.observations import VehicleObservation, create_vehicle_observation


class VehicleObservationTests(unittest.TestCase):
    def test_observation_is_created_without_simulated_vehicle(self) -> None:
        track = Track.create_oval(4.0, 2.0, 100)
        state = VehicleState(4.0, 0.0, 1.57, 2.0)
        geometry = VehicleGeometry(0.5, 0.3, 0.1)
        observation = create_vehicle_observation("ego", state, geometry, track)
        self.assertIsInstance(observation, VehicleObservation)
        self.assertIs(observation.state, state)
        self.assertIs(observation.geometry, geometry)
        self.assertAlmostEqual(observation.track_position.s, 0.0)

    def test_empty_observation_id_is_rejected(self) -> None:
        track = Track.create_oval(4.0, 2.0, 100)
        state = VehicleState(4.0, 0.0, 1.57, 2.0)
        geometry = VehicleGeometry(0.5, 0.3, 0.1)
        with self.assertRaises(ValueError):
            create_vehicle_observation("", state, geometry, track)


if __name__ == "__main__":
    unittest.main()
