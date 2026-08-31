import unittest

import numpy as np

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.track_geometry import TrackPosition, track_to_world
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import VehicleState
from sim.simulated_vehicle import SimulatedVehicle
from strategy.offset_path import (
    LateralManeuver,
    OffsetPathConfig,
    generate_offset_track,
)


class OvertakingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(
            np.array(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))),
            width=4.0,
        )

    def test_existing_pure_pursuit_tracks_temporary_open_path(self) -> None:
        temporary = generate_offset_track(
            self.track,
            TrackPosition(2.0, 0.0, 0),
            LateralManeuver(2.0, 0.0, 1.0, 4.0),
            config=OffsetPathConfig(sample_spacing=0.2, path_length=8.0),
        )
        controller = PurePursuitController(
            PurePursuitConfig(2.0, 3.0, 1.0, 2.0, 0.6)
        )
        point = track_to_world(2.0, 0.0, self.track)
        command, target = controller.compute_command(
            VehicleState(float(point[0]), float(point[1]), 0.0, 2.0),
            temporary,
            wheelbase=0.33,
        )
        self.assertGreater(command.steering_angle, 0.0)
        self.assertGreater(target.lookahead_index, 0)

    def test_non_ego_centerline_step_is_unchanged(self) -> None:
        config = VehicleModelConfig(0.33, 0.02, 0.5, 0.0, 5.0)
        controller_config = PurePursuitConfig(1.0, 2.0, 1.0, 2.0, 0.5)
        initial = VehicleState(0.0, 0.0, 0.0, 1.0)
        baseline = SimulatedVehicle(
            "baseline",
            initial,
            KinematicBicycleModel(config),
            PurePursuitController(controller_config),
        )
        non_ego = SimulatedVehicle(
            "non-ego",
            initial,
            KinematicBicycleModel(config),
            PurePursuitController(controller_config),
        )
        baseline.step(self.track)
        non_ego.step(self.track)
        self.assertEqual(non_ego.state, baseline.state)
        self.assertEqual(non_ego.trajectory, baseline.trajectory)


if __name__ == "__main__":
    unittest.main()
