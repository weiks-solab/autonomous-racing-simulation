import unittest

import numpy as np

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.track_geometry import world_to_track
from core.vehicle_state import VehicleState


class OpenTrackCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(
            np.array(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0))),
            closed=False,
        )

    def test_projection_does_not_add_final_to_first_segment(self) -> None:
        position = world_to_track(
            VehicleState(0.5, 0.5, 0.0, 0.0),
            self.track,
        )
        self.assertEqual(position.centerline_index, 0)
        self.assertAlmostEqual(position.s, 0.5)
        self.assertAlmostEqual(position.d, 0.5)

    def test_nearest_lookup_uses_sampled_points(self) -> None:
        self.assertEqual(self.track.nearest_point_index(0.5, 0.5), 0)

    def test_lookahead_stops_at_final_point(self) -> None:
        controller = PurePursuitController(
            PurePursuitConfig(100.0, 1.0, 1.0, 1.0, 0.5)
        )
        _, target = controller.compute_command(
            VehicleState(1.8, 0.1, 0.0, 1.0),
            self.track,
            wheelbase=0.33,
        )
        self.assertEqual(target.lookahead_index, 2)
        self.assertEqual((target.x, target.y), (2.0, 2.0))

    def test_forward_search_beyond_path_does_not_wrap(self) -> None:
        self.assertEqual(self.track.forward_point_index(1, 100.0), 2)
        self.assertEqual(self.track.forward_point_index(2, 1.0), 2)


if __name__ == "__main__":
    unittest.main()
