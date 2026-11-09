"""Exercise the existing controller, offset paths and data entry with both maps."""

import unittest
from unittest.mock import patch

import numpy as np

import main
from core.track_geometry import track_length, world_to_track
from core.vehicle_geometry import oriented_rectangle_corners
from core.vehicle_state import VehicleState
from generate_training_data import _demo_settings
from strategy.offset_path import (
    LateralManeuver, continue_lateral_maneuver, generate_offset_track,
)


class TrackSwitchingTests(unittest.TestCase):
    def test_simulation_and_data_collection_use_the_same_selected_geometry(self) -> None:
        for name in ("rounded_rectangle", "oval"):
            with self.subTest(name=name), patch("main.TRACK_NAME", name), \
                 patch("main.TRACK_WIDTH", 2.4):
                expected = main.create_demo_track()
                settings, _, _ = _demo_settings(no_defense=False)
                np.testing.assert_array_equal(settings.track.centerline, expected.centerline)
                self.assertEqual(settings.track.width, 2.4)
                self.assertEqual(settings.track.closed, expected.closed)

    def test_existing_controller_completes_two_laps_on_both_maps(self) -> None:
        for name in ("rounded_rectangle", "oval"):
            with self.subTest(name=name), patch("main.TRACK_NAME", name):
                track = main.create_demo_track()
                vehicle = main._create_vehicle("solo", 0.0, 10.0, main.MODEL_CONFIG, track)
                positions = [world_to_track(vehicle.state, track)]
                for _ in range(round(21.0 / main.MODEL_CONFIG.time_step)):
                    vehicle.step(track)
                    positions.append(world_to_track(vehicle.state, track))
                length = track_length(track)
                increments = (np.diff([p.s for p in positions]) + length / 2) % length - length / 2
                self.assertGreater(float(increments.sum()), 2 * length)
                self.assertGreaterEqual(float(increments.min()), -1e-9)
                self.assertLess(max(abs(p.d) for p in positions), 0.3)

    def test_lateral_paths_remain_trackable_through_all_four_corners(self) -> None:
        with patch("main.TRACK_NAME", "rounded_rectangle"):
            track = main.create_demo_track()
        for offset in (-0.6, 0.6):
            with self.subTest(offset=offset):
                vehicle = main._create_vehicle("offset", 0.0, 10.0, main.MODEL_CONFIG, track)
                position = world_to_track(vehicle.state, track)
                maneuver = LateralManeuver(position.s, 0.0, offset, 4.0)
                progress = 0.0
                length = track_length(track)
                for _ in range(round(15.0 / main.MODEL_CONFIG.time_step)):
                    maneuver = continue_lateral_maneuver(maneuver, position.s, length)
                    path = generate_offset_track(track, position, maneuver, main.OFFSET_PATH_CONFIG)
                    vehicle.step(path)
                    next_position = world_to_track(vehicle.state, track)
                    progress += (next_position.s - position.s + length / 2) % length - length / 2
                    position = next_position
                    for x, y in oriented_rectangle_corners(vehicle.state, vehicle.geometry):
                        corner = VehicleState(float(x), float(y), vehicle.state.yaw, vehicle.state.velocity)
                        self.assertLess(abs(world_to_track(corner, track).d), track.width / 2)
                self.assertGreater(progress, 2 * length)
                self.assertAlmostEqual(position.d, offset, delta=0.25)


if __name__ == "__main__":
    unittest.main()
