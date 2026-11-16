"""Exercise map files independently of the ignored external download."""

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

import main
from core.track_factory import TrackConfig, create_track
from core.track_geometry import track_length, track_to_world, world_to_track
from core.track_loader import PROJECT_ROOT, find_centerline, load_centerline, project_file
from core.vehicle_state import VehicleState
from generate_training_data import _demo_settings
from sim.map_background import load_map_background
from sim.visualizer import SimulationVisualizer, VisualizationConfig
from strategy.assignment import StrategyAssignment


class ImportedTrackTests(unittest.TestCase):
    def setUp(self) -> None:
        cache = PROJECT_ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=cache, prefix="map-test-")
        self.root = Path(self.temporary.name)
        self.folder = self.root / "Test Track"
        self.folder.mkdir()
        self.csv = self.folder / "TestTrack_centerline.csv"
        angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        self.points = np.column_stack((6 * np.cos(angles), 4 * np.sin(angles)))
        self.data = np.column_stack((self.points, np.full((120, 2), 1.1)))
        self._write_csv(self.data)
        # 非對稱像素可用來辨識上下顛倒；格式同下載的 YAML。
        self.pixels = np.array([[0.0, 0.2], [0.7, 1.0]], dtype=float)
        plt.imsave(self.folder / "TestTrack_map.png", self.pixels, cmap="gray", vmin=0, vmax=1)
        self.yaml = self.folder / "TestTrack_map.yaml"
        self._write_yaml()

    def tearDown(self) -> None:
        plt.close("all")
        self.temporary.cleanup()

    def _write_csv(self, data) -> None:
        np.savetxt(self.csv, data, delimiter=",", header="x_m,y_m,w_tr_right_m,w_tr_left_m")

    def _write_yaml(self, resolution="0.5", origin="[10.0, -3.0, 0.0]") -> None:
        self.yaml.write_text(
            f'image: "TestTrack_map.png"\nresolution: {resolution}\norigin: {origin}\n'
            "negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.196\n", encoding="utf-8",
        )

    def test_loading_uses_metres_and_csv_width(self) -> None:
        path = find_centerline("Test Track", self.root)
        track = create_track(TrackConfig(name="Test Track",
                                         map_root=str(self.root), sample_spacing=0.3, width=99))
        self.assertEqual(path, self.csv)
        self.assertAlmostEqual(track.width, 2.2)
        np.testing.assert_allclose(track.centerline[0], self.points[0])
        self.assertFalse(np.array_equal(track.centerline[0], track.centerline[-1]))
        steps = np.linalg.norm(np.roll(track.centerline, -1, axis=0) - track.centerline, axis=1)
        self.assertLessEqual(float(steps.max()), 0.3 + 1e-9)
        self.assertGreater(float(steps.min()), 0.29)

    def test_duplicate_samples_and_repeated_endpoint_do_not_create_zero_segments(self) -> None:
        original = load_centerline(self.csv)
        self._write_csv(np.vstack((self.data[0], self.data, self.data[-1], self.data[0])))
        cleaned = load_centerline(self.csv)
        np.testing.assert_allclose(cleaned.centerline, original.centerline)

    def test_invalid_columns_coordinates_widths_and_closure_are_rejected(self) -> None:
        cases = [self.data[:, :2], self.data[:2], np.full_like(self.data, np.nan)]
        for width in (0.0, -1.0):
            changed = self.data.copy()
            changed[:, 2:] = width
            cases.append(changed)
        asymmetric = self.data.copy()
        asymmetric[:, 2] = 0.9
        variable = self.data.copy()
        variable[20, 3] = 0.7
        cases.extend((asymmetric, variable, self.data[:50], np.tile(self.data[0], (4, 1))))
        for index, data in enumerate(cases):
            with self.subTest(index=index):
                self._write_csv(data)
                with self.assertRaises(ValueError):
                    load_centerline(self.csv)
        self._write_csv(self.data)
        for spacing in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                load_centerline(self.csv, spacing)

    def test_repo_path_and_ambiguous_file_checks(self) -> None:
        with self.assertRaises(ValueError):
            project_file(PROJECT_ROOT.parent / "outside.csv")
        for name in ("..", "../escape", "missing"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                find_centerline(name, self.root)
        (self.folder / "duplicate_centerline.csv").write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            find_centerline("Test Track", self.root)

    def test_track_coordinates_and_pursuit_cross_the_imported_seam(self) -> None:
        track = load_centerline(self.csv, 0.2)
        length = track_length(track)
        for s in (length - 0.05, length + 0.05):
            xy = track_to_world(s, 0.0, track)
            position = world_to_track(VehicleState(*xy, 0, 1), track)
            self.assertAlmostEqual(position.s, s % length, places=8)
        vehicle = main._create_vehicle("solo", length - 0.5, 3, main.MODEL_CONFIG, track)
        _, target = vehicle.controller.compute_command(vehicle.state, track, main.MODEL_CONFIG.wheelbase)
        self.assertLess(target.lookahead_index, target.nearest_index)

    def test_background_transform_retains_pixel_orientation_and_optional_visibility(self) -> None:
        self._write_yaml(origin=f"[10, -3, {math.pi / 2}]")
        background = load_map_background(self.yaml)
        track = load_centerline(self.csv)
        visualizer = SimulationVisualizer(track, VisualizationConfig(), background)
        artist = visualizer.map_artist
        self.assertEqual(artist.origin, "upper")
        np.testing.assert_allclose(artist.get_extent(), (0, 1, 0, 1))
        # 像素影像左下、右上角經旋轉後，應落在指定世界座標。
        world = visualizer.axes.transData.inverted().transform(
            artist.get_transform().transform([[0, 0], [1, 1]]))
        np.testing.assert_allclose(world, [[10, -3], [9, -2]], atol=1e-10)
        self.assertLess(float(artist.get_array()[0, 0, 0]), float(artist.get_array()[-1, 0, 0]))
        hidden = SimulationVisualizer(track, VisualizationConfig(show_map_background=False), background)
        self.assertIsNone(hidden.map_artist)
        visualizer.figure.canvas.draw()

    def test_invalid_map_metadata_and_outside_image_are_rejected(self) -> None:
        for resolution, origin in (("0", "[0,0,0]"), ("nan", "[0,0,0]"),
                                   ("0.5", "[0,0]"), ("0.5", "[0,0,1e999]")):
            self._write_yaml(resolution, origin)
            with self.subTest(resolution=resolution, origin=origin), self.assertRaises(ValueError):
                load_map_background(self.yaml)
        self._write_yaml()
        self.yaml.write_text(self.yaml.read_text().replace('"TestTrack_map.png"',
                             json.dumps(str(PROJECT_ROOT.parent / "outside.png"))), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_map_background(self.yaml)

    def test_data_collection_shares_external_geometry(self) -> None:
        with patch("main.TRACK_NAME", "Test Track"), \
             patch("main.TRACK_MAP_ROOT", str(self.root)):
            track = main.create_demo_track()
            settings, _, _ = _demo_settings(False)
        np.testing.assert_array_equal(settings.track.centerline, track.centerline)


    def test_imported_multi_vehicle_does_not_load_old_gp(self) -> None:
        specs = (("first", 0.0, 3.0), ("second", 4.0, 2.0), ("third", 12.0, 2.0))
        assignments = {"first": StrategyAssignment("relative_overtake")}
        with patch("main.TRACK_MAP_ROOT", str(self.root)), patch("main.TRACK_NAME", "Test Track"), \
             patch("main.VEHICLE_SPECS", specs), patch("main.STRATEGY_ASSIGNMENTS", assignments), \
             patch("main.SIMULATION_SECONDS", 0.1), patch("main.load_gp_routes") as gp, \
             patch("main.SimulationVisualizer") as display, redirect_stdout(io.StringIO()):
            main.run_simulation()
        gp.assert_not_called()
        self.assertFalse(display.call_args.args[1].show_gp_predictions)
        self.assertIsNotNone(display.call_args.kwargs["background"])
        self.assertEqual(set(display.return_value.update_vehicles.call_args.kwargs["states"]),
                         {"first", "second", "third"})



if __name__ == "__main__":
    unittest.main()
