"""顯示策略實際使用的預測，並驗證整圈 GP 圖的資料與輸出。"""

from contextlib import redirect_stdout
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from core.track import Track
from core.track_geometry import track_length, track_to_world_points
import plot_gp_route
from sim.gp_visualization import create_gp_route_figure, sample_gp_route
from sim.visualizer import SimulationVisualizer, VisualizationConfig
from strategy.gp_prediction import GpRoute, PROJECT_ROOT
from strategy.relative_overtaking import RelativeOvertakingConfig, RelativeOvertakingStrategy
from tests.test_bicycle_prediction import observation, prediction_config
from tests.test_gp_prediction import LinearModel


class GpVisualizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.route = GpRoute(LinearModel(offset=0.2))
        self.ego = observation(self.track, "ego", 20.0, 0.0, 6.0)
        self.front = observation(self.track, "front", 22.0, 0.6, 3.0)
        self.strategy = RelativeOvertakingStrategy(
            RelativeOvertakingConfig(prediction_horizon=1.0), prediction_config(),
            {"front": prediction_config(), "third": prediction_config()}, {"front": self.route},
        )
        self.snapshot = self.strategy.update(self.ego, (self.front,), self.track, 0.0).candidate_snapshot

    def tearDown(self) -> None:
        plt.close("all")

    def update(self, visualizer: SimulationVisualizer, snapshots: dict) -> None:
        states = {"ego": self.ego.state, "front": self.front.state}
        visualizer.update_vehicles(states, {key: [state] for key, state in states.items()},
                                   {key: None for key in states}, candidate_snapshots=snapshots)

    def test_snapshot_reuses_collision_states_and_excludes_non_gp_opponents(self) -> None:
        third = observation(self.track, "third", 40.0, -0.6, 3.0)
        snapshot = self.strategy.update(self.ego, (self.front, third), self.track, 0.1).candidate_snapshot
        self.assertEqual([item.vehicle_id for item in snapshot.gp_predictions], ["front"])
        actual = self.strategy.last_other_predictions["front"]
        self.assertIs(snapshot.gp_predictions[0].states, actual.states)
        self.assertIs(snapshot.gp_predictions[0].times, actual.times)

    def test_live_line_matches_prediction_reuses_artists_and_clears_old_data(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig(show_candidate_paths=False))
        self.update(visualizer, {"ego": self.snapshot})
        artists = visualizer._gp_prediction_artists[("ego", "front")]
        points = np.array([(state.x, state.y) for state in self.snapshot.gp_predictions[0].states])
        np.testing.assert_allclose(np.column_stack(artists.line.get_data()), points)
        self.assertTrue(artists.line.get_visible())
        self.assertIn("GP ego -> front", artists.label.get_text())
        self.assertIn("@ 0.00s, +1.00s", artists.label.get_text())
        counts = len(visualizer.axes.lines), len(visualizer.axes.texts)
        self.ego = observation(self.track, "ego", 21.0, 0.0, 6.0)
        self.update(visualizer, {"ego": self.snapshot})
        self.assertEqual(counts, (len(visualizer.axes.lines), len(visualizer.axes.texts)))
        np.testing.assert_allclose(np.column_stack(artists.line.get_data()), points)
        self.update(visualizer, {})
        self.assertFalse(artists.line.get_visible())
        self.assertFalse(artists.label.get_visible())

    def test_gp_visibility_and_labels_are_independent_of_candidate_visibility(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig(show_candidate_paths=False))
        self.update(visualizer, {"ego": self.snapshot})
        artists = visualizer._gp_prediction_artists[("ego", "front")]
        visualizer.config = replace(visualizer.config, show_gp_prediction_labels=False)
        self.update(visualizer, {"ego": self.snapshot})
        self.assertTrue(artists.line.get_visible())
        self.assertFalse(artists.label.get_visible())
        visualizer.config = replace(visualizer.config, show_gp_predictions=False)
        self.update(visualizer, {"ego": self.snapshot})
        self.assertFalse(artists.line.get_visible())

    def test_multiple_observers_do_not_overwrite_one_another(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig(show_candidate_paths=False))
        states = {"ego": self.ego.state, "front": self.front.state,
                  "third": observation(self.track, "third", 10, 0, 4).state}
        visualizer.update_vehicles(states, {key: [value] for key, value in states.items()},
                                   {key: None for key in states},
                                   candidate_snapshots={"ego": self.snapshot, "third": self.snapshot})
        self.assertEqual(set(visualizer._gp_prediction_artists), {("ego", "front"), ("third", "front")})
        first = visualizer._gp_prediction_artists[("ego", "front")]
        second = visualizer._gp_prediction_artists[("third", "front")]
        self.assertIsNot(first.line, second.line)
        self.assertNotEqual(first.label.get_text(), second.label.get_text())

    def test_full_lap_samples_and_plot_share_the_same_gp_values(self) -> None:
        samples = sample_gp_route(self.track, self.route, 100)
        self.assertEqual(len(samples.s), 100)
        self.assertEqual(samples.s[0], 0.0)
        self.assertLess(samples.s[-1], track_length(self.track))
        np.testing.assert_allclose(samples.d, 0.2)
        np.testing.assert_allclose(samples.points, track_to_world_points(samples.s, samples.d, self.track))
        figure = create_gp_route_figure(self.track, samples)
        map_line = next(line for line in figure.axes[0].lines if line.get_label() == "GP mean route")
        np.testing.assert_allclose(np.column_stack(map_line.get_data())[:-1], samples.points)
        np.testing.assert_allclose(np.column_stack(map_line.get_data())[-1], samples.points[0])
        np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), samples.d)
        with self.assertRaises(ValueError):
            sample_gp_route(self.track, self.route, 2)

    def test_offline_command_exports_png_samples_and_track_metadata(self) -> None:
        cache = PROJECT_ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            output = Path(directory) / "route.png"
            model = Path(directory) / "model.pkl"
            model.write_bytes(b"placeholder; model loader is replaced in this test")
            with patch("plot_gp_route.GpRoute.load", return_value=self.route), \
                 patch("main.create_demo_track", return_value=self.track), redirect_stdout(io.StringIO()):
                self.assertEqual(plot_gp_route.main(["--model", str(model), "--points", "17",
                                                     "--output", str(output)]), 0)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))
            with output.with_suffix(".csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 17)
            self.assertEqual(metadata["point_count"], 17)
            self.assertEqual(metadata["track"]["width"], self.track.width)
            np.testing.assert_allclose([float(row["d_m"]) for row in rows], 0.2)


if __name__ == "__main__":
    unittest.main()
