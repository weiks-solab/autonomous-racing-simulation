import unittest

import matplotlib.pyplot as plt

from core.pure_pursuit import PursuitTarget
from core.track import Track
from core.track_geometry import world_to_track
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from sim.visualizer import SimulationVisualizer, VisualizationConfig


class SimulationVisualizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track.create_oval(4.0, 2.0, 50, width=1.5)
        self.state = VehicleState(4.0, 0.0, 1.57, 1.0)
        self.target = PursuitTarget(0, 2, 3.9, 0.5)
        self.geometry = VehicleGeometry(0.5, 0.3, 0.1)
        self.position = world_to_track(self.state, self.track)

    def tearDown(self) -> None:
        plt.close("all")

    def test_all_dynamic_elements_can_be_hidden(self) -> None:
        visualizer = SimulationVisualizer(
            self.track,
            VisualizationConfig(
                show_vehicle_heading=False,
                show_vehicle_footprints=False,
                show_track_positions=False,
                show_vehicle_markers=False,
                show_trajectories=False,
                show_pursuit_targets=False,
            ),
        )
        self._update(visualizer)
        artists = visualizer._vehicle_artists["vehicle"]
        self.assertFalse(artists.trajectory.get_visible())
        self.assertFalse(artists.marker.get_visible())
        self.assertFalse(artists.target.get_visible())
        self.assertFalse(artists.heading.get_visible())
        self.assertFalse(artists.footprint.get_visible())
        self.assertFalse(artists.track_position_text.get_visible())
        self.assertFalse(artists.collision_text.get_visible())

    def test_all_dynamic_elements_can_be_shown(self) -> None:
        visualizer = SimulationVisualizer(
            self.track,
            VisualizationConfig(show_track_positions=True),
        )
        self._update(visualizer)
        artists = visualizer._vehicle_artists["vehicle"]
        self.assertTrue(artists.trajectory.get_visible())
        self.assertTrue(artists.marker.get_visible())
        self.assertTrue(artists.target.get_visible())
        self.assertTrue(artists.heading.get_visible())
        self.assertTrue(artists.footprint.get_visible())
        self.assertTrue(artists.track_position_text.get_visible())
        self.assertFalse(artists.collision_text.get_visible())

    def test_collision_turns_rectangle_red_and_shows_status(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        self._update(visualizer, collision_pairs={("vehicle", "other")}, include_other=True)
        artists = visualizer._vehicle_artists["vehicle"]
        self.assertTrue(artists.collision_text.get_visible())
        self.assertEqual(
            artists.collision_text.get_text(),
            "COLLISION",
        )
        self.assertGreater(artists.footprint.get_facecolor()[0], 0.9)

    def test_collision_status_clears_on_the_next_safe_update(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        self._update(visualizer, collision_pairs={("vehicle", "other")}, include_other=True)
        artists = visualizer._vehicle_artists["vehicle"]
        self._update(visualizer, collision_pairs=set(), include_other=True)
        self.assertFalse(artists.collision_text.get_visible())
        self.assertEqual(artists.footprint.get_alpha(), 0.25)

    def test_updates_reuse_existing_artists(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        self._update(visualizer)
        artists_before = visualizer._vehicle_artists["vehicle"]
        self._update(visualizer)
        self.assertIs(
            visualizer._vehicle_artists["vehicle"],
            artists_before,
        )

    def test_single_vehicle_update_api_remains_compatible(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        visualizer.update(self.state, [self.state], self.target)
        self.assertIn("vehicle", visualizer._vehicle_artists)

    def _update(
        self,
        visualizer: SimulationVisualizer,
        collision_pairs: set[tuple[str, str]] | None = None,
        include_other: bool = False,
    ) -> None:
        states = {"vehicle": self.state}
        trajectories = {"vehicle": [self.state]}
        targets = {"vehicle": self.target}
        geometries = {"vehicle": self.geometry}
        positions = {"vehicle": self.position}
        if include_other:
            other_state = VehicleState(-4.0, 0.0, -1.57, 1.0)
            states["other"] = other_state
            trajectories["other"] = [other_state]
            targets["other"] = None
            geometries["other"] = self.geometry
            positions["other"] = world_to_track(other_state, self.track)
        visualizer.update_vehicles(
            states=states,
            trajectories=trajectories,
            targets=targets,
            geometries=geometries,
            track_positions=positions,
            collision_pairs=collision_pairs,
        )


if __name__ == "__main__":
    unittest.main()
