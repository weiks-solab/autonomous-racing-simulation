import unittest
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from core.pure_pursuit import PursuitTarget
from core.track import Track
from core.track_geometry import world_to_track
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from sim.visualizer import SimulationVisualizer, VisualizationConfig
from strategy.base import CandidatePath, CandidateSnapshot
from strategy.offset_path import LateralManeuver


class SimulationVisualizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track.create_oval(4.0, 2.0, 50, width=1.5)
        self.state = VehicleState(4.0, 0.0, 1.57, 1.0)
        self.target = PursuitTarget(0, 2, 3.9, 0.5)
        self.geometry = VehicleGeometry(0.5, 0.3, 0.1)
        self.position = world_to_track(self.state, self.track)
        self.snapshot = CandidateSnapshot(
            evaluated_at=1.0,
            position=self.position,
            distances=(0.0, 0.5, 1.0),
            candidates=(
                CandidatePath(
                    LateralManeuver(self.position.s, self.position.d, 0.3, 1.0),
                    selected=True, race_cost=1.0, blocking_cost=2.0,
                ),
                CandidatePath(
                    LateralManeuver(self.position.s, self.position.d, -0.3, 1.0),
                    name="RIGHT", feasible=False, reason="predicted overlap",
                ),
            ),
        )

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

    def test_strategy_status_and_temporary_path_are_optional(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        temporary = Track(
            self.track.centerline[:10],
            closed=False,
            width=self.track.width,
        )
        self._update(visualizer)
        artists = visualizer._vehicle_artists["vehicle"]
        self.assertFalse(artists.strategy_text.get_visible())
        self.assertFalse(artists.temporary_path.get_visible())

        visualizer.update_vehicles(
            states={"vehicle": self.state},
            trajectories={"vehicle": [self.state]},
            targets={"vehicle": self.target},
            strategy_labels={"vehicle": "OVERTAKE\nside=LEFT"},
            temporary_paths={"vehicle": temporary},
        )
        self.assertTrue(artists.strategy_text.get_visible())
        self.assertTrue(artists.temporary_path.get_visible())

    def test_single_vehicle_update_api_remains_compatible(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        visualizer.update(self.state, [self.state], self.target)
        self.assertIn("vehicle", visualizer._vehicle_artists)

    def test_candidate_costs_rejections_and_timestamp_are_displayed(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        self._update(visualizer, candidate_snapshot=self.snapshot)
        visualizer.figure.canvas.draw()
        artists = visualizer._vehicle_artists["vehicle"]
        selected, rejected = artists.candidate_paths
        self.assertTrue(selected.line.get_visible())
        self.assertEqual(selected.line.get_linestyle(), "-")
        self.assertIn("J=3.00", selected.label.get_text())
        self.assertTrue(selected.label.get_text().endswith("*"))
        self.assertEqual(rejected.line.get_linestyle(), ":")
        self.assertIn("RIGHT", rejected.label.get_text())
        self.assertIn("predicted overlap", rejected.label.get_text())
        self.assertEqual(artists.candidate_time.get_text(), "candidates @ 1.00s")

    def test_candidate_snapshots_reuse_artists_and_clear_old_lines(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        self._update(visualizer, candidate_snapshot=self.snapshot)
        artists = visualizer._vehicle_artists["vehicle"]
        first_line = artists.candidate_paths[0].line
        original_points = np.array(first_line.get_data())
        line_count, text_count = len(visualizer.axes.lines), len(visualizer.axes.texts)
        self.state = replace(self.state, x=3.9, y=0.5)
        self._update(visualizer, candidate_snapshot=self.snapshot)
        np.testing.assert_array_equal(first_line.get_data(), original_points)
        smaller = replace(self.snapshot, evaluated_at=1.2, candidates=self.snapshot.candidates[:1])
        self._update(visualizer, candidate_snapshot=smaller)
        self.assertFalse(artists.candidate_paths[1].line.get_visible())
        self.assertFalse(artists.candidate_paths[1].label.get_visible())
        self._update(visualizer)
        self.assertFalse(artists.candidate_time.get_visible())
        self.assertTrue(all(not item.line.get_visible() for item in artists.candidate_paths))
        self.assertTrue(all(not item.label.get_visible() for item in artists.candidate_paths))
        self._update(visualizer, candidate_snapshot=self.snapshot)
        self.assertIs(artists.candidate_paths[0].line, first_line)
        self.assertEqual((len(visualizer.axes.lines), len(visualizer.axes.texts)),
                         (line_count, text_count))

    def test_candidate_paths_and_labels_can_be_hidden(self) -> None:
        for paths, labels in ((False, True), (True, False)):
            with self.subTest(paths=paths, labels=labels):
                visualizer = SimulationVisualizer(
                    self.track,
                    VisualizationConfig(show_candidate_paths=paths, show_candidate_labels=labels),
                )
                self._update(visualizer, candidate_snapshot=self.snapshot)
                artists = visualizer._vehicle_artists["vehicle"]
                self.assertTrue(all(item.line.get_visible() == paths
                                    for item in artists.candidate_paths))
                self.assertTrue(all(not item.label.get_visible() for item in artists.candidate_paths))
                self.assertFalse(artists.candidate_time.get_visible())

    def test_candidate_snapshots_reject_unknown_vehicle_ids(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        with self.assertRaisesRegex(ValueError, "unknown vehicle ID"):
            visualizer.update_vehicles(
                states={"vehicle": self.state}, trajectories={"vehicle": [self.state]},
                targets={"vehicle": None}, candidate_snapshots={"other": self.snapshot},
            )

    def test_rejected_cost_candidate_shows_collision_reason_and_no_selection(self) -> None:
        rejected = replace(
            self.snapshot.candidates[0], selected=False, feasible=False,
            reason="predicted overlap with other @ 0.40s", cost_terms=(("progress", -3.0),),
        )
        snapshot = replace(self.snapshot, candidates=(rejected,))
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        self._update(visualizer, candidate_snapshot=snapshot)
        visualizer.figure.canvas.draw()
        artists = visualizer._vehicle_artists["vehicle"].candidate_paths[0]
        self.assertEqual(artists.line.get_linestyle(), ":")
        self.assertEqual(artists.line.get_color(), "0.5")
        self.assertIn("other @ 0.40s", artists.label.get_text())
        self.assertNotIn("J=", artists.label.get_text())
        self.assertFalse(artists.label.get_text().endswith("*"))

    def _update(
        self,
        visualizer: SimulationVisualizer,
        collision_pairs: set[tuple[str, str]] | None = None,
        include_other: bool = False,
        candidate_snapshot: CandidateSnapshot | None = None,
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
            candidate_snapshots=None if candidate_snapshot is None else {"vehicle": candidate_snapshot},
        )


if __name__ == "__main__":
    unittest.main()
