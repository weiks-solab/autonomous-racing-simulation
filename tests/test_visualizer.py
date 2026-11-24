import unittest
import io
from dataclasses import replace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from core.pure_pursuit import PursuitTarget
from core.track import Track
from core.track_geometry import world_to_track
from core.vehicle_geometry import CollisionConfig, VehicleGeometry, rectangles_overlap
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

    def test_realtime_reuses_background_and_rebuilds_after_view_changes(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig(realtime=True))
        canvas = visualizer.figure.canvas
        with patch("sim.visualizer.plt.pause"), patch.object(canvas, "draw", wraps=canvas.draw) as draw:
            self._update(visualizer, candidate_snapshot=self.snapshot)
            cached = visualizer._background
            self.state = replace(self.state, x=3.8, y=0.2)
            self._update(visualizer, candidate_snapshot=self.snapshot)
            self.assertIs(visualizer._background, cached)
            self.assertEqual(draw.call_count, 1)
            visualizer.axes.set_xlim(-5, 5)
            self._update(visualizer)
            self.assertEqual(draw.call_count, 2)
            visualizer.figure.set_size_inches(8, 6)
            canvas.draw()  # GUI resize normally triggers this full draw.
            self.assertEqual(draw.call_count, 3)
            self._update(visualizer)
            self.assertEqual(draw.call_count, 3)
        plt.ioff()

    def test_blit_save_and_complete_redraw_include_dynamic_artists(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig(realtime=True))
        canvas = visualizer.figure.canvas
        with patch("sim.visualizer.plt.pause"):
            self._update(visualizer, candidate_snapshot=self.snapshot)
            self._update(visualizer)  # Old candidate lines must disappear without ghosting.
            cached_pixels = np.asarray(canvas.buffer_rgba()).copy()
            png = io.BytesIO()
            visualizer.figure.savefig(png, format="png", dpi=visualizer.figure.dpi)
            self.assertIsNone(visualizer._background)
            png.seek(0)
            saved_pixels = np.round(plt.imread(png) * 255).astype(np.uint8)
            visualizer._use_blit = False
            for artist in visualizer._dynamic_artists():
                artist.set_animated(False)
            canvas.draw()
            complete_pixels = np.asarray(canvas.buffer_rgba()).copy()
            np.testing.assert_array_equal(saved_pixels, complete_pixels)
            np.testing.assert_array_equal(cached_pixels, complete_pixels)
        plt.ioff()

    def test_backend_without_blit_retains_original_update_path(self) -> None:
        with patch("matplotlib.backends.backend_agg.FigureCanvasAgg.supports_blit", False):
            visualizer = SimulationVisualizer(self.track, VisualizationConfig(realtime=True))
        with patch("sim.visualizer.plt.pause"), patch.object(visualizer.figure.canvas, "draw_idle") as draw:
            self._update(visualizer)
        draw.assert_called()
        self.assertFalse(visualizer._use_blit)
        plt.ioff()

    def test_blit_processes_events_without_pause_or_repeated_show(self) -> None:
        for interval in (0.001, 0.0):
            with self.subTest(interval=interval):
                visualizer = SimulationVisualizer(
                    self.track, VisualizationConfig(realtime=True, pause_seconds=interval))
                plt.ioff()
                canvas = visualizer.figure.canvas
                with patch("sim.visualizer.plt.pause") as pause, \
                     patch("sim.visualizer.plt.show") as show, \
                     patch.object(canvas, "start_event_loop") as loop, \
                     patch.object(canvas, "draw", wraps=canvas.draw) as draw:
                    self._update(visualizer)
                    self._update(visualizer)
                pause.assert_not_called()
                show.assert_not_called()
                self.assertEqual(draw.call_count, 1)
                self.assertEqual(loop.call_count, 2 if interval > 0 else 0)
                if interval > 0:
                    loop.assert_called_with(interval)

    def test_wait_processes_events_and_skips_non_positive_time(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig(realtime=True))
        plt.ioff()
        canvas = visualizer.figure.canvas
        with patch("sim.visualizer.plt.pause") as pause, \
             patch.object(canvas, "start_event_loop") as loop:
            for seconds in (0.0, -0.01, float("nan")):
                visualizer.wait(seconds)
            loop.assert_not_called()
            visualizer.wait(0.02)
        loop.assert_called_once_with(0.02)
        pause.assert_not_called()

        with patch("matplotlib.backends.backend_agg.FigureCanvasAgg.supports_blit", False):
            fallback = SimulationVisualizer(self.track, VisualizationConfig(realtime=True))
        plt.ioff()
        with patch("sim.visualizer.plt.pause") as pause:
            fallback.wait(0.02)
            fallback.wait(0.0)
        pause.assert_called_once_with(0.02)

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
        self._update(visualizer, collision_pairs={("vehicle", "other")}, include_other=True,
                     other_state=self.state)
        artists = visualizer._vehicle_artists["vehicle"]
        self.assertTrue(artists.collision_text.get_visible())
        self.assertEqual(
            artists.collision_text.get_text(),
            "COLLISION",
        )
        self.assertGreater(artists.footprint.get_facecolor()[0], 0.9)

    def test_collision_status_clears_on_the_next_safe_update(self) -> None:
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        self._update(visualizer, collision_pairs={("vehicle", "other")}, include_other=True,
                     other_state=self.state)
        artists = visualizer._vehicle_artists["vehicle"]
        self._update(visualizer, collision_pairs=set(), include_other=True)
        self.assertFalse(artists.collision_text.get_visible())
        self.assertEqual(artists.footprint.get_alpha(), 0.25)

    def test_only_collision_envelope_overlap_is_orange_then_body_collision_is_red(self) -> None:
        self.geometry = replace(self.geometry, collision=CollisionConfig(0.6, 0.8))
        other = replace(self.state, x=self.state.x + 0.5)
        self.assertTrue(rectangles_overlap(self.state, self.geometry, other, self.geometry))
        body = replace(self.geometry, collision=None)
        self.assertFalse(rectangles_overlap(self.state, body, other, body))
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        self._update(visualizer, collision_pairs={("vehicle", "other")}, include_other=True, other_state=other)
        artists = visualizer._vehicle_artists["vehicle"]
        self.assertEqual(artists.collision_text.get_text(), "ENVELOPE OVERLAP")
        self.assertEqual(artists.collision_text.get_color(), "darkorange")
        self._update(visualizer, collision_pairs={("vehicle", "other")}, include_other=True, other_state=self.state)
        self.assertEqual(artists.collision_text.get_text(), "COLLISION")
        self.assertEqual(artists.collision_text.get_color(), "red")
        self._update(visualizer, collision_pairs=set(), include_other=True)
        self.assertFalse(artists.collision_text.get_visible())

    def test_envelope_switch_hides_orange_but_preserves_body_collision(self) -> None:
        self.geometry = replace(self.geometry, collision=CollisionConfig(0.6, 0.8))
        visualizer = SimulationVisualizer(self.track, VisualizationConfig(show_envelope_overlap=False))
        other = replace(self.state, x=self.state.x + 0.5)
        self._update(visualizer, collision_pairs={("vehicle", "other")}, include_other=True, other_state=other)
        artists = visualizer._vehicle_artists["vehicle"]
        self.assertFalse(artists.collision_text.get_visible())
        np.testing.assert_allclose(artists.footprint.get_facecolor()[:3], artists.color[:3])
        self.assertEqual(artists.footprint.get_alpha(), 0.25)
        self._update(visualizer, collision_pairs={("vehicle", "other")}, include_other=True, other_state=self.state)
        self.assertTrue(artists.collision_text.get_visible())
        self.assertEqual(artists.collision_text.get_text(), "COLLISION")
        self.assertEqual(artists.collision_text.get_color(), "red")

    def test_physical_overlap_is_detected_even_with_smaller_collision_envelope(self) -> None:
        self.geometry = replace(self.geometry, collision=CollisionConfig(0.1, 0.1))
        other = replace(self.state, y=self.state.y + 0.3)
        self.assertFalse(rectangles_overlap(self.state, self.geometry, other, self.geometry))
        visualizer = SimulationVisualizer(self.track, VisualizationConfig())
        self._update(visualizer, collision_pairs=set(), include_other=True, other_state=other)
        self.assertEqual(visualizer._vehicle_artists["vehicle"].collision_text.get_text(), "COLLISION")

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
        other_state: VehicleState | None = None,
    ) -> None:
        states = {"vehicle": self.state}
        trajectories = {"vehicle": [self.state]}
        targets = {"vehicle": self.target}
        geometries = {"vehicle": self.geometry}
        positions = {"vehicle": self.position}
        if include_other:
            other_state = other_state or VehicleState(-4.0, 0.0, -1.57, 1.0)
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
