"""相對進度策略可獨立指派，與原有策略及訓練資料入口共存。"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import main
from core.track_geometry import signed_longitudinal_gap, track_length, world_to_track
from generate_training_data import _demo_settings, _run_metadata, _select_pairs
from sim.data_generation import (
    EpisodeScenario, RandomizationConfig, VehicleInitialCondition, run_episode,
)
from strategy.assignment import StrategyAssignment
from training_data import TrainingDataCollector, VehiclePair


class RelativeOvertakingIntegrationTests(unittest.TestCase):
    def run_scene(self, assignments: dict[str, StrategyAssignment], overlap: bool = False) -> tuple[dict, str]:
        specs = (("rear", 0.0, 6.0), ("front", 0.0 if overlap else 4.0, 5.0),
                 ("third", 20.0, 4.0))
        output = io.StringIO()
        with patch("main.VEHICLE_SPECS", specs), patch("main.STRATEGY_ASSIGNMENTS", assignments), \
             patch("main.SIMULATION_SECONDS", 0.14), \
             patch("main.SimulationVisualizer") as visualizer, redirect_stdout(output):
            main.run_simulation()
        return visualizer.return_value.update_vehicles.call_args.kwargs, output.getvalue()

    def test_new_strategy_coexists_with_progress_and_original_cost_attack(self) -> None:
        frame, output = self.run_scene({
            "rear": StrategyAssignment("relative_overtake"),
            "front": StrategyAssignment("progress"),
            "third": StrategyAssignment("cost_overtake"),
        })
        for vehicle_id, name in (("rear", "relative_overtake"), ("front", "progress"),
                                 ("third", "cost_overtake")):
            self.assertIn(f"{vehicle_id} strategy: {name}", output)
        snapshot = frame["candidate_snapshots"]["rear"]
        self.assertEqual(len(snapshot.candidates), 9)
        self.assertEqual(sum(item.selected for item in snapshot.candidates), 1)
        self.assertTrue(all({name for name, _ in item.cost_terms} == {"relative_gap"}
                            for item in snapshot.candidates if item.feasible))
        self.assertTrue(all({name for name, _ in item.cost_terms} == {"progress", "pass"}
                            for item in frame["candidate_snapshots"]["third"].candidates if item.feasible))

    def test_two_assigned_vehicles_have_independent_predictions_and_controller_configs(self) -> None:
        config = replace(main.RELATIVE_OVERTAKING_CONFIG, prediction_horizon=2.0)
        with patch("main.RELATIVE_OVERTAKING_CONFIG", config):
            frame, _ = self.run_scene({
                "rear": StrategyAssignment("relative_overtake"),
                "third": StrategyAssignment("relative_overtake"),
            })
        rear, third = (frame["candidate_snapshots"][name] for name in ("rear", "third"))
        self.assertIsNot(rear, third)
        self.assertNotEqual(rear.position, third.position)
        self.assertGreater(rear.candidates[0].predicted_states[-1].velocity,
                           third.candidates[0].predicted_states[-1].velocity)
        for vehicle_id, snapshot in (("rear", rear), ("third", third)):
            step = round(snapshot.evaluated_at / main.MODEL_CONFIG.time_step)
            self.assertTrue(all(item.predicted_states[0] == frame["trajectories"][vehicle_id][step]
                                for item in snapshot.candidates))

    def test_all_rejected_keeps_simulation_running_and_reports_actual_overlap(self) -> None:
        frame, output = self.run_scene({"rear": StrategyAssignment("relative_overtake")}, overlap=True)
        self.assertIn("NO_FEASIBLE_PATH", frame["strategy_labels"]["rear"])
        self.assertIn("Detected collision pairs: front/rear", output)
        self.assertTrue(all(not item.selected and not item.feasible
                            for item in frame["candidate_snapshots"]["rear"].candidates))
        self.assertEqual(len(frame["trajectories"]["rear"]), 8)

    def test_relative_cost_stays_active_after_the_actual_overtake(self) -> None:
        specs = (("rear", 0.0, 6.0), ("front", 1.5, 3.0))
        assignments = {"rear": StrategyAssignment("relative_overtake"),
                       "front": StrategyAssignment("progress")}
        with patch("main.VEHICLE_SPECS", specs), patch("main.STRATEGY_ASSIGNMENTS", assignments), \
             patch("main.SIMULATION_SECONDS", 4.0), patch("main.SimulationVisualizer") as visualizer, \
             redirect_stdout(io.StringIO()):
            main.run_simulation()
        frame = visualizer.return_value.update_vehicles.call_args.kwargs
        track = visualizer.call_args.args[0]
        gap = signed_longitudinal_gap(world_to_track(frame["states"]["rear"], track),
                                      world_to_track(frame["states"]["front"], track), track_length(track))
        self.assertLess(gap, -main.RELATIVE_OVERTAKING_CONFIG.detection_range)
        self.assertIn("ATTACK", frame["strategy_labels"]["rear"])
        self.assertIn("target=front", frame["strategy_labels"]["rear"])
        candidates = frame["candidate_snapshots"]["rear"].candidates
        costs = [item.cost_terms[0][1] for item in candidates if item.feasible]
        selected = next(item for item in candidates if item.selected)
        self.assertLess(selected.cost_terms[0][1], 0.0)
        self.assertEqual(selected.cost_terms[0][1], min(costs))
        self.assertGreater(max(costs) - min(costs), 1e-6)

    def test_training_entry_copies_config_and_retains_progress_opponent_pairing(self) -> None:
        config = replace(main.RELATIVE_OVERTAKING_CONFIG, transition_length=5.0, prediction_horizon=0.2)
        with patch("main.RELATIVE_OVERTAKING_CONFIG", config):
            settings, _, _ = _demo_settings(no_defense=False)
        self.assertEqual(settings.relative_overtaking_config, config)
        vehicles = (
            VehicleInitialCondition("ego", 0.0, 0.0, 4.0, 6.0),
            VehicleInitialCondition("opponent", 4.0, 0.0, 3.0, 5.0),
        )
        assignments = {"ego": StrategyAssignment("relative_overtake"),
                       "opponent": StrategyAssignment("progress")}
        scenario = EpisodeScenario("relative", vehicles, assignments)
        collector = TrainingDataCollector(settings.model_config.time_step)
        run_episode(scenario, settings, collector, 0.04)
        self.assertEqual(len(collector.episodes[0].frames), 3)
        pairs = _select_pairs(None, ("ego", "opponent"), assignments)
        self.assertEqual(pairs, (VehiclePair("ego", "opponent"),))
        metadata = _run_metadata(settings, (scenario,), pairs, 42, 0.04, RandomizationConfig())
        self.assertEqual(metadata["simulation_configs"]["relative_overtaking_config"]["transition_length"], 5.0)
        self.assertEqual(_select_pairs(None, ("ego", "opponent"),
                                      {"ego": StrategyAssignment("relative_overtake")}), pairs)


if __name__ == "__main__":
    unittest.main()
