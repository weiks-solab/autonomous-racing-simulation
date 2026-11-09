"""確認主程式與訓練資料入口均可指派多台獨立的純進度車。"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import main
from generate_training_data import _demo_settings, _run_metadata, _select_pairs
from sim.data_generation import (
    EpisodeScenario, RandomizationConfig, VehicleInitialCondition, run_episode,
)
from strategy.assignment import StrategyAssignment
from training_data import TrainingDataCollector, VehiclePair


class ProgressIntegrationTests(unittest.TestCase):
    def run_scene(self, assignments: dict[str, StrategyAssignment], rear_s: float = 0.0) -> tuple[dict, str]:
        specs = (("rear", rear_s, 6.0), ("front", 4.0, 5.0), ("follower", 20.0, 4.0))
        output = io.StringIO()
        with patch("main.VEHICLE_SPECS", specs), \
             patch("main.STRATEGY_ASSIGNMENTS", assignments), \
             patch("main.SIMULATION_SECONDS", 0.12), \
             patch("main.SimulationVisualizer") as visualizer, redirect_stdout(output):
            main.run_simulation()
        return visualizer.return_value.update_vehicles.call_args.kwargs, output.getvalue()

    def test_front_progress_trajectory_is_unchanged_when_another_vehicle_overlaps(self) -> None:
        assignments = {"front": StrategyAssignment("progress")}
        clear, _ = self.run_scene(assignments)
        overlap, output = self.run_scene(assignments, rear_s=4.0)
        self.assertEqual(clear["trajectories"]["front"], overlap["trajectories"]["front"])
        self.assertEqual(clear["candidate_snapshots"]["front"], overlap["candidate_snapshots"]["front"])
        self.assertIn("front strategy: progress", output)
        self.assertIn("Detected collision pairs: front/rear", output)
        self.assertNotIn("follower", overlap["candidate_snapshots"])

    def test_multiple_progress_vehicles_keep_independent_states_and_speed_configs(self) -> None:
        config = replace(main.PROGRESS_CONFIG, prediction_horizon=2.0)
        with patch("main.PROGRESS_CONFIG", config):
            frame, _ = self.run_scene({
                "rear": StrategyAssignment("progress"),
                "follower": StrategyAssignment("progress"),
                "front": StrategyAssignment("cost_overtake"),
            })
        rear = frame["candidate_snapshots"]["rear"]
        follower = frame["candidate_snapshots"]["follower"]
        self.assertIsNot(rear, follower)
        self.assertNotEqual(rear.position, follower.position)
        self.assertGreater(rear.candidates[0].predicted_states[-1].velocity,
                           follower.candidates[0].predicted_states[-1].velocity)
        for vehicle_id, snapshot in (("rear", rear), ("follower", follower)):
            step = round(snapshot.evaluated_at / main.MODEL_CONFIG.time_step)
            state = frame["trajectories"][vehicle_id][step]
            self.assertTrue(all(item.predicted_states[0] == state for item in snapshot.candidates))
            self.assertTrue(all(len(item.cost_terms) == 1 for item in snapshot.candidates))

    def test_training_entry_copies_progress_config_and_records_multiple_progress_vehicles(self) -> None:
        config = replace(main.PROGRESS_CONFIG, transition_length=5.0, prediction_horizon=0.2)
        with patch("main.PROGRESS_CONFIG", config):
            settings, _, _ = _demo_settings(no_defense=True)
        self.assertEqual(settings.progress_config, config)
        assignments = {name: StrategyAssignment("progress") for name in ("ego", "opponent")}
        scenario = EpisodeScenario("progress", (
            VehicleInitialCondition("ego", 0.0, 0.0, 4.0, 6.0),
            VehicleInitialCondition("opponent", 4.0, 0.0, 3.0, 5.0),
        ), assignments)
        collector = TrainingDataCollector(settings.model_config.time_step)
        run_episode(scenario, settings, collector, 0.04)
        frames = collector.episodes[0].frames
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(item.maneuver is not None for item in frames[-1].observations.values()))
        pairs = _select_pairs(None, ("ego", "opponent"), assignments)
        self.assertEqual(set(pairs), {VehiclePair("ego", "opponent"), VehiclePair("opponent", "ego")})
        metadata = _run_metadata(settings, (scenario,), pairs, 42, 0.04, RandomizationConfig())
        self.assertEqual(metadata["simulation_configs"]["progress_config"]["transition_length"], 5.0)


if __name__ == "__main__":
    unittest.main()
