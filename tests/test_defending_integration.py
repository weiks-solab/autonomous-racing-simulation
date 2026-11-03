"""用完整模擬確認防守生效，也保留被動車與多車支援。"""

from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import numpy as np

import main
from core.track import Track
from core.track_geometry import track_length, world_to_track
from strategy.assignment import StrategyAssignment


def run_without_window(defense_enabled: bool) -> tuple[dict, str, Track]:
    output = io.StringIO()
    with patch("main.SimulationVisualizer") as visualizer, redirect_stdout(output):
        main.run_simulation(defense_enabled=defense_enabled)
    final_frame = visualizer.return_value.update_vehicles.call_args.kwargs
    track = visualizer.call_args.args[0]
    return final_frame, output.getvalue(), track


class DefendingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 固定舊策略場景，讓回歸測試不依賴使用者正在調整的示範車數與策略。
        cls.attacker_id, cls.defender_id = "vehicle-1", "vehicle-2"
        specs = (("vehicle-1", 0.0, 6.0), ("vehicle-2", 6.0, 5.0),
                 ("vehicle-3", -20.0, 4.0))
        assignments = {
            "vehicle-1": StrategyAssignment("overtake"),
            "vehicle-2": StrategyAssignment("defend", target_vehicle_id="vehicle-1"),
        }
        with patch("main.VEHICLE_SPECS", specs), patch("main.STRATEGY_ASSIGNMENTS", assignments):
            cls.active, cls.active_output, cls.track = run_without_window(True)
            cls.passive, cls.passive_output, _ = run_without_window(False)

    def test_rule_based_scene_completes_two_laps_and_reports_defense(self) -> None:
        self.assertIn("Detected collision pairs:", self.active_output)
        self.assertIn("DEFEND", self.active_output)
        self.assertIn(self.defender_id, self.active["strategy_labels"])
        positions = [
            world_to_track(state, self.track)
            for state in self.active["trajectories"][self.attacker_id]
        ]
        length = track_length(self.track)
        steps = (np.diff([position.s for position in positions]) + length / 2) % length - length / 2
        self.assertGreater(float(np.sum(steps)) / length, 2.0)
        self.assertLess(max(abs(position.d) for position in positions), main.TRACK_WIDTH / 2)

    def test_existing_collision_is_reported_and_simulation_continues(self) -> None:
        specs = (("rear", 0.0, 6.0), ("front", 0.0, 5.0))
        assignments = {"front": StrategyAssignment("defend", target_vehicle_id="rear")}
        with patch("main.VEHICLE_SPECS", specs), \
             patch("main.STRATEGY_ASSIGNMENTS", assignments), \
             patch("main.SIMULATION_SECONDS", 0.4):
            frame, output, _ = run_without_window(True)
        self.assertIn("Detected collision pairs: front/rear", output)
        for trajectory in frame["trajectories"].values():
            self.assertEqual(len(trajectory), 1 + int(0.4 / main.MODEL_CONFIG.time_step))
            self.assertGreater(trajectory[-1].velocity, trajectory[0].velocity)

    def test_passive_vehicle_is_unchanged_and_defender_actually_moves(self) -> None:
        self.assertEqual(
            self.active["trajectories"]["vehicle-3"],
            self.passive["trajectories"]["vehicle-3"],
        )
        active = self.active["trajectories"][self.defender_id]
        passive = self.passive["trajectories"][self.defender_id]
        maximum_change = max(
            np.hypot(first.x - second.x, first.y - second.y)
            for first, second in zip(active, passive)
        )
        self.assertGreater(maximum_change, 0.1)
        self.assertNotIn(self.defender_id, self.passive["strategy_labels"])
        self.assertIn("Detected collision pairs: none", self.passive_output)

    def test_more_vehicles_can_be_added_without_changing_strategy_code(self) -> None:
        specs = (*main.VEHICLE_SPECS, ("fourth", 15.0, 2.0), ("fifth", 22.0, 2.0))
        with patch("main.VEHICLE_SPECS", specs), patch("main.SIMULATION_SECONDS", 0.4):
            frame, _, _ = run_without_window(True)
        self.assertEqual(set(frame["states"]), {spec[0] for spec in specs})

    def test_strategies_can_be_reassigned_to_arbitrary_vehicle_ids(self) -> None:
        specs = (("red", 0.0, 6.0), ("blue", 6.0, 3.0), ("green", 20.0, 4.0))
        assignments = {
            "red": StrategyAssignment("overtake"),
            "blue": StrategyAssignment("defend", target_vehicle_id="red"),
        }
        with patch("main.VEHICLE_SPECS", specs), \
             patch("main.STRATEGY_ASSIGNMENTS", assignments), \
             patch("main.SIMULATION_SECONDS", 2.0):
            frame, output, _ = run_without_window(True)
        self.assertEqual(set(frame["strategy_labels"]), {"red", "blue"})
        self.assertIn("red strategy: overtake", output)
        self.assertIn("blue strategy: defend", output)
        self.assertIn("green strategy: centerline", output)

    def test_single_centerline_car_needs_no_attacker_or_defender(self) -> None:
        with patch("main.VEHICLE_SPECS", (("solo", 0.0, 3.0),)), \
             patch("main.STRATEGY_ASSIGNMENTS", {}), \
             patch("main.SIMULATION_SECONDS", 0.4):
            frame, output, _ = run_without_window(True)
        self.assertEqual(set(frame["states"]), {"solo"})
        self.assertEqual(frame["strategy_labels"], {})
        self.assertIn("solo strategy: centerline", output)


if __name__ == "__main__":
    unittest.main()
