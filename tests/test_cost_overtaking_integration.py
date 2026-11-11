"""確認成本式進攻可由策略表替換，且各車配置與觀測保持獨立。"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import main
from strategy.assignment import StrategyAssignment


class CostOvertakingIntegrationTests(unittest.TestCase):
    def run_scene(
        self, assignments: dict, defense_enabled: bool = True,
        specs: tuple[tuple[str, float, float], ...] | None = None,
    ) -> tuple[dict, str]:
        if specs is None:
            specs = (("red", 0.0, 6.0), ("blue", 6.0, 5.0), ("green", -15.0, 4.0))
        output = io.StringIO()
        config = replace(main.COST_OVERTAKING_CONFIG,
                         candidate_offsets=(-0.6, -0.3, 0.0, 0.3, 0.6))
        with patch("main.VEHICLE_SPECS", specs), \
             patch("main.STRATEGY_ASSIGNMENTS", assignments), \
             patch("main.COST_OVERTAKING_CONFIG", config), \
             patch("main.SIMULATION_SECONDS", 0.4), \
             patch("main.SimulationVisualizer") as visualizer, redirect_stdout(output):
            main.run_simulation(defense_enabled=defense_enabled)
        return visualizer.return_value.update_vehicles.call_args.kwargs, output.getvalue()

    def test_strategy_table_switches_between_cost_and_original_attack(self) -> None:
        cost, output = self.run_scene({"red": StrategyAssignment("cost_overtake")})
        self.assertIn("red strategy: cost_overtake", output)
        snapshot = cost["candidate_snapshots"]["red"]
        self.assertEqual(len(snapshot.candidates), 5)
        self.assertEqual(sum(candidate.selected for candidate in snapshot.candidates), 1)
        for candidate in snapshot.candidates:
            self.assertGreater(len(candidate.predicted_states), 1)
            if candidate.feasible:
                self.assertEqual({name for name, _ in candidate.cost_terms}, {"progress", "pass"})
            else:
                self.assertFalse(candidate.selected)
                self.assertIn("predicted overlap", candidate.reason)

        original, output = self.run_scene({"red": StrategyAssignment("overtake")})
        self.assertIn("red strategy: overtake", output)
        self.assertNotIn("red", original["candidate_snapshots"])

    def test_multiple_cost_attackers_keep_their_own_speed_and_state(self) -> None:
        # 此案例要觀察接近各車目標速度的狀態，不依賴示範預測時長。
        config = replace(main.COST_OVERTAKING_CONFIG, prediction_horizon=5.0)
        with patch("main.COST_OVERTAKING_CONFIG", config):
            frame, _ = self.run_scene({
                "red": StrategyAssignment("cost_overtake"),
                "green": StrategyAssignment("cost_overtake"),
            })
        red, green = frame["candidate_snapshots"]["red"], frame["candidate_snapshots"]["green"]
        self.assertIsNot(red, green)
        self.assertNotEqual(red.position, green.position)
        self.assertGreater(red.candidates[0].predicted_states[-1].velocity, 5.9)
        self.assertLess(green.candidates[0].predicted_states[-1].velocity, 4.01)
        for vehicle_id, snapshot in (("red", red), ("green", green)):
            step = round(snapshot.evaluated_at / main.MODEL_CONFIG.time_step)
            actual_state = frame["trajectories"][vehicle_id][step]
            self.assertEqual(snapshot.candidates[0].predicted_states[0], actual_state)

    def test_disabling_defense_preserves_cost_attack_and_independent_follower(self) -> None:
        assignments = {
            "red": StrategyAssignment("cost_overtake"),
            "blue": StrategyAssignment("defend", target_vehicle_id="red"),
        }
        active, _ = self.run_scene(assignments)
        passive, output = self.run_scene(assignments, defense_enabled=False)
        self.assertEqual(active["trajectories"]["green"], passive["trajectories"]["green"])
        self.assertIn("red strategy: cost_overtake", output)
        self.assertIn("blue strategy: centerline", output)
        self.assertIn("red", passive["candidate_snapshots"])
        self.assertNotIn("blue", passive["candidate_snapshots"])

    def test_all_candidates_rejected_does_not_stop_simulation(self) -> None:
        frame, output = self.run_scene(
            {"red": StrategyAssignment("cost_overtake")},
            specs=(("red", 0.0, 6.0), ("blue", 0.0, 5.0)),
        )
        self.assertIn("NO_FEASIBLE_PATH", frame["strategy_labels"]["red"])
        self.assertIn("continuing previous path", output)
        self.assertIn("Detected collision pairs: blue/red", output)
        for trajectory in frame["trajectories"].values():
            self.assertEqual(len(trajectory), 1 + round(0.4 / main.MODEL_CONFIG.time_step))
            self.assertGreater(trajectory[-1].velocity, 0.0)
        snapshot = frame["candidate_snapshots"]["red"]
        self.assertTrue(all(not item.feasible and not item.selected for item in snapshot.candidates))


if __name__ == "__main__":
    unittest.main()
