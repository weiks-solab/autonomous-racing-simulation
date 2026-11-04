"""候選快照必須對應實際評估時刻，並保留原策略結果。"""

from dataclasses import replace
import unittest

import numpy as np

from core.track import Track
from core.track_geometry import track_to_world
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from sim.candidate_paths import candidate_path_points
from strategy.assignment import DefendingStrategy, OvertakingStrategy
from strategy.defending import DefenseConfig
from strategy.observations import VehicleObservation, create_vehicle_observation
from strategy.overtaking import OvertakingConfig


class CandidateSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))), width=2.0)
        self.geometry = VehicleGeometry(0.55, 0.3, 0.12)

    def observation(self, vehicle_id: str, s: float, d: float, speed: float) -> VehicleObservation:
        point = track_to_world(s, d, self.track)
        return create_vehicle_observation(
            vehicle_id, VehicleState(float(point[0]), float(point[1]), 0.0, speed),
            self.geometry, self.track,
        )

    def test_defense_snapshot_updates_only_when_costs_are_recomputed(self) -> None:
        strategy = DefendingStrategy(DefenseConfig(), "rear")
        defender = self.observation("front", 20.0, 0.0, 3.0)
        rear = self.observation("rear", 14.0, 0.6, 5.0)
        first = strategy.update(defender, (rear,), self.track, 0.0)
        snapshot = first.candidate_snapshot
        self.assertEqual(len(snapshot.candidates), 5)
        selected = next(candidate for candidate in snapshot.candidates if candidate.selected)
        self.assertEqual(selected.maneuver, first.maneuver)

        moved = self.observation("front", 21.0, 0.2, 4.0)
        held = strategy.update(moved, (rear,), self.track, 0.1)
        self.assertIs(held.candidate_snapshot, snapshot)
        self.assertEqual(snapshot.position, defender.track_position)
        self.assertEqual(snapshot.distances, ())
        self.assertEqual(len(selected.predicted_states), 11)
        self.assertIs(selected.predicted_states[0], defender.state)

        updated = strategy.update(moved, (rear,), self.track, 0.3)
        self.assertIsNot(updated.candidate_snapshot, snapshot)
        self.assertEqual(updated.candidate_snapshot.evaluated_at, 0.3)
        self.assertEqual(updated.candidate_snapshot.position, moved.track_position)
        self.assertTrue(all(candidate.predicted_states[0] is moved.state
                            for candidate in updated.candidate_snapshot.candidates))

    def test_defense_drawn_samples_match_the_scored_path(self) -> None:
        config = DefenseConfig(
            prediction_horizon=0.55, prediction_step=0.25, collision_check_enabled=False,
        )
        strategy = DefendingStrategy(config, "rear")
        defender = self.observation("front", 20.0, 0.0, 3.0)
        rear = self.observation("rear", 14.0, 0.6, 5.0)
        strategy.update(defender, (rear,), self.track, 0.0)
        moved = self.observation("front", 21.0, 0.2, 3.0)
        result = strategy.update(moved, (rear,), self.track, 0.3)
        snapshot = result.candidate_snapshot
        self.assertEqual(snapshot.distances, ())
        for candidate in snapshot.candidates:
            points = candidate_path_points(snapshot, candidate, self.track)
            offsets = points[:, 1]
            self.assertAlmostEqual(offsets[0], moved.track_position.d)
            self.assertAlmostEqual(
                candidate.race_cost,
                -config.progress_weight * (points[-1, 0] - moved.state.x)
                + config.smooth_weight * np.sum(np.diff(offsets) ** 2),
            )
            self.assertAlmostEqual(
                candidate.blocking_cost,
                config.q_block * np.sum((offsets[1:] - rear.track_position.d) ** 2),
            )

    def test_attack_snapshot_stays_fixed_while_side_is_locked_and_clears_on_return(self) -> None:
        strategy = OvertakingStrategy(OvertakingConfig())
        attacker = self.observation("rear", 20.0, 0.0, 5.0)
        target = self.observation("front", 24.0, 0.0, 3.0)
        preparing = strategy.update(attacker, (target,), self.track, 0.0)
        self.assertIsNone(preparing.candidate_snapshot)
        passing = strategy.update(attacker, (target,), self.track, 0.02)
        self.assertEqual(passing.mode, "overtake")
        snapshot = passing.candidate_snapshot
        self.assertEqual(snapshot.evaluated_at, 0.02)
        self.assertEqual({candidate.name for candidate in snapshot.candidates}, {"LEFT", "RIGHT"})
        self.assertEqual(sum(candidate.selected for candidate in snapshot.candidates), 1)
        self.assertTrue(all(candidate.race_cost is None for candidate in snapshot.candidates))
        selected = next(candidate for candidate in snapshot.candidates if candidate.selected)
        self.assertEqual(selected.maneuver, passing.maneuver)

        moved = self.observation("rear", 21.0, 0.1, 5.0)
        continued = strategy.update(moved, (target,), self.track, 0.5)
        self.assertIs(continued.candidate_snapshot, snapshot)
        self.assertEqual(snapshot.position, attacker.track_position)
        returned = strategy.update(
            self.observation("rear", 30.0, 0.6, 5.0), (target,), self.track, 1.0,
        )
        self.assertEqual(returned.mode, "return")
        self.assertIsNone(returned.candidate_snapshot)
        following = strategy.update(
            self.observation("rear", 31.0, 0.0, 5.0), (target,), self.track, 1.2,
        )
        self.assertIsNone(following.candidate_snapshot)

    def test_rejected_attack_candidates_are_kept_and_refresh_during_prepare(self) -> None:
        config = OvertakingConfig(prediction_horizon=0.55, prediction_step=0.25)
        strategy = OvertakingStrategy(config)
        track = replace(self.track, width=0.6)
        attacker = self.observation("rear", 20.0, 0.0, 5.0)
        target = self.observation("front", 24.0, 0.0, 3.0)
        strategy.update(attacker, (target,), track, 0.0)
        result = strategy.update(attacker, (target,), track, 0.02)
        snapshot = result.candidate_snapshot
        self.assertEqual(result.mode, "prepare_overtake")
        np.testing.assert_allclose(snapshot.distances, (0.0, 1.25, 2.5, 2.75))
        self.assertEqual(len(snapshot.candidates), 2)
        for candidate in snapshot.candidates:
            self.assertFalse(candidate.selected)
            self.assertFalse(candidate.feasible)
            self.assertEqual(candidate.reason, "track width exceeded")
        moved = self.observation("rear", 20.1, 0.0, 5.0)
        refreshed = strategy.update(moved, (target,), track, 0.04)
        self.assertIsNot(refreshed.candidate_snapshot, snapshot)
        self.assertEqual(refreshed.candidate_snapshot.position, moved.track_position)

    def test_candidate_drawing_wraps_at_finish_line(self) -> None:
        strategy = DefendingStrategy(DefenseConfig(collision_check_enabled=False), "rear")
        defender = self.observation("front", 399.0, 0.0, 3.0)
        rear = self.observation("rear", 394.0, 0.6, 5.0)
        result = strategy.update(defender, (rear,), self.track, 0.0)
        snapshot = result.candidate_snapshot
        center = next(candidate for candidate in snapshot.candidates
                      if candidate.maneuver.target_d == 0.0)
        points = candidate_path_points(snapshot, center, self.track)
        np.testing.assert_allclose(points[-1], (3.5, 0.0), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
