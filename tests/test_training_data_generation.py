from dataclasses import replace
import math
import unittest

from core.pure_pursuit import PurePursuitConfig
from core.track import Track
from core.track_geometry import world_to_track
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_model import VehicleModelConfig
from sim.data_generation import (
    EpisodeScenario,
    SimulationSettings,
    VehicleInitialCondition,
    randomized_scenarios,
    run_episode,
)
from strategy.assignment import StrategyAssignment
from strategy.offset_path import OffsetPathConfig
from strategy.overtaking import OvertakingConfig
from training_data import TrainingDataCollector, VehiclePair, build_dataset


class TrainingEpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimulationSettings(
            Track.create_oval(10.0, 5.0, 120),
            VehicleModelConfig(0.33, 0.02, math.radians(50.0), 0.0, 10.0),
            VehicleGeometry(0.55, 0.3, 0.12),
            PurePursuitConfig(2.0, 5.0, 1.5, 3.0, math.radians(50.0)),
            offset_path_config=OffsetPathConfig(path_length=4.0),
        )
        self.vehicles = (
            VehicleInitialCondition("attacker", 0.0, 0.3, 6.0, 7.0),
            VehicleInitialCondition("defender", 4.0, -0.1, 5.0, 5.0),
            VehicleInitialCondition("follower", 20.0, 0.0, 3.0, 3.0),
        )
        self.pair = VehiclePair("attacker", "defender")

    def test_headless_episode_records_all_vehicles_and_true_model_transition(self) -> None:
        scenario = EpisodeScenario("three-cars", self.vehicles, {})
        collector = TrainingDataCollector(0.02)
        run_episode(scenario, self.settings, collector, 0.08)
        frames = collector.episodes[0].frames
        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[-1].time_seconds, 0.08)
        self.assertEqual(set(frames[-1].observations), {item.vehicle_id for item in self.vehicles})
        for frame in frames:
            for observation in frame.observations.values():
                expected = world_to_track(observation.state, self.settings.track)
                self.assertAlmostEqual(observation.track_position.d, expected.d)
        # Explicit Euler 的第一步位置只由當下速度與朝向決定，可獨立驗證。
        for vehicle_id, current in frames[0].observations.items():
            future = frames[1].observations[vehicle_id].state
            self.assertAlmostEqual(future.x, current.state.x + current.state.velocity * math.cos(current.state.yaw) * 0.02)
            self.assertAlmostEqual(future.y, current.state.y + current.state.velocity * math.sin(current.state.yaw) * 0.02)
        dataset = build_dataset(collector.episodes, self.settings.track, (self.pair,))
        self.assertEqual(dataset.X.shape, (4, 4))
        self.assertAlmostEqual(
            dataset.Y[0, 0],
            frames[1].observations["defender"].track_position.d - frames[0].observations["defender"].track_position.d,
        )

    def test_repeated_episodes_reset_strategy_and_vehicle_state(self) -> None:
        assignments = {"attacker": StrategyAssignment("cost_overtake"),
                       "defender": StrategyAssignment("defend", "attacker")}
        collector = TrainingDataCollector(0.02)
        for name in ("first", "second"):
            run_episode(EpisodeScenario(name, self.vehicles, assignments), self.settings, collector, 0.12)
        first, second = collector.episodes
        for left, right in zip(first.frames, second.frames):
            self.assertEqual(left.observations, right.observations)

    def test_original_overtake_with_short_configured_path_still_runs(self) -> None:
        scenario = EpisodeScenario("overtake", self.vehicles, {"attacker": StrategyAssignment("overtake")})
        settings = replace(self.settings, overtaking_config=OvertakingConfig(transition_length=6.0))
        collector = TrainingDataCollector(0.02)
        run_episode(scenario, settings, collector, 0.04)
        self.assertEqual(len(collector.episodes[0].frames), 3)

    def test_disabling_defense_matches_passive_opponent(self) -> None:
        active_assignment = {"defender": StrategyAssignment("defend", "attacker")}
        disabled, passive = TrainingDataCollector(0.02), TrainingDataCollector(0.02)
        run_episode(
            EpisodeScenario("disabled", self.vehicles, active_assignment),
            replace(self.settings, defense_enabled=False), disabled, 0.12,
        )
        run_episode(EpisodeScenario("passive", self.vehicles, {}), self.settings, passive, 0.12)
        for left, right in zip(disabled.episodes[0].frames, passive.episodes[0].frames):
            self.assertEqual(left.observations, right.observations)

    def test_random_initial_conditions_are_seeded_and_multi_vehicle(self) -> None:
        specs = tuple((item.vehicle_id, item.start_s, item.target_speed) for item in self.vehicles)
        first = randomized_scenarios(self.settings, specs, {}, 2, 42)
        second = randomized_scenarios(self.settings, specs, {}, 2, 42)
        self.assertEqual(first, second)
        self.assertNotEqual(first, randomized_scenarios(self.settings, specs, {}, 2, 43))
        self.assertEqual(len(first[0].vehicles), 3)
        self.assertNotEqual(first[0].vehicles, first[1].vehicles)
        collector = TrainingDataCollector(0.02)
        for scenario in first:
            run_episode(scenario, self.settings, collector, 0.04)
        self.assertEqual(len(collector.episodes), 2)

    def test_invalid_initial_footprints_and_time_settings_are_rejected(self) -> None:
        collector = TrainingDataCollector(0.02)
        overlapping = (self.vehicles[0], replace(self.vehicles[0], vehicle_id="other"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            run_episode(EpisodeScenario("overlap", overlapping, {}), self.settings, collector, 0.04)
        outside = (replace(self.vehicles[0], start_d=1.0),)
        with self.assertRaisesRegex(ValueError, "boundaries"):
            run_episode(EpisodeScenario("outside", outside, {}), self.settings, collector, 0.04)
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            run_episode(EpisodeScenario("timing", self.vehicles, {}), self.settings, collector, 0.03)
        with self.assertRaisesRegex(ValueError, "same time_step"):
            run_episode(EpisodeScenario("timing", self.vehicles, {}), self.settings, TrainingDataCollector(0.1), 0.04)


if __name__ == "__main__":
    unittest.main()
