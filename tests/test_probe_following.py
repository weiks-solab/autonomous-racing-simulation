"""試探跟車的成本、速度選取、跨圈間距與正式執行入口。"""

from contextlib import redirect_stdout
from dataclasses import astuple, replace
import io
import unittest
from unittest.mock import patch

import numpy as np

import main
from core.track import Track
from core.track_factory import TrackConfig, create_track
from core.track_geometry import track_length
from generate_training_data import _demo_settings, _run_metadata, _select_pairs
from sim.data_generation import (
    EpisodeScenario, RandomizationConfig, SimulationSettings, VehicleInitialCondition, run_episode,
)
from strategy.assignment import StrategyAssignment
from strategy.bicycle_prediction import BicyclePrediction, predict_bicycle_trajectory
from strategy.probe_following import ProbeFollowingConfig, ProbeFollowingStrategy, body_progress_bounds
from tests.test_bicycle_prediction import observation, prediction_config
from training_data import TrainingDataCollector, VehiclePair


class ProbeFollowingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.prediction = prediction_config(4.0)
        self.config = ProbeFollowingConfig(target_gap=2.5)
        self.ego = observation(self.track, "rear", 20.0, 0.0, 3.0)
        self.front = observation(self.track, "front", 22.5, 0.0, 3.0)

    def strategy(self, config: ProbeFollowingConfig | None = None) -> ProbeFollowingStrategy:
        return ProbeFollowingStrategy(
            self.config if config is None else config, self.prediction,
            {"front": prediction_config(3.0), "third": prediction_config(3.0)},
        )

    def test_far_gap_selects_catch_up_and_close_gap_selects_lower_speed(self) -> None:
        for front_s, expected in ((25.0, 4.0), (21.4, 2.0)):
            with self.subTest(front_s=front_s):
                strategy = self.strategy()
                front = observation(self.track, "front", front_s, 0.0, 3.0)
                decision = strategy.update(self.ego, (front,), self.track, 0.0)
                self.assertEqual(decision.speed_limit, expected)
                selected = next(candidate for candidate in strategy.last_candidates
                                if candidate.maneuver == decision.maneuver
                                and candidate.speed_limit == decision.speed_limit)
                self.assertEqual(selected.total_cost, min(candidate.total_cost
                                                         for candidate in strategy.last_candidates if candidate.feasible))
                self.assertEqual({name for name, _ in selected.cost_terms}, {"gap", "speed", "probe", "switch"})
                shown = next(candidate for candidate in decision.candidate_snapshot.candidates if candidate.selected)
                self.assertEqual(shown.speed_limit, decision.speed_limit)
                self.assertEqual(shown.predicted_states, selected.prediction.states)

    def test_cost_terms_match_normalized_gap_and_terminal_progress_rate(self) -> None:
        strategy = self.strategy()
        strategy.update(self.ego, (self.front,), self.track, 0.0)
        center = next(candidate for candidate in strategy.last_candidates
                      if candidate.maneuver.target_d == 0.0 and candidate.speed_limit == 3.0)
        terms = dict(center.cost_terms)
        self.assertAlmostEqual(terms["gap"], 0.0, delta=1e-12)
        self.assertAlmostEqual(terms["speed"], 0.0, delta=1e-12)
        self.assertAlmostEqual(terms["probe"], self.config.probe_weight)
        self.assertAlmostEqual(terms["switch"], 0.0)

    def test_cruise_candidate_is_selected_for_a_far_front_vehicle(self) -> None:
        prediction = replace(prediction_config(12.0), model=replace(self.prediction.model, max_velocity=30.0))
        strategy = ProbeFollowingStrategy(self.config, prediction, {"front": prediction_config(10.0)})
        rear = observation(self.track, "rear", 20.0, 0.0, 10.0)
        front = observation(self.track, "front", 28.0, 0.0, 10.0)
        decision = strategy.update(rear, (front,), self.track, 0.0)
        center = [candidate for candidate in strategy.last_candidates if candidate.maneuver.target_d == 0.0]
        self.assertEqual([candidate.speed_limit for candidate in center], [9.0, 10.0, 11.0, 12.0])
        self.assertEqual(decision.speed_limit, 12.0)
        self.assertTrue(next(candidate for candidate in center if candidate.speed_limit == 12.0).feasible)
        self.assertEqual(next(candidate.speed_limit for candidate in decision.candidate_snapshot.candidates
                              if candidate.selected), 12.0)

    def test_cruise_candidate_is_deduplicated_and_never_exceeds_cruise(self) -> None:
        for front_speed, expected in ((0.0, (0.0, 1.0, 4.0)),
                                      (3.0, (2.0, 3.0, 4.0)), (6.0, (4.0,))):
            with self.subTest(front_speed=front_speed):
                strategy = self.strategy()
                front = observation(self.track, "front", 28.0, 0.0, front_speed)
                strategy.update(self.ego, (front,), self.track, 0.0)
                speeds = tuple(candidate.speed_limit for candidate in strategy.last_candidates
                               if candidate.maneuver.target_d == 0.0)
                self.assertEqual(speeds, expected)

    def test_cruise_candidate_still_rejects_predicted_passing(self) -> None:
        prediction = prediction_config(8.0)
        strategy = ProbeFollowingStrategy(self.config, prediction, {"front": prediction_config(3.0)})
        front = observation(self.track, "front", 21.4, 0.0, 3.0)
        decision = strategy.update(self.ego, (front,), self.track, 0.0)
        cruise = [candidate for candidate in strategy.last_candidates if candidate.speed_limit == 8.0]
        self.assertTrue(cruise)
        self.assertTrue(all(not candidate.feasible for candidate in cruise))
        self.assertEqual(decision.speed_limit, 2.0)

    def test_entire_rollout_rejects_passing_even_with_lateral_separation(self) -> None:
        strategy = self.strategy()
        rear = observation(self.track, "rear", 20.0, -0.6, 3.0)
        front = observation(self.track, "front", 20.8, 0.6, 3.0)
        strategy.update(rear, (front,), self.track, 0.0)
        for candidate in strategy.last_candidates:
            self.assertTrue(all(clearance >= self.config.clearance for clearance in candidate.clearances)
                            if candidate.feasible else bool(candidate.reason))
        # 已並排行駛時，即使兩車沒有車身重疊，也無法滿足車頭留在車尾後方的條件。
        too_close = observation(self.track, "front", 20.5, 0.6, 3.0)
        rejected = self.strategy().update(rear, (too_close,), self.track, 0.0)
        self.assertEqual(rejected.mode, "no_feasible_path")
        self.assertTrue(all(not candidate.feasible for candidate in rejected.candidate_snapshot.candidates))

    def test_mid_horizon_clearance_violation_is_rejected_despite_positive_end_gap(self) -> None:
        strategy = self.strategy()
        own_states = tuple(replace(self.ego.state, x=x) for x in (20.0, 23.0, 23.0))
        front_states = tuple(replace(self.front.state, x=x) for x in (22.5, 22.6, 24.5))
        own = BicyclePrediction(own_states, (0.0, 3.0, 3.0), (0.0, 0.5, 1.0))
        front = BicyclePrediction(front_states, (0.0, 0.1, 2.0), own.times)
        with patch("strategy.probe_following.predict_bicycle_trajectory", return_value=own):
            candidate = strategy._evaluate_candidate(
                self.ego, self.track, 0.0, 3.0, 2.5, 0.0, front,
                body_progress_bounds(front, self.front.geometry, self.track), (),
            )
        self.assertGreater(candidate.gaps[-1], 0.0)
        self.assertLess(min(candidate.gaps), 0.0)
        self.assertFalse(candidate.feasible)
        self.assertEqual(candidate.cost_terms, ())
        with self.assertRaises(ValueError):
            _ = candidate.total_cost

    def test_same_maneuver_origin_is_preserved_when_replanning_speed(self) -> None:
        strategy = self.strategy()
        first = strategy.update(self.ego, (self.front,), self.track, 0.0)
        self.assertEqual(first.target_d, 0.6)
        rear = observation(self.track, "rear", 20.1, 0.02, 3.0)
        front = observation(self.track, "front", 22.6, 0.0, 2.8)
        second = strategy.update(rear, (front,), self.track, 0.1)
        self.assertIs(second.maneuver, first.maneuver)

    def test_phase_waits_for_actual_arrival_and_hold_time(self) -> None:
        strategy = self.strategy(replace(self.config, prediction_horizon=0.1))
        for time in (0.0, 2.0, 4.0):
            strategy.update(self.ego, (self.front,), self.track, time)
            self.assertEqual(strategy.phase_index, 0)
        arrived = observation(self.track, "rear", 20.0, 0.6, 3.0)
        for time in (5.0, 5.5):
            strategy.update(arrived, (self.front,), self.track, time)
            self.assertEqual(strategy.phase_index, 0)
        decision = strategy.update(arrived, (self.front,), self.track, 7.0)
        self.assertEqual(strategy.phase_index, 1)
        self.assertEqual(decision.mode, "return")

    def test_custom_probe_sequence_sets_order_and_side(self) -> None:
        config = replace(self.config, prediction_horizon=0.1, hold_seconds=1.0,
                         probe_sequence=(-0.45, 0.0, 0.3, 0.0))
        strategy = self.strategy(config)
        decision = strategy.update(self.ego, (self.front,), self.track, 0.0)
        self.assertEqual((decision.mode, decision.target_d), ("probe_right", -0.45))
        arrived = observation(self.track, "rear", 20.0, -0.45, 3.0)
        strategy.update(arrived, (self.front,), self.track, 1.0)
        decision = strategy.update(arrived, (self.front,), self.track, 2.0)
        self.assertEqual((strategy.phase_index, decision.mode), (1, "return"))

    def test_default_sequence_is_unchanged_and_invalid_sequences_are_rejected(self) -> None:
        self.assertEqual(ProbeFollowingConfig(probe_offset=0.5).sequence, (0.5, 0.0, -0.5, 0.0))
        self.assertEqual(ProbeFollowingConfig(probe_sequence=[0.3, 0]).probe_sequence, (0.3, 0.0))
        for sequence in ((), (float("nan"),)):
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                ProbeFollowingConfig(probe_sequence=sequence)

    def test_probe_target_is_held_even_when_center_has_lower_cost(self) -> None:
        config = replace(self.config, gap_weight=0.0, speed_weight=0.0, probe_weight=0.0)
        strategy = self.strategy(config)
        decision = strategy.update(self.ego, (self.front,), self.track, 0.0)
        center = [candidate for candidate in strategy.last_candidates
                  if candidate.feasible and candidate.maneuver.target_d == 0.0]
        side = [candidate for candidate in strategy.last_candidates
                if candidate.feasible and candidate.maneuver.target_d == 0.6]
        self.assertLess(min(candidate.total_cost for candidate in center),
                        min(candidate.total_cost for candidate in side))
        self.assertEqual(decision.target_d, 0.6)
        self.assertIsNone(strategy.hold_started_at)

    def test_gap_and_tracking_variation_do_not_interrupt_started_hold(self) -> None:
        strategy = self.strategy(replace(self.config, prediction_horizon=0.1))
        strategy.update(self.ego, (self.front,), self.track, 0.0)
        arrived = observation(self.track, "rear", 20.0, 0.6, 3.0)
        strategy.update(arrived, (self.front,), self.track, 0.1)
        outside = observation(self.track, "front", 24.0, 0.0, 3.0)
        drifted = observation(self.track, "rear", 20.0, 0.4, 3.0)
        for time in (0.5, 1.0, 2.0):
            decision = strategy.update(drifted, (outside,), self.track, time)
            self.assertEqual(decision.target_d, 0.6)
            self.assertEqual(strategy.hold_started_at, 0.1)
        decision = strategy.update(drifted, (outside,), self.track, 2.1)
        self.assertEqual(decision.target_d, 0.0)
        self.assertEqual(decision.mode, "return")

    def test_infeasible_probe_target_interrupts_hold_and_allows_other_path(self) -> None:
        strategy = self.strategy()
        strategy.update(self.ego, (self.front,), self.track, 0.0)
        arrived = observation(self.track, "rear", 20.0, 0.6, 3.0)
        strategy.update(arrived, (self.front,), self.track, 0.1)
        self.assertEqual(strategy.hold_started_at, 0.1)
        evaluate = strategy._evaluate_candidate

        def blocked_side(*args, **kwargs):
            candidate = evaluate(*args, **kwargs)
            return replace(candidate, reason="blocked side") if candidate.maneuver.target_d == 0.6 else candidate

        with patch.object(strategy, "_evaluate_candidate", side_effect=blocked_side):
            decision = strategy.update(arrived, (self.front,), self.track, 0.2)
        self.assertEqual(decision.target_d, 0.0)
        self.assertFalse(strategy.phase_active)
        self.assertIsNone(strategy.hold_started_at)

    def test_target_gap_remains_continuous_across_finish(self) -> None:
        track = create_track(TrackConfig(point_count=100))
        length = track_length(track)
        strategy = self.strategy(replace(self.config, prediction_horizon=0.1))
        for time, rear_s, front_s in ((0.0, length - 1.0, 1.5), (0.2, 0.2, 2.7)):
            rear = observation(track, "rear", rear_s, 0.0, 3.0)
            front = observation(track, "front", front_s, 0.0, 3.0)
            strategy.update(rear, (front,), track, time)
            self.assertAlmostEqual(strategy.target_state.gap, 2.5, delta=0.01)

    def test_target_is_locked_other_predictions_are_shared_and_instances_independent(self) -> None:
        strategy = self.strategy(replace(self.config, prediction_horizon=0.1))
        third = observation(self.track, "third", 25.0, -0.6, 3.0)
        with patch("strategy.probe_following.predict_bicycle_trajectory",
                   wraps=predict_bicycle_trajectory) as predictor:
            first = strategy.update(self.ego, (self.front, third), self.track, 0.0)
        ids = [call.args[0].vehicle_id for call in predictor.call_args_list]
        self.assertEqual(ids.count("front"), 1)
        self.assertEqual(ids.count("third"), 1)
        closer = observation(self.track, "third", 21.5, -0.6, 3.0)
        second = strategy.update(self.ego, (self.front, closer), self.track, 0.2)
        self.assertEqual(first.target_vehicle_id, second.target_vehicle_id)
        strategy.phase_index = 2
        fresh = self.strategy()
        self.assertEqual(fresh.phase_index, 0)
        self.assertIsNone(fresh.target_state)
        absent = strategy.update(self.ego, (), self.track, 0.4)
        self.assertEqual(absent.mode, "cruise")
        self.assertIsNone(absent.speed_limit)
        self.assertEqual(strategy.last_candidates, ())
        self.assertFalse(strategy.phase_active)

    def test_invalid_config_observations_and_time_are_rejected(self) -> None:
        for kwargs in ({"target_gap": 0.0}, {"prediction_horizon": float("nan")},
                       {"gap_weight": -1.0}, {"speed_scale": 0.0}, {"clearance": 3.0},
                       {"speed_step": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ProbeFollowingConfig(**kwargs)
        for others in ((self.ego,), (self.front, self.front)):
            with self.assertRaises(ValueError):
                self.strategy().update(self.ego, others, self.track, 0.0)
        for time in (-1.0, float("nan")):
            with self.assertRaises(ValueError):
                self.strategy().update(self.ego, (), self.track, time)
        with self.assertRaises(ValueError):
            self.strategy().update(self.ego, (), replace(self.track, closed=False), 0.0)
        strategy = self.strategy()
        strategy.update(self.ego, (), self.track, 1.0)
        with self.assertRaises(ValueError):
            strategy.update(self.ego, (), self.track, 0.5)


class ProbeFollowingIntegrationTests(unittest.TestCase):
    def test_selected_cruise_candidate_reaches_actual_episode_step(self) -> None:
        track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        config = prediction_config(12.0)
        settings = SimulationSettings(
            track, replace(config.model, max_velocity=30.0),
            observation(track, "rear", 20.0, 0.0, 10.0).geometry,
            config.controller, offset_path_config=config.path,
            probe_following_config=ProbeFollowingConfig(target_gap=2.5),
        )
        scenario = EpisodeScenario(
            "cruise-catch-up", (VehicleInitialCondition("rear", 20.0, 0.0, 10.0, 12.0),
                                VehicleInitialCondition("front", 28.0, 0.0, 10.0, 10.0)),
            {"rear": StrategyAssignment("probe_follow")},
        )
        decisions = []
        original_update = ProbeFollowingStrategy.update

        def record_update(strategy, *args, **kwargs):
            decision = original_update(strategy, *args, **kwargs)
            decisions.append(decision)
            return decision

        collector = TrainingDataCollector(settings.model_config.time_step)
        with patch.object(ProbeFollowingStrategy, "update", new=record_update):
            run_episode(scenario, settings, collector, 0.2)
        frames = collector.episodes[0].frames
        self.assertEqual(decisions[0].speed_limit, 12.0)
        self.assertGreater(frames[1].observations["rear"].state.velocity, 10.0)
        for index, decision in enumerate(decisions):
            if abs(decision.candidate_snapshot.evaluated_at - frames[index].time_seconds) < 1e-9:
                selected = next(candidate for candidate in decision.candidate_snapshot.candidates if candidate.selected)
                np.testing.assert_allclose(astuple(frames[index + 1].observations["rear"].state),
                                           astuple(selected.predicted_states[1]), atol=1e-12, rtol=0.0)
        self.assertFalse(any(frame.collision_pairs for frame in frames))

    def test_general_assignments_register_probe_without_demo_flag(self) -> None:
        specs = (("vehicle-1", 4.0, 10.0), ("vehicle-2", 0.0, 12.0))
        assignments = {
            "vehicle-1": StrategyAssignment("defend", target_vehicle_id="vehicle-2"),
            "vehicle-2": StrategyAssignment("probe_follow"),
        }
        with patch("main.VEHICLE_SPECS", specs), patch("main.STRATEGY_ASSIGNMENTS", assignments), \
             patch("main.TRACK_NAME", "rounded_rectangle"), patch("main.SIMULATION_SECONDS", 0.12), \
             patch("strategy.assignment.ProbeFollowingStrategy", wraps=ProbeFollowingStrategy) as factory, \
             patch("main.SimulationVisualizer") as visualizer, redirect_stdout(io.StringIO()):
            main.run_simulation()
        factory.assert_called_once()
        self.assertIs(factory.call_args.args[0], main.PROBE_FOLLOWING_CONFIG)
        frame = visualizer.return_value.update_vehicles.call_args.kwargs
        selected = next(candidate for candidate in frame["candidate_snapshots"]["vehicle-2"].candidates
                        if candidate.selected)
        self.assertIsNotNone(selected.speed_limit)
        self.assertEqual({name for name, _ in selected.cost_terms}, {"gap", "speed", "probe", "switch"})
        self.assertEqual(len(frame["trajectories"]["vehicle-2"]), 7)

    def test_main_demo_registers_strategy_and_uses_rounded_track(self) -> None:
        with patch("main.SIMULATION_SECONDS", 0.12), patch("main.SimulationVisualizer") as visualizer, \
             patch("main.load_gp_routes") as load_gp, redirect_stdout(io.StringIO()):
            main.run_simulation(probe_follow_demo=True)
        self.assertEqual(set(visualizer.return_value.update_vehicles.call_args.kwargs["states"]), {"rear", "front"})
        self.assertEqual(len(visualizer.call_args.args[0].centerline), 500)
        load_gp.assert_not_called()

    def test_data_entry_registers_config_selects_pair_and_executes_predictions(self) -> None:
        track = create_track(TrackConfig(point_count=100))
        config = prediction_config(4.0)
        settings = SimulationSettings(track, config.model, observation(track, "rear", 1.0, 0.0, 3.0).geometry,
                                      config.controller, offset_path_config=config.path,
                                      probe_following_config=ProbeFollowingConfig(target_gap=2.5))
        scenario = EpisodeScenario(
            "probe", (VehicleInitialCondition("rear", 1.0, 0.0, 3.0, 4.0),
                      VehicleInitialCondition("front", 5.0, 0.0, 3.0, 3.0)),
            {"rear": StrategyAssignment("probe_follow")},
        )
        decisions = []
        original_update = ProbeFollowingStrategy.update

        def record_update(strategy, *args, **kwargs):
            result = original_update(strategy, *args, **kwargs)
            decisions.append(result)
            return result

        collector = TrainingDataCollector(config.model.time_step)
        with patch.object(ProbeFollowingStrategy, "update", new=record_update):
            run_episode(scenario, settings, collector, 0.24)
        frames = collector.episodes[0].frames
        for index, decision in enumerate(decisions):
            if decision.candidate_snapshot.evaluated_at == index * config.model.time_step:
                selected = next(candidate for candidate in decision.candidate_snapshot.candidates if candidate.selected)
                np.testing.assert_allclose(astuple(frames[index + 1].observations["rear"].state),
                                           astuple(selected.predicted_states[1]), atol=1e-12, rtol=0.0)
        self.assertGreater(frames[-1].observations["rear"].state.velocity, 3.0)
        self.assertEqual(_select_pairs(None, ("rear", "front"), dict(scenario.assignments)),
                         (VehiclePair("rear", "front"),))
        metadata = _run_metadata(settings, (scenario,), (VehiclePair("rear", "front"),),
                                 42, 0.24, RandomizationConfig())
        self.assertEqual(metadata["simulation_configs"]["probe_following_config"]["target_gap"], 2.5)
        with patch("main.create_demo_track", return_value=track):
            copied, _, _ = _demo_settings(False)
        self.assertEqual(copied.probe_following_config, main.PROBE_FOLLOWING_CONFIG)


if __name__ == "__main__":
    unittest.main()
