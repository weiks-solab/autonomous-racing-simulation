"""速度要求、離散速度候選，以及固定決策下的預測／執行一致性。"""

from contextlib import redirect_stdout
from dataclasses import asdict, astuple, replace
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np

import main
from core.pure_pursuit import PurePursuitController, compute_speed_acceleration
from core.track import Track
from core.vehicle_geometry import CollisionConfig
from core.vehicle_model import KinematicBicycleModel
from core.vehicle_state import ControlCommand, VehicleState
from sim.candidate_paths import candidate_path_label
from sim.data_generation import (
    EpisodeScenario, SimulationSettings, VehicleInitialCondition, run_episode,
)
from sim.simulated_vehicle import SimulatedVehicle
from strategy.assignment import StrategyAssignment
from strategy.base import CandidatePath, StrategyDecision
from strategy.bicycle_prediction import BicyclePredictionConfig, predict_bicycle_trajectory
from strategy.offset_path import LateralManeuver
from strategy.relative_overtaking import (
    RelativeCandidate, RelativeOvertakingConfig, RelativeOvertakingStrategy,
)
from tests.test_bicycle_prediction import observation, prediction_config
from training_data import TrainingDataCollector


class SpeedRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.config = prediction_config()
        self.vehicle = observation(self.track, "ego", 20.0, 0.0, 6.0)
        self.maneuver = LateralManeuver(20.0, 0.0, 0.0, 3.0)

    def make_runtime(self, state: VehicleState | None = None) -> SimulatedVehicle:
        return SimulatedVehicle(
            "ego", self.vehicle.state if state is None else state,
            KinematicBicycleModel(self.config.model),
            PurePursuitController(self.config.controller), self.vehicle.geometry,
        )

    def test_omitted_limit_keeps_original_symmetric_speed_control(self) -> None:
        controller = PurePursuitController(self.config.controller)
        self.assertEqual(controller.config.max_deceleration, controller.config.max_acceleration)
        # 舊版比例控制在 v=0、5、6、7、10 m/s 時的輸出。
        for velocity, expected in ((0.0, 3.0), (5.0, 1.5), (6.0, 0.0),
                                   (7.0, -1.5), (10.0, -3.0)):
            with self.subTest(velocity=velocity):
                state = replace(self.vehicle.state, velocity=velocity)
                command, _ = controller.compute_command(state, self.track, 0.33)
                explicit, _ = controller.compute_command(state, self.track, 0.33, speed_limit=None)
                self.assertEqual(command.acceleration, expected)
                self.assertEqual(command, explicit)

    def test_unlimited_rollout_preserves_cruise_motion(self) -> None:
        default = predict_bicycle_trajectory(
            self.vehicle, self.track, self.maneuver, self.config, 1.0,
        )
        explicit = predict_bicycle_trajectory(
            self.vehicle, self.track, self.maneuver, self.config, 1.0, speed_limit=None,
        )
        self.assertEqual(default, explicit)
        for state, time in zip(default.states, default.times):
            np.testing.assert_allclose(astuple(state), (20.0 + 6.0 * time, 0.0, 0.0, 6.0),
                                       rtol=0.0, atol=1e-12)

    def test_limit_above_cruise_uses_original_cruise_target(self) -> None:
        controller = self.config.controller
        self.assertEqual(compute_speed_acceleration(7.0, controller, 9.0), -1.5)
        self.assertEqual(compute_speed_acceleration(1.0, controller, 2.0), 1.5)

    def test_deceleration_is_gradual_and_has_its_own_limit(self) -> None:
        config = replace(self.config, controller=replace(
            self.config.controller, max_acceleration=1.0, max_deceleration=2.5,
        ))
        vehicle = replace(self.vehicle, state=replace(self.vehicle.state, velocity=8.0))
        command, _ = PurePursuitController(config.controller).compute_command(
            vehicle.state, self.track, config.model.wheelbase, speed_limit=2.0,
        )
        self.assertEqual(command.acceleration, -2.5)
        prediction = predict_bicycle_trajectory(
            vehicle, self.track, self.maneuver, config, 4.0, speed_limit=2.0,
        )
        speeds = np.array([state.velocity for state in prediction.states])
        self.assertAlmostEqual(speeds[1], 8.0 - 2.5 * config.model.time_step)
        self.assertGreater(speeds[1], 2.0)
        self.assertTrue(np.all(np.diff(speeds) < 0.0))
        self.assertTrue(np.all(-np.diff(speeds) <= 2.5 * config.model.time_step + 1e-12))
        self.assertLess(speeds[-1], 2.2)

    def test_zero_limit_stops_gradually_and_none_restores_cruise(self) -> None:
        runtime = self.make_runtime()
        controller_config = runtime.controller.config
        time_step = self.config.model.time_step
        for _ in range(600):
            previous_speed = runtime.state.velocity
            command, _ = runtime.step(self.track, speed_limit=0.0)
            self.assertGreaterEqual(command.acceleration, -controller_config.max_deceleration)
            self.assertLessEqual(command.acceleration, 0.0)
            self.assertGreaterEqual(runtime.state.velocity, 0.0)
            self.assertLessEqual(runtime.state.velocity, previous_speed)
            self.assertLessEqual(previous_speed - runtime.state.velocity,
                                 controller_config.max_deceleration * time_step + 1e-12)
        # 比例控制漸近停止，以 1e-6 m/s 作為接近停止的容許誤差。
        self.assertGreater(runtime.state.velocity, 0.0)
        self.assertLess(runtime.state.velocity, 1e-6)
        for _ in range(300):
            previous_speed = runtime.state.velocity
            command, _ = runtime.step(self.track, speed_limit=None)
            self.assertGreater(command.acceleration, 0.0)
            self.assertLessEqual(command.acceleration, controller_config.max_acceleration)
            self.assertLessEqual(runtime.state.velocity - previous_speed,
                                 controller_config.max_acceleration * time_step + 1e-12)
        self.assertGreater(runtime.state.velocity, 5.9)
        self.assertIs(runtime.controller.config, controller_config)
        self.assertEqual(controller_config.target_speed, 6.0)

    def test_constant_speed_opponent_keeps_its_original_meaning(self) -> None:
        track = Track.create_oval(10.0, 5.0, 300)
        vehicle = observation(track, "front", 1.0, 0.0, 4.0)
        maneuver = LateralManeuver(vehicle.track_position.s, 0.0, 0.0, 3.0)
        original = predict_bicycle_trajectory(
            vehicle, track, maneuver, self.config, 0.8, constant_speed=True,
        )
        limited = predict_bicycle_trajectory(
            vehicle, track, maneuver, self.config, 0.8, constant_speed=True, speed_limit=0.0,
        )
        self.assertEqual(limited, original)
        self.assertTrue(all(state.velocity == 4.0 for state in limited.states))
        self.assertNotEqual(limited.states[-1].yaw, vehicle.state.yaw)

    def test_invalid_limits_are_rejected_at_public_boundaries(self) -> None:
        prediction = predict_bicycle_trajectory(
            self.vehicle, self.track, self.maneuver, self.config, 0.1,
        )
        runtime = self.make_runtime()
        for value in (-1.0, float("nan"), float("inf"), -float("inf")):
            factories = (
                lambda: StrategyDecision("cruise", speed_limit=value),
                lambda: CandidatePath(self.maneuver, speed_limit=value),
                lambda: RelativeCandidate(self.maneuver, prediction, None, speed_limit=value),
                lambda: runtime.step(self.track, speed_limit=value),
                lambda: predict_bicycle_trajectory(
                    self.vehicle, self.track, self.maneuver, self.config, 0.1, speed_limit=value,
                ),
                lambda: predict_bicycle_trajectory(
                    self.vehicle, self.track, self.maneuver, self.config, 0.1,
                    constant_speed=True, speed_limit=value,
                ),
            )
            for index, factory in enumerate(factories):
                with self.subTest(value=value, boundary=index), self.assertRaisesRegex(
                    ValueError, "speed_limit",
                ):
                    factory()
        self.assertEqual(runtime.trajectory, [self.vehicle.state])
        for value in (None, 0.0, 2.0):
            self.assertEqual(StrategyDecision("cruise", speed_limit=value).speed_limit, value)

    def test_invalid_deceleration_limits_are_rejected(self) -> None:
        for value in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.config.controller, max_deceleration=value)


class SpeedCandidateIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.prediction = prediction_config()
        self.config = RelativeOvertakingConfig(
            candidate_offsets=(0.0,), prediction_horizon=1.0,
            speed_candidates_enabled=True, speed_candidate_delta=1.0,
        )
        self.ego = observation(self.track, "ego", 20.0, 0.0, 6.0)
        self.front = observation(self.track, "front", 22.5, 0.0, 3.0)

    def strategy(self, config: RelativeOvertakingConfig | None = None) -> RelativeOvertakingStrategy:
        return RelativeOvertakingStrategy(
            self.config if config is None else config, self.prediction,
            {"front": prediction_config(5.0)},
        )

    def execute_fixed_decision(
        self, track: Track, initial: VehicleState, config: BicyclePredictionConfig,
        decision: StrategyDecision, horizon: float,
    ) -> tuple[list[VehicleState], list[ControlCommand]]:
        """走正式 main → StrategyManager → step → controller，固定決策且不重規劃。"""
        runtime = SimulatedVehicle(
            "ego", initial, KinematicBicycleModel(config.model),
            PurePursuitController(config.controller), self.ego.geometry,
        )
        commands: list[ControlCommand] = []
        frozen_strategy = Mock()
        frozen_strategy.update.return_value = decision

        def record_step(reference: Track, speed_limit: float | None = None) -> tuple:
            command, target = SimulatedVehicle.step(runtime, reference, speed_limit=speed_limit)
            commands.append(command)
            return command, target

        with patch.multiple(
            main, TRACK_NAME="rounded_rectangle", MODEL_CONFIG=config.model,
            OFFSET_PATH_CONFIG=config.path, GP_PREDICTION_ENABLED=False,
            VEHICLE_SPECS=(("ego", 0.0, config.controller.target_speed),),
            STRATEGY_ASSIGNMENTS={"ego": StrategyAssignment("relative_overtake")},
            SIMULATION_SECONDS=horizon,
        ), patch("main.create_demo_track", return_value=track), \
             patch("main._create_vehicle", return_value=runtime), \
             patch("main.RelativeOvertakingStrategy", return_value=frozen_strategy), \
             patch("main.SimulationVisualizer"), \
             patch.object(runtime, "step", side_effect=record_step), \
             redirect_stdout(io.StringIO()):
            main.run_simulation()
        self.assertEqual(len(commands), round(horizon / config.model.time_step))
        self.assertEqual(runtime.controller.config.target_speed, config.controller.target_speed)
        return runtime.trajectory, commands

    def assert_same_states(self, actual: list[VehicleState], expected: tuple[VehicleState, ...]) -> None:
        self.assertEqual(len(actual), len(expected))
        for step, (executed, predicted) in enumerate(zip(actual, expected)):
            with self.subTest(step=step):
                np.testing.assert_allclose(astuple(executed), astuple(predicted), rtol=0.0, atol=1e-10)

    def test_fixed_decision_matches_rollout_on_straight_and_curved_paths(self) -> None:
        for shape, track, start_s in (
            ("straight", self.track, 20.0),
            ("curve", Track.create_oval(10.0, 5.0, 300), 2.0),
        ):
            for speed_limit in (None, 2.0, 0.0):
                with self.subTest(shape=shape, speed_limit=speed_limit):
                    ego = observation(track, "ego", start_s, 0.0, 6.0)
                    maneuver = LateralManeuver(ego.track_position.s, 0.0, 0.45, 3.0)
                    decision = StrategyDecision(
                        "attack", target_d=0.45, maneuver=maneuver, speed_limit=speed_limit,
                    )
                    prediction = predict_bicycle_trajectory(
                        ego, track, maneuver, self.prediction, 1.0, speed_limit=speed_limit,
                    )
                    actual, commands = self.execute_fixed_decision(
                        track, ego.state, self.prediction, decision, 1.0,
                    )
                    self.assert_same_states(actual, prediction.states)
                    if speed_limit is not None:
                        self.assertLess(commands[0].acceleration, 0.0)

    def test_strategy_selects_feasible_lower_speed_and_execution_matches(self) -> None:
        baseline = self.strategy(replace(self.config, speed_candidates_enabled=False))
        baseline_decision = baseline.update(self.ego, (self.front,), self.track, 0.0)
        self.assertEqual(baseline_decision.mode, "no_feasible_path")
        self.assertIsNone(baseline_decision.speed_limit)
        self.assertEqual(len(baseline.last_candidates), 1)
        self.assertIsNotNone(baseline.last_candidates[0].collision)

        strategy = self.strategy()
        with patch("strategy.relative_overtaking.predict_bicycle_trajectory",
                   wraps=predict_bicycle_trajectory) as predictor:
            decision = strategy.update(self.ego, (self.front,), self.track, 0.0)
        self.assertEqual(sum(call.args[0].vehicle_id == "front" for call in predictor.call_args_list), 1)
        self.assertEqual(tuple(candidate.speed_limit for candidate in strategy.last_candidates),
                         (None, 3.0, 2.0))
        self.assertEqual(strategy.last_candidates[:1], baseline.last_candidates)
        self.assertEqual(strategy.last_other_predictions, baseline.last_other_predictions)
        self.assertTrue(all(candidate.prediction.times == strategy.last_candidates[0].prediction.times
                            for candidate in strategy.last_candidates))
        self.assertTrue(all(candidate.prediction.states[0] is self.ego.state
                            for candidate in strategy.last_candidates))
        self.assertEqual(decision.speed_limit, 3.0)
        selected = next(candidate for candidate in strategy.last_candidates
                        if candidate.speed_limit == decision.speed_limit)
        self.assertTrue(selected.feasible)
        self.assertEqual(selected.maneuver, baseline.last_candidates[0].maneuver)
        self.assertEqual(decision.maneuver, selected.maneuver)
        self.assertEqual(selected.total_cost, min(candidate.total_cost for candidate in strategy.last_candidates
                                                  if candidate.feasible))
        shown = next(candidate for candidate in decision.candidate_snapshot.candidates if candidate.selected)
        self.assertEqual(shown.speed_limit, decision.speed_limit)
        self.assertEqual(shown.predicted_states, selected.prediction.states)
        labels = [candidate_path_label(candidate) for candidate in decision.candidate_snapshot.candidates]
        self.assertEqual(len(set(labels)), 3)
        self.assertIn("speed_limit=3.00m/s", candidate_path_label(shown))

        actual, commands = self.execute_fixed_decision(
            self.track, self.ego.state, self.prediction, decision, self.config.prediction_horizon,
        )
        self.assertLess(commands[0].acceleration, 0.0)
        self.assertGreater(actual[1].velocity, decision.speed_limit)
        self.assert_same_states(actual, selected.prediction.states)

    def test_speed_candidates_continue_existing_maneuver_origin(self) -> None:
        config = replace(self.config, candidate_offsets=(0.0, 0.6), prediction_horizon=0.1,
                         extra_speed_candidate_deltas=(4.0,))
        strategy = self.strategy(config)
        maneuver = LateralManeuver(19.5, 0.0, 0.45, 3.0)
        strategy.decision = StrategyDecision("attack", target_d=0.45, maneuver=maneuver)
        decision = strategy.update(self.ego, (self.front,), self.track, 0.0)
        self.assertEqual(len(strategy.last_candidates), 5)
        slower = [candidate for candidate in strategy.last_candidates if candidate.speed_limit is not None]
        self.assertEqual(len(slower), 3)
        self.assertTrue(all(candidate.maneuver is maneuver for candidate in slower))
        self.assertTrue(all(candidate.prediction.states[0] is self.ego.state for candidate in slower))
        self.assertTrue(all(item.speed_limit is None for item in strategy.last_candidates[:2]))
        self.assertEqual(len(decision.candidate_snapshot.candidates), 5)

    def test_speed_candidate_filtering_and_deduplication(self) -> None:
        for front_speed, delta, expected in (
            (3.0, 1.0, (3.0, 2.0)), (3.0, 0.0, (3.0,)),
            (0.0, 1.0, (0.0,)), (6.0, 1.0, (5.0,)), (7.0, 1.0, ()),
        ):
            with self.subTest(front_speed=front_speed, delta=delta):
                config = replace(self.config, speed_candidate_delta=delta, prediction_horizon=0.1)
                strategy = self.strategy(config)
                front = replace(self.front, state=replace(self.front.state, velocity=front_speed))
                strategy.update(self.ego, (front,), self.track, 0.0)
                self.assertEqual(tuple(candidate.speed_limit for candidate in strategy.last_candidates
                                       if candidate.speed_limit is not None), expected)

    def test_extra_speed_candidates_are_clamped_filtered_and_deduplicated(self) -> None:
        for front_speed, deltas, expected in (
            (3.0, (4.0, 1.0, 4.0, 8.0), (3.0, 2.0, 0.0)),
            (9.0, (4.0, 4.0, 7.0), (5.0, 2.0)),
            (0.0, (4.0,), (0.0,)),
        ):
            with self.subTest(front_speed=front_speed):
                config = replace(self.config, extra_speed_candidate_deltas=deltas, prediction_horizon=0.1)
                strategy = self.strategy(config)
                front = replace(self.front, state=replace(self.front.state, velocity=front_speed))
                strategy.update(self.ego, (front,), self.track, 0.0)
                self.assertEqual(tuple(candidate.speed_limit for candidate in strategy.last_candidates
                                       if candidate.speed_limit is not None), expected)
        disabled = self.strategy(replace(config, speed_candidates_enabled=False))
        disabled.update(self.ego, (self.front,), self.track, 0.0)
        self.assertEqual(len(disabled.last_candidates), 1)

    def test_extra_delta_rejects_negative_and_nonfinite_values(self) -> None:
        for delta in (-1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                replace(self.config, extra_speed_candidate_deltas=(delta,))

    def test_minus_four_adds_a_feasible_solution_with_wide_collision_rectangles(self) -> None:
        # 同一條直線：匹配及 -1 都會碰撞，-4 透過較強的正常減速可行。
        geometry = replace(self.ego.geometry, collision=CollisionConfig(0.8, 0.7))
        self.ego = replace(observation(self.track, "ego", 20.0, 0.0, 11.0), geometry=geometry)
        front = replace(observation(self.track, "front", 21.3, 0.0, 9.0), geometry=geometry)
        self.prediction = replace(self.prediction,
            model=replace(self.prediction.model, max_velocity=30.0),
            controller=replace(self.prediction.controller, target_speed=11.0,
                               max_acceleration=5.0, max_deceleration=5.0))
        baseline = self.strategy()
        previous = baseline.update(self.ego, (front,), self.track, 0.0)
        self.assertEqual(previous.mode, "no_feasible_path")
        strategy = self.strategy(replace(self.config, extra_speed_candidate_deltas=(4.0,)))
        decision = strategy.update(self.ego, (front,), self.track, 0.0)
        self.assertEqual([candidate.speed_limit for candidate in strategy.last_candidates], [None, 9.0, 8.0, 5.0])
        self.assertEqual(strategy.last_candidates[:-1], baseline.last_candidates)
        self.assertEqual(decision.speed_limit, 5.0)
        self.assertEqual(decision.target_d, 0.0)
        selected = strategy.last_candidates[-1]
        self.assertTrue(selected.feasible)
        actual, commands = self.execute_fixed_decision(self.track, self.ego.state, self.prediction, decision, 1.0)
        self.assertEqual(commands[0].acceleration, -5.0)
        self.assertAlmostEqual(actual[1].velocity, 10.9)
        self.assert_same_states(actual, selected.prediction.states)

    def test_extra_speed_cannot_remove_existing_overlap(self) -> None:
        strategy = self.strategy(replace(self.config, extra_speed_candidate_deltas=(4.0,)))
        overlapping = observation(self.track, "front", 20.4, 0.0, 3.0)
        decision = strategy.update(self.ego, (overlapping,), self.track, 0.0)
        self.assertEqual(decision.mode, "no_feasible_path")
        self.assertEqual(len(strategy.last_candidates), 4)
        self.assertTrue(all(candidate.collision.time_seconds == 0.0 for candidate in strategy.last_candidates))

    def test_no_front_target_and_passed_target_add_no_speed_candidates(self) -> None:
        strategy = self.strategy()
        strategy.update(self.ego, (), self.track, 0.0)
        self.assertEqual(len(strategy.last_candidates), 1)
        strategy.update(self.ego, (self.front,), self.track, 0.2)
        moved = observation(self.track, "ego", 24.0, 0.0, 6.0)
        decision = strategy.update(moved, (self.front,), self.track, 0.4)
        self.assertEqual(decision.target_vehicle_id, "front")
        self.assertLess(strategy.target_state.gap, 0.0)
        self.assertEqual(len(strategy.last_candidates), 1)
        self.assertIsNone(decision.speed_limit)

    def test_all_candidates_rejected_keeps_no_feasible_path(self) -> None:
        strategy = self.strategy()
        previous = strategy.update(self.ego, (self.front,), self.track, 0.0)
        too_close = observation(self.track, "front", 21.0, 0.0, 3.0)
        decision = strategy.update(self.ego, (too_close,), self.track, 0.2)
        self.assertEqual(decision.mode, "no_feasible_path")
        self.assertIs(decision.maneuver, previous.maneuver)
        self.assertEqual(decision.speed_limit, previous.speed_limit)
        self.assertEqual(len(strategy.last_candidates), 3)
        self.assertTrue(all(not candidate.feasible for candidate in strategy.last_candidates))
        self.assertTrue(all(candidate.collision.time_seconds > 0.0 for candidate in strategy.last_candidates))
        self.assertTrue(all(not candidate.selected for candidate in decision.candidate_snapshot.candidates))

    def test_default_switch_matches_disabled_baseline(self) -> None:
        default = RelativeOvertakingConfig(candidate_offsets=(0.0,), prediction_horizon=1.0)
        disabled = replace(self.config, speed_candidates_enabled=False)
        self.assertFalse(default.speed_candidates_enabled)
        front = observation(self.track, "front", 22.5, -0.6, 3.0)
        first, second = self.strategy(default), self.strategy(disabled)
        left = first.update(self.ego, (front,), self.track, 0.0)
        right = second.update(self.ego, (front,), self.track, 0.0)
        self.assertEqual(left, right)
        self.assertEqual(len(first.last_candidates), 1)
        self.assertIsNone(left.speed_limit)
        baseline = first.last_candidates[0]
        self.assertTrue(baseline.feasible)
        self.assertAlmostEqual(baseline.total_cost, -0.5)
        actual, _ = self.execute_fixed_decision(
            self.track, self.ego.state, self.prediction, left, 1.0,
        )
        self.assert_same_states(actual, baseline.prediction.states)
        # 關閉功能後，仍是原本 6 m/s 的直線巡航。
        for state, time in zip(actual, baseline.prediction.times):
            np.testing.assert_allclose(astuple(state), (20.0 + 6.0 * time, 0.0, 0.0, 6.0),
                                       rtol=0.0, atol=1e-12)

    def test_invalid_speed_candidate_delta_is_rejected(self) -> None:
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "speed_candidate_delta"):
                replace(self.config, speed_candidate_delta=value)

    def test_training_episode_executes_selected_speed_limit(self) -> None:
        settings = SimulationSettings(
            self.track, self.prediction.model, self.ego.geometry, self.prediction.controller,
            offset_path_config=self.prediction.path, relative_overtaking_config=self.config,
        )
        scenario = EpisodeScenario(
            "speed-candidates",
            (VehicleInitialCondition("ego", 20.0, 0.0, 6.0, 6.0),
             VehicleInitialCondition("front", 22.5, 0.0, 3.0, 3.0)),
            {"ego": StrategyAssignment("relative_overtake")},
        )
        collector = TrainingDataCollector(self.prediction.model.time_step)
        strategy = self.strategy()
        decision = strategy.update(self.ego, (self.front,), self.track, 0.0)
        selected = next(candidate for candidate in strategy.last_candidates
                        if candidate.speed_limit == decision.speed_limit)
        frozen_strategy = Mock()
        frozen_strategy.update.return_value = decision
        with patch("sim.data_generation.RelativeOvertakingStrategy", return_value=frozen_strategy):
            run_episode(scenario, settings, collector, 1.0)
        actual = [frame.observations["ego"].state for frame in collector.episodes[0].frames]
        self.assertLess(actual[1].velocity, actual[0].velocity)
        self.assert_same_states(actual, selected.prediction.states)

    def run_closed_loop_case(self) -> dict:
        """沿用整合案例，透過原有雙車步進記錄真實重規劃與控制結果。"""
        settings = SimulationSettings(
            self.track, self.prediction.model, self.ego.geometry, self.prediction.controller,
            offset_path_config=self.prediction.path, relative_overtaking_config=self.config,
        )
        scenario = EpisodeScenario(
            "closed-loop-speed-candidates",
            (
                VehicleInitialCondition(
                    "ego", self.ego.track_position.s, self.ego.track_position.d,
                    self.ego.state.velocity, self.prediction.controller.target_speed,
                ),
                VehicleInitialCondition(
                    "front", self.front.track_position.s, self.front.track_position.d,
                    self.front.state.velocity, self.front.state.velocity,
                ),
            ),
            {"ego": StrategyAssignment("relative_overtake")},
        )
        strategy = self.strategy()
        original_update = strategy.update
        original_step = SimulatedVehicle.step
        decisions: list[StrategyDecision] = []
        steps: list[dict] = []

        def record_update(*args, **kwargs) -> StrategyDecision:
            # 呼叫真正策略並原樣回傳決策，記錄器只觀察結果。
            decision = original_update(*args, **kwargs)
            decisions.append(decision)
            return decision

        def record_step(
            vehicle: SimulatedVehicle, reference: Track, speed_limit: float | None = None,
        ) -> tuple:
            before = vehicle.state.velocity
            command, target = original_step(vehicle, reference, speed_limit=speed_limit)
            if vehicle.vehicle_id == "ego":
                decision = decisions[-1]
                steps.append({
                    "step_index": len(steps),
                    "time_s": len(steps) * self.prediction.model.time_step,
                    "evaluated_at_s": decision.candidate_snapshot.evaluated_at,
                    "mode": decision.mode,
                    "target_d_m": decision.target_d,
                    "speed_limit_mps": decision.speed_limit,
                    "executed_speed_limit_mps": speed_limit,
                    "acceleration_mps2": command.acceleration,
                    "velocity_before_mps": before,
                    "velocity_after_mps": vehicle.state.velocity,
                })
            return command, target

        collector = TrainingDataCollector(self.prediction.model.time_step)
        # 工廠提供真實策略物件。兩個包裝函式皆保留原輸入與原輸出。
        with patch("sim.data_generation.RelativeOvertakingStrategy", return_value=strategy), \
             patch.object(strategy, "update", side_effect=record_update), \
             patch.object(SimulatedVehicle, "step", new=record_step):
            run_episode(scenario, settings, collector, 1.0)

        frames = collector.episodes[0].frames
        replans = [step for step in steps if step["time_s"] == step["evaluated_at_s"]]
        first_lower = next((step for step in replans
                            if step["mode"] != "no_feasible_path"
                            and step["speed_limit_mps"] is not None
                            and step["speed_limit_mps"] < self.prediction.controller.target_speed), None)
        return {
            "duration_s": frames[-1].time_seconds,
            "time_step_s": self.prediction.model.time_step,
            "strategy_config": asdict(self.config),
            "model_config": asdict(self.prediction.model),
            "controller_config": asdict(self.prediction.controller),
            "path_config": asdict(self.prediction.path),
            "geometry": asdict(self.ego.geometry),
            "scenario": {"vehicles": [asdict(vehicle) for vehicle in scenario.vehicles]},
            "track_centerline_m": self.track.centerline.tolist(),
            "track_width_m": self.track.width,
            "initial_candidates": [
                {"speed_limit_mps": candidate.speed_limit, "feasible": candidate.feasible}
                for candidate in decisions[0].candidate_snapshot.candidates
            ],
            "replanning_count_including_initial": len(replans),
            "replanning_times_s": [step["time_s"] for step in replans],
            "first_lower_speed_step": first_lower,
            "front_speeds_mps": [frame.observations["front"].state.velocity for frame in frames],
            "collision_checks_including_initial": len(frames),
            "collision_frames": [
                {"time_s": frame.time_seconds, "pairs": sorted(frame.collision_pairs)}
                for frame in frames if frame.collision_pairs
            ],
            "steps": steps,
        }

    def test_closed_loop_replanning_slows_without_collision(self) -> None:
        self.closed_loop_result = self.run_closed_loop_case()
        result = self.closed_loop_result
        self.assertEqual(result["duration_s"], 1.0)
        self.assertEqual(len(result["steps"]), 50)
        initial = result["initial_candidates"]
        self.assertFalse(next(candidate for candidate in initial
                              if candidate["speed_limit_mps"] is None)["feasible"])
        self.assertTrue(any(candidate["feasible"] and candidate["speed_limit_mps"] is not None
                            for candidate in initial))
        self.assertGreater(result["replanning_count_including_initial"], 1, "策略沒有再次重規劃")
        self.assertTrue(any(step["time_s"] > 0.0 and step["time_s"] == step["evaluated_at_s"]
                            and step["mode"] != "no_feasible_path"
                            and step["speed_limit_mps"] is not None
                            and step["speed_limit_mps"] < self.prediction.controller.target_speed
                            for step in result["steps"]), "後續重規劃未選中低速候選")
        first = result["first_lower_speed_step"]
        self.assertIsNotNone(first, "未選中低於巡航速度的候選")
        self.assertLess(first["acceleration_mps2"], 0.0, f"第 {first['step_index']} 步未輸出負加速度")
        self.assertLess(first["velocity_after_mps"], first["velocity_before_mps"], "實際車速未下降")
        self.assertTrue(all(step["speed_limit_mps"] == step["executed_speed_limit_mps"]
                            and step["target_d_m"] == 0.0 for step in result["steps"]))
        self.assertTrue(all(speed == self.front.state.velocity for speed in result["front_speeds_mps"]))
        self.assertEqual(result["collision_frames"], [], "驗證期間發生車身碰撞")


if __name__ == "__main__":
    unittest.main()
