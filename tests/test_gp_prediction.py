"""確認 GP 查詢、時間對齊、碰撞篩選與原相對成本確實接在一起。"""

from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import main
from core.track import Track
from core.track_geometry import track_length, world_to_track
from strategy.assignment import StrategyAssignment
from strategy.collision_check import PredictedVehicle, first_predicted_collision
from strategy.gp_prediction import GpRoute, PROJECT_ROOT, load_gp_routes, predict_gp_trajectory
from strategy.offset_path import LateralManeuver
from strategy.bicycle_prediction import predict_bicycle_trajectory
from strategy.relative_overtaking import RelativeOvertakingConfig, RelativeOvertakingStrategy
from tests.test_bicycle_prediction import observation, prediction_config


class LinearModel:
    """可檢查查詢位置的替身，不需要擬合或安裝 sklearn。"""

    def __init__(self, offset: float = 0.0, slope: float = 0.0) -> None:
        self.offset, self.slope = offset, slope
        self.queries: list[np.ndarray] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.queries.append(X.copy())
        return self.offset + self.slope * X[:, 0]


class GpPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))))
        self.front = observation(self.track, "front", 22.0, 0.6, 3.0)
        self.ego = observation(self.track, "ego", 20.0, 0.0, 6.0)
        self.model = LinearModel()
        self.route = GpRoute(self.model)

    def strategy(self, routes: dict | None = None) -> RelativeOvertakingStrategy:
        return RelativeOvertakingStrategy(
            RelativeOvertakingConfig(prediction_horizon=1.0), prediction_config(),
            {"front": prediction_config(5.0), "third": prediction_config(4.0)},
            gp_routes=routes,
        )

    def test_queries_opponent_future_s_and_uses_gp_curve_heading(self) -> None:
        model = LinearModel(offset=-2.0, slope=0.1)
        result = predict_gp_trajectory(self.front, self.track, GpRoute(model), 4.0, 0.105, 0.02)
        self.assertEqual(result.states[0], self.front.state)
        self.assertEqual(result.times[-1], 0.105)
        self.assertEqual(len(result.times), 7)
        self.assertAlmostEqual(result.progress[-1], 0.42)
        self.assertAlmostEqual(result.states[-1].x, 22.42)
        self.assertAlmostEqual(result.states[-1].y, 0.242)
        self.assertAlmostEqual(result.states[-1].yaw, math.atan(0.1))
        self.assertEqual(len(model.queries), 1)
        np.testing.assert_allclose(model.queries[0][:6, 0], 22.0 + np.array(result.progress[1:]))

    def test_wrapped_gp_query_keeps_unwrapped_progress(self) -> None:
        track = Track.create_oval(10.0, 10.0, 500)
        length = track_length(track)
        front = observation(track, "front", length - 0.1, 0.0, 3.0)
        result = predict_gp_trajectory(front, track, self.route, 3.0, 0.1, 0.02)
        self.assertAlmostEqual(result.progress[-1], 0.3)
        self.assertTrue(np.all(self.model.queries[0] >= 0.0))
        self.assertTrue(np.all(self.model.queries[0] < length))
        self.assertAlmostEqual(world_to_track(result.states[-1], track).s, 0.2, delta=0.01)

    def test_stopped_vehicle_stays_at_observed_position(self) -> None:
        result = predict_gp_trajectory(self.front, self.track, self.route, 0.0, 1.0, 0.02)
        self.assertTrue(all(state == self.front.state for state in result.states))
        self.assertEqual(self.model.queries, [])
        self.assertEqual(result.offsets, (0.6,) * len(result.times))

    def test_offsets_keep_observed_d_then_follow_gp(self) -> None:
        model = LinearModel(offset=-2.0, slope=0.1)
        result = predict_gp_trajectory(self.front, self.track, GpRoute(model), 4.0, 0.105, 0.02)
        self.assertEqual(len(result.offsets), len(result.times))
        self.assertAlmostEqual(result.offsets[0], 0.6)
        np.testing.assert_allclose(result.offsets[1:], -2.0 + 0.1 * (22.0 + np.array(result.progress[1:])))

    def test_current_overlap_is_not_hidden_by_gp(self) -> None:
        route = GpRoute(LinearModel(offset=-0.6))
        other = observation(self.track, "front", 20.0, 0.0, 3.0)
        opponent = predict_gp_trajectory(other, self.track, route, 3.0, 1.0, 0.02)
        own = predict_bicycle_trajectory(self.ego, self.track, LateralManeuver(20, 0, 0, 3),
                                         prediction_config(), 1.0)
        hit = first_predicted_collision(own, self.ego.geometry,
                                        (PredictedVehicle("front", other.geometry, opponent),))
        self.assertEqual(hit.time_seconds, 0.0)

    def test_same_route_at_different_times_is_not_a_collision(self) -> None:
        other = observation(self.track, "front", 45.0, 0.0, 3.0)
        opponent = predict_gp_trajectory(other, self.track, self.route, 3.0, 1.0, 0.02)
        own = predict_bicycle_trajectory(self.ego, self.track, LateralManeuver(20, 0, 0, 3),
                                         prediction_config(), 1.0)
        self.assertIsNone(first_predicted_collision(own, self.ego.geometry,
                          (PredictedVehicle("front", other.geometry, opponent),)))

    def test_gp_changes_collision_filter_and_preserves_relative_cost(self) -> None:
        original = self.strategy()
        original.update(self.ego, (self.front,), self.track, 0.0)
        self.assertTrue(next(c for c in original.last_candidates if c.maneuver.target_d == 0).feasible)
        strategy = self.strategy({"front": self.route})
        decision = strategy.update(self.ego, (self.front,), self.track, 0.0)
        center = next(c for c in strategy.last_candidates if c.maneuver.target_d == 0)
        self.assertFalse(center.feasible)
        self.assertEqual(center.collision.vehicle_id, "front")
        self.assertGreater(center.collision.time_seconds, 0.0)
        self.assertEqual(len(self.model.queries), 1)  # 九條候選共用一份對手預測。
        feasible = [c for c in strategy.last_candidates if c.feasible]
        self.assertTrue(feasible)
        for candidate in feasible:
            self.assertAlmostEqual(candidate.total_cost, 2.0 + 3.0 - candidate.prediction.progress[-1])
        selected = next(c for c in feasible if c.maneuver.target_d == decision.target_d)
        self.assertEqual(selected.total_cost, min(c.total_cost for c in feasible))

    def test_observed_speed_and_independent_attacker_history(self) -> None:
        first, second = (self.strategy({"front": self.route}) for _ in range(2))
        first.update(self.ego, (self.front,), self.track, 0.0)
        moved = observation(self.track, "front", 22.4, 0.6, 3.0)
        first.update(self.ego, (moved,), self.track, 0.1)
        second.update(self.ego, (moved,), self.track, 0.1)
        self.assertAlmostEqual(first.last_target_prediction.progress[-1], 4.0)
        self.assertAlmostEqual(second.last_target_prediction.progress[-1], 3.0)

    def test_observed_speed_across_finish(self) -> None:
        track = Track.create_oval(10.0, 10.0, 500)
        length = track_length(track)
        strategy = self.strategy({"front": self.route})
        ego = observation(track, "ego", 10.0, 0.0, 6.0)
        for time, s in ((0.0, length - 0.1), (0.1, 0.1)):
            strategy.update(ego, (observation(track, "front", s, 0.0, 3.0),), track, time)
        self.assertAlmostEqual(strategy.last_other_predictions["front"].progress[-1], 2.0, delta=0.01)

    def test_unconfigured_opponent_keeps_original_prediction(self) -> None:
        first, second = self.strategy(), self.strategy({"third": self.route})
        for strategy in (first, second):
            strategy.update(self.ego, (self.front,), self.track, 0.0)
        self.assertEqual(first.last_other_predictions, second.last_other_predictions)
        self.assertEqual(first.last_candidates, second.last_candidates)

    def test_main_loads_models_once_and_can_disable_loading(self) -> None:
        specs = (("ego", 0.0, 6.0), ("front", 4.0, 5.0))
        assignments = {"ego": StrategyAssignment("relative_overtake")}
        for enabled in (True, False):
            with self.subTest(enabled=enabled), patch("main.VEHICLE_SPECS", specs), \
                 patch("main.TRACK_NAME", "rounded_rectangle"), \
                 patch("main.STRATEGY_ASSIGNMENTS", assignments), patch("main.SIMULATION_SECONDS", 0.14), \
                 patch("main.GP_PREDICTION_ENABLED", enabled), \
                 patch("main.GP_MODEL_PATHS", {"front": "models/example.pkl"}), \
                 patch("main.load_gp_routes", return_value={"front": self.route}) as load, \
                 patch("main.SimulationVisualizer"), redirect_stdout(io.StringIO()):
                main.run_simulation()
                self.assertEqual(load.call_count, int(enabled))

    def test_loader_checks_schema_and_reuses_same_file(self) -> None:
        cache = PROJECT_ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            path = Path(directory) / "model.pkl"
            saved = {"pipeline": LinearModel(offset=0.6), "feature_names": ("opponent_s",),
                     "target_spec": {"name": "opponent_current_d"},
                     "target_timing": "current", "label_horizon_seconds": 0.0}
            path.write_bytes(pickle.dumps(saved))
            routes = load_gp_routes({"front": path, "third": path})
            self.assertIs(routes["front"], routes["third"])
            np.testing.assert_allclose(routes["front"].predict_d(np.array([1.0, 2.0])), [0.6, 0.6])
            for key, bad_value in (("feature_names", ("opponent_d",)), ("target_timing", "future"),
                                   ("label_horizon_seconds", 0.02),
                                   ("target_spec", {"name": "opponent_delta_d"})):
                path.write_bytes(pickle.dumps({**saved, key: bad_value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    GpRoute.load(path)
            with self.assertRaisesRegex(FileNotFoundError, "GP_MODEL_PATHS"):
                GpRoute.load(Path(directory) / "missing.pkl")


if __name__ == "__main__":
    unittest.main()
