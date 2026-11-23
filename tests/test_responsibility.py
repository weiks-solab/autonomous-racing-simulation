"""避撞責任約束：ℓ 的形狀、逐對檢查，以及進攻與防守策略的接入。"""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from core.track import Track
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.collision_check import PredictedCollision
from strategy.defending import CostBasedDefenseStrategy, DefenseConfig
from strategy.defense_conditions import predict_rear
from strategy.relative_overtaking import RelativeOvertakingConfig, RelativeOvertakingStrategy
from core.vehicle_geometry import CollisionConfig
from strategy.responsibility import (
    PairPrediction, ResponsibilityConfig, ResponsibilityResult, body_centers, center_gap_now,
    collision_length, evaluate_responsibility, first_responsibility_violation, leader_penalty,
    responsibility_bound, responsibility_constraint,
)
from tests.test_bicycle_prediction import observation as bicycle_observation, prediction_config
from tests.test_defending_strategy import observation as defense_observation

CONFIG = ResponsibilityConfig()  # r_col=0.6、a=5、b=4.5
SOFT = ResponsibilityConfig(leader_soft=True)  # 當下在前只計懲罰
OVERLAP = ResponsibilityConfig(overlap_rule=True)  # 重疊中兩車負責，脫離後只剩後車
FOLLOWER_LIMIT = 1.0 / (1.0 + math.exp(4.5))  # 後車 ℓ 的上限 ≈ 0.011 m²
GEOMETRY = VehicleGeometry(0.55, 0.3, 0.12)


def straight_states(x: float, y: float, count: int, step: float = 0.0) -> tuple[VehicleState, ...]:
    """沿 +x 的等速狀態；step 為每個時間點前進的距離。"""
    return tuple(VehicleState(x + index * step, y, 0.0, 0.0) for index in range(count))


class ResponsibilityBoundTests(unittest.TestCase):
    def test_zero_when_side_by_side_and_bounded_above_by_a_small_value(self) -> None:
        self.assertEqual(float(responsibility_bound(0.0, CONFIG)), 0.0)
        ahead = responsibility_bound(np.linspace(0.01, 10.0, 50), CONFIG)
        self.assertTrue(np.all(ahead >= 0.0))
        self.assertTrue(np.all(ahead <= FOLLOWER_LIMIT + 1e-12))  # tanh 計算的捨入誤差
        self.assertAlmostEqual(float(responsibility_bound(10.0, CONFIG)), FOLLOWER_LIMIT)

    def test_leader_is_relieved_and_matches_the_paper_figure(self) -> None:
        behind = responsibility_bound(np.linspace(-10.0, -0.01, 50), CONFIG)
        self.assertTrue(np.all(behind <= 0.0))
        self.assertAlmostEqual(float(responsibility_bound(-10.0, CONFIG)), FOLLOWER_LIMIT - 1.0)
        # 手算：Δs=−0.7 → 1/(1+e^4.5) − 1/(1+e^1)
        expected = FOLLOWER_LIMIT - 1.0 / (1.0 + math.e)
        self.assertAlmostEqual(float(responsibility_bound(-0.7, CONFIG)), expected)
        self.assertAlmostEqual(expected, -0.2580, places=4)

    def test_monotonic_and_one_vehicle_of_each_pair_carries_the_responsibility(self) -> None:
        values = np.linspace(-3.0, 3.0, 301)
        bounds = responsibility_bound(values, CONFIG)
        self.assertTrue(np.all(np.diff(bounds) > 0.0))
        mirrored = responsibility_bound(-values, CONFIG)
        self.assertTrue(np.all(bounds * mirrored <= 0.0))

    def test_extreme_values_do_not_overflow_and_invalid_inputs_are_rejected(self) -> None:
        extreme = responsibility_bound((-1e6, 1e6), CONFIG)
        self.assertTrue(np.all(np.isfinite(extreme)))
        with self.assertRaisesRegex(ValueError, "finite"):
            responsibility_bound((0.0, math.nan), CONFIG)
        for kwargs in ({"collision_radius": 0.0}, {"collision_radius": math.inf},
                       {"slope": 0.0}, {"slope": -1.0}, {"offset": math.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ResponsibilityConfig(**kwargs)


class ResponsibilityViolationTests(unittest.TestCase):
    times = (0.0, 0.1, 0.2)

    def pair(self, vehicle_id: str, x: float, y: float, delta_s: float) -> PairPrediction:
        return PairPrediction(vehicle_id, GEOMETRY, straight_states(x, y, 3),
                              np.full(3, delta_s))

    def test_body_centers_use_the_collision_rectangle_center(self) -> None:
        centers = body_centers((VehicleState(1.0, 2.0, math.pi / 2.0, 0.0),), GEOMETRY)
        np.testing.assert_allclose(centers, ((1.0, 2.12),), atol=1e-12)

    def test_same_distance_rejects_the_follower_but_not_the_leader(self) -> None:
        own = straight_states(0.0, 0.0, 3)
        # 中心距離 0.55 m < r_col；後車須保持 r_col，前車領先 0.55 m 已放寬。
        follower = first_responsibility_violation(
            self.times, own, GEOMETRY, (self.pair("front", 0.55, 0.0, 0.55),), CONFIG)
        leader = first_responsibility_violation(
            self.times, own, GEOMETRY, (self.pair("rear", -0.55, 0.0, -0.55),), CONFIG)
        self.assertEqual(follower, PredictedCollision("front", 0.1, kind="responsibility"))
        self.assertIsNone(leader)

    def test_side_by_side_both_vehicles_are_responsible(self) -> None:
        own = straight_states(0.0, 0.0, 3)
        for delta_s in (-0.02, 0.0, 0.02):
            with self.subTest(delta_s=delta_s):
                result = first_responsibility_violation(
                    self.times, own, GEOMETRY, (self.pair("side", 0.0, 0.55, delta_s),), CONFIG)
                self.assertIsNotNone(result)

    def test_current_time_is_not_checked(self) -> None:
        own = straight_states(0.0, 0.0, 3)
        # 只有 t=0 太近，之後拉開。
        other = PairPrediction("side", GEOMETRY,
                               (VehicleState(0.0, 0.3, 0.0, 0.0),) + straight_states(0.0, 1.0, 2),
                               np.zeros(3))
        values = responsibility_constraint(own, GEOMETRY, other, CONFIG)
        self.assertLess(values[0], 0.0)
        self.assertIsNone(first_responsibility_violation(self.times, own, GEOMETRY, (other,), CONFIG))

    def test_pairs_are_checked_independently_and_the_earliest_violation_wins(self) -> None:
        own = straight_states(0.0, 0.0, 3)
        far_behind = self.pair("behind", -3.0, 0.0, -3.0)
        # 前車 t=0.1 仍遠、t=0.2 才進入 r_col。
        closing = PairPrediction("ahead", GEOMETRY, straight_states(1.5, 0.0, 3, -0.5),
                                 np.array((1.5, 1.0, 0.5)))
        side = self.pair("beside", 0.0, 0.5, 0.0)
        result = first_responsibility_violation(
            self.times, own, GEOMETRY, (far_behind, closing), CONFIG)
        self.assertEqual(result, PredictedCollision("ahead", 0.2, kind="responsibility"))
        both = first_responsibility_violation(
            self.times, own, GEOMETRY, (closing, side, far_behind), CONFIG)
        self.assertEqual(both.vehicle_id, "beside")
        self.assertEqual(both.time_seconds, 0.1)
        self.assertIn("responsibility violation with beside", both.reason)
        with self.assertRaisesRegex(ValueError, "same sample times"):
            responsibility_constraint(own[:2], GEOMETRY, side, CONFIG)


class LeaderSoftTests(unittest.TestCase):
    times = (0.0, 0.1, 0.2)

    def pair(self, vehicle_id: str, x: float, delta_s) -> PairPrediction:
        return PairPrediction(vehicle_id, GEOMETRY, straight_states(x, 0.0, 3),
                              np.broadcast_to(np.asarray(delta_s, dtype=float), (3,)).copy())

    def test_disabled_matches_the_hard_constraint(self) -> None:
        own = straight_states(0.0, 0.0, 3)
        others = (self.pair("rear", -0.4, -0.4), self.pair("front", 0.4, 0.4))
        self.assertEqual(
            evaluate_responsibility(self.times, own, GEOMETRY, others, CONFIG),
            ResponsibilityResult(first_responsibility_violation(
                self.times, own, GEOMETRY, others, CONFIG)),
        )
        # 原本的硬約束下，前車在 0.4 m 內也會被剔除。
        leader_only = evaluate_responsibility(self.times, own, GEOMETRY, others[:1], CONFIG)
        self.assertIsNotNone(leader_only.violation)

    def test_leader_is_penalized_instead_of_rejected(self) -> None:
        own = straight_states(0.0, 0.0, 3)
        result = evaluate_responsibility(
            self.times, own, GEOMETRY, (self.pair("rear", -0.4, -0.4),), SOFT)
        self.assertIsNone(result.violation)
        # 中心距離 0.4 m，不足 0.2 m；k=1、2 平均 0.2² = 0.04 m²。
        self.assertAlmostEqual(result.leader_penalty, 0.04)
        clear = evaluate_responsibility(
            self.times, own, GEOMETRY, (self.pair("rear", -1.0, -1.0),), SOFT)
        self.assertEqual(clear, ResponsibilityResult(None, 0.0))

    def test_follower_and_side_by_side_stay_hard(self) -> None:
        own = straight_states(0.0, 0.0, 3)
        for name, x, delta_s in (("front", 0.4, 0.4), ("side", 0.0, 0.0)):
            with self.subTest(name=name):
                result = evaluate_responsibility(
                    self.times, own, GEOMETRY, (self.pair(name, x, delta_s),), SOFT)
                self.assertEqual(result.violation.vehicle_id, name)
                self.assertEqual(result.leader_penalty, 0.0)

    def test_role_uses_the_current_gap_and_pairs_are_split(self) -> None:
        own = straight_states(0.0, 0.0, 3)
        # 預測中對方會超前，但決策當下自車在前，因此整段只計懲罰。
        passing = self.pair("passing", -0.4, (-0.4, 0.1, 0.5))
        result = evaluate_responsibility(self.times, own, GEOMETRY, (passing,), SOFT)
        self.assertIsNone(result.violation)
        self.assertGreater(result.leader_penalty, 0.0)
        mixed = evaluate_responsibility(
            self.times, own, GEOMETRY, (passing, self.pair("front", 0.4, 0.4)), SOFT)
        self.assertEqual(mixed.violation.vehicle_id, "front")
        self.assertAlmostEqual(mixed.leader_penalty, result.leader_penalty)
        self.assertEqual(leader_penalty(own[:1], GEOMETRY, PairPrediction(
            "x", GEOMETRY, own[:1], np.zeros(1)), SOFT), 0.0)

    def test_invalid_leader_soft_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "leader_soft"):
            ResponsibilityConfig(leader_soft="yes")


class OverlapRuleTests(unittest.TestCase):
    """GEOMETRY 沒有另設碰撞框，L̄ = 車長 0.55 m。"""

    times = (0.0, 0.1, 0.2)

    def test_overlapping_leader_is_responsible_without_bound_relief(self) -> None:
        own = straight_states(0.0, 0.0, 3)
        # 當下領先 0.3 m（重疊中），預測再拉開到 0.5 m，中心距離仍小於 r_col。
        other = PairPrediction("rear", GEOMETRY, straight_states(-0.3, 0.0, 3, -0.1),
                               np.array((-0.3, -0.4, -0.5)))
        result = evaluate_responsibility(self.times, own, GEOMETRY, (other,), OVERLAP)
        self.assertEqual(result.violation, PredictedCollision("rear", 0.1, kind="responsibility"))
        # 對照：a=40 的 ℓ 與 leader_soft 都會讓前車免責。
        steep = ResponsibilityConfig(slope=40.0)
        self.assertIsNone(evaluate_responsibility(self.times, own, GEOMETRY, (other,), steep).violation)
        self.assertIsNone(evaluate_responsibility(self.times, own, GEOMETRY, (other,), SOFT).violation)

    def test_clear_leader_is_free_even_if_predicted_to_be_caught(self) -> None:
        own = straight_states(0.0, 0.0, 3)
        # 當下領先 0.7 m（已脫離），預測中對方追到 0.36 m 內。
        catching = PairPrediction(
            "rear", GEOMETRY,
            (VehicleState(-0.7, 0.2, 0.0, 0.0), VehicleState(-0.3, 0.2, 0.0, 0.0),
             VehicleState(0.0, 0.2, 0.0, 0.0)),
            np.array((-0.7, -0.3, 0.0)))
        result = evaluate_responsibility(self.times, own, GEOMETRY, (catching,), OVERLAP)
        self.assertIsNone(result.violation)
        self.assertGreater(result.leader_penalty, 0.0)  # 策略權重為 0 時不影響成本

    def test_clear_follower_is_responsible(self) -> None:
        own = straight_states(0.0, 0.0, 3, 0.3)
        front = PairPrediction("front", GEOMETRY, straight_states(0.7, 0.0, 3),
                               np.array((0.7, 0.4, 0.1)))
        result = evaluate_responsibility(self.times, own, GEOMETRY, (front,), OVERLAP)
        self.assertEqual(result.violation, PredictedCollision("front", 0.1, kind="responsibility"))
        self.assertEqual(result.leader_penalty, 0.0)

    def test_overlap_length_uses_collision_boxes_and_body_centers(self) -> None:
        long_box = VehicleGeometry(0.55, 0.3, 0.12, collision=CollisionConfig(1.0, 0.4))
        self.assertEqual(collision_length(GEOMETRY), 0.55)
        self.assertEqual(collision_length(long_box), 1.0)
        own = straight_states(0.0, 0.0, 3)
        other_states = straight_states(-0.8, 0.0, 3)
        short = PairPrediction("rear", GEOMETRY, other_states, np.full(3, -0.8))
        long = PairPrediction("rear", long_box, other_states, np.full(3, -0.8))
        # r_col=1.0 時中心距離 0.8 m 不足；0.8 m 超過 0.55 m 框長而免責，換成 1.0 m 碰撞框時仍重疊而負責。
        wide = ResponsibilityConfig(collision_radius=1.0, overlap_rule=True)
        free = evaluate_responsibility(self.times, own, GEOMETRY, (short,), wide)
        self.assertIsNone(free.violation)
        self.assertGreater(free.leader_penalty, 0.0)
        self.assertIsNotNone(evaluate_responsibility(self.times, own, long_box, (long,), wide).violation)
        # 後軸差 −0.65 m，但對方中心比後軸多前移 0.15 m，中心差 −0.5 m 仍重疊。
        forward_center = VehicleGeometry(0.55, 0.3, 0.27)
        shifted = PairPrediction("rear", forward_center, straight_states(-0.65, 0.0, 3),
                                 np.full(3, -0.65))
        self.assertAlmostEqual(center_gap_now(GEOMETRY, shifted), -0.5)
        self.assertIsNotNone(
            evaluate_responsibility(self.times, own, GEOMETRY, (shifted,), OVERLAP).violation)

    def test_invalid_overlap_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap_rule must be a bool"):
            ResponsibilityConfig(overlap_rule="yes")
        with self.assertRaisesRegex(ValueError, "cannot both"):
            ResponsibilityConfig(leader_soft=True, overlap_rule=True)


class RelativeOvertakingResponsibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))), width=2.0)
        self.base = RelativeOvertakingConfig(prediction_horizon=1.0)
        self.config = replace(self.base, responsibility=CONFIG)

    def strategy(self, config: RelativeOvertakingConfig) -> RelativeOvertakingStrategy:
        return RelativeOvertakingStrategy(
            config, prediction_config(),
            {name: prediction_config(5.0) for name in ("front", "behind")},
        )

    def test_disabled_keeps_the_rectangle_filter(self) -> None:
        ego = bicycle_observation(self.track, "ego", 20.0, 0.0, 6.0)
        front = bicycle_observation(self.track, "front", 22.0, -0.6, 3.0)
        with patch("strategy.relative_overtaking.evaluate_responsibility") as check:
            self.strategy(self.base).update(ego, (front,), self.track, 0.0)
        check.assert_not_called()
        with patch("strategy.relative_overtaking.first_predicted_collision") as overlap:
            self.strategy(self.config).update(ego, (front,), self.track, 0.0)
        overlap.assert_not_called()

    def test_attacker_behind_must_keep_clear_of_the_front_vehicle(self) -> None:
        ego = bicycle_observation(self.track, "ego", 20.0, 0.0, 6.0)
        front = bicycle_observation(self.track, "front", 21.0, -0.6, 3.0)
        strategy = self.strategy(self.config)
        decision = strategy.update(ego, (front,), self.track, 0.0)
        cut_in = next(item for item in strategy.last_candidates if item.maneuver.target_d == -0.6)
        self.assertFalse(cut_in.feasible)
        self.assertEqual(cut_in.collision.kind, "responsibility")
        self.assertEqual(cut_in.collision.vehicle_id, "front")
        self.assertEqual(decision.mode, "attack")
        self.assertTrue(any(item.feasible for item in strategy.last_candidates))

    def test_attacker_ahead_is_relieved_but_the_same_gap_behind_is_rejected(self) -> None:
        # 同車道、同速、後軸相距 0.55 m：矩形剛好接觸，中心距離小於 r_col。
        ego = bicycle_observation(self.track, "ego", 20.0, 0.0, 6.0)
        behind = bicycle_observation(self.track, "behind", 19.45, 0.0, 6.0)
        leader = self.strategy(self.config)
        leader.update(ego, (behind,), self.track, 0.0)
        center = next(item for item in leader.last_candidates if item.maneuver.target_d == 0.0)
        self.assertTrue(center.feasible)
        rectangle = self.strategy(self.base)
        rectangle.update(ego, (behind,), self.track, 0.0)
        self.assertTrue(all(not item.feasible for item in rectangle.last_candidates))

        follower_ego = bicycle_observation(self.track, "ego", 19.45, 0.0, 6.0)
        front = bicycle_observation(self.track, "front", 20.0, 0.0, 6.0)
        follower = self.strategy(self.config)
        follower.update(follower_ego, (front,), self.track, 0.0)
        center = next(item for item in follower.last_candidates if item.maneuver.target_d == 0.0)
        self.assertFalse(center.feasible)
        self.assertEqual(center.collision.kind, "responsibility")

    def test_invalid_responsibility_setting_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RelativeOvertakingConfig(responsibility=0.6)
        for weight in (-1.0, math.nan):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                RelativeOvertakingConfig(leader_penalty_weight=weight)

    def test_leading_attacker_pays_a_penalty_instead_of_rejection(self) -> None:
        ego = bicycle_observation(self.track, "ego", 20.0, 0.0, 6.0)
        behind = bicycle_observation(self.track, "behind", 19.5, 0.0, 6.0)
        hard = self.strategy(self.config)
        hard.update(ego, (behind,), self.track, 0.0)
        hard_center = next(item for item in hard.last_candidates if item.maneuver.target_d == 0.0)
        self.assertFalse(hard_center.feasible)  # a=5 時領先 0.5 m 仍要負責
        soft = self.strategy(replace(self.config, responsibility=SOFT, leader_penalty_weight=10.0))
        decision = soft.update(ego, (behind,), self.track, 0.0)
        center = next(item for item in soft.last_candidates if item.maneuver.target_d == 0.0)
        self.assertTrue(center.feasible)
        self.assertGreater(center.leader_cost, 0.0)
        self.assertAlmostEqual(
            center.total_cost,
            center.relative_cost + center.safety_cost + center.deviation_cost + center.leader_cost,
        )
        terms = dict(next(item for item in decision.candidate_snapshot.candidates
                          if item.maneuver.target_d == 0.0).cost_terms)
        self.assertAlmostEqual(terms["leader"], center.leader_cost)
        free = self.strategy(replace(self.config, responsibility=SOFT))
        free.update(ego, (behind,), self.track, 0.0)
        self.assertTrue(all(item.leader_cost == 0.0 for item in free.last_candidates))

    def test_overlap_rule_makes_an_overlapping_leader_avoid_the_car_behind(self) -> None:
        ego = bicycle_observation(self.track, "ego", 20.0, 0.0, 6.0)
        config = replace(self.config, responsibility=OVERLAP)
        for behind_s, expected in ((19.7, False), (19.2, True)):
            with self.subTest(behind_s=behind_s):
                behind = bicycle_observation(self.track, "behind", behind_s, -0.55, 6.0)
                strategy = self.strategy(config)
                strategy.update(ego, (behind,), self.track, 0.0)
                toward = next(item for item in strategy.last_candidates
                              if item.maneuver.target_d == -0.6)
                center = next(item for item in strategy.last_candidates
                              if item.maneuver.target_d == 0.0)
                # 領先 0.3 m 仍重疊，切向後車被剔除；領先 0.8 m 已脫離，不受限。
                self.assertEqual(toward.feasible, expected)
                self.assertTrue(center.feasible)
                if not expected:
                    self.assertEqual(toward.collision.kind, "responsibility")

    def test_following_attacker_is_still_rejected_with_leader_soft(self) -> None:
        ego = bicycle_observation(self.track, "ego", 19.45, 0.0, 6.0)
        front = bicycle_observation(self.track, "front", 20.0, 0.0, 6.0)
        strategy = self.strategy(replace(self.config, responsibility=SOFT, leader_penalty_weight=10.0))
        strategy.update(ego, (front,), self.track, 0.0)
        center = next(item for item in strategy.last_candidates if item.maneuver.target_d == 0.0)
        self.assertFalse(center.feasible)
        self.assertEqual(center.collision.kind, "responsibility")


class DefenseResponsibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))), width=2.0)
        self.config = DefenseConfig(
            collision_check_enabled=False, q_block=1e6, collision_horizon=1.0,
            responsibility=CONFIG,
        )
        self.defender = defense_observation("front", 20.0, 0.0, 3.0, self.track)

    def candidate(self, strategy: CostBasedDefenseStrategy, target_d: float):
        return next(item for item in strategy.last_candidates if item.target_d == target_d)

    def test_alongside_attacker_blocks_the_cut_across_candidate(self) -> None:
        alongside = defense_observation("rear", 19.9, 0.6, 3.0, self.track)
        strategy = CostBasedDefenseStrategy(self.config, "rear")
        decision = strategy.update(self.defender, (alongside,), self.track, 0.0)
        cut = self.candidate(strategy, 0.6)
        self.assertFalse(cut.feasible)
        self.assertEqual(cut.collision.kind, "responsibility")
        self.assertTrue(self.candidate(strategy, 0.0).feasible)
        self.assertLess(decision.target_d, 0.6)
        # 原本設定（守方無碰撞條件）會直接對齊攻方。
        free = CostBasedDefenseStrategy(replace(self.config, responsibility=None), "rear")
        self.assertEqual(free.update(self.defender, (alongside,), self.track, 0.0).target_d, 0.6)

    def test_attacker_clearly_behind_does_not_restrict_blocking(self) -> None:
        behind = defense_observation("rear", 18.0, 0.6, 3.0, self.track)
        strategy = CostBasedDefenseStrategy(self.config, "rear")
        decision = strategy.update(self.defender, (behind,), self.track, 0.0)
        self.assertTrue(all(item.feasible for item in strategy.last_candidates))
        self.assertEqual(decision.target_d, 0.6)

    def test_leading_defender_is_relieved_within_the_collision_radius(self) -> None:
        # 同車道、後車中心落後 0.55 m：距離小於 r_col，但守方在前而放寬。
        close = defense_observation("rear", 19.45, 0.0, 3.0, self.track)
        strategy = CostBasedDefenseStrategy(self.config, "rear")
        strategy.update(self.defender, (close,), self.track, 0.0)
        self.assertTrue(self.candidate(strategy, 0.0).feasible)

    def test_every_other_vehicle_is_checked_and_the_front_one_binds(self) -> None:
        attacker = defense_observation("rear", 15.0, 0.6, 3.0, self.track)
        third = defense_observation("third", 20.5, 0.0, 3.0, self.track)
        strategy = CostBasedDefenseStrategy(self.config, "rear")
        strategy.update(self.defender, (attacker, third), self.track, 0.0)
        self.assertEqual(len(strategy.last_other_predictions), 2)
        self.assertEqual(strategy.last_other_gaps, (-5.0, 0.5))
        center = self.candidate(strategy, 0.0)
        self.assertFalse(center.feasible)
        self.assertEqual(center.collision.vehicle_id, "third")

    def test_bicycle_rollout_uses_the_same_check_times(self) -> None:
        alongside = defense_observation("rear", 19.9, 0.6, 3.0, self.track)
        strategy = CostBasedDefenseStrategy(self.config, "rear", prediction_config(3.0))
        strategy.update(self.defender, (alongside,), self.track, 0.0)
        self.assertFalse(self.candidate(strategy, 0.6).feasible)
        self.assertTrue(self.candidate(strategy, 0.0).feasible)

    def test_disabled_responsibility_keeps_the_original_rectangle_condition(self) -> None:
        config = replace(self.config, responsibility=None, collision_check_enabled=True)
        strategy = CostBasedDefenseStrategy(config, "rear")
        strategy.update(self.defender, (defense_observation("rear", 18.0, 0.6, 3.0, self.track),),
                        self.track, 0.0)
        self.assertEqual(strategy.last_other_predictions, ())
        self.assertIsNotNone(strategy.last_rear_prediction)

    def test_predicted_pass_no_longer_forces_the_leading_defender_to_yield(self) -> None:
        # 攻方落後 2 m、快 3 m/s，等速預測約 0.67 s 後穿過守方。
        faster = defense_observation("rear", 18.0, 0.6, 6.0, self.track)
        base = replace(self.config, q_block=15.0)
        hard = CostBasedDefenseStrategy(base, "rear")
        hard.update(self.defender, (faster,), self.track, 0.0)
        self.assertFalse(self.candidate(hard, 0.6).feasible)

        free = CostBasedDefenseStrategy(replace(base, responsibility=SOFT), "rear")
        decision = free.update(self.defender, (faster,), self.track, 0.0)
        self.assertTrue(all(item.feasible for item in free.last_candidates))
        self.assertEqual(decision.target_d, 0.6)
        self.assertEqual(self.candidate(free, 0.6).leader_cost, 0.0)

        weighted = CostBasedDefenseStrategy(
            replace(base, responsibility=SOFT, leader_penalty_weight=1e4), "rear")
        decision = weighted.update(self.defender, (faster,), self.track, 0.0)
        blocking = self.candidate(weighted, 0.6)
        self.assertTrue(blocking.feasible)
        self.assertGreater(blocking.leader_cost, 0.0)
        self.assertAlmostEqual(blocking.race_cost - self.candidate(free, 0.6).race_cost,
                               blocking.leader_cost)
        self.assertLess(decision.target_d, 0.6)

    def test_overlap_rule_for_the_defender(self) -> None:
        config = replace(self.config, responsibility=OVERLAP)
        # 攻方落後 0.3 m 並排（重疊中）：守方不能切向攻方。
        alongside = defense_observation("rear", 19.7, 0.6, 3.0, self.track)
        strategy = CostBasedDefenseStrategy(config, "rear")
        strategy.update(self.defender, (alongside,), self.track, 0.0)
        self.assertFalse(self.candidate(strategy, 0.6).feasible)
        self.assertEqual(self.candidate(strategy, 0.6).collision.kind, "responsibility")
        self.assertTrue(self.candidate(strategy, 0.0).feasible)
        soft = CostBasedDefenseStrategy(replace(config, responsibility=SOFT), "rear")
        soft.update(self.defender, (alongside,), self.track, 0.0)
        self.assertTrue(self.candidate(soft, 0.6).feasible)  # leader_soft 下守方免責
        # 攻方落後 0.8 m、快 3 m/s（預測會超越）：尚未重疊，守方仍可阻擋。
        faster = defense_observation("rear", 19.2, 0.6, 6.0, self.track)
        blocking = CostBasedDefenseStrategy(config, "rear")
        decision = blocking.update(self.defender, (faster,), self.track, 0.0)
        self.assertTrue(all(item.feasible for item in blocking.last_candidates))
        self.assertEqual(decision.target_d, 0.6)

    def test_leader_soft_keeps_the_front_vehicle_hard_for_the_defender(self) -> None:
        attacker = defense_observation("rear", 19.6, 0.0, 3.0, self.track)
        # 前車在前 0.9 m、慢 1 m/s：留在中心約 0.34 s 後進入 r_col，橫移到 −0.6 則可避開。
        third = defense_observation("third", 20.9, 0.2, 2.0, self.track)
        config = replace(self.config, responsibility=SOFT, leader_penalty_weight=100.0,
                         transition_length=1.0)
        strategy = CostBasedDefenseStrategy(config, "rear")
        strategy.update(self.defender, (attacker, third), self.track, 0.0)
        center = self.candidate(strategy, 0.0)
        self.assertFalse(center.feasible)
        self.assertEqual(center.collision.vehicle_id, "third")
        away = self.candidate(strategy, -0.6)
        self.assertTrue(away.feasible)
        self.assertGreater(away.leader_cost, 0.0)  # 與後方 0.4 m 的攻方仍在 r_col 內

    def test_rear_prediction_records_progress_and_invalid_setting_is_rejected(self) -> None:
        rear = defense_observation("rear", 18.0, 0.6, 3.0, self.track)
        prediction = predict_rear(rear, self.track, 1.0, 0.1)
        self.assertEqual(len(prediction.progress), len(prediction.times))
        self.assertAlmostEqual(prediction.progress[-1], 3.0)
        with self.assertRaises(ValueError):
            DefenseConfig(responsibility="on")
        with self.assertRaises(ValueError):
            DefenseConfig(leader_penalty_weight=-1.0)


if __name__ == "__main__":
    unittest.main()
