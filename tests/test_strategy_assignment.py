"""策略分配、各車獨立狀態與資訊更新順序。"""

from dataclasses import replace
import unittest
from unittest.mock import Mock

from core.track import Track
from core.track_geometry import track_to_world
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.assignment import StrategyAssignment, StrategyManager
from strategy.base import StrategyDecision
from strategy.centerline import CenterlineStrategy
from strategy.defending import CostBasedDefenseStrategy, DefenseConfig
from strategy.observations import create_vehicle_observation
from strategy.offset_path import LateralManeuver
from strategy.overtaking import OvertakingConfig, RuleBasedOvertakingStrategy


class StrategyAssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track.create_oval(10.0, 5.0, 200, width=2.0)
        geometry = VehicleGeometry(0.55, 0.3, 0.12)
        self.observations = {}
        for vehicle_id, s, speed in (("red", 0.0, 5.0), ("blue", 5.0, 3.0),
                                     ("green", 15.0, 2.0)):
            point = track_to_world(s, 0.0, self.track)
            self.observations[vehicle_id] = create_vehicle_observation(
                vehicle_id, VehicleState(float(point[0]), float(point[1]), 1.57, speed),
                geometry, self.track,
            )
        self.overtaking_config = OvertakingConfig()
        self.defense_config = DefenseConfig()

    def manager(
        self,
        assignments: dict[str, StrategyAssignment],
        defense_enabled: bool = True,
    ) -> StrategyManager:
        return StrategyManager(
            tuple(self.observations), assignments,
            self.overtaking_config, self.defense_config, defense_enabled,
        )

    def test_unassigned_vehicle_follows_centerline(self) -> None:
        manager = self.manager({})
        decisions = manager.update(self.observations, self.track, 0.0)
        self.assertEqual(set(decisions), set(self.observations))
        for decision in decisions.values():
            self.assertEqual(decision.mode, "centerline")
            self.assertIsNone(decision.maneuver)

    def test_overtaking_adapter_preserves_original_algorithm(self) -> None:
        manager = self.manager({"red": StrategyAssignment("overtake")})
        original = RuleBasedOvertakingStrategy(self.overtaking_config)
        others = tuple(value for key, value in self.observations.items() if key != "red")
        for time in (0.0, 0.02):
            expected = original.update(self.observations["red"], others, self.track)
            actual = manager.update(self.observations, self.track, time)["red"]
            self.assertEqual(actual.mode, expected.mode.value)
            self.assertEqual(actual.target_d, expected.target_d)
            self.assertEqual(actual.maneuver, expected.maneuver)
            self.assertEqual(actual.reason, expected.reason)

    def test_same_strategy_creates_independent_instances(self) -> None:
        manager = self.manager({
            "red": StrategyAssignment("overtake"),
            "green": StrategyAssignment("overtake"),
        })
        decisions = manager.update(self.observations, self.track, 0.0)
        self.assertEqual(decisions["red"].mode, "prepare_overtake")
        self.assertEqual(decisions["green"].mode, "follow")
        self.assertIsNot(manager.strategies["red"], manager.strategies["green"])

    def test_multiple_defenders_have_independent_targets_and_state(self) -> None:
        manager = self.manager({
            "blue": StrategyAssignment("defend", target_vehicle_id="red"),
            "green": StrategyAssignment("defend", target_vehicle_id="blue"),
        })
        decisions = manager.update(self.observations, self.track, 0.0)
        self.assertEqual(decisions["blue"].mode, "defend")
        self.assertEqual(decisions["green"].mode, "cruise")
        self.assertIsNot(manager.strategies["blue"], manager.strategies["green"])

    def test_no_defense_switch_only_replaces_defending_assignments(self) -> None:
        manager = self.manager({
            "red": StrategyAssignment("overtake"),
            "blue": StrategyAssignment("defend", target_vehicle_id="red"),
        }, defense_enabled=False)
        self.assertEqual(manager.assignments["red"].name, "overtake")
        self.assertEqual(manager.assignments["blue"].name, "centerline")
        decisions = manager.update(self.observations, self.track, 0.0)
        self.assertEqual(decisions["blue"].mode, "centerline")

    def test_defender_receives_no_other_plans_in_either_update_order(self) -> None:
        for order in (("blue", "red", "green"), ("red", "green", "blue")):
            with self.subTest(order=order):
                self.check_plan_visibility(order)

    def check_plan_visibility(self, order: tuple[str, ...]) -> None:
        manager = StrategyManager(
            order,
            {"blue": StrategyAssignment("defend", target_vehicle_id="red"),
             "red": StrategyAssignment("overtake")},
            self.overtaking_config, self.defense_config,
        )
        previous = LateralManeuver(5.0, 0.0, -0.3, 3.0)
        new_attack = LateralManeuver(0.0, 0.0, 0.6, 6.0)
        observations = dict(self.observations)
        observations["blue"] = replace(observations["blue"], maneuver=previous)
        previous_attack = LateralManeuver(0.0, 0.0, -0.6, 6.0)
        observations["red"] = replace(observations["red"], maneuver=previous_attack)
        observations["green"] = replace(
            observations["green"], maneuver=LateralManeuver(15.0, 0.0, 0.3, 3.0),
        )
        attacker = Mock()
        attacker.update.return_value = StrategyDecision("overtake", 0.6, maneuver=new_attack)
        defender = Mock()
        defender.update.return_value = StrategyDecision("defend")
        manager.strategies["red"] = attacker
        manager.strategies["blue"] = defender
        manager.update(observations, self.track, 0.0)

        seen_by_attacker = {other.vehicle_id: other for other in attacker.update.call_args.args[1]}
        seen_by_defender = {other.vehicle_id: other for other in defender.update.call_args.args[1]}
        self.assertEqual(seen_by_attacker["blue"].maneuver, previous)
        for vehicle_id, other in seen_by_defender.items():
            self.assertIsNone(other.maneuver)
            self.assertEqual(other.state, observations[vehicle_id].state)
            self.assertEqual(other.track_position, observations[vehicle_id].track_position)
        self.assertEqual(observations["blue"].maneuver, previous)
        self.assertEqual(observations["red"].maneuver, previous_attack)

    def test_defense_adapter_preserves_original_algorithm(self) -> None:
        manager = self.manager({"blue": StrategyAssignment("defend", target_vehicle_id="red")})
        original = CostBasedDefenseStrategy(self.defense_config, "red")
        others = tuple(value for key, value in self.observations.items() if key != "blue")
        expected = original.update(self.observations["blue"], others, self.track, 0.0)
        actual = manager.update(self.observations, self.track, 0.0)["blue"]
        self.assertEqual(actual.mode, expected.mode.value)
        self.assertEqual(actual.target_d, expected.target_d)
        self.assertEqual(actual.maneuver, expected.maneuver)
        self.assertEqual(actual.reason, expected.reason)

    def test_invalid_assignments_are_rejected_before_running(self) -> None:
        for kwargs in ({"name": ""}, {"name": "defend"},
                       {"name": "defend", "target_vehicle_id": ""},
                       {"name": "centerline", "target_vehicle_id": "red"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StrategyAssignment(**kwargs)
        for assignments in (
            {"missing": StrategyAssignment("overtake")},
            {"red": StrategyAssignment("unknown")},
            {"blue": StrategyAssignment("defend", target_vehicle_id="blue")},
            {"blue": StrategyAssignment("defend", target_vehicle_id="missing")},
        ):
            with self.subTest(assignments=assignments), self.assertRaises(ValueError):
                self.manager(assignments)

    def test_registered_strategy_factory_creates_independent_instances(self) -> None:
        factory = Mock(side_effect=lambda vehicle_id: CenterlineStrategy())
        manager = StrategyManager(
            tuple(self.observations),
            {"red": StrategyAssignment("custom"), "green": StrategyAssignment("custom")},
            self.overtaking_config, self.defense_config,
            strategy_factories={"custom": factory},
        )
        self.assertEqual([call.args[0] for call in factory.call_args_list], ["red", "green"])
        self.assertIsNot(manager.strategies["red"], manager.strategies["green"])
        self.assertEqual(manager.update(self.observations, self.track, 0.0)["red"].mode, "centerline")

    def test_missing_or_mismatched_observations_are_rejected(self) -> None:
        manager = self.manager({})
        with self.assertRaises(ValueError):
            manager.update({}, self.track, 0.0)
        observations = dict(self.observations)
        observations["red"] = replace(observations["red"], vehicle_id="wrong")
        with self.assertRaises(ValueError):
            manager.update(observations, self.track, 0.0)


if __name__ == "__main__":
    unittest.main()
