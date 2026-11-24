"""決策週期不受浮點累積誤差影響；同一橫移目標保持超過一圈時，參考線不會重播過渡。"""

import math
import unittest

import numpy as np

from core.track import Track
from core.track_factory import TrackConfig, create_track
from core.track_geometry import track_length, world_to_track
from core.vehicle_state import VehicleState
from strategy.common import decision_due
from strategy.defending import CostBasedDefenseStrategy, DefenseConfig
from strategy.offset_path import OffsetPathConfig, generate_offset_track
from strategy.overtaking import OvertakingConfig, OvertakingMode, RuleBasedOvertakingStrategy
from strategy.progress import ProgressConfig, ProgressStrategy
from tests.test_bicycle_prediction import observation as bicycle_observation
from tests.test_bicycle_prediction import prediction_config
from tests.test_defending_strategy import observation


def _first_offset(track: Track, vehicle, maneuver) -> float:
    """實際交給 Pure Pursuit 的臨時路徑，第一點相對中心線的 d。"""
    path = generate_offset_track(track, vehicle.track_position, maneuver, OffsetPathConfig(0.2, 7.0))
    x, y = path.centerline[0]
    return world_to_track(VehicleState(float(x), float(y), 0.0, 0.0), track).d


class DecisionPeriodTests(unittest.TestCase):
    def test_accumulated_step_times_do_not_skip_a_period(self) -> None:
        # dt=0.02 累加的時間有浮點誤差；0.1 s 週期應每 5 步準時決策一次。
        last, due_steps = -math.inf, []
        for step in range(300):
            if decision_due(step * 0.02, last, 0.1):
                due_steps.append(step)
                last = step * 0.02
        self.assertEqual(len(due_steps), 60)
        self.assertEqual(set(np.diff(due_steps)), {5})

    def test_strategy_replans_every_five_simulation_steps(self) -> None:
        track = Track(np.array(((0, 0), (100, 0), (100, 100), (0, 100))), width=2.0)
        config = ProgressConfig(candidate_offsets=(0.0,), decision_period=0.1, prediction_horizon=0.04)
        strategy = ProgressStrategy(config, prediction_config())
        vehicle = bicycle_observation(track, "ego", 20.0, 0.0, 3.0)
        evaluated = []
        for step in range(300):
            strategy.update(vehicle, (), track, step * 0.02)
            if strategy.last_evaluation_time == step * 0.02:
                evaluated.append(step)
        self.assertEqual(len(evaluated), 60)


class HeldManeuverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = create_track(TrackConfig(point_count=300))
        self.length = track_length(self.track)

    def test_defender_keeps_its_offset_after_a_full_lap_on_the_same_target(self) -> None:
        # 只留 0 與 0.6 兩個候選，並加大對齊權重，讓目標穩定保持在 0.6。
        config = DefenseConfig(collision_check_enabled=False, q_block=50.0, candidate_offsets=(0.0, 0.6))
        strategy = CostBasedDefenseStrategy(config, "ego")
        start = 5.0
        for step in range(int(self.length / 0.5) + 3):
            s = start + 0.5 * step
            # 第一次決策時防守車在中心線，之後假設已追上 0.6 m 的目標。
            defender = observation("defender", s, 0.0 if step == 0 else 0.6, 3.0, self.track)
            ego = observation("ego", s - 3.0, 0.6, 3.0, self.track)
            decision = strategy.update(defender, (ego,), self.track, step * 0.2)
            self.assertAlmostEqual(decision.target_d, 0.6)
        self.assertGreater(0.5 * step, self.length)  # 同一目標已保持超過一圈
        self.assertAlmostEqual(_first_offset(self.track, defender, decision.maneuver), 0.6, places=3)

    def test_locked_overtake_keeps_its_offset_after_a_full_lap(self) -> None:
        strategy = RuleBasedOvertakingStrategy(OvertakingConfig(prediction_horizon=2.0))
        start = 5.0
        for step in range(int(self.length / 0.5) + 3):
            s = start + 0.5 * step
            ego_d = 0.0 if step < 2 else 0.6
            ego = observation("ego", s, ego_d, 5.0, self.track)
            target = observation("target", s + 3.0, 0.0, 3.0, self.track)
            decision = strategy.update(ego, (target,), self.track)
        self.assertIs(decision.mode, OvertakingMode.OVERTAKE)
        self.assertAlmostEqual(decision.target_d, 0.6)
        self.assertAlmostEqual(_first_offset(self.track, ego, decision.maneuver), 0.6, places=3)


if __name__ == "__main__":
    unittest.main()
