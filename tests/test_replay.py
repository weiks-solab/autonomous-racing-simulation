"""先算後播：播放節奏、跳幀，以及與一般模式相同的最後畫面。"""

from contextlib import redirect_stdout
import ctypes
from dataclasses import replace
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import main
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_state import VehicleState
from sim.replay import ReplayFrame, play_frames, precise_timer


def _frames(count: int, step: float = 0.02) -> tuple[list[ReplayFrame], list[SimpleNamespace]]:
    vehicle = SimpleNamespace(
        vehicle_id="a", geometry=VehicleGeometry(0.5, 0.3, 0.1),
        trajectory=[VehicleState(float(index), 0.0, 0.0, 1.0) for index in range(count + 1)],
    )
    frames = [
        ReplayFrame(
            time_seconds=(index + 1) * step,
            states={"a": vehicle.trajectory[index + 1]},
            trajectory_lengths={"a": index + 2},
            targets={"a": None}, track_positions={}, collision_pairs=frozenset(),
            strategy_labels={}, temporary_paths={}, candidate_snapshots={},
        )
        for index in range(count)
    ]
    return frames, [vehicle]


class FakeDisplay:
    """以假時鐘記錄播放：畫一幀花 draw_seconds，wait 讓時鐘前進要求的秒數。"""

    def __init__(self, draw_seconds: float) -> None:
        self.now = 0.0
        self.draw_seconds = draw_seconds
        self.shown: list[tuple[float, float]] = []  # (幀的模擬時間 x 座標, 顯示時的時鐘)
        self.waits: list[float] = []

    def clock(self) -> float:
        return self.now

    def update_vehicles(self, **kwargs) -> None:
        self.shown.append((kwargs["states"]["a"].x, self.now))
        self.now += self.draw_seconds

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += max(seconds, 0.0)


class PlayFramesTests(unittest.TestCase):
    def test_fast_drawing_shows_every_frame_on_schedule(self) -> None:
        frames, vehicles = _frames(10)
        display = FakeDisplay(draw_seconds=0.005)
        drawn = play_frames(display, frames, vehicles, clock=display.clock)
        self.assertEqual(drawn, 10)
        shown_at = [time for _, time in display.shown]
        np.testing.assert_allclose(np.diff(shown_at), 0.02, atol=1e-12)
        np.testing.assert_allclose(display.waits, 0.015, atol=1e-12)

    def test_speed_scales_the_schedule(self) -> None:
        frames, vehicles = _frames(10)
        display = FakeDisplay(draw_seconds=0.001)
        play_frames(display, frames, vehicles, speed=2.0, clock=display.clock)
        np.testing.assert_allclose(np.diff([time for _, time in display.shown]), 0.01, atol=1e-12)

    def test_slow_drawing_skips_frames_but_keeps_real_time_and_last_frame(self) -> None:
        frames, vehicles = _frames(50)
        display = FakeDisplay(draw_seconds=0.05)
        drawn = play_frames(display, frames, vehicles, clock=display.clock)
        self.assertLess(drawn, 50)
        self.assertEqual(display.shown[-1][0], 50.0)  # 最後一幀一定顯示
        # 每次顯示的幀都對應當下牆鐘，不會越播越慢。
        for frame_x, shown_at in display.shown:
            self.assertLessEqual(shown_at - (frame_x - 1) * 0.02, 0.05 + 1e-12)

    def test_empty_and_invalid_inputs(self) -> None:
        display = FakeDisplay(0.0)
        self.assertEqual(play_frames(display, [], [], clock=display.clock), 0)
        frames, vehicles = _frames(2)
        for speed in (0.0, -1.0, float("nan")):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                play_frames(display, frames, vehicles, speed=speed)

    def test_precise_timer_only_changes_the_windows_timer(self) -> None:
        winmm = Mock()
        with patch("sim.replay.sys.platform", "linux"), \
             patch.object(ctypes, "WinDLL", create=True, return_value=winmm) as loader:
            with precise_timer():
                pass
        loader.assert_not_called()
        with patch("sim.replay.sys.platform", "win32"), \
             patch.object(ctypes, "WinDLL", create=True, return_value=winmm):
            with self.assertRaises(RuntimeError), precise_timer():
                raise RuntimeError("playback interrupted")
        winmm.timeBeginPeriod.assert_called_once_with(1)
        winmm.timeEndPeriod.assert_called_once_with(1)


def _run(**kwargs) -> tuple[Mock, str]:
    output = io.StringIO()
    with patch("main.SIMULATION_SECONDS", 0.2), patch("main.SimulationVisualizer") as visualizer, \
         redirect_stdout(output):
        main.run_simulation(**kwargs)
    return visualizer, output.getvalue()


class ReplaySimulationTests(unittest.TestCase):
    def test_replay_ends_on_the_same_frame_and_does_not_change_the_simulation(self) -> None:
        normal, normal_output = _run()
        replayed, replay_output = _run(replay=True)
        self.assertEqual(replayed.call_count, 1)
        self.assertTrue(replayed.call_args.args[1].realtime)
        self.assertGreaterEqual(replayed.return_value.update_vehicles.call_count, 1)
        expected = normal.return_value.update_vehicles.call_args.kwargs
        actual = replayed.return_value.update_vehicles.call_args.kwargs
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            if key == "temporary_paths":
                self.assertEqual(set(actual[key]), set(expected[key]))
                for vehicle_id, path in expected[key].items():
                    np.testing.assert_array_equal(actual[key][vehicle_id].centerline, path.centerline)
            else:
                self.assertEqual(actual[key], expected[key], key)
        # 播放只多出計算進度，其餘終端摘要與一般模式相同。
        progress = [line for line in replay_output.splitlines() if line.startswith("Computing replay")]
        self.assertTrue(progress and progress[-1].startswith("Computing replay: 100%"))
        self.assertEqual(
            [line for line in replay_output.splitlines() if not line.startswith("Computing replay")],
            normal_output.splitlines(),
        )
        replayed.return_value.show.assert_called_once()

    def test_hidden_candidates_are_not_stored_for_playback(self) -> None:
        options = replace(main.DISPLAY_OPTIONS, show_candidate_paths=False, show_gp_predictions=False)
        with patch("main.DISPLAY_OPTIONS", options):
            replayed, _ = _run(replay=True)
        self.assertEqual(replayed.return_value.update_vehicles.call_args.kwargs["candidate_snapshots"], {})


if __name__ == "__main__":
    unittest.main()
