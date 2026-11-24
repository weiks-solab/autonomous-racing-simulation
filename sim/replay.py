"""先算完再播放：記錄每一步交給畫面的資料，之後依牆鐘時間平順播放。

即時模式在同一執行緒邊算邊畫，重新決策的計算尖峰會讓畫面停頓；
播放模式把計算與顯示分開，畫面節奏只取決於模擬時間與播放倍率。
"""

from bisect import bisect_right
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import math
import sys
import time
from typing import Any, Callable, Collection, Iterator, Mapping, Protocol, Sequence

from core.pure_pursuit import PursuitTarget
from core.track import Track
from core.track_geometry import TrackPosition, world_to_track
from core.vehicle_state import VehicleState
from sim.simulated_vehicle import SimulatedVehicle
from strategy.base import CandidateSnapshot


@dataclass(frozen=True)
class ReplayFrame:
    """一個模擬步之後的畫面資料；軌跡只存長度，播放時再從完整軌跡切片。"""

    time_seconds: float
    states: Mapping[str, VehicleState]
    trajectory_lengths: Mapping[str, int]
    targets: Mapping[str, PursuitTarget | None]
    track_positions: Mapping[str, TrackPosition]
    collision_pairs: frozenset[tuple[str, str]]
    strategy_labels: Mapping[str, str]
    temporary_paths: Mapping[str, Track]
    candidate_snapshots: Mapping[str, CandidateSnapshot]


class FrameDisplay(Protocol):
    def update_vehicles(self, **kwargs: Any) -> None: ...

    def wait(self, seconds: float) -> None: ...


def record_frame(
    time_seconds: float,
    vehicles: Sequence[SimulatedVehicle],
    track: Track,
    collision_pairs: Collection[tuple[str, str]],
    strategy_labels: Mapping[str, str],
    temporary_paths: Mapping[str, Track],
    candidate_snapshots: Mapping[str, CandidateSnapshot],
) -> ReplayFrame:
    """保存目前這一步要畫的內容；資料都是不可變物件，之後不會被模擬改動。"""
    return ReplayFrame(
        time_seconds=time_seconds,
        states={vehicle.vehicle_id: vehicle.state for vehicle in vehicles},
        trajectory_lengths={vehicle.vehicle_id: len(vehicle.trajectory) for vehicle in vehicles},
        targets={vehicle.vehicle_id: vehicle.current_target for vehicle in vehicles},
        track_positions={
            vehicle.vehicle_id: world_to_track(vehicle.state, track) for vehicle in vehicles
        },
        collision_pairs=frozenset(collision_pairs),
        strategy_labels=dict(strategy_labels),
        temporary_paths=dict(temporary_paths),
        candidate_snapshots=dict(candidate_snapshots),
    )


def frame_kwargs(frame: ReplayFrame, vehicles: Sequence[SimulatedVehicle]) -> dict[str, Any]:
    """還原成 SimulationVisualizer.update_vehicles 的參數。"""
    return {
        "states": dict(frame.states),
        "trajectories": {
            vehicle.vehicle_id: vehicle.trajectory[:frame.trajectory_lengths[vehicle.vehicle_id]]
            for vehicle in vehicles
        },
        "targets": dict(frame.targets),
        "geometries": {vehicle.vehicle_id: vehicle.geometry for vehicle in vehicles},
        "track_positions": dict(frame.track_positions),
        "collision_pairs": set(frame.collision_pairs),
        "strategy_labels": dict(frame.strategy_labels),
        "temporary_paths": dict(frame.temporary_paths),
        "candidate_snapshots": dict(frame.candidate_snapshots),
    }


def play_frames(
    display: FrameDisplay,
    frames: Sequence[ReplayFrame],
    vehicles: Sequence[SimulatedVehicle],
    speed: float = 1.0,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    """依模擬時間播放；畫面跟不上就跳過中間幀，維持指定倍率。回傳實際畫出的幀數。"""
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"playback speed must be positive and finite (got {speed!r})")
    if not frames:
        return 0
    times = [frame.time_seconds for frame in frames]
    start = clock()
    index = drawn = 0
    while True:
        display.update_vehicles(**frame_kwargs(frames[index], vehicles))
        drawn += 1
        if index == len(frames) - 1:
            return drawn
        # 牆鐘對應的模擬時間；落後時直接跳到最新應顯示的幀，超前時等到下一幀。
        now = times[0] + (clock() - start) * speed
        index = max(index + 1, min(bisect_right(times, now) - 1, len(frames) - 1))
        display.wait((times[index] - times[0]) / speed - (clock() - start))


@contextmanager
def precise_timer() -> Iterator[None]:
    """Windows 預設計時器刻度約 15.6 ms，播放期間暫時改為 1 ms，讓每幀的等待準確。"""
    if sys.platform != "win32":
        yield
        return
    winmm = ctypes.WinDLL("winmm")
    winmm.timeBeginPeriod(1)
    try:
        yield
    finally:
        winmm.timeEndPeriod(1)
