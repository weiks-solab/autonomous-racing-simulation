"""Single-vehicle lap verification using the existing model and controller."""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Callable

import matplotlib
from matplotlib.figure import Figure
import numpy as np

from core.track import Track
from core.track_geometry import track_length, world_to_track
from core.track_loader import PROJECT_ROOT, project_file
from core.vehicle_geometry import oriented_rectangle_corners
from core.vehicle_state import VehicleState
from sim.map_background import MapBackground
from sim.simulated_vehicle import SimulatedVehicle


@dataclass(frozen=True)
class FollowingResult:
    completed: bool
    requested_laps: float
    max_seconds: float
    laps: float
    progress_m: float
    track_length_m: float
    simulation_seconds: float
    wall_seconds: float
    step_count: int
    max_abs_d_m: float
    rms_d_m: float
    max_body_corner_abs_d_m: float
    reference_outside_samples: int
    body_outside_samples: int
    s: tuple[float, ...]
    d: tuple[float, ...]


def follow_centerline(
    vehicle: SimulatedVehicle, track: Track, laps: float,
    max_seconds: float, on_step: Callable[[int], None] | None = None,
) -> FollowingResult:
    """Stop on measured forward progress, with a simulation-time limit.

    Boundary checks project all four physical body corners onto the CSV road;
    these are sampled checks, not swept-body or raster-obstacle collision tests.
    """
    if not track.closed:
        raise ValueError("lap verification requires a closed track")
    if any(not math.isfinite(v) or v <= 0.0 for v in (laps, max_seconds)):
        raise ValueError("laps and max_seconds must be positive and finite")
    if vehicle.controller.config.target_speed <= 0.0:
        raise ValueError("single-vehicle target speed must be positive")
    if len(vehicle.trajectory) != 1:
        raise ValueError("lap verification requires a fresh vehicle")
    length = track_length(track)
    dt = vehicle.model.config.time_step
    positions = [world_to_track(vehicle.state, track)]
    progress = 0.0
    started = time.perf_counter()
    for steps in range(1, math.floor(max_seconds / dt) + 1):
        vehicle.step(track)
        position = world_to_track(vehicle.state, track)
        progress += (position.s - positions[-1].s + length / 2.0) % length - length / 2.0
        positions.append(position)
        if on_step is not None:
            on_step(steps)
        if progress >= laps * length:
            break
    # 即使 timeout，也輸出實際進度與誤差，不宣稱完成兩圈。
    corner_maxima = []
    for state in vehicle.trajectory:
        corner_maxima.append(max(
            abs(world_to_track(VehicleState(float(x), float(y), state.yaw, state.velocity), track).d)
            for x, y in oriented_rectangle_corners(state, vehicle.geometry)
        ))
    d = np.array([p.d for p in positions])
    return FollowingResult(
        completed=progress >= laps * length,
        requested_laps=laps, max_seconds=max_seconds, laps=progress / length, progress_m=progress,
        track_length_m=length, simulation_seconds=(len(positions) - 1) * dt,
        wall_seconds=time.perf_counter() - started, step_count=len(positions) - 1,
        max_abs_d_m=float(np.max(np.abs(d))), rms_d_m=float(np.sqrt(np.mean(d * d))),
        max_body_corner_abs_d_m=float(max(corner_maxima)),
        reference_outside_samples=int(np.count_nonzero(np.abs(d) > track.width / 2.0 + 1e-9)),
        body_outside_samples=int(np.count_nonzero(np.array(corner_maxima) > track.width / 2.0 + 1e-9)),
        s=tuple(p.s for p in positions), d=tuple(d),
    )


def save_following_result(
    directory: str | Path, result: FollowingResult, vehicle: SimulatedVehicle,
    track: Track, figure: Figure, background: MapBackground | None,
) -> Path:
    """Save a small handoff bundle without replacing an existing experiment."""
    directory = project_file(directory)
    directory.mkdir(parents=True, exist_ok=False)
    summary = asdict(result)
    summary.pop("s")
    summary.pop("d")
    summary.update({
        "date": datetime.now(timezone(timedelta(hours=8))).date().isoformat(),
        "branch": _git_value("branch", "--show-current"),
        "base_commit": _git_value("rev-parse", "HEAD"),
        "version_note": "基於 base_commit，含未提交修改。",
        "scope": "單車中心線循跡；未驗證多車超車、GP 或 PNG 障礙物碰撞。",
        "backend": matplotlib.get_backend(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "vehicle_id": vehicle.vehicle_id,
        "initial_state": asdict(vehicle.trajectory[0]),
        "vehicle_geometry": asdict(vehicle.geometry),
        "model_config": asdict(vehicle.model.config),
        "controller_config": asdict(vehicle.controller.config),
        "track": {
            "closed": track.closed, "width_m": track.width,
            "point_count": len(track.centerline),
            "source": None if track.source is None else asdict(track.source),
            "direction": "依輸入中心線順序；d 向左為正；未旋轉或縮放 CSV。",
            "centerline_sha256": hashlib.sha256(track.centerline.astype("<f8").tobytes()).hexdigest(),
        },
        "map": None if background is None else {
            "resolution_m_per_pixel": background.resolution, "origin": background.origin,
            "files": {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (background.yaml_file, background.image_file)},
        },
        "trajectory_csv_stride_steps": 10,
        "source_sha256": {
            p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest()
            for p in ("main.py", "core/track.py", "core/track_factory.py", "core/track_loader.py",
                      "core/pure_pursuit.py", "core/vehicle_model.py", "sim/track_following.py",
                      "sim/visualizer.py", "sim/map_background.py")
        },
    })
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    with (directory / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "time_s", "x_m", "y_m", "yaw_rad", "velocity_mps", "s_m", "d_m"))
        indices = sorted(set(range(0, len(vehicle.trajectory), 10)) | {result.step_count})
        for index in indices:
            state = vehicle.trajectory[index]
            writer.writerow((index, index * vehicle.model.config.time_step, state.x, state.y,
                             state.yaw, state.velocity, result.s[index], result.d[index]))
    np.savetxt(directory / "centerline.csv", track.centerline, delimiter=",", header="x_m,y_m")
    figure.tight_layout()
    figure.savefig(directory / "following.png", dpi=180)
    return directory


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
