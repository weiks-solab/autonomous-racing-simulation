"""使用目前 main 的配置，記錄橫向目標保持時間；不指定控制命令。"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")

from core.track_geometry import signed_longitudinal_gap, track_length
from generate_training_data import _demo_settings
from sim.data_generation import EpisodeScenario, VehicleInitialCondition, run_episode
from strategy.probe_following import ProbeFollowingStrategy
from training_data import TrainingDataCollector
import main


def verify(label: str) -> dict:
    settings, specs, assignments = _demo_settings(False)
    rear_id = next(key for key, assignment in assignments.items() if assignment.name == "probe_follow")
    front_id = next(key for key, assignment in assignments.items()
                    if assignment.name == "defend" and assignment.target_vehicle_id == rear_id)
    scenario = EpisodeScenario("probe-hold", tuple(
        VehicleInitialCondition(key, s, 0.0, 0.0, speed) for key, s, speed in specs
    ), assignments)
    decisions = []
    original_update = ProbeFollowingStrategy.update

    def record_update(strategy, *args, **kwargs):
        decision = original_update(strategy, *args, **kwargs)
        decisions.append(decision)
        return decision

    collector = TrainingDataCollector(settings.model_config.time_step)
    with patch.object(ProbeFollowingStrategy, "update", new=record_update):
        run_episode(scenario, settings, collector, main.SIMULATION_SECONDS)
    frames = collector.episodes[0].frames
    length = track_length(settings.track)
    gap = signed_longitudinal_gap(frames[0].observations[rear_id].track_position,
                                 frames[0].observations[front_id].track_position, length)
    rows = []
    for index, decision in enumerate(decisions):
        if index:
            gap += signed_longitudinal_gap(frames[index - 1].observations[front_id].track_position,
                                           frames[index].observations[front_id].track_position, length)
            gap -= signed_longitudinal_gap(frames[index - 1].observations[rear_id].track_position,
                                           frames[index].observations[rear_id].track_position, length)
        rows.append({"time_s": frames[index].time_seconds, "mode": decision.mode,
                     "target_d_m": decision.target_d, "actual_d_m": frames[index].observations[rear_id].track_position.d,
                     "speed_limit_mps": decision.speed_limit, "gap_m": gap})
    segments = []
    start = 0
    for end in range(1, len(rows) + 1):
        if end < len(rows) and rows[end]["target_d_m"] == rows[start]["target_d_m"]:
            continue
        target = rows[start]["target_d_m"]
        if target != 0.0:
            end_time = end * settings.model_config.time_step
            arrival = next((row["time_s"] for row in rows[start:end]
                            if abs(row["actual_d_m"] - target) <= settings.probe_following_config.lateral_tolerance), None)
            segments.append({"target_d_m": target, "start_s": rows[start]["time_s"], "end_s": end_time,
                             "first_arrival_s": arrival,
                             "held_after_arrival_s": None if arrival is None else end_time - arrival,
                             "simulation_ended": end == len(rows)})
        start = end
    with (OUTPUT / f"hold_{label}_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "executed_at_taipei": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree": "含未提交修改", "track_name": main.TRACK_NAME,
        "seconds": main.SIMULATION_SECONDS, "vehicle_specs": specs,
        "assignments": {key: asdict(value) for key, value in assignments.items()},
        "track_parameters": {"length": main.TRACK_RECTANGLE_LENGTH, "height": main.TRACK_RECTANGLE_HEIGHT,
                             "radius": main.TRACK_CORNER_RADIUS, "width": settings.track.width,
                             "point_count": len(settings.track.centerline)},
        "configs": {name: asdict(getattr(settings, name)) for name in (
            "probe_following_config", "defense_config", "model_config", "controller_config", "geometry", "offset_path_config")},
        "lateral_target_segments": segments,
        "collision_pairs": sorted({pair for frame in frames for pair in frame.collision_pairs}),
        "no_feasible_path_steps": sum(decision.mode == "no_feasible_path" for decision in decisions),
    }
    (OUTPUT / f"hold_{label}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def plot_comparison() -> None:
    """以保存的資料比較目標切換與實際偏移，不重新執行模擬。"""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for axis, label in zip(axes, ("before", "after")):
        with (OUTPUT / f"hold_{label}_trace.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        times = [float(row["time_s"]) for row in rows]
        axis.step(times, [float(row["target_d_m"]) for row in rows], where="post", label="target d")
        axis.plot(times, [float(row["actual_d_m"]) for row in rows], label="actual d", alpha=0.8)
        axis.set(ylabel="Lateral offset (m)", title=label)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    figure.tight_layout()
    figure.savefig(OUTPUT / "hold_comparison.png", dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", choices=("before", "after"))
    label = parser.parse_args().label
    result = verify(label)
    if label == "after" and (OUTPUT / "hold_before_trace.csv").exists():
        plot_comparison()
    print(json.dumps({key: result[key] for key in (
        "lateral_target_segments", "collision_pairs", "no_feasible_path_steps")}, ensure_ascii=False, indent=2))
