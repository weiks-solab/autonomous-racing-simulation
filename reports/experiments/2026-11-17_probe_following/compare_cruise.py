"""固定前次 3 m 初距的圓角場景，只比較有無原巡航速度候選。"""

from dataclasses import asdict, astuple, replace
from datetime import datetime, timedelta, timezone
import csv
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

import matplotlib.pyplot as plt
import numpy as np

import main
from core.track_geometry import signed_longitudinal_gap, track_length
from generate_training_data import _demo_settings
from sim.data_generation import EpisodeScenario, SimulationSettings, VehicleInitialCondition, run_episode
from sim.simulated_vehicle import SimulatedVehicle
from strategy.probe_following import ProbeFollowingStrategy
from training_data import TrainingDataCollector


def previous_speed_limits(strategy: ProbeFollowingStrategy, front_speed: float) -> tuple[float, ...]:
    """僅在比較的 baseline 還原舊候選；執行與預測控制仍用正式程式。"""
    cruise, step = strategy.prediction_config.controller.target_speed, strategy.config.speed_step
    return tuple(dict.fromkeys(min(cruise, max(0.0, front_speed + delta)) for delta in (-step, 0.0, step)))


def measure(settings: SimulationSettings, scenario: EpisodeScenario, baseline: bool) -> tuple[dict, list[dict]]:
    decisions, commands = [], []
    original_update, original_step = ProbeFollowingStrategy.update, SimulatedVehicle.step

    def record_update(strategy, *args, **kwargs):
        decision = original_update(strategy, *args, **kwargs)
        decisions.append(decision)
        return decision

    def record_step(vehicle, *args, **kwargs):
        command, target = original_step(vehicle, *args, **kwargs)
        if vehicle.vehicle_id == "vehicle-2":
            commands.append(command)
        return command, target

    speeds = previous_speed_limits if baseline else ProbeFollowingStrategy._speed_limits
    collector = TrainingDataCollector(settings.model_config.time_step)
    with patch.object(ProbeFollowingStrategy, "_speed_limits", new=speeds), \
         patch.object(ProbeFollowingStrategy, "update", new=record_update), \
         patch.object(SimulatedVehicle, "step", new=record_step):
        run_episode(scenario, settings, collector, 15.0)
    frames = collector.episodes[0].frames
    length = track_length(settings.track)
    gaps = [signed_longitudinal_gap(frames[0].observations["vehicle-2"].track_position,
                                   frames[0].observations["vehicle-1"].track_position, length)]
    for before, after in zip(frames, frames[1:]):
        advances = {key: signed_longitudinal_gap(before.observations[key].track_position,
                                               after.observations[key].track_position, length)
                    for key in ("vehicle-1", "vehicle-2")}
        gaps.append(gaps[-1] + advances["vehicle-1"] - advances["vehicle-2"])
    rows, replans, errors = [], [], []
    for index, frame in enumerate(frames):
        decision = decisions[min(index, len(decisions) - 1)]
        rows.append({"time_s": frame.time_seconds, "gap_m": gaps[index], "mode": decision.mode,
                     "rear_velocity_mps": frame.observations["vehicle-2"].state.velocity,
                     "front_velocity_mps": frame.observations["vehicle-1"].state.velocity,
                     "speed_limit_mps": decision.speed_limit,
                     "acceleration_mps2": commands[index].acceleration if index < len(commands) else None})
        if index < len(decisions) and abs(decision.candidate_snapshot.evaluated_at - frame.time_seconds) < 1e-9:
            replans.append(rows[-1])
            selected = next((candidate for candidate in decision.candidate_snapshot.candidates if candidate.selected), None)
            if selected is not None:
                actual = np.asarray(astuple(frames[index + 1].observations["vehicle-2"].state))
                errors.append(float(np.max(np.abs(actual - astuple(selected.predicted_states[1])))))
    config = settings.probe_following_config
    # 排除起步就在試探區間的樣本，量測離開後第一次回到範圍的時刻。
    outside = next((index for index, gap in enumerate(gaps)
                    if abs(gap - config.target_gap) > config.probe_gap_tolerance), None)
    returned = None if outside is None else next((frames[index].time_seconds for index in range(outside + 1, len(gaps))
                                                if abs(gaps[index] - config.target_gap) <= config.probe_gap_tolerance), None)
    return {
        "initial_gap_m": gaps[0], "maximum_gap_m": max(gaps), "final_gap_m": gaps[-1],
        "first_return_to_probe_window_s": returned, "replanning_count": len(replans),
        "cruise_selections": sum(abs(row["speed_limit_mps"] - 12.0) < 1e-9 for row in replans),
        "mode_duration_s": {mode: sum(row["mode"] == mode for row in rows[:-1]) * settings.model_config.time_step
                            for mode in sorted({row["mode"] for row in rows[:-1]})},
        "maximum_prediction_first_step_error": max(errors),
        "collision_pairs": sorted({pair for frame in frames for pair in frame.collision_pairs}),
        "passing_samples": sum(gap <= 0.0 for gap in gaps),
    }, rows


if __name__ == "__main__":
    settings, _, assignments = _demo_settings(False)
    settings = replace(settings, track=main.create_demo_track("rounded_rectangle"))
    scenario = EpisodeScenario("cruise-comparison", (
        VehicleInitialCondition("vehicle-1", 3.0, 0.0, 0.0, 10.0),
        VehicleInitialCondition("vehicle-2", 0.0, 0.0, 0.0, 12.0),
    ), assignments)
    summary = {
        "executed_at_taipei": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree": "含未提交修改", "seconds": 15.0,
        "scenario": {"vehicles": [asdict(vehicle) for vehicle in scenario.vehicles],
                     "assignments": {key: asdict(value) for key, value in scenario.assignments.items()}},
        "track": {"name": "rounded_rectangle", "length": main.TRACK_RECTANGLE_LENGTH,
                  "height": main.TRACK_RECTANGLE_HEIGHT, "radius": main.TRACK_CORNER_RADIUS,
                  "width": settings.track.width, "point_count": len(settings.track.centerline)},
        "configs": {name: asdict(getattr(settings, name)) for name in (
            "probe_following_config", "defense_config", "controller_config", "model_config", "geometry", "offset_path_config")},
    }
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for name, baseline in (("baseline", True), ("with_cruise", False)):
        summary[name], rows = measure(settings, scenario, baseline)
        with (OUTPUT / f"cruise_{name}_trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        axes[0].plot([row["time_s"] for row in rows], [row["gap_m"] for row in rows], label=name)
        axes[1].plot([row["time_s"] for row in rows], [row["rear_velocity_mps"] for row in rows], label=name)
        print(name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    config = settings.probe_following_config
    axes[0].axhspan(config.target_gap - config.probe_gap_tolerance,
                    config.target_gap + config.probe_gap_tolerance, color="0.9", label="probe window")
    axes[0].axhline(config.target_gap, color="0.5", linestyle="--", label="target gap")
    axes[0].set_ylabel("Rear-axle gap (m)")
    axes[1].set(xlabel="Time (s)", ylabel="Rear speed (m/s)")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "cruise_comparison.png", dpi=130)
    plt.close(figure)
    (OUTPUT / "cruise_comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
