"""以正式雙車步進測試圓角賽道，保存試探動作與前車反應。"""

from dataclasses import asdict, astuple
from datetime import datetime, timedelta, timezone
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")

import numpy as np
import matplotlib.pyplot as plt

import main
from core.pure_pursuit import PurePursuitConfig
from core.track_factory import TrackConfig
from core.track_geometry import signed_longitudinal_gap, track_boundaries, track_length
from sim.data_generation import EpisodeScenario, SimulationSettings, VehicleInitialCondition, run_episode
from sim.simulated_vehicle import SimulatedVehicle
from strategy.bicycle_prediction import BicyclePrediction
from strategy.probe_following import ProbeFollowingStrategy, body_progress_bounds
from training_data import TrainingDataCollector


def verify() -> dict:
    track_config = TrackConfig(
        name="rounded_rectangle", point_count=main.TRACK_POINT_COUNT, width=main.TRACK_WIDTH,
        rectangle_length=main.TRACK_RECTANGLE_LENGTH,
        rectangle_height=main.TRACK_RECTANGLE_HEIGHT, corner_radius=main.TRACK_CORNER_RADIUS,
    )
    track = main.create_demo_track("rounded_rectangle")
    settings = SimulationSettings(
        track, main.MODEL_CONFIG, main.VEHICLE_GEOMETRY,
        PurePursuitConfig(main.LOOKAHEAD_DISTANCE, 0.0, main.SPEED_GAIN,
                          main.MAX_ACCELERATION, main.MODEL_CONFIG.max_steering_angle,
                          max_deceleration=main.MAX_DECELERATION),
        defense_config=main.DEFENSE_CONFIG, offset_path_config=main.OFFSET_PATH_CONFIG,
        probe_following_config=main.PROBE_FOLLOWING_CONFIG,
    )
    scenario = EpisodeScenario(
        "rounded-probe", tuple(VehicleInitialCondition(key, s, 0.0, 0.0, speed)
                               for key, s, speed in main.PROBE_VEHICLE_SPECS),
        main.PROBE_STRATEGY_ASSIGNMENTS,
    )
    decisions, commands = [], []
    original_update, original_step = ProbeFollowingStrategy.update, SimulatedVehicle.step

    def record_update(strategy, *args, **kwargs):
        decision = original_update(strategy, *args, **kwargs)
        decisions.append(decision)
        return decision

    def record_step(vehicle, *args, **kwargs):
        command, target = original_step(vehicle, *args, **kwargs)
        if vehicle.vehicle_id == "rear":
            commands.append(command)
        return command, target

    collector = TrainingDataCollector(settings.model_config.time_step)
    with patch.object(ProbeFollowingStrategy, "update", new=record_update), \
         patch.object(SimulatedVehicle, "step", new=record_step):
        run_episode(scenario, settings, collector, main.SIMULATION_SECONDS)
    frames = collector.episodes[0].frames
    length = track_length(track)
    progress, predictions = {}, {}
    for key in ("rear", "front"):
        positions = [frame.observations[key].track_position for frame in frames]
        increments = [signed_longitudinal_gap(a, b, length) for a, b in zip(positions, positions[1:])]
        progress[key] = np.concatenate(([0.0], np.cumsum(increments)))
        predictions[key] = BicyclePrediction(
            tuple(frame.observations[key].state for frame in frames),
            tuple(progress[key]), tuple(frame.time_seconds for frame in frames),
        )
    initial_gap = signed_longitudinal_gap(frames[0].observations["rear"].track_position,
                                         frames[0].observations["front"].track_position, length)
    gaps = initial_gap + progress["front"] - progress["rear"]
    rear_bounds = body_progress_bounds(predictions["rear"], settings.geometry, track)
    front_bounds = body_progress_bounds(predictions["front"], settings.geometry, track)
    clearances = gaps + front_bounds[:, 0] - rear_bounds[:, 1]
    rows, first_step_errors = [], []
    for index, (decision, command) in enumerate(zip(decisions, commands)):
        rear, front = (frames[index].observations[key] for key in ("rear", "front"))
        selected = next((candidate for candidate in decision.candidate_snapshot.candidates
                         if candidate.selected), None)
        if selected is not None and decision.candidate_snapshot.evaluated_at == frames[index].time_seconds:
            expected = astuple(selected.predicted_states[1])
            actual = astuple(frames[index + 1].observations["rear"].state)
            first_step_errors.append(float(np.max(np.abs(np.asarray(actual) - expected))))
        rows.append({
            "time_s": frames[index].time_seconds, "mode": decision.mode,
            "evaluated_at_s": decision.candidate_snapshot.evaluated_at,
            "gap_m": float(gaps[index]), "clearance_m": float(clearances[index]),
            "rear_s_m": rear.track_position.s, "front_s_m": front.track_position.s,
            "rear_d_m": rear.track_position.d, "front_d_m": front.track_position.d,
            "rear_velocity_mps": rear.state.velocity, "front_velocity_mps": front.state.velocity,
            "rear_x_m": rear.state.x, "rear_y_m": rear.state.y, "rear_yaw_rad": rear.state.yaw,
            "front_x_m": front.state.x, "front_y_m": front.state.y, "front_yaw_rad": front.state.yaw,
            "target_d_m": decision.target_d, "speed_limit_mps": decision.speed_limit,
            "acceleration_mps2": command.acceleration,
            **dict(decision.candidate_snapshot.candidates[0].cost_terms if selected is None else selected.cost_terms),
        })
    with (OUTPUT / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure = plt.figure(figsize=(12, 6))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.2, 1.0))
    map_axis = figure.add_subplot(grid[:, 0])
    gap_axis = figure.add_subplot(grid[0, 1])
    lateral_axis = figure.add_subplot(grid[1, 1])
    for boundary in track_boundaries(track):
        map_axis.plot(boundary[:, 0], boundary[:, 1], color="0.7", linewidth=1)
    map_axis.plot(track.centerline[:, 0], track.centerline[:, 1], ":", color="0.7")
    times = predictions["rear"].times
    for key, color in (("rear", "tab:blue"), ("front", "tab:orange")):
        states = predictions[key].states
        map_axis.plot([state.x for state in states], [state.y for state in states], color=color, label=key)
        lateral_axis.plot(times, [frame.observations[key].track_position.d for frame in frames],
                          color=color, label=key)
    map_axis.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Rounded rectangle trajectories")
    map_axis.legend()
    gap_axis.plot(times, gaps, color="tab:blue", label="actual gap")
    gap_axis.axhline(settings.probe_following_config.target_gap, linestyle="--", color="0.4", label="target gap")
    gap_axis.set(xlabel="time (s)", ylabel="rear-axle gap (m)")
    gap_axis.legend()
    lateral_axis.set(xlabel="time (s)", ylabel="lateral offset d (m)")
    lateral_axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "probe_following.png", dpi=140)
    plt.close(figure)
    modes = [row["mode"] for row in rows]
    collisions = sorted({pair for frame in frames for pair in frame.collision_pairs})
    passed = (not collisions and float(clearances.min()) >= settings.probe_following_config.clearance
              and "probe_left" in modes and "probe_right" in modes and "no_feasible_path" not in modes
              and float(gaps[-1]) < initial_gap and max(first_step_errors) < 1e-10)
    return {
        "executed_at_taipei": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "worktree": "含未提交修改", "python": platform.python_version(), "numpy": np.__version__,
        "track_config": asdict(track_config), "length_m": length,
        "scenario": {"vehicles": [asdict(vehicle) for vehicle in scenario.vehicles],
                     "assignments": {key: asdict(value) for key, value in scenario.assignments.items()}},
        "configs": {name: asdict(getattr(settings, name)) for name in (
            "model_config", "geometry", "controller_config", "probe_following_config",
            "defense_config", "offset_path_config",
        )},
        "seconds": frames[-1].time_seconds, "step_count": len(rows),
        "replanning_count": len({row["evaluated_at_s"] for row in rows}),
        "initial_gap_m": initial_gap, "final_gap_m": float(gaps[-1]),
        "gap_range_m": [float(gaps.min()), float(gaps.max())],
        "gap_mae_last_5s_m": float(np.mean(np.abs(gaps[-251:] - settings.probe_following_config.target_gap))),
        "minimum_body_clearance_m": float(clearances.min()),
        "rear_laps": float(progress["rear"][-1] / length),
        "rear_d_range_m": [min(row["rear_d_m"] for row in rows), max(row["rear_d_m"] for row in rows)],
        "front_d_range_m": [min(row["front_d_m"] for row in rows), max(row["front_d_m"] for row in rows)],
        "modes": sorted(set(modes)), "collision_pairs": collisions,
        "first_step_max_abs_error": max(first_step_errors),
        "negative_acceleration_steps": sum(row["acceleration_mps2"] < 0 for row in rows),
        "faster_than_front_steps": sum(row["rear_velocity_mps"] > row["front_velocity_mps"] + 0.05 for row in rows),
        "passed": passed,
    }


if __name__ == "__main__":
    summary = verify()
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "passed", "seconds", "replanning_count", "initial_gap_m", "final_gap_m",
        "gap_range_m", "gap_mae_last_5s_m", "minimum_body_clearance_m", "rear_d_range_m",
        "front_d_range_m", "modes", "collision_pairs", "first_step_max_abs_error",
    )}, ensure_ascii=False, indent=2))
    sys.exit(0 if summary["passed"] else 1)
