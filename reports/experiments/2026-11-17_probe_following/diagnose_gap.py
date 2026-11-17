"""只觀察目前 main 配置，量測車距與速度候選，另評估未執行的巡航候選。"""

from dataclasses import asdict
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

import numpy as np

import main
from core.track_geometry import signed_longitudinal_gap, track_length
from generate_training_data import _demo_settings
from sim.data_generation import EpisodeScenario, VehicleInitialCondition, run_episode
from sim.simulated_vehicle import SimulatedVehicle
from strategy.probe_following import ProbeFollowingConfig, ProbeFollowingStrategy, body_progress_bounds
from training_data import TrainingDataCollector


def diagnose() -> dict:
    settings, specs, assignments = _demo_settings(False)
    rear_id = next(key for key, value in assignments.items() if value.name == "probe_follow")
    front_id = next(key for key, value in assignments.items()
                    if value.name == "defend" and value.target_vehicle_id == rear_id)
    scenario = EpisodeScenario(
        "gap-diagnosis", tuple(VehicleInitialCondition(key, s, 0.0, 0.0, speed)
                               for key, s, speed in specs), assignments,
    )
    decisions, commands, comparisons = [], [], []
    original_update, original_step = ProbeFollowingStrategy.update, SimulatedVehicle.step

    def record_update(strategy, vehicle, others, track, time_seconds):
        previous_time = strategy.last_evaluation_time
        result = original_update(strategy, vehicle, others, track, time_seconds)
        if strategy.last_evaluation_time == previous_time:
            return result
        selected = next((candidate for candidate in strategy.last_candidates
                         if candidate.feasible and candidate.maneuver == result.maneuver
                         and candidate.speed_limit == result.speed_limit), None)
        front = next(other for other in others if other.vehicle_id == front_id)
        decisions.append({
            "time_s": time_seconds, "gap_m": strategy.target_state.gap,
            "mode": result.mode, "rear_velocity_mps": vehicle.state.velocity,
            "front_velocity_mps": front.state.velocity, "speed_limit_mps": result.speed_limit,
            "maximum_candidate_mps": max(candidate.speed_limit for candidate in strategy.last_candidates),
            "selected_cost": None if selected is None else selected.total_cost,
            "selected_speed_cost": None if selected is None else dict(selected.cost_terms)["speed"],
            "rejected_count": sum(not candidate.feasible for candidate in strategy.last_candidates),
        })
        # 在三個指定時刻，只計算「若提供巡航速度」的成本，原決策仍照常執行。
        if result.mode == "follow" and any(abs(time_seconds - time) < 1e-8 for time in (4.0, 8.0, 12.0)):
            obstacles = tuple(strategy._predict_other(other, track) for other in others)
            predicted_front = next(other.prediction for other in obstacles if other.vehicle_id == front_id)
            candidate = strategy._evaluate_candidate(
                vehicle, track, result.target_d, strategy.prediction_config.controller.target_speed,
                strategy.target_state.gap, 0.0, predicted_front,
                body_progress_bounds(predicted_front, front.geometry, track), obstacles,
            )
            comparisons.append({
                "time_s": time_seconds, "gap_m": strategy.target_state.gap,
                "selected_speed_limit_mps": result.speed_limit,
                "selected_cost_terms": None if selected is None else dict(selected.cost_terms),
                "unused_cruise_speed_mps": candidate.speed_limit,
                "unused_cruise_feasible": candidate.feasible,
                "unused_cruise_reason": candidate.reason,
                "unused_cruise_cost_terms": dict(candidate.cost_terms),
            })
        return result

    def record_step(vehicle, *args, **kwargs):
        before = vehicle.state.velocity
        command, target = original_step(vehicle, *args, **kwargs)
        if vehicle.vehicle_id == rear_id:
            commands.append({"before_mps": before, "after_mps": vehicle.state.velocity,
                             "acceleration_mps2": command.acceleration})
        return command, target

    collector = TrainingDataCollector(settings.model_config.time_step)
    with patch.object(ProbeFollowingStrategy, "update", new=record_update), \
         patch.object(SimulatedVehicle, "step", new=record_step):
        run_episode(scenario, settings, collector, main.SIMULATION_SECONDS)
    frames = collector.episodes[0].frames
    length = track_length(settings.track)
    initial_gap = signed_longitudinal_gap(frames[0].observations[rear_id].track_position,
                                         frames[0].observations[front_id].track_position, length)
    progress = {}
    for key in (rear_id, front_id):
        positions = [frame.observations[key].track_position for frame in frames]
        progress[key] = np.concatenate(([0.0], np.cumsum([
            signed_longitudinal_gap(a, b, length) for a, b in zip(positions, positions[1:])
        ])))
    gaps = initial_gap + progress[front_id] - progress[rear_id]
    samples = []
    for second in (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, main.SIMULATION_SECONDS):
        index = round(second / settings.model_config.time_step)
        frame = frames[index]
        decision = decisions[min(index // round(settings.probe_following_config.decision_period
                                                 / settings.model_config.time_step), len(decisions) - 1)]
        samples.append({
            "time_s": second, "gap_m": float(gaps[index]),
            "rear_velocity_mps": frame.observations[rear_id].state.velocity,
            "front_velocity_mps": frame.observations[front_id].state.velocity,
            "speed_limit_mps": decision["speed_limit_mps"],
        })
    with (OUTPUT / "gap_diagnosis_decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    return {
        "executed_at_taipei": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree": "含未提交修改",
        "track": main.TRACK_NAME, "seconds": main.SIMULATION_SECONDS,
        "vehicle_specs": specs, "probe_class_default_target_gap_m": ProbeFollowingConfig().target_gap,
        "effective_probe_config": asdict(settings.probe_following_config),
        "defense_config": asdict(settings.defense_config),
        "model_config": asdict(settings.model_config), "controller_template": asdict(settings.controller_config),
        "gap_range_m": [float(gaps.min()), float(gaps.max())], "samples": samples,
        "mode_counts_at_replans": {mode: sum(row["mode"] == mode for row in decisions)
                                    for mode in sorted({row["mode"] for row in decisions})},
        "replans_selecting_fastest_candidate": sum(abs(row["speed_limit_mps"] - row["maximum_candidate_mps"]) < 1e-8
                                                   for row in decisions),
        "replanning_count": len(decisions), "unused_cruise_comparisons": comparisons,
        "first_step": commands[0],
        "collision_pairs": sorted({pair for frame in frames for pair in frame.collision_pairs}),
        "progress_last_5s_m": {key: float(values[-1] - values[-251]) for key, values in progress.items()},
    }


if __name__ == "__main__":
    summary = diagnose()
    (OUTPUT / "gap_diagnosis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
