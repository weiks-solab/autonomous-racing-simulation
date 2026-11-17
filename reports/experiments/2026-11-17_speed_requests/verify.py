"""重現固定速度候選案例，保存逐步一致性摘要、測試與 Agg 執行日誌。"""

from dataclasses import asdict, astuple
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")
os.environ["TEMP"] = str(PROJECT_ROOT / ".cache")
os.environ["TMP"] = os.environ["TEMP"]
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import matplotlib
import numpy as np

import main
from core.track import Track
from core.vehicle_state import VehicleState
from strategy.base import StrategyDecision
from strategy.bicycle_prediction import predict_bicycle_trajectory
from strategy.offset_path import LateralManeuver
from tests.test_bicycle_prediction import observation
from tests.test_speed_requests import SpeedCandidateIntegrationTests, SpeedRequestTests


def state_errors(actual: list[VehicleState], predicted: tuple[VehicleState, ...]) -> list[float]:
    difference = np.array([astuple(state) for state in actual]) - np.array([
        astuple(state) for state in predicted
    ])
    return np.max(np.abs(difference), axis=0).tolist()


def verify_candidates() -> dict:
    case = SpeedCandidateIntegrationTests()
    case.setUp()
    strategy = case.strategy()
    decision = strategy.update(case.ego, (case.front,), case.track, 0.0)
    selected = next(candidate for candidate in strategy.last_candidates
                    if candidate.speed_limit == decision.speed_limit)
    actual, commands = case.execute_fixed_decision(
        case.track, case.ego.state, case.prediction, decision, 1.0,
    )
    case.assert_same_states(actual, selected.prediction.states)
    return {
        "track_centerline_m": case.track.centerline.tolist(),
        "track_width_m": case.track.width,
        "initial_states_xy_yaw_velocity": {
            "ego": astuple(case.ego.state), "front": astuple(case.front.state),
        },
        "model_config": asdict(case.prediction.model),
        "controller_config": asdict(case.prediction.controller),
        "path_config": asdict(case.prediction.path),
        "strategy_config": asdict(case.config),
        "vehicle_geometry": asdict(case.ego.geometry),
        "selected_maneuver": asdict(selected.maneuver),
        "dt_s": case.prediction.model.time_step,
        "horizon_s": case.config.prediction_horizon,
        "cruise_speed_mps": case.prediction.controller.target_speed,
        "speed_gain_per_s": case.prediction.controller.speed_gain,
        "max_acceleration_mps2": case.prediction.controller.max_acceleration,
        "max_deceleration_mps2": case.prediction.controller.max_deceleration,
        "front_prediction": "constant_speed bicycle, fixed current d",
        "candidates": [
            {
                "speed_limit_mps": candidate.speed_limit,
                "target_d_m": candidate.maneuver.target_d,
                "feasible_within_horizon": candidate.feasible,
                "collision_time_s": None if candidate.collision is None
                else candidate.collision.time_seconds,
                "relative_gap_cost_m": candidate.predicted_gap,
                "end_velocity_mps": candidate.prediction.states[-1].velocity,
            }
            for candidate in strategy.last_candidates
        ],
        "selected_speed_limit_mps": decision.speed_limit,
        "first_actual_acceleration_mps2": commands[0].acceleration,
        "first_actual_velocity_mps": actual[1].velocity,
        "compared_states_including_initial": len(actual),
        "max_abs_error_xy_yaw_velocity": state_errors(actual, selected.prediction.states),
    }


def verify_paths() -> list[dict]:
    case = SpeedCandidateIntegrationTests()
    case.setUp()
    results = []
    for shape, track, start_s in (
        ("straight", case.track, 20.0),
        ("curve", Track.create_oval(10.0, 5.0, 300), 2.0),
    ):
        for speed_limit in (None, 2.0, 0.0):
            ego = observation(track, "ego", start_s, 0.0, 6.0)
            maneuver = LateralManeuver(ego.track_position.s, 0.0, 0.45, 3.0)
            decision = StrategyDecision("attack", target_d=0.45, maneuver=maneuver,
                                        speed_limit=speed_limit)
            prediction = predict_bicycle_trajectory(
                ego, track, maneuver, case.prediction, 1.0, speed_limit=speed_limit,
            )
            actual, _ = case.execute_fixed_decision(track, ego.state, case.prediction, decision, 1.0)
            case.assert_same_states(actual, prediction.states)
            results.append({
                "shape": shape, "speed_limit_mps": speed_limit, "horizon_s": 1.0,
                "dt_s": case.prediction.model.time_step,
                "compared_states_including_initial": len(actual),
                "max_abs_error_xy_yaw_velocity": state_errors(actual, prediction.states),
            })
    return results


def verify_stop_and_resume() -> dict:
    case = SpeedRequestTests()
    case.setUp()
    runtime = case.make_runtime()
    for _ in range(600):
        runtime.step(case.track, speed_limit=0.0)
    stopped = runtime.state.velocity
    first_command, _ = runtime.step(case.track, speed_limit=None)
    for _ in range(299):
        runtime.step(case.track, speed_limit=None)
    return {
        "initial_velocity_mps": 6.0, "stop_duration_s": 12.0,
        "stop_tolerance_mps": 1e-6, "stopped_velocity_mps": stopped,
        "resume_duration_s": 6.0, "resumed_velocity_mps": runtime.state.velocity,
        "first_resume_acceleration_mps2": first_command.acceleration,
        "original_cruise_speed_mps": runtime.controller.config.target_speed,
    }


def run_logged(arguments: list[str], filename: str) -> int:
    completed = subprocess.run(
        [sys.executable, *arguments], cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    (OUTPUT / filename).write_text(completed.stdout, encoding="utf-8")
    print(f"{filename} exit_code={completed.returncode}", flush=True)
    return completed.returncode


if __name__ == "__main__":
    summary = {
        "executed_at_taipei": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True,
        ).strip(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip(),
        "worktree": "含未提交修改",
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "backend": matplotlib.get_backend(),
        "main_smoke_settings": {
            "track_name": main.TRACK_NAME, "seconds": main.SIMULATION_SECONDS,
            "vehicle_specs": main.VEHICLE_SPECS,
            "strategy_assignments": {
                key: asdict(value) for key, value in main.STRATEGY_ASSIGNMENTS.items()
            },
            "model_config": asdict(main.MODEL_CONFIG),
            "relative_overtaking_config": asdict(main.RELATIVE_OVERTAKING_CONFIG),
            "progress_config": asdict(main.PROGRESS_CONFIG),
            "path_config": asdict(main.OFFSET_PATH_CONFIG),
            "max_acceleration": main.MAX_ACCELERATION,
            "max_deceleration": main.MAX_DECELERATION,
        },
        "lower_speed_case": verify_candidates(),
        "fixed_decision_paths": verify_paths(),
        "stop_and_resume": verify_stop_and_resume(),
    }
    summary["test_exit_code"] = run_logged(["-m", "unittest", "-v"], "unittest.log")
    summary["main_exit_code"] = run_logged(["main.py"], "main_agg.log")
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print("summary.json saved", flush=True)
    if summary["test_exit_code"] != 0 or summary["main_exit_code"] != 0:
        sys.exit(1)
