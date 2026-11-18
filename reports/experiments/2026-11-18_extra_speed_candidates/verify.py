"""用當下 main 場景比較有無額外 -4 m/s 候選，正式決策照常執行。"""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")

import main
from strategy.relative_overtaking import RelativeOvertakingStrategy


def measure(extra_deltas: tuple[float, ...]) -> dict:
    rows, hits = [], []
    update = RelativeOvertakingStrategy.update
    check = main.find_vehicle_collisions
    step_count = 0

    def record_update(strategy, vehicle, others, track, time_seconds):
        previous = strategy.last_evaluation_time
        decision = update(strategy, vehicle, others, track, time_seconds)
        if strategy.last_evaluation_time != previous:
            front = next((other for other in others if other.vehicle_id == decision.target_vehicle_id), None)
            extra_selected = (front is not None and decision.speed_limit is not None
                              and abs(decision.speed_limit - max(0.0, front.state.velocity - 4.0)) < 1e-8)
            rows.append({"time_s": time_seconds, "mode": decision.mode,
                         "speed_limit_mps": decision.speed_limit, "extra_selected": extra_selected,
                         "feasible_count": sum(candidate.feasible for candidate in strategy.last_candidates)})
        return decision

    def record_collisions(vehicles):
        nonlocal step_count
        step_count += 1
        result = check(vehicles)
        if result:
            hits.append(step_count * main.MODEL_CONFIG.time_step)
        return result

    config = replace(main.RELATIVE_OVERTAKING_CONFIG, extra_speed_candidate_deltas=extra_deltas)
    with patch.object(main, "RELATIVE_OVERTAKING_CONFIG", config), \
         patch.object(RelativeOvertakingStrategy, "update", new=record_update), \
         patch.object(main, "find_vehicle_collisions", new=record_collisions), \
         patch.object(main, "SimulationVisualizer", MagicMock()), redirect_stdout(io.StringIO()):
        main.run_simulation()
    return {"config": asdict(config), "planning_count": len(rows),
            "limited_selections": sum(row["speed_limit_mps"] is not None for row in rows),
            "extra_selections": sum(row["extra_selected"] for row in rows),
            "no_feasible_count": sum(row["mode"] == "no_feasible_path" for row in rows),
            "first_collision_s": min(hits) if hits else None,
            "first_no_feasible": next((row for row in rows if row["mode"] == "no_feasible_path"), None)}


if __name__ == "__main__":
    summary = {
        "executed_at_taipei": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree": "含未提交修改", "track": main.TRACK_NAME,
        "seconds": main.SIMULATION_SECONDS, "vehicle_specs": main.VEHICLE_SPECS,
        "assignments": {key: asdict(value) for key, value in main.STRATEGY_ASSIGNMENTS.items()},
        "gp_enabled": main.GP_PREDICTION_ENABLED,
        "track_parameters": {"length": main.TRACK_RECTANGLE_LENGTH, "height": main.TRACK_RECTANGLE_HEIGHT,
                             "radius": main.TRACK_CORNER_RADIUS, "width": main.TRACK_WIDTH,
                             "point_count": main.TRACK_POINT_COUNT},
        "model": asdict(main.MODEL_CONFIG), "geometry": asdict(main.VEHICLE_GEOMETRY),
        "defense": asdict(main.DEFENSE_CONFIG), "offset_path": asdict(main.OFFSET_PATH_CONFIG),
        "control": {"lookahead": main.LOOKAHEAD_DISTANCE, "gain": main.SPEED_GAIN,
                    "max_acceleration": main.MAX_ACCELERATION, "max_deceleration": main.MAX_DECELERATION},
    }
    for name, deltas in (("baseline", ()), ("with_extra", (4.0,))):
        summary[name] = measure(deltas)
        print(name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
