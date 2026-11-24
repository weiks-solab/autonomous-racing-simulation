"""量測 B2（決策週期容差）與 B3（橫移完成後接續）修正前後的場景差異。

用法（專案根目錄）：
    python reports/experiments/2026-11-24_decision_period_and_maneuver_fix/measure_fixes.py <輸出.json>

每個場景記錄：各車的決策次數與間隔（步）、碰撞步數、各車累積縱向進度與最終 d。
只包裝策略 update 與碰撞偵測來計數，不改變模擬行為。
"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np  # noqa: E402

import main  # noqa: E402
from core.track_geometry import signed_longitudinal_gap, track_length, world_to_track  # noqa: E402
from strategy.assignment import StrategyAssignment  # noqa: E402
from strategy.cost_overtaking import CostBasedOvertakingStrategy  # noqa: E402
from strategy.defending import CostBasedDefenseStrategy  # noqa: E402
from strategy.progress import ProgressStrategy  # noqa: E402
from strategy.relative_overtaking import RelativeOvertakingStrategy  # noqa: E402

TIMED = (RelativeOvertakingStrategy, CostBasedOvertakingStrategy, ProgressStrategy, CostBasedDefenseStrategy)
DEFEND_V1 = StrategyAssignment("defend", target_vehicle_id="vehicle-1")
SCENARIOS = {
    "progress_centerline_6s": {
        "SIMULATION_SECONDS": 6.0,
        "VEHICLE_SPECS": (("vehicle-1", 0.0, 11.0), ("vehicle-2", 4.0, 9.0), ("vehicle-3", -12.0, 8.0)),
        "STRATEGY_ASSIGNMENTS": {"vehicle-1": StrategyAssignment("progress"),
                                 "vehicle-3": StrategyAssignment("progress")},
    },
    "cost_overtake_defend_6s": {
        "SIMULATION_SECONDS": 6.0,
        "STRATEGY_ASSIGNMENTS": {"vehicle-1": StrategyAssignment("cost_overtake"), "vehicle-2": DEFEND_V1},
    },
    "default_relative_defend_6s": {"SIMULATION_SECONDS": 6.0},
    "overtake_defend_12s": {
        "SIMULATION_SECONDS": 12.0,
        "VEHICLE_SPECS": (("vehicle-1", 0.0, 6.0), ("vehicle-2", 6.0, 5.0), ("vehicle-3", -20.0, 4.0)),
        "STRATEGY_ASSIGNMENTS": {"vehicle-1": StrategyAssignment("overtake"), "vehicle-2": DEFEND_V1},
    },
}


def run(patches: dict) -> dict:
    decisions: dict[str, list[float]] = {}
    collision_steps = 0
    originals = {cls: cls.update for cls in TIMED}
    original_find = main.find_vehicle_collisions

    def counted(cls):
        def wrapper(self, vehicle, others, track, time_seconds):
            result = originals[cls](self, vehicle, others, track, time_seconds)
            if self.last_evaluation_time == time_seconds:
                decisions.setdefault(vehicle.vehicle_id, []).append(time_seconds)
            return result
        return wrapper

    def tracked_find(vehicles):
        nonlocal collision_steps
        pairs = original_find(vehicles)
        collision_steps += bool(pairs)
        return pairs

    patchers = [patch.object(cls, "update", counted(cls)) for cls in TIMED]
    for item in patchers:
        item.start()
    try:
        with patch.multiple(main, **patches), patch("main.find_vehicle_collisions", tracked_find), \
                patch("main.SimulationVisualizer") as visualizer, redirect_stdout(io.StringIO()):
            main.run_simulation()
    finally:
        for item in patchers:
            item.stop()
    frame = visualizer.return_value.update_vehicles.call_args.kwargs
    track = visualizer.call_args.args[0]
    length = track_length(track)
    progress, final_d = {}, {}
    for vehicle_id, states in sorted(frame["trajectories"].items()):
        positions = [world_to_track(state, track) for state in states]
        progress[vehicle_id] = round(sum(
            signed_longitudinal_gap(first, second, length) for first, second in zip(positions, positions[1:])
        ), 4)
        final_d[vehicle_id] = round(positions[-1].d, 4)
    dt = main.MODEL_CONFIG.time_step
    return {
        "decisions": {
            vehicle_id: {"count": len(times),
                         "interval_steps": sorted({int(round(step)) for step in np.diff(times) / dt})}
            for vehicle_id, times in sorted(decisions.items())
        },
        "collision_steps": collision_steps,
        "progress_m": progress,
        "final_d_m": final_d,
    }


if __name__ == "__main__":
    output = Path(sys.argv[1])
    results = {}
    for name, patches in SCENARIOS.items():
        results[name] = run(patches)
        print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
    report = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.strip(),
        "scenarios": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
