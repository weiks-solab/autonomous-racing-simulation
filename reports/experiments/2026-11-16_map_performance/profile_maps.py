"""短場景比較策略計算、PNG 重繪與賽道幾何查詢；不更動正式程式。"""

from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import main
from sim.visualizer import SimulationVisualizer
from strategy.assignment import StrategyManager


def describe(values):
    return {"calls": len(values), "total_ms": sum(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "max_ms": max(values)}


def run(name, background, gp):
    samples = {key: [] for key in ("planner_all_steps", "planner_replan", "planner_reuse",
                                   "update_artists", "draw", "simulation_step_with_frame")}
    previous = None
    last_update_start = None
    original = StrategyManager.update

    def update(self, *args, **kwargs):
        nonlocal previous, last_update_start
        start = perf_counter()
        decisions = original(self, *args, **kwargs)
        ms = (perf_counter() - start) * 1000
        times = tuple(d.candidate_snapshot.evaluated_at if d.candidate_snapshot else None
                      for d in decisions.values())
        samples["planner_all_steps"].append(ms)
        samples["planner_replan" if times != previous else "planner_reuse"].append(ms)
        previous = times
        last_update_start = start
        return decisions

    class MeasuredVisualizer(SimulationVisualizer):
        def __init__(self, track, config, background=None):
            super().__init__(track, replace(config, realtime=False), background)

        def update_vehicles(self, **kwargs):
            start = perf_counter()
            super().update_vehicles(**kwargs)
            drawn = perf_counter()
            self.figure.canvas.draw()
            end = perf_counter()
            samples["update_artists"].append((drawn - start) * 1000)
            samples["draw"].append((end - drawn) * 1000)
            samples["simulation_step_with_frame"].append((end - last_update_start) * 1000)

        def show(self):
            plt.close(self.figure)

    options = replace(main.create_display_options(True), show_map_background=background)
    start = perf_counter()
    with patch("main.TRACK_NAME", name), patch("main.SIMULATION_SECONDS", 2.0), \
         patch("main.GP_PREDICTION_ENABLED", gp), patch("main.create_display_options", return_value=options), \
         patch("main.SimulationVisualizer", MeasuredVisualizer), \
         patch.object(StrategyManager, "update", update), redirect_stdout(io.StringIO()):
        main.run_simulation(realtime=True)
        track = main.create_demo_track()
    return {"name": name, "png": background, "gp_enabled": gp,
            "track_points": len(track.centerline), "wall_seconds": perf_counter() - start,
            "timings": {key: describe(values) for key, values in samples.items() if values}}


def main_profile():
    directory = Path(__file__).resolve().parent
    cases = []
    for name, png, gp in (("rounded_rectangle", False, True), ("rounded_rectangle", False, False),
                           ("Oschersleben", False, False), ("Oschersleben", True, False)):
        cases.append(run(name, png, gp))
        print(name, png, gp, round(cases[-1]["timings"]["draw"]["mean_ms"], 2), flush=True)

    # 另行剖析無視窗執行，避免把 profiler 開銷混進上述繪圖比較。
    profiler = cProfile.Profile()
    with ExitStack() as stack:
        stack.enter_context(patch("main.TRACK_NAME", "Oschersleben"))
        stack.enter_context(patch("main.SIMULATION_SECONDS", 2.0))
        stack.enter_context(patch("main.SimulationVisualizer"))
        stack.enter_context(redirect_stdout(io.StringIO()))
        profiler.runcall(main.run_simulation)
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, function), (primitive, calls, self_time, cumulative, _) in stats.stats.items():
        try:
            relative = Path(filename).relative_to(ROOT).as_posix()
        except ValueError:
            continue
        rows.append({"file": relative, "line": line, "function": function, "calls": calls,
                     "self_seconds": self_time, "cumulative_seconds": cumulative})
    report = {
        "date": "2026-11-16", "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "version_note": "基於 base_commit，含未提交修改。",
        "scope": "各案例只執行一次，2 秒／100 步；Agg 每步強制重畫，無 GUI 等待，非實際 GUI FPS。",
        "cases": cases, "profile_total_seconds": stats.total_tt,
        "profile_functions": sorted(rows, key=lambda r: -r["cumulative_seconds"])[:30],
    }
    (directory / "profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main_profile()
