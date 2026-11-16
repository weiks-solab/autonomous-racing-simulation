"""快取後比較兩圖的穩定繪圖、決策尖峰與幾何預測；不改正式程式。"""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
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
from strategy.relative_overtaking import RelativeOvertakingStrategy


def summary(values):
    return {"calls": len(values), "mean_ms": float(np.mean(values)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": max(values)}


def run(name, gp):
    values = {key: [] for key in ("replan", "reuse", "render", "step", "opponent_prediction")}
    draws = 0
    previous = None
    step_start = 0
    manager_update = StrategyManager.update
    predict_other = RelativeOvertakingStrategy._predict_other

    def update(self, *args):
        nonlocal step_start, previous
        step_start = perf_counter()
        result = manager_update(self, *args)
        elapsed = (perf_counter() - step_start) * 1000
        times = tuple(d.candidate_snapshot.evaluated_at if d.candidate_snapshot else None
                      for d in result.values())
        values["replan" if times != previous else "reuse"].append(elapsed)
        previous = times
        return result

    def opponent(self, *args):
        start = perf_counter()
        result = predict_other(self, *args)
        values["opponent_prediction"].append((perf_counter() - start) * 1000)
        return result

    class MeasuredVisualizer(SimulationVisualizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            plt.ioff()
            self.figure.canvas.mpl_connect("draw_event", self.count_draw)

        def count_draw(self, event):
            nonlocal draws
            draws += 1

        def update_vehicles(self, **kwargs):
            start = perf_counter()
            super().update_vehicles(**kwargs)
            end = perf_counter()
            values["render"].append((end - start) * 1000)
            values["step"].append((end - step_start) * 1000)

        def show(self):
            plt.close(self.figure)

    with patch("main.TRACK_NAME", name), patch("main.SIMULATION_SECONDS", 8.0), \
         patch("main.GP_PREDICTION_ENABLED", gp), patch("main.SimulationVisualizer", MeasuredVisualizer), \
         patch.object(StrategyManager, "update", update), \
         patch.object(RelativeOvertakingStrategy, "_predict_other", opponent), \
         patch("sim.visualizer.plt.pause"), redirect_stdout(io.StringIO()):
        main.run_simulation(realtime=True)
        points = len(main.create_demo_track().centerline)
    return {"track": name, "gp_enabled": gp, "point_count": points, "full_draws": draws,
            "measurements": {key: summary(items) for key, items in values.items()},
            "steady_render": summary(values["render"][1:])}


if __name__ == "__main__":
    cases = []
    for name, gp in (("rounded_rectangle", True), ("rounded_rectangle", False), ("Oschersleben", False)):
        cases.append(run(name, gp))
        print(json.dumps(cases[-1]), flush=True)
    report = {"date": "2026-11-16", "branch": subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "version_note": "基於 base_commit，含未提交修改。",
        "scope": "各一次8秒400步；Agg，無GUI事件等待，非實測GUI FPS。",
        "cases": cases, "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
        for p in ("main.py", "sim/visualizer.py", "strategy/bicycle_prediction.py", "core/track_geometry.py")}}
    (Path(__file__).parent / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
