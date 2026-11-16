"""比較整張重繪與即時背景快取；兩者均無 GUI 等待。"""

from contextlib import redirect_stdout
from dataclasses import replace
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


def run(cached):
    samples = []
    full_draws = 0
    final_states = None

    class MeasuredVisualizer(SimulationVisualizer):
        def __init__(self, track, config, background=None):
            super().__init__(track, replace(config, realtime=cached), background)
            plt.ioff()  # Agg 沒有 GUI 排程，避免互動模式自動觸發額外 draw_idle。
            self.figure.canvas.mpl_connect("draw_event", self.count_draw)

        def count_draw(self, event):
            nonlocal full_draws
            full_draws += 1

        def update_vehicles(self, **kwargs):
            nonlocal final_states
            start = perf_counter()
            super().update_vehicles(**kwargs)
            if not cached:
                self.figure.canvas.draw()
            samples.append((perf_counter() - start) * 1000)
            final_states = {key: (state.x, state.y, state.yaw, state.velocity)
                            for key, state in kwargs["states"].items()}

        def show(self):
            if cached:
                self.figure.savefig(Path(__file__).parent / "cached_frame.png", dpi=160)
            plt.close(self.figure)

    with patch("main.TRACK_NAME", "Oschersleben"), patch("main.SIMULATION_SECONDS", 2.0), \
         patch("main.SimulationVisualizer", MeasuredVisualizer), \
         patch("sim.visualizer.plt.pause"), redirect_stdout(io.StringIO()):
        main.run_simulation(realtime=True)
    # 存圖也會發出 draw_event，不列入 100 幀動畫的完整重繪數。
    return {"frames": len(samples), "full_draws": full_draws - int(cached),
            "mean_ms": float(np.mean(samples)), "p50_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)), "first_frame_ms": samples[0],
            "steady_mean_ms": float(np.mean(samples[1:])), "final_states": final_states}


if __name__ == "__main__":
    baseline, cached = run(False), run(True)
    assert baseline["final_states"] == cached["final_states"]
    assert baseline["full_draws"] == 100 and cached["full_draws"] == 1
    report = {
        "date": "2026-11-16", "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "version_note": "基於 base_commit，含未提交修改。",
        "scope": "各一次、2 秒100步，Agg；量測圖元更新與渲染，不含策略或 GUI 等待，非實際 FPS。",
        "full_redraw": baseline, "background_cache": cached,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in ("sim/visualizer.py", "main.py", "tests/test_visualizer.py")},
    }
    (Path(__file__).parent / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"full_redraw_ms": baseline["mean_ms"], "cached_ms": cached["mean_ms"],
                      "full_draws": [baseline["full_draws"], cached["full_draws"]]}))
