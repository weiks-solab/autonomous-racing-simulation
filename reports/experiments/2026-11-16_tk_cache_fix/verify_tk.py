"""以真實 TkAgg 隱藏視窗比較舊 pause 行為與修正後事件迴圈。"""
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
os.environ["MPLBACKEND"] = "TkAgg"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import main
from sim.visualizer import SimulationVisualizer


def run(old_pause):
    samples, draw_counts = [], []
    final_states = None

    class MeasuredVisualizer(SimulationVisualizer):
        def __init__(self, *args, **kwargs):
            # 保留 config.realtime=True，但不顯示視窗；使用真正 Tk canvas 與事件迴圈。
            with patch("sim.visualizer.plt.ion"):
                super().__init__(*args, **kwargs)
            self.figure.canvas.manager.window.withdraw()
            self.figure.canvas.manager._shown = True
            self.draws = 0
            self.figure.canvas.mpl_connect("draw_event", self.count_draw)
            if old_pause:
                original_loop = self.figure.canvas.start_event_loop

                def old_event_loop(interval):
                    # 重現舊 plt.pause 的 stale 檢查、show 與事件等待，避免遞迴。
                    if self.figure.stale:
                        self.figure.canvas.draw_idle()
                    self.figure.canvas.manager.show()
                    original_loop(interval)

                self.figure.canvas.start_event_loop = old_event_loop

        def count_draw(self, event):
            self.draws += 1

        def update_vehicles(self, **kwargs):
            nonlocal final_states
            start = perf_counter()
            super().update_vehicles(**kwargs)
            samples.append((perf_counter() - start) * 1000)
            draw_counts.append(self.draws)
            final_states = {key: (s.x, s.y, s.yaw, s.velocity) for key, s in kwargs["states"].items()}

        def show(self):
            if not old_pause:
                self.figure.savefig(Path(__file__).parent / "tk_fixed.png", dpi=160)
            plt.close(self.figure)

    with mpl.rc_context({"figure.raise_window": False}), \
         patch("main.TRACK_NAME", "Oschersleben"), patch("main.SIMULATION_SECONDS", 2.0), \
         patch("main.SimulationVisualizer", MeasuredVisualizer), redirect_stdout(io.StringIO()):
        main.run_simulation(realtime=True)
    return {"frames": len(samples), "full_draws": draw_counts[-1], "mean_ms": float(np.mean(samples)),
            "p95_ms": float(np.percentile(samples, 95)),
            "steady_mean_ms": float(np.mean(samples[1:])), "final_states": final_states}


if __name__ == "__main__":
    old, fixed = run(True), run(False)
    assert old["full_draws"] >= 100 and fixed["full_draws"] == 1
    assert old["final_states"] == fixed["final_states"]
    report = {"date": "2026-11-16", "branch": subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "version_note": "基於 base_commit，含未提交修改。", "backend": mpl.get_backend(),
        "scope": "真實 TkAgg canvas 與事件迴圈，視窗隱藏；各一次2秒100步，非人工GUI操作或實際FPS。",
        "old_pause_behavior": old, "fixed_event_loop": fixed,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in ("main.py", "sim/visualizer.py", "tests/test_visualizer.py")}}
    (Path(__file__).parent / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"old_ms": old["mean_ms"], "fixed_ms": fixed["mean_ms"],
                      "full_draws": [old["full_draws"], fixed["full_draws"]]}))
