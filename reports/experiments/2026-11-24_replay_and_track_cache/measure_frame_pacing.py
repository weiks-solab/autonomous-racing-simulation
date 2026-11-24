"""比較 --realtime 與 --replay 的幀間隔（Agg，不開視窗）。

用法（專案根目錄）：
    python reports/experiments/2026-11-24_replay_and_track_cache/measure_frame_pacing.py <輸出.json> [模擬秒數]

Agg 沒有真正的 GUI 事件迴圈，這裡把 start_event_loop 換成精準的忙等，模擬 GUI 等待；
記錄每次 update_vehicles 開始的牆鐘時間。實際 TkAgg 視窗另含畫面傳輸與 Tk 計時誤差，需目視確認。
"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np  # noqa: E402
from matplotlib.backend_bases import FigureCanvasBase  # noqa: E402

import main  # noqa: E402
from sim.visualizer import SimulationVisualizer  # noqa: E402


def _busy_wait(self, timeout: float = 0.0) -> None:
    end = time.perf_counter() + max(timeout, 0.0)
    while time.perf_counter() < end:
        pass


def measure(seconds: float, **kwargs) -> dict:
    stamps: list[float] = []
    original = SimulationVisualizer.update_vehicles

    def stamped(self, *args, **inner):
        stamps.append(time.perf_counter())
        return original(self, *args, **inner)

    with patch.object(SimulationVisualizer, "update_vehicles", stamped), \
         patch.object(SimulationVisualizer, "show", lambda self: None), \
         patch.object(FigureCanvasBase, "start_event_loop", _busy_wait), \
         patch("main.SIMULATION_SECONDS", seconds), redirect_stdout(io.StringIO()):
        started = time.perf_counter()
        main.run_simulation(**kwargs)
        wall = time.perf_counter() - started
    intervals = np.diff(stamps) * 1000.0
    shown = len(stamps)
    return {
        "mode": kwargs,
        "wall_seconds_total": round(wall, 2),
        "frames_shown": shown,
        "playback_seconds": round(stamps[-1] - stamps[0], 3) if shown > 1 else 0.0,
        "interval_ms": {
            "mean": round(float(intervals.mean()), 2),
            "median": round(float(np.median(intervals)), 2),
            "p95": round(float(np.percentile(intervals, 95)), 2),
            "max": round(float(intervals.max()), 2),
            "share_over_50ms": round(float(np.mean(intervals > 50.0)), 3),
        },
    }


if __name__ == "__main__":
    output = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    results = [measure(seconds, realtime=True), measure(seconds, replay=True)]
    for item in results:
        print(json.dumps(item, ensure_ascii=False), flush=True)
    report = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.strip(),
        "python": platform.python_version(),
        "simulation_seconds": seconds,
        "frame_period_ms": main.MODEL_CONFIG.time_step * 1000.0,
        "note": "Agg＋忙等模擬 GUI 等待；不含 TkAgg 畫面傳輸。",
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
