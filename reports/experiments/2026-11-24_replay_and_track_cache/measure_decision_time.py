"""量測預設場景每次重新決策的耗時（不開視窗），用來比較賽道幾何快取前後。

用法（專案根目錄）：
    python reports/experiments/2026-11-24_replay_and_track_cache/measure_decision_time.py <輸出.json> [模擬秒數] [重複次數]

只包裝策略的 update 計時，不改變模擬行為；請勿與其他重度程序同時執行，以免互相干擾。
"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np  # noqa: E402

import main  # noqa: E402
from strategy.defending import CostBasedDefenseStrategy  # noqa: E402
from strategy.relative_overtaking import RelativeOvertakingStrategy  # noqa: E402

STRATEGIES = {"relative_overtake": RelativeOvertakingStrategy, "defend": CostBasedDefenseStrategy}


def run_once(seconds: float) -> dict:
    samples: dict[str, list[float]] = {name: [] for name in STRATEGIES}
    originals = {name: cls.update for name, cls in STRATEGIES.items()}

    def timed(name):
        def wrapper(self, vehicle, others, track, time_seconds):
            started = time.perf_counter()
            result = originals[name](self, vehicle, others, track, time_seconds)
            if self.last_evaluation_time == time_seconds:  # 只計重新決策，不計沿用
                samples[name].append((time.perf_counter() - started) * 1000.0)
            return result
        return wrapper

    patches = [patch.object(cls, "update", timed(name)) for name, cls in STRATEGIES.items()]
    for item in patches:
        item.start()
    try:
        with patch("main.SIMULATION_SECONDS", seconds), patch("main.SimulationVisualizer"), \
                redirect_stdout(io.StringIO()):
            started = time.perf_counter()
            main.run_simulation()
            wall = time.perf_counter() - started
    finally:
        for item in patches:
            item.stop()
    return {
        "wall_seconds": round(wall, 3),
        "decisions": {
            name: {
                "count": len(values),
                "mean_ms": round(statistics.mean(values), 2),
                "median_ms": round(statistics.median(values), 2),
                "p95_ms": round(float(np.percentile(values, 95)), 2),
            }
            for name, values in samples.items() if values
        },
    }


if __name__ == "__main__":
    output = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    runs = []
    for index in range(repeats):
        runs.append(run_once(seconds))
        print(json.dumps(runs[-1], ensure_ascii=False), flush=True)
    report = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.strip(),
        "uncommitted_files": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                            capture_output=True, text=True).stdout.splitlines(),
        "python": platform.python_version(),
        "scenario": {"track": main.TRACK_NAME, "simulation_seconds": seconds,
                     "assignments": {key: value.name for key, value in main.STRATEGY_ASSIGNMENTS.items()}},
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
