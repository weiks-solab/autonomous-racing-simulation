"""只改 TRACK_NAME，以原有多車競速入口執行並保存最後畫面。"""

from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
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
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

import main
from core.track_geometry import track_length, world_to_track
from sim.visualizer import SimulationVisualizer


def verify() -> None:
    directory = Path(__file__).resolve().parent
    result = {}

    class RecordingVisualizer(SimulationVisualizer):
        def update_vehicles(self, **kwargs):
            super().update_vehicles(**kwargs)
            length = track_length(self.track)
            result["vehicles"] = {}
            for vehicle_id, trajectory in kwargs["trajectories"].items():
                positions = [world_to_track(state, self.track) for state in trajectory]
                increments = (np.diff([p.s for p in positions]) + length / 2) % length - length / 2
                result["vehicles"][vehicle_id] = {
                    "sample_count": len(trajectory),
                    "strategy": main.STRATEGY_ASSIGNMENTS[vehicle_id].name,
                    "progress_m": float(increments.sum()),
                    "max_abs_d_m": max(abs(p.d) for p in positions),
                    "initial_state": asdict(trajectory[0]), "final_state": asdict(trajectory[-1]),
                }
            result["track"] = {"source": {"name": main.TRACK_NAME, "centerline_file": main.find_centerline(main.TRACK_NAME, main.TRACK_MAP_ROOT).relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(main.find_centerline(main.TRACK_NAME, main.TRACK_MAP_ROOT).read_bytes()).hexdigest()}, "width_m": self.track.width,
                               "length_m": length, "point_count": len(self.track.centerline)}
            result["background_visible"] = self.map_artist is not None
            result["candidate_vehicle_ids"] = sorted(kwargs["candidate_snapshots"])

        def show(self):
            self.axes.set_title("Oschersleben | original multi-vehicle strategies | 21 s")
            self.figure.tight_layout()
            self.figure.savefig(directory / "multivehicle.png", dpi=180)
            plt.close(self.figure)

    console = io.StringIO()
    # 只改地圖與保存畫面的方法，車輛、策略、速度及時間均沿用 main.py。
    with patch("main.TRACK_NAME", "Oschersleben"), \
         patch("main.SimulationVisualizer", RecordingVisualizer), redirect_stdout(console):
        main.run_simulation()
    result.update({
        "date": "2026-11-16", "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "version_note": "基於 base_commit，含未提交修改。",
        "scope": "只確認原有多車策略在下載圖可執行；不是成功率或改善比較。Agg，未人工操作 GUI。",
        "simulation_seconds": main.SIMULATION_SECONDS,
        "vehicle_specs": main.VEHICLE_SPECS,
        "assignments": {key: asdict(value) for key, value in main.STRATEGY_ASSIGNMENTS.items()},
        "model_config": asdict(main.MODEL_CONFIG),
        "progress_config": asdict(main.PROGRESS_CONFIG),
        "relative_overtaking_config": asdict(main.RELATIVE_OVERTAKING_CONFIG),
        "offset_path_config": asdict(main.OFFSET_PATH_CONFIG),
        "lookahead_distance_m": main.LOOKAHEAD_DISTANCE,
        "gp_used": False,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in ("main.py", "core/track_factory.py", "core/track_loader.py",
                                    "strategy/relative_overtaking.py", "strategy/progress.py",
                                    "sim/visualizer.py", "sim/map_background.py")},
    })
    assert result["background_visible"]
    assert set(result["vehicles"]) == {spec[0] for spec in main.VEHICLE_SPECS}
    assert all(v["sample_count"] == 1051 for v in result["vehicles"].values())
    assert result["candidate_vehicle_ids"] == sorted(result["vehicles"])
    (directory / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "simulation.log").write_text(console.getvalue(), encoding="utf-8")
    print(console.getvalue())


if __name__ == "__main__":
    verify()
