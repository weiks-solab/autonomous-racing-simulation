"""產生整圈圖，並以即時更新流程保存 2 秒場景的最後一幀。"""

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import main
import plot_gp_route
from sim.visualizer import SimulationVisualizer


def verify() -> None:
    directory = Path(__file__).resolve().parent
    plot_gp_route.main(["--output", str(directory / "gp_route.png")])
    frames = 0
    snapshots = {}

    class RecordingVisualizer(SimulationVisualizer):
        def __init__(self, track, config) -> None:
            # 保留每步即時更新流程，但使用 Agg，不開 GUI 或暫停。
            super().__init__(track, replace(config, realtime=False))

        def update_vehicles(self, **kwargs) -> None:
            nonlocal frames, snapshots
            super().update_vehicles(**kwargs)
            frames += 1
            snapshots = kwargs["candidate_snapshots"]

        def show(self) -> None:
            self.figure.tight_layout()
            self.figure.savefig(directory / "gp_live_preview.png", dpi=160)
            visible = [artists for artists in self._gp_prediction_artists.values() if artists.line.get_visible()]
            assert visible, "即時更新必須顯示 GP 預測。"
            summary = {
                "date": "2026-11-16", "branch": "master",
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                                       text=True).strip(),
                "version_note": "基於 base_commit，含未提交修改。",
                "scope": "Agg 後端執行即時逐步更新流程；沒有人工操作 GUI。",
                "simulation_seconds": 2.0, "update_frames": frames,
                "gp_labels": [artists.label.get_text() for artists in visible],
                "gp_point_counts": [len(artists.line.get_xdata()) for artists in visible],
                "snapshot_evaluation_times": {key: snapshot.evaluated_at for key, snapshot in snapshots.items()},
                "source_sha256": {
                    path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
                    for path in ("plot_gp_route.py", "sim/gp_visualization.py", "sim/visualizer.py",
                                 "main.py", "strategy/base.py", "strategy/relative_overtaking.py")
                },
            }
            (directory / "live_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )

    output = io.StringIO()
    with patch("main.SIMULATION_SECONDS", 2.0), patch("main.SimulationVisualizer", RecordingVisualizer), \
         redirect_stdout(output):
        main.run_simulation(realtime=True)
    (directory / "live.log").write_text(output.getvalue(), encoding="utf-8")
    print(f"Saved whole-lap route and live preview; {frames} update frames.")


if __name__ == "__main__":
    verify()
