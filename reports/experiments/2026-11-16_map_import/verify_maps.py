"""盤點本機地圖，並以 Agg 驗證 PNG 疊圖及即時更新流程。"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import subprocess
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

import main
from core.track_factory import TrackConfig, create_track
from core.track_geometry import track_length
from sim.visualizer import SimulationVisualizer


def verify() -> None:
    directory = Path(__file__).resolve().parent
    version = {
        "date": "2026-11-16",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT,
                                          text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                               text=True).strip(),
        "version_note": "基於 base_commit，含未提交修改。",
    }
    inventory = []
    for folder in sorted((PROJECT_ROOT / main.TRACK_MAP_ROOT).iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        track = create_track(TrackConfig(name=folder.name, source="f1tenth"))
        background = main.create_demo_background(track)
        # 世界座標轉回 PNG 像素，獨立核對原點、解析度與上下方向。
        x, y, yaw = background.origin
        cosine, sine = np.cos(yaw), np.sin(yaw)
        local = (track.centerline - (x, y)) @ np.array([[cosine, -sine], [sine, cosine]])
        columns = np.floor(local[:, 0] / background.resolution).astype(int)
        rows = background.pixels.shape[0] - 1 - np.floor(local[:, 1] / background.resolution).astype(int)
        height, width = background.pixels.shape[:2]
        inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        pixels = background.pixels[rows[inside], columns[inside]]
        brightness = pixels if pixels.ndim == 1 else np.mean(pixels[:, :3], axis=1)
        entry = {
            "name": folder.name, "source": track.source.centerline_file,
            "centerline_file_sha256": track.source.sha256,
            "point_count": len(track.centerline), "length_m": track_length(track),
            "width_m": track.width, "sample_spacing_m": track.source.sample_spacing,
            "resolution_m_per_pixel": background.resolution, "origin": background.origin,
            "image_shape": background.pixels.shape,
            "centerline_samples_inside_image": int(np.count_nonzero(inside)),
            "centerline_samples_on_light_pixels": int(np.count_nonzero(brightness > 0.8)),
        }
        assert np.all(inside), entry
        assert np.all(brightness > 0.8), entry
        inventory.append(entry)
    (directory / "map_inventory.json").write_text(
        json.dumps({**version, "scope": "靜態載入與像素對齊；非所有地圖循跡驗證。",
                    "upstream_repository": "https://github.com/f1tenth/f1tenth_racetracks.git",
                    "upstream_commit": subprocess.check_output(
                        ["git", "-C", str(PROJECT_ROOT / main.TRACK_MAP_ROOT), "rev-parse", "HEAD"],
                        text=True).strip(),
                    "maps": inventory}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    frames = 0

    class RecordingVisualizer(SimulationVisualizer):
        def __init__(self, track, config, background=None):
            super().__init__(track, replace(config, realtime=False), background)

        def update_vehicles(self, **kwargs):
            nonlocal frames
            frames += 1
            super().update_vehicles(**kwargs)
            assert len(self.axes.images) == 1

    console = io.StringIO()
    with patch("main.SimulationVisualizer", RecordingVisualizer), redirect_stdout(console):
        result = main.run_single_vehicle("Oschersleben", realtime=True, laps=2,
                                         max_seconds=2, show=False)
    assert not result.completed
    assert frames == 11
    (directory / "live_check.json").write_text(
        json.dumps({**version, "scope": "Agg 即時更新流程，未人工操作 GUI。",
                    "simulation_seconds": result.simulation_seconds,
                    "update_frames": frames, "map_artists_per_frame": 1,
                    "timeout_reported": not result.completed}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plt.close("all")
    print(f"Loaded and aligned {len(inventory)} maps; live update frames: {frames}.")


if __name__ == "__main__":
    verify()
