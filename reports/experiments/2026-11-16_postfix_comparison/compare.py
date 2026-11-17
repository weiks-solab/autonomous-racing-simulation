"""重測事件迴圈修正後的兩圖差異；不修改正式程式。"""
import hashlib
import importlib.util
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
path = ROOT / "reports/experiments/2026-11-16_cache_comparison/compare_tracks.py"
spec = importlib.util.spec_from_file_location("cache_comparison", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from core.track_geometry import track_length

cases = []
for name, gp in (("rounded_rectangle", True), ("rounded_rectangle", False), ("Oschersleben", False)):
    with patch.object(FigureCanvasAgg, "start_event_loop"):
        case = module.run(name, gp)
    with patch("main.TRACK_NAME", name):
        track = module.main.create_demo_track()
        case["length_m"] = track_length(track)
        case["centerline_span_m"] = (track.centerline.max(axis=0)-track.centerline.min(axis=0)).tolist()
    cases.append(case)
    print(json.dumps(case), flush=True)
report = {"date": "2026-11-16", "branch": subprocess.check_output(
    ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
    "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "version_note": "基於 base_commit；本次只新增量測紀錄，未改正式程式。",
    "scope": "各一次8秒400步；Agg，不含GUI事件等待，非實際GUI FPS。",
    "cases": cases, "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    for p in ("main.py", "sim/visualizer.py", "core/track_geometry.py", "strategy/relative_overtaking.py")}}
(Path(__file__).parent / "summary.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
