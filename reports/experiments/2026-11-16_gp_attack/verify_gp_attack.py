"""以既有模型執行 main.py 的預設完整場景；不開視窗、不重訓。"""

from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import main
import numpy
import sklearn


def main_check() -> None:
    directory = Path(__file__).resolve().parent
    output = io.StringIO()
    collision_frames = 0
    collision_pairs: set[tuple[str, str]] = set()
    original_check = main.find_vehicle_collisions

    def record_collisions(vehicles: list) -> set[tuple[str, str]]:
        nonlocal collision_frames
        pairs = original_check(vehicles)
        collision_frames += bool(pairs)
        collision_pairs.update(pairs)
        return pairs

    started = time.perf_counter()
    with patch("main.SimulationVisualizer") as visualizer, \
         patch("main.find_vehicle_collisions", side_effect=record_collisions), redirect_stdout(output):
        main.run_simulation()
    elapsed = time.perf_counter() - started
    frame = visualizer.return_value.update_vehicles.call_args.kwargs
    sources = ("main.py", "strategy/gp_prediction.py", "strategy/relative_overtaking.py")
    summary = {
        "date": "2026-11-16", "branch": "master",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                                text=True).strip(),
        "version_note": "基於 base_commit，含未提交修改；歷史訓練版本未記錄。",
        "scope": "預設完整場景的接入執行驗證；僅替換視覺化，不宣稱策略效果改善。",
        "seconds": main.SIMULATION_SECONDS, "time_step": main.MODEL_CONFIG.time_step,
        "vehicle_specs": main.VEHICLE_SPECS,
        "assignments": {key: asdict(value) for key, value in main.STRATEGY_ASSIGNMENTS.items()},
        "relative_config": asdict(main.RELATIVE_OVERTAKING_CONFIG),
        "model_config": asdict(main.MODEL_CONFIG), "geometry": asdict(main.VEHICLE_GEOMETRY),
        "progress_config": asdict(main.PROGRESS_CONFIG),
        "controller": {"lookahead_distance": main.LOOKAHEAD_DISTANCE,
                       "speed_gain": main.SPEED_GAIN, "max_acceleration": main.MAX_ACCELERATION},
        "track": {"name": main.TRACK_NAME, "rectangle_length": main.TRACK_RECTANGLE_LENGTH,
                  "rectangle_height": main.TRACK_RECTANGLE_HEIGHT, "corner_radius": main.TRACK_CORNER_RADIUS,
                  "point_count": main.TRACK_POINT_COUNT, "width": main.TRACK_WIDTH},
        "gp_enabled": main.GP_PREDICTION_ENABLED, "gp_model_paths": main.GP_MODEL_PATHS,
        "gp_model_sha256": {key: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
                            for key, path in main.GP_MODEL_PATHS.items()},
        "source_sha256": {path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
                          for path in sources},
        "versions": {"python": sys.version, "numpy": numpy.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": elapsed, "actual_collision_frames": collision_frames,
        "actual_collision_pairs": sorted(collision_pairs),
        "trajectory_point_counts": {key: len(value) for key, value in frame["trajectories"].items()},
        "model_not_archived": "model.pkl 僅在本機且被忽略；重現需提供相同模型或另訓練相容模型。",
    }
    (directory / "smoke.log").write_text(output.getvalue(), encoding="utf-8")
    (directory / "smoke_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(output.getvalue())
    print(f"Completed {main.SIMULATION_SECONDS:g}s; collision frames={collision_frames}; wall={elapsed:.2f}s")


if __name__ == "__main__":
    main_check()
