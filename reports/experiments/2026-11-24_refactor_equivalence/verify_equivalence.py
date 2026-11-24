"""重構等價驗證：用公開介面跑固定場景，輸出逐步狀態與畫面資料的 SHA-256。

用法（專案根目錄）：
    python reports/experiments/2026-11-24_refactor_equivalence/verify_equivalence.py run <輸出.json> [場景 ...]
    python reports/experiments/2026-11-24_refactor_equivalence/verify_equivalence.py compare <a.json> <b.json>

重構前後各跑一次再比對；除執行秒數外全部相同，才表示模擬行為不變。
只 patch main 的常數與 SimulationVisualizer，不修改任何設定檔。
"""

from contextlib import redirect_stdout
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
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

import main  # noqa: E402
from strategy.assignment import StrategyAssignment  # noqa: E402

GP_MODEL = "models/gp/20261111T064250Z_dbc5bc95/model.pkl"
IMPORTED_TRACK = "Oschersleben"
TIMING_KEYS = {"seconds"}  # 只供參考，比對時忽略


def canonical(value):
    """轉成只含基本型別的結構；浮點以 hex 表示，不受 repr 或 numpy 型別影響。"""
    if isinstance(value, Enum):
        return canonical(value.value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value, dtype=float)
        return ["ndarray", list(array.shape), hashlib.sha256(array.tobytes()).hexdigest()]
    if is_dataclass(value) and not isinstance(value, type):
        return [type(value).__name__,
                [[item.name, canonical(getattr(value, item.name))] for item in fields(value)]]
    if hasattr(value, "items"):
        return ["mapping", sorted(([canonical(k), canonical(v)] for k, v in value.items()), key=repr)]
    if isinstance(value, (set, frozenset)):
        return ["set", sorted((canonical(item) for item in value), key=repr)]
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    raise TypeError(f"unsupported value for canonical hashing: {type(value)!r}")


def digest(value) -> str:
    text = json.dumps(canonical(value), separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_main(patches: dict, **kwargs) -> dict:
    """以 main.run_simulation 跑完整場景，記錄最後一次交給畫面的全部資料。"""
    output = io.StringIO()
    started = time.perf_counter()
    with patch.multiple(main, **patches), patch("main.SimulationVisualizer") as visualizer, \
            redirect_stdout(output):
        main.run_simulation(**kwargs)
    elapsed = time.perf_counter() - started
    frame = visualizer.return_value.update_vehicles.call_args.kwargs
    args, call_kwargs = visualizer.call_args.args, visualizer.call_args.kwargs
    background = call_kwargs.get("background", args[2] if len(args) > 2 else None)
    text = output.getvalue()
    return {
        "track": digest(args[0]),
        "display_options": digest(args[1]),
        "background": None if background is None else Path(background.image_file).name,
        "trajectories": {
            vehicle_id: {"steps": len(states), "sha256": digest(states)}
            for vehicle_id, states in sorted(frame["trajectories"].items())
        },
        "final_frame": digest(frame),
        "stdout_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "stdout_tail": text.strip().splitlines()[-1],
        "seconds": round(elapsed, 2),
    }


def run_data_generation() -> dict:
    """無視窗資料生成入口：比對每一幀觀測、碰撞與每步決策和控制命令。"""
    from generate_training_data import _demo_settings
    from sim.data_generation import RandomizationConfig, randomized_scenarios, run_episode
    from training_data import TrainingDataCollector

    started = time.perf_counter()
    settings, specs, assignments = _demo_settings(False)
    scenario = randomized_scenarios(settings, specs, assignments, 1, 42, RandomizationConfig())[0]
    collector = TrainingDataCollector(settings.model_config.time_step)
    steps = []

    def observe(step, observation, decision, command):
        steps.append((step, observation.vehicle_id, decision.mode, decision.target_d,
                      decision.speed_limit, decision.maneuver, decision.reason,
                      decision.details, command))

    run_episode(scenario, settings, collector, 2.0, step_observer=observe)
    episode = collector.episodes[0]
    return {
        "scenario": digest(scenario),
        "frames": {"count": len(episode.frames), "sha256": digest(collector.episodes)},
        "decisions_and_commands": {"count": len(steps), "sha256": digest(steps)},
        "seconds": round(time.perf_counter() - started, 2),
    }


def _overtake_scene(path_length: float | None) -> dict:
    patches = {
        "SIMULATION_SECONDS": 8.0,
        "VEHICLE_SPECS": (("vehicle-1", 0.0, 6.0), ("vehicle-2", 6.0, 5.0), ("vehicle-3", -20.0, 4.0)),
        "STRATEGY_ASSIGNMENTS": {
            "vehicle-1": StrategyAssignment("overtake"),
            "vehicle-2": StrategyAssignment("defend", target_vehicle_id="vehicle-1"),
        },
    }
    if path_length is not None:
        patches["OFFSET_PATH_CONFIG"] = replace(main.OFFSET_PATH_CONFIG, path_length=path_length)
    return run_main(patches)


SCENARIOS = {
    # 目前 main 的預設場景：relative_overtake 對 defend、避撞責任 overlap_rule。
    "default_6s": lambda: run_main({"SIMULATION_SECONDS": 6.0}),
    "default_no_defense_3s": lambda: run_main({"SIMULATION_SECONDS": 3.0}, defense_enabled=False),
    "cost_overtake_defend_3s": lambda: run_main({
        "SIMULATION_SECONDS": 3.0,
        "STRATEGY_ASSIGNMENTS": {
            "vehicle-1": StrategyAssignment("cost_overtake"),
            "vehicle-2": StrategyAssignment("defend", target_vehicle_id="vehicle-1"),
        },
    }),
    "progress_centerline_3s": lambda: run_main({
        "SIMULATION_SECONDS": 3.0,
        "VEHICLE_SPECS": (("vehicle-1", 0.0, 11.0), ("vehicle-2", 4.0, 9.0), ("vehicle-3", -12.0, 8.0)),
        "STRATEGY_ASSIGNMENTS": {
            "vehicle-1": StrategyAssignment("progress"),
            "vehicle-3": StrategyAssignment("progress"),
        },
    }),
    "probe_follow_demo_3s": lambda: run_main({"SIMULATION_SECONDS": 3.0}, probe_follow_demo=True),
    # 規則式超車：舊版 main 須把參考路徑延長到 6 m 才能執行。
    "overtake_defend_path6_8s": lambda: _overtake_scene(6.0),
    # 不 patch 路徑長度；重構前預期報錯，重構後應與上一個場景相同。
    "overtake_defend_auto_path_8s": lambda: _overtake_scene(None),
    # 關閉責任約束：進攻用矩形重疊剔除，防守啟用指定對象的碰撞條件。
    "rectangle_collision_checks_3s": lambda: run_main({
        "SIMULATION_SECONDS": 3.0,
        "RELATIVE_OVERTAKING_CONFIG": replace(main.RELATIVE_OVERTAKING_CONFIG, responsibility=None),
        "DEFENSE_CONFIG": replace(main.DEFENSE_CONFIG, responsibility=None, collision_check_enabled=True),
    }),
    "gp_relative_3s": lambda: run_main({
        "SIMULATION_SECONDS": 3.0, "GP_PREDICTION_ENABLED": True,
        "GP_MODEL_PATHS": {"vehicle-2": GP_MODEL},
    }),
    "imported_track_2s": lambda: run_main({"SIMULATION_SECONDS": 2.0, "TRACK_NAME": IMPORTED_TRACK}),
    "data_generation_2s": run_data_generation,
}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def run(output: Path, names: list[str]) -> None:
    results = {}
    for name in names or SCENARIOS:
        print(f"running {name} ...", flush=True)
        try:
            results[name] = SCENARIOS[name]()
        except Exception as error:  # 舊版預期會失敗的場景也要留下紀錄。
            results[name] = {"error": f"{type(error).__name__}: {error}"}
        print(f"  {json.dumps(results[name], ensure_ascii=False)}", flush=True)
    report = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "uncommitted_files": _git("status", "--porcelain").splitlines(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "scenarios": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                      encoding="utf-8")


def _strip_timing(value):
    if isinstance(value, dict):
        return {key: _strip_timing(item) for key, item in value.items() if key not in TIMING_KEYS}
    return value


def compare(first: Path, second: Path) -> int:
    a = json.loads(first.read_text(encoding="utf-8"))["scenarios"]
    b = json.loads(second.read_text(encoding="utf-8"))["scenarios"]
    different = 0
    for name in sorted(set(a) | set(b)):
        same = name in a and name in b and _strip_timing(a[name]) == _strip_timing(b[name])
        different += not same
        print(f"{'SAME' if same else 'DIFF'}  {name}")
    print(f"{len(set(a) | set(b)) - different} same, {different} different")
    return 1 if different else 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(Path(sys.argv[2]), sys.argv[3:])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        raise SystemExit(compare(Path(sys.argv[2]), Path(sys.argv[3])))
    else:
        raise SystemExit(__doc__)
