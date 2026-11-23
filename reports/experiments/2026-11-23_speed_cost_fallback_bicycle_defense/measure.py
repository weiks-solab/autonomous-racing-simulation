"""以 main.py 目前設定跑一次 Agg 模擬，記錄碰撞、模式次數與每次決策耗時。

用法：python reports/experiments/2026-11-23_speed_cost_fallback_bicycle_defense/measure.py <輸出 json>
修改前後使用同一支腳本；只包裝策略 update 計時，不改變模擬行為。
"""

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import main  # noqa: E402
from core.track_geometry import signed_longitudinal_gap, track_length, world_to_track  # noqa: E402
from strategy.defending import CostBasedDefenseStrategy  # noqa: E402
from strategy.relative_overtaking import RelativeOvertakingStrategy  # noqa: E402

records: dict[str, list[dict]] = {"relative_overtake": [], "defend": []}


def _wrap(cls, key: str) -> None:
    original = cls.update

    def timed(self, vehicle, others, track, time_seconds):
        start = time.perf_counter()
        result = original(self, vehicle, others, track, time_seconds)
        elapsed = time.perf_counter() - start
        if self.last_evaluation_time == time_seconds:
            records[key].append({
                "t": time_seconds, "ms": elapsed * 1000.0,
                "mode": result.mode if isinstance(result.mode, str) else result.mode.value,
                "speed_limit": getattr(result, "speed_limit", None),
                "candidates": len(self.last_candidates),
            })
        return result

    cls.update = timed


_wrap(RelativeOvertakingStrategy, "relative_overtake")
_wrap(CostBasedDefenseStrategy, "defend")

steps: list[dict] = []
original_find = main.find_vehicle_collisions


def tracked_find(vehicles):
    pairs = original_find(vehicles)
    steps.append({"states": {v.vehicle_id: v.state for v in vehicles}, "pairs": sorted(pairs)})
    return pairs


main.find_vehicle_collisions = tracked_find


def _summary(items: list[dict]) -> dict:
    if not items:
        return {"evaluations": 0}
    times = [item["ms"] for item in items]
    modes: dict[str, int] = {}
    for item in items:
        modes[item["mode"]] = modes.get(item["mode"], 0) + 1
    return {
        "evaluations": len(items),
        "mean_ms": sum(times) / len(times), "max_ms": max(times),
        "total_s": sum(times) / 1000.0,
        "mode_counts": modes,
        "speed_limited_evaluations": sum(item["speed_limit"] is not None for item in items),
        "max_candidates": max(item["candidates"] for item in items),
    }


def main_entry(output: Path) -> None:
    start = time.perf_counter()
    main.run_simulation(realtime=False)
    wall = time.perf_counter() - start
    track = main.create_demo_track()
    length = track_length(track)
    ids = list(steps[0]["states"])
    progress = {vehicle_id: 0.0 for vehicle_id in ids}
    previous = {vehicle_id: world_to_track(steps[0]["states"][vehicle_id], track) for vehicle_id in ids}
    for step in steps[1:]:
        for vehicle_id in ids:
            position = world_to_track(step["states"][vehicle_id], track)
            progress[vehicle_id] += signed_longitudinal_gap(previous[vehicle_id], position, length)
            previous[vehicle_id] = position
    starts = {vehicle_id: s for vehicle_id, s, _ in main.VEHICLE_SPECS}
    dt = main.MODEL_CONFIG.time_step
    collision_steps = [index for index, step in enumerate(steps) if step["pairs"]]
    result = {
        "track": main.TRACK_NAME, "simulation_seconds": main.SIMULATION_SECONDS,
        "assignments": {key: value.name for key, value in main.STRATEGY_ASSIGNMENTS.items()},
        "wall_seconds_including_plot": wall,
        "final_s_from_start": {key: starts[key] + progress[key] for key in ids},
        "collision_steps": len(collision_steps),
        "first_collision_time": None if not collision_steps else (collision_steps[0] + 1) * dt,
        "collision_pairs": sorted({tuple(pair) for step in steps for pair in step["pairs"]}),
        "strategies": {key: _summary(value) for key, value in records.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _apply_ablation(arguments: list[str]) -> dict:
    """對照組選項：--safety-weight W 覆寫安全項權重；--geometric-defense 讓防守車改回幾何預測。"""
    from dataclasses import replace
    applied: dict = {}
    if "--safety-weight" in arguments:
        weight = float(arguments[arguments.index("--safety-weight") + 1])
        main.RELATIVE_OVERTAKING_CONFIG = replace(main.RELATIVE_OVERTAKING_CONFIG, safety_weight=weight)
        applied["safety_weight"] = weight
    if "--geometric-defense" in arguments:
        original_manager = main.StrategyManager

        def geometric_manager(*args, **kwargs):
            kwargs["prediction_configs"] = None
            return original_manager(*args, **kwargs)

        main.StrategyManager = geometric_manager
        applied["defense_prediction"] = "geometric"
    return applied


if __name__ == "__main__":
    ablation = _apply_ablation(sys.argv[2:])
    if ablation:
        print(f"ablation: {ablation}")
    main_entry(Path(sys.argv[1]))
