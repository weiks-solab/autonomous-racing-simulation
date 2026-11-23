"""以 main.py 目前設定跑一次 Agg 模擬，記錄碰撞、模式次數、決策耗時、平均 |d| 與目標切換次數。

用法：python reports/experiments/2026-11-23_collision_responsibility/measure.py <輸出 json> [--no-responsibility]
複製自 2026-11-23_centerline_deviation_cost/measure.py，另加：
- --no-responsibility：把進攻、防守設定的 responsibility 改回 None（原本碰撞篩選）。
- 每次決策的責任違反剔除數，以及進攻車領先防守車的首次時間。
只包裝策略 update 計時，不改變模擬行為。
"""

from dataclasses import replace
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


def _rejections(candidates) -> dict[str, int]:
    counts = {"overlap": 0, "responsibility": 0}
    for candidate in candidates:
        if candidate.collision is not None:
            kind = getattr(candidate.collision, "kind", "overlap")
            counts[kind] = counts.get(kind, 0) + 1
    return counts


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
                "target_d": result.target_d,
                "candidates": len(self.last_candidates),
                "rejected": _rejections(self.last_candidates),
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
        "target_d_switches": sum(
            abs(first["target_d"] - second["target_d"]) > 1e-9
            for first, second in zip(items, items[1:])
        ),
        "max_candidates": max(item["candidates"] for item in items),
        "rejected_candidates": {
            kind: sum(item["rejected"].get(kind, 0) for item in items)
            for kind in ("overlap", "responsibility")
        },
        "evaluations_with_rejection": sum(any(item["rejected"].values()) for item in items),
    }


def main_entry(output: Path, applied: dict) -> None:
    start = time.perf_counter()
    main.run_simulation(realtime=False)
    wall = time.perf_counter() - start
    track = main.create_demo_track()
    length = track_length(track)
    ids = list(steps[0]["states"])
    starts = {vehicle_id: s for vehicle_id, s, _ in main.VEHICLE_SPECS}
    progress = {vehicle_id: starts[vehicle_id] for vehicle_id in ids}
    previous = {vehicle_id: world_to_track(steps[0]["states"][vehicle_id], track) for vehicle_id in ids}
    absolute_d = {vehicle_id: [abs(previous[vehicle_id].d)] for vehicle_id in ids}
    dt = main.MODEL_CONFIG.time_step
    attacker = next((key for key, value in main.STRATEGY_ASSIGNMENTS.items()
                     if value.name == "relative_overtake"), None)
    defender = next((key for key, value in main.STRATEGY_ASSIGNMENTS.items()
                     if value.name == "defend"), None)
    first_lead = None
    # 每步的累積 s 與 d，供對照剔除發生當下的實際相對位置。
    series = [{"s": dict(progress), "d": {key: previous[key].d for key in ids}}]
    for index, step in enumerate(steps[1:], start=1):
        for vehicle_id in ids:
            position = world_to_track(step["states"][vehicle_id], track)
            progress[vehicle_id] += signed_longitudinal_gap(previous[vehicle_id], position, length)
            absolute_d[vehicle_id].append(abs(position.d))
            previous[vehicle_id] = position
        series.append({"s": dict(progress), "d": {key: previous[key].d for key in ids}})
        if (first_lead is None and attacker in progress and defender in progress
                and progress[attacker] > progress[defender]):
            first_lead = (index + 1) * dt

    def rejection_timeline(items: list[dict]) -> list[dict]:
        """決策時刻 t 使用第 round(t/dt)-1 步後的狀態（t=0 用第一步）。"""
        timeline = []
        for item in items:
            if not any(item["rejected"].values()):
                continue
            state = series[max(round(item["t"] / dt) - 1, 0)]
            entry = {"t": round(item["t"], 3), "rejected": item["rejected"],
                     "selected_target_d": item["target_d"],
                     "d": {key: round(value, 3) for key, value in state["d"].items()}}
            if attacker is not None and defender is not None:
                entry["s_attacker_minus_defender"] = round(
                    state["s"][attacker] - state["s"][defender], 3)
            timeline.append(entry)
        return timeline
    collision_steps = [index for index, step in enumerate(steps) if step["pairs"]]
    result = {
        "track": main.TRACK_NAME, "simulation_seconds": main.SIMULATION_SECONDS,
        "assignments": {key: value.name for key, value in main.STRATEGY_ASSIGNMENTS.items()},
        "applied": applied,
        "wall_seconds_including_plot": wall,
        "final_s_from_start": progress,
        "first_attacker_lead_time": first_lead,
        "mean_abs_d": {key: sum(values) / len(values) for key, values in absolute_d.items()},
        "collision_steps": len(collision_steps),
        "first_collision_time": None if not collision_steps else (collision_steps[0] + 1) * dt,
        "collision_pairs": sorted({tuple(pair) for step in steps for pair in step["pairs"]}),
        "strategies": {key: _summary(value) for key, value in records.items()},
        "rejection_timeline": {key: rejection_timeline(value) for key, value in records.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _apply_options(arguments: list[str]) -> dict:
    """--no-responsibility：兩邊的 responsibility 改回 None。
    --radius R／--slope A／--offset B：覆寫兩邊共用的 r_col、a、b。
    --defense-collision-horizon H：覆寫防守車的檢查時域（責任約束也使用此值）。
    """
    applied: dict = {}
    responsibility = getattr(main, "RESPONSIBILITY_CONFIG", None)
    overrides = {}
    for flag, field in (("--radius", "collision_radius"), ("--slope", "slope"),
                        ("--offset", "offset")):
        if flag in arguments:
            overrides[field] = float(arguments[arguments.index(flag) + 1])
    if overrides and responsibility is not None:
        responsibility = replace(responsibility, **overrides)
        applied.update(overrides)
        for name in ("RELATIVE_OVERTAKING_CONFIG", "DEFENSE_CONFIG"):
            config = getattr(main, name)
            if getattr(config, "responsibility", None) is not None:
                setattr(main, name, replace(config, responsibility=responsibility))
    if "--defense-collision-horizon" in arguments:
        horizon = float(arguments[arguments.index("--defense-collision-horizon") + 1])
        main.DEFENSE_CONFIG = replace(main.DEFENSE_CONFIG, collision_horizon=horizon)
        applied["defense_collision_horizon"] = horizon
    if "--no-responsibility" in arguments:
        for name in ("RELATIVE_OVERTAKING_CONFIG", "DEFENSE_CONFIG"):
            config = getattr(main, name)
            if hasattr(config, "responsibility"):
                setattr(main, name, replace(config, responsibility=None))
        applied["responsibility"] = None
    applied["attacker_responsibility"] = repr(getattr(main.RELATIVE_OVERTAKING_CONFIG, "responsibility", None))
    applied["defender_responsibility"] = repr(getattr(main.DEFENSE_CONFIG, "responsibility", None))
    return applied


if __name__ == "__main__":
    options = _apply_options(sys.argv[2:])
    print(f"options: {options}")
    main_entry(Path(sys.argv[1]), options)
