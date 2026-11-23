"""以 main.py 目前設定跑一次 Agg 模擬，比較責任約束的「前車懲罰」設定。

用法：python reports/experiments/2026-11-23_responsibility_leader_penalty/measure.py <輸出 json> [選項]
複製自 2026-11-23_collision_responsibility/measure.py，另加：
- --leader-soft：當下在前的車改為成本懲罰（後車維持硬約束）。
- --defense-leader-weight W／--attack-leader-weight W：兩邊的前車懲罰權重。
- 阻擋指標：攻方落後 0～3 m 時，兩車平均 |d_守 − d_攻|（越小越貼近阻擋）。
- 選中候選的前車懲罰。
只包裝策略 update，不改變模擬行為。
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
BLOCKING_RANGE = 3.0  # m；攻方落後此距離內才計算阻擋指標


def _rejections(candidates) -> dict[str, int]:
    counts = {"overlap": 0, "responsibility": 0}
    for candidate in candidates:
        if candidate.collision is not None:
            kind = getattr(candidate.collision, "kind", "overlap")
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def _selected_leader_cost(strategy, result) -> float:
    """選中候選的前車懲罰（已乘權重）；舊版沒有此欄位時回傳 0。"""
    for candidate in strategy.last_candidates:
        if not candidate.feasible:
            continue
        target = getattr(candidate, "target_d", None)
        if target is None:
            target = candidate.maneuver.target_d
        if (target == result.target_d
                and getattr(candidate, "speed_limit", None) == getattr(result, "speed_limit", None)):
            return float(getattr(candidate, "leader_cost", 0.0) or 0.0)
    return 0.0


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
                "selected_leader_cost": _selected_leader_cost(self, result),
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
    leader_costs = [item["selected_leader_cost"] for item in items]
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
        "evaluations_selected_with_leader_cost": sum(value > 0.0 for value in leader_costs),
        "selected_leader_cost_max": max(leader_costs),
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
    blocking: list[float] = []
    series = [{"s": dict(progress), "d": {key: previous[key].d for key in ids}}]
    for index, step in enumerate(steps[1:], start=1):
        for vehicle_id in ids:
            position = world_to_track(step["states"][vehicle_id], track)
            progress[vehicle_id] += signed_longitudinal_gap(previous[vehicle_id], position, length)
            absolute_d[vehicle_id].append(abs(position.d))
            previous[vehicle_id] = position
        series.append({"s": dict(progress), "d": {key: previous[key].d for key in ids}})
        if attacker in progress and defender in progress:
            behind = progress[defender] - progress[attacker]
            if first_lead is None and behind < 0.0:
                first_lead = (index + 1) * dt
            if first_lead is None and 0.0 < behind <= BLOCKING_RANGE:
                blocking.append(abs(previous[defender].d - previous[attacker].d))

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
        "blocking_mean_abs_delta_d_before_lead": (
            None if not blocking else sum(blocking) / len(blocking)),
        "blocking_samples": len(blocking),
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


def _value(arguments: list[str], flag: str) -> float | None:
    return float(arguments[arguments.index(flag) + 1]) if flag in arguments else None


def _apply_options(arguments: list[str]) -> dict:
    """覆寫 main 設定；未給選項時完全沿用 main。"""
    applied: dict = {}
    responsibility = getattr(main, "RESPONSIBILITY_CONFIG", None)
    overrides: dict = {}
    for flag, field in (("--radius", "collision_radius"), ("--slope", "slope"),
                        ("--offset", "offset")):
        value = _value(arguments, flag)
        if value is not None:
            overrides[field] = value
    if "--leader-soft" in arguments:
        overrides["leader_soft"] = True
    if overrides and responsibility is not None:
        responsibility = replace(responsibility, **overrides)
        applied.update(overrides)
        for name in ("RELATIVE_OVERTAKING_CONFIG", "DEFENSE_CONFIG"):
            config = getattr(main, name)
            if getattr(config, "responsibility", None) is not None:
                setattr(main, name, replace(config, responsibility=responsibility))
    for flag, name, field in (
            ("--defense-leader-weight", "DEFENSE_CONFIG", "leader_penalty_weight"),
            ("--attack-leader-weight", "RELATIVE_OVERTAKING_CONFIG", "leader_penalty_weight"),
            ("--defense-collision-horizon", "DEFENSE_CONFIG", "collision_horizon")):
        value = _value(arguments, flag)
        if value is not None:
            setattr(main, name, replace(getattr(main, name), **{field: value}))
            applied[f"{name}.{field}"] = value
    if "--no-responsibility" in arguments:
        for name in ("RELATIVE_OVERTAKING_CONFIG", "DEFENSE_CONFIG"):
            setattr(main, name, replace(getattr(main, name), responsibility=None))
        applied["responsibility"] = None
    for name in ("RELATIVE_OVERTAKING_CONFIG", "DEFENSE_CONFIG"):
        config = getattr(main, name)
        applied[f"{name}.responsibility"] = repr(getattr(config, "responsibility", None))
        applied[f"{name}.leader_penalty_weight"] = getattr(config, "leader_penalty_weight", None)
    applied["DEFENSE_CONFIG.collision_horizon"] = main.DEFENSE_CONFIG.collision_horizon
    applied["DEFENSE_CONFIG.decision_period"] = main.DEFENSE_CONFIG.decision_period
    return applied


if __name__ == "__main__":
    options = _apply_options(sys.argv[2:])
    print(f"options: {options}")
    main_entry(Path(sys.argv[1]), options)
