"""換目標成本 0.3 與 0.5 的差別：記錄攻方每次決策的所有候選成本，找出兩者第一次分歧的原因。

run 模式直接呼叫 main.run_simulation（與 python main.py 相同的組裝），只把畫面換成假物件，並包裝 StrategyManager 記錄決策。
  python why_switch.py run <scene> <weight>   scene：main（main 的預設場景）或 demo（相距 6 m 的示範場景）
  python why_switch.py compare                 比對已存的結果，輸出 why_switch.json
"""

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

from dataclasses import replace  # noqa: E402
import io  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402

import main  # noqa: E402
from core.track_geometry import signed_longitudinal_gap, track_length  # noqa: E402
from sim.simulated_vehicle import VehicleSpec  # noqa: E402

ATTACKER, DEFENDER = "vehicle-1", "vehicle-2"


def run(scene: str, weight: float, overrides: dict[str, float] | None = None) -> None:
    """overrides：額外改 relative 設定，例如 {"progress_weight": 2.0, "safety_weight": 5.0}。"""
    overrides = overrides or {}
    records, track_holder = [], {}
    original = main.StrategyManager

    class Recording(original):
        def update(self, observations, track, time_seconds):
            decisions = super().update(observations, track, time_seconds)
            decision = decisions[ATTACKER]
            snapshot = decision.candidate_snapshot
            track_holder["track"] = track
            if snapshot is not None and abs(snapshot.evaluated_at - time_seconds) < 1e-9:
                rear, front = observations[ATTACKER], observations[DEFENDER]
                records.append({
                    "t": round(time_seconds, 2), "target_d": decision.target_d, "speed_limit": decision.speed_limit,
                    "mode": decision.mode, "details": list(decision.details),
                    "rear_s": rear.track_position.s, "rear_d": rear.track_position.d, "rear_v": rear.state.velocity,
                    "front_s": front.track_position.s, "front_d": front.track_position.d,
                    "front_v": front.state.velocity,
                    "candidates": [{"target_d": c.maneuver.target_d, "speed_limit": c.speed_limit,
                                    "feasible": c.feasible, "reason": c.reason,
                                    "terms": dict(c.cost_terms)} for c in snapshot.candidates],
                })
            return decisions

    paper = replace(main.PAPER_ROLLOUT_OVERTAKING_CONFIG,
                    relative=replace(main.PAPER_ROLLOUT_OVERTAKING_CONFIG.relative, switch_weight=weight,
                                     **overrides))
    collision_steps = []
    detect = main.find_vehicle_collisions

    def counting(vehicles):
        pairs = detect(vehicles)
        collision_steps.append(bool(pairs))
        return pairs

    patches = {"PAPER_ROLLOUT_OVERTAKING_CONFIG": paper, "StrategyManager": Recording,
               "find_vehicle_collisions": counting}
    if scene == "demo":
        patches.update(SIMULATION_SECONDS=20.0, VEHICLE_SPECS=(
            VehicleSpec(ATTACKER, start_s=0.0, target_speed=7.0), VehicleSpec(DEFENDER, start_s=6.0, target_speed=6.0)))
    output = io.StringIO()
    with patch.multiple(main, **patches), patch("main.SimulationVisualizer"), redirect_stdout(output):
        main.run_simulation()
    track = track_holder["track"]
    length, gap = track_length(track), None
    previous = None
    for record in records:
        # 前車 s − 攻方 s，以決策間的進度累加，跨圈不跳變。
        if previous is None:
            gap = signed_longitudinal_gap(_pos(record, "rear"), _pos(record, "front"), length)
        else:
            gap += (_delta(previous, record, "front", length) - _delta(previous, record, "rear", length))
        record["gap"] = gap
        previous = record
    summary_lines = [line for line in output.getvalue().splitlines() if "collision" in line.lower()]
    suffix = "".join(f"_{key.split('_')[0]}{value:g}" for key, value in sorted(overrides.items()))
    path = HERE / f"why_{scene}_{weight:g}{suffix}.json"
    path.write_text(json.dumps({"scene": scene, "weight": weight, "ellipse_sigma": paper.ellipse_sigma,
                                "progress_weight": paper.relative.progress_weight,
                                "safety_weight": paper.relative.safety_weight,
                                "stdout_collision": summary_lines, "collision_steps": sum(collision_steps),
                                "decisions": records}) + "\n",
                    encoding="utf-8")
    first_lead = next((r["t"] for r in records if r["gap"] < -0.6), None)
    targets = [r["target_d"] for r in records]
    print(f"{scene} weight={weight:g} {overrides} progress={paper.relative.progress_weight:g}: first lead {first_lead}, switches "
          f"{sum(a != b for a, b in zip(targets, targets[1:]))}, collision steps {sum(collision_steps)}, "
          f"{summary_lines}")


class _Pos:
    def __init__(self, s):
        self.s = s


def _pos(record, who):
    return _Pos(record[f"{who}_s"])


def _delta(previous, record, who, length):
    return signed_longitudinal_gap(_pos(previous, who), _pos(record, who), length)


def total(candidate) -> float:
    return sum(candidate["terms"].values())


def explain(decision, previous_target) -> dict:
    """該次決策：最佳候選、維持原目標的最佳候選，以及兩者不含換目標成本的差。"""
    feasible = [c for c in decision["candidates"] if c["feasible"]]
    best = min(feasible, key=total)
    kept = [c for c in feasible if c["target_d"] == previous_target]
    kept_best = min(kept, key=total) if kept else None
    switch_best = min((c for c in feasible if c["target_d"] != previous_target), key=total, default=None)

    def raw(c):
        return None if c is None else total(c) - c["terms"].get("switch", 0.0)

    return {"t": decision["t"], "gap": round(decision["gap"], 3), "rear_d": round(decision["rear_d"], 3),
            "front_d": round(decision["front_d"], 3), "previous_target": previous_target,
            "chosen": {"target_d": best["target_d"], "speed_limit": best["speed_limit"],
                       "terms": {k: round(v, 3) for k, v in best["terms"].items()}},
            "best_kept_without_switch_cost": None if kept_best is None else {
                "speed_limit": kept_best["speed_limit"], "cost": round(raw(kept_best), 3),
                "terms": {k: round(v, 3) for k, v in kept_best["terms"].items()}},
            "best_other_without_switch_cost": None if switch_best is None else {
                "target_d": switch_best["target_d"], "speed_limit": switch_best["speed_limit"],
                "cost": round(raw(switch_best), 3),
                "terms": {k: round(v, 3) for k, v in switch_best["terms"].items()}},
            "gain_of_switching": None if kept_best is None or switch_best is None
            else round(raw(kept_best) - raw(switch_best), 3)}


def compare() -> None:
    report = {}
    runs = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(HERE.glob("why_*_*.json"))}
    for name, data in runs.items():
        decisions = data["decisions"]
        targets = [d["target_d"] for d in decisions]
        lead = next((d for d in decisions if d["gap"] < -0.6), None)
        report[name] = {"first_lead_s": None if lead is None else lead["t"],
                        "switches": sum(a != b for a, b in zip(targets, targets[1:])),
                        "final_gap_m": round(decisions[-1]["gap"], 3), "collisions": data["stdout_collision"],
                        "switch_times": [d["t"] for d, p in zip(decisions[1:], decisions) if d["target_d"] != p["target_d"]]}
    # 同場景、不同權重：第一次分歧的決策與當時的成本。
    for scene in ("main", "demo"):
        keys = sorted(name for name in runs if name.startswith(f"why_{scene}_"))
        for a in keys:
            for b in keys:
                if a >= b:
                    continue
                da, db = runs[a]["decisions"], runs[b]["decisions"]
                index = next((i for i, (x, y) in enumerate(zip(da, db))
                              if (x["target_d"], x["speed_limit"]) != (y["target_d"], y["speed_limit"])), None)
                if index is None:
                    continue
                previous = da[index - 1]["target_d"] if index > 0 else 0.0
                report[f"{a}_vs_{b}"] = {"first_difference_s": da[index]["t"],
                                         a: explain(da[index], previous), b: explain(db[index], previous)}
    (HERE / "why_switch.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        # 例：run main 0.3 progress_weight=2 safety_weight=5
        extra = {key: float(value) for key, value in (item.split("=") for item in sys.argv[4:])}
        run(sys.argv[2], float(sys.argv[3]), extra)
    else:
        compare()
