"""選 switch_weight 的依據：不加換目標成本時，每次換目標比「維持原目標的最佳候選」省下多少成本。

沿用 paper_rollout_overtake 示範的場景與工具，只跑 k=2、k=0 各前 10 s。
執行：python reports/experiments/2026-11-25_paper_switch_cost/switch_gain.py
"""

import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location(
    "paper_demo", PROJECT_ROOT / "reports/experiments/2026-11-25_paper_rollout_overtake/demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

from sim.data_generation import EpisodeScenario, VehicleInitialCondition, run_episode  # noqa: E402
from strategy.assignment import StrategyAssignment  # noqa: E402
from training_data import TrainingDataCollector  # noqa: E402

SECONDS = 10.0


def gains(settings) -> list[float]:
    rear, front = demo.main.INTERACTION_VEHICLE_SPECS
    scenario = EpisodeScenario("switch-gain", (
        VehicleInitialCondition("rear", rear.start_s, 0.0, 0.0, rear.target_speed),
        VehicleInitialCondition("front", front.start_s, 0.0, 0.0, front.target_speed),
    ), {"rear": StrategyAssignment("paper_rollout_overtake"),
        "front": StrategyAssignment("defend", target_vehicle_id="rear")})
    dt, previous, out = settings.model_config.time_step, [0.0], []

    def observe(step, observation, decision, command):
        snapshot = decision.candidate_snapshot
        if observation.vehicle_id != "rear" or snapshot is None or abs(snapshot.evaluated_at - step * dt) > 1e-9:
            return
        costs = [(c.maneuver.target_d, sum(value for _, value in c.cost_terms))
                 for c in snapshot.candidates if c.feasible and c.cost_terms]
        if decision.target_d != previous[0] and costs:
            best = min(cost for _, cost in costs)
            kept = [cost for target, cost in costs if target == previous[0]]
            # 原目標全部不可行時記為無限大（任何權重都擋不住這次換目標）。
            out.append(float(min(kept) - best) if kept else float("inf"))
        previous[0] = decision.target_d

    run_episode(scenario, settings, TrainingDataCollector(dt), SECONDS, step_observer=observe)
    return out


def main() -> None:
    settings, _ = demo.build_settings()
    base = replace(demo.main.PAPER_ROLLOUT_OVERTAKING_CONFIG,
                   relative=replace(demo.main.RELATIVE_OVERTAKING_CONFIG, switch_weight=0.0))
    report = {}
    for k in (2.0, 0.0):
        values = gains(replace(settings, paper_rollout_overtaking_config=replace(base, ellipse_sigma=k)))
        finite = np.array([v for v in values if np.isfinite(v)])
        report[f"k={k:g}"] = {
            "switches": len(values), "original_target_infeasible": len(values) - len(finite),
            "gain_percentiles_m": {str(p): float(np.percentile(finite, p)) for p in (25, 50, 75, 90)},
            "gain_max_m": float(finite.max()),
        }
        print(f"k={k:g}: {report[f'k={k:g}']}", flush=True)
    (HERE / "switch_gain.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
