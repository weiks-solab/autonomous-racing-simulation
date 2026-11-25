"""paper_rollout_overtake 加上最單純的換目標成本（目標 d 不同就加 0.5 m）後，重跑同一個單場示範。

場景、指標與圖沿用 ../2026-11-25_paper_rollout_overtake/demo.py；只跑 k=2、k=0 兩個設定，
並與該示範中未加換目標成本的結果並列。
執行：python reports/experiments/2026-11-25_paper_switch_cost/demo_switch.py
"""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
BEFORE = PROJECT_ROOT / "reports/experiments/2026-11-25_paper_rollout_overtake"
spec = importlib.util.spec_from_file_location("paper_demo", BEFORE / "demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

SWITCH_WEIGHT = 0.5
KEYS = ("first_lead_s", "final_gap_m", "min_gap_m", "collision_steps", "target_switches",
        "no_feasible_path_decisions", "gp_active_share", "feasible_candidates_before_lead", "decision_ms")


def main() -> None:
    demo.HERE = HERE   # 圖輸出到本目錄
    demo.RUNS = {
        "paper_k2": ("paper_rollout_overtake", f"paper_rollout_overtake, k=2, switch {SWITCH_WEIGHT:g}", "#eb6834"),
        "paper_k0": ("paper_rollout_overtake", f"paper_rollout_overtake, k=0, switch {SWITCH_WEIGHT:g}", "#eda100"),
    }
    settings, load_seconds = demo.build_settings()
    relative = replace(demo.main.RELATIVE_OVERTAKING_CONFIG, switch_weight=SWITCH_WEIGHT)
    results = {}
    for name, k in (("paper_k2", 2.0), ("paper_k0", 0.0)):
        config = replace(demo.main.PAPER_ROLLOUT_OVERTAKING_CONFIG, relative=relative, ellipse_sigma=k, seed=0)
        results[name] = demo.run(name, replace(settings, paper_rollout_overtaking_config=config))
        print(f"{name}: {len(results[name]['decisions'])} decisions in {results[name]['wall_seconds']:.1f} s",
              flush=True)
    after = {name: demo.metrics(name, result, settings.track) for name, result in results.items()}
    before = json.loads((BEFORE / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "switch_weight": SWITCH_WEIGHT,
        "with_switch_cost": after,
        "without_switch_cost": {name: {key: before[name][key] for key in KEYS}
                                for name in ("paper_k2", "paper_k0", "relative_overtake", "rollout_overtake")},
        "without_switch_cost_prediction": {name: before[name].get("prediction_gp_active")
                                           for name in ("paper_k2", "paper_k0")},
        "scene": {**before["scene"], "paper_model_load_seconds": load_seconds, "ellipse_sigma": [2.0, 0.0],
                  "switch_weight": SWITCH_WEIGHT},
    }
    summary["snapshot_times_s"] = demo.plot_snapshots(results["paper_k2"], settings.track)
    demo.plot_timeseries(results, settings.track)
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
