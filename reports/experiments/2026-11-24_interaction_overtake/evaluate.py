"""互動 GP 超車的成對評估：同一批初始條件，比較 relative_overtake、previous、candidate。

python evaluate.py run --strategy candidate     # 各策略可分開、並行執行，結果寫入 runs_<strategy>.json
python evaluate.py summarize                    # 讀取三份結果，輸出 summary.json、per_episode.csv、圖
"""

import argparse
import csv
from collections import Counter
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

import main
from core.track_geometry import signed_longitudinal_gap, track_length
from generate_training_data import _demo_settings
from sim.data_generation import EpisodeScenario, VehicleInitialCondition, run_episode
from strategy.assignment import StrategyAssignment
from strategy.interaction_gp import InteractionGpModel
from training_data import TrainingDataCollector

HERE = Path(__file__).resolve().parent
STRATEGIES = ("relative", "previous", "candidate")
EPISODES, SECONDS, SEED = 20, 20.0, 7
LEAD_MARGIN = 0.6      # 攻方 s 超過前車此距離（碰撞框長）才算領先，m
SPEED_ADVANTAGE = 1.0  # 攻方巡航速度＝前車＋此值，m/s；GP 訓練中後車最多快約 1.3 m/s


def scenarios(track_len: float) -> list[EpisodeScenario]:
    rng = np.random.default_rng(SEED)
    result = []
    for number in range(EPISODES):
        origin, gap, speed = rng.uniform(0, track_len), rng.uniform(2, 5), rng.uniform(3, 6)
        result.append(EpisodeScenario(f"io-{number:03d}", (
            VehicleInitialCondition("rear", float(origin), 0.0, float(speed + SPEED_ADVANTAGE),
                                    float(speed + SPEED_ADVANTAGE)),
            VehicleInitialCondition("front", float(origin + gap), 0.0, float(speed), float(speed)),
        ), {}))
    return result


def settings_for(strategy: str):
    settings, _, _ = _demo_settings(False)
    config = main.INTERACTION_OVERTAKING_CONFIG
    settings = replace(settings, track=main.create_demo_track("rounded_rectangle"),
                       relative_overtaking_config=config.relative)  # 基準與新策略用相同候選與成本
    if strategy == "relative":
        return settings, "relative_overtake"
    model = InteractionGpModel.load(main.INTERACTION_GP_MODEL_PATH)
    return replace(settings, interaction_overtaking_config=replace(config, conditioning=strategy),
                   interaction_model=model), "interaction_overtake"


def run(strategy: str) -> None:
    settings, name = settings_for(strategy)
    track_len = track_length(settings.track)
    dt = settings.model_config.time_step
    results = []
    for base in scenarios(track_len):
        scenario = replace(base, assignments={
            "rear": StrategyAssignment(name), "front": StrategyAssignment("defend", target_vehicle_id="rear")})
        decisions = []

        def observe(step, observation, decision, command):
            snapshot = decision.candidate_snapshot
            # 只記錄真正重新評估的那一步。
            if observation.vehicle_id != "rear" or snapshot is None or abs(snapshot.evaluated_at - step * dt) > 1e-9:
                return
            gp = next((item for item in snapshot.gp_predictions if item.delta_d_mean is not None), None)
            off = next((text[7:-1] for text in decision.details if text.startswith("gp=off(")), None)
            decisions.append({"t": step * dt, "mode": decision.mode, "target_d": decision.target_d,
                              "speed_limit": decision.speed_limit,
                              "gp_mean": None if gp is None else gp.delta_d_mean,
                              "gp_std": None if gp is None else gp.delta_d_std,
                              "gp_d_target": None if gp is None else gp.d_target, "gp_off": off})

        collector = TrainingDataCollector(dt)
        started = time.perf_counter()
        run_episode(scenario, settings, collector, SECONDS, step_observer=observe)
        frames = collector.episodes[0].frames
        results.append({"episode": scenario.episode_id, "wall_seconds": time.perf_counter() - started,
                        "front_speed": base.vehicles[1].target_speed,
                        "initial_gap": base.vehicles[1].start_s - base.vehicles[0].start_s,
                        "relative_gap": relative_gaps(frames, track_len, base.vehicles[1].start_s - base.vehicles[0].start_s),
                        "front_d": [f.observations["front"].track_position.d for f in frames],
                        "collision_steps": sum(1 for f in frames if f.collision_pairs),
                        "decisions": decisions})
        print(f"{strategy} {scenario.episode_id} done in {results[-1]['wall_seconds']:.1f} s", flush=True)
    (HERE / f"runs_{strategy}.json").write_text(json.dumps({
        "strategy": strategy, "episodes": EPISODES, "seconds": SECONDS, "seed": SEED, "time_step": dt,
        "speed_advantage": SPEED_ADVANTAGE, "lead_margin": LEAD_MARGIN,
        "config": asdict(main.INTERACTION_OVERTAKING_CONFIG), "model": main.INTERACTION_GP_MODEL_PATH,
        "results": results}) + "\n", encoding="utf-8")


def relative_gaps(frames, track_len: float, initial_gap: float) -> list[float]:
    """前車 s − 攻方 s，沿途累積進度，跨圈不跳變；負值表示攻方在前。"""
    gaps, value = [initial_gap], initial_gap
    for previous, current in zip(frames, frames[1:]):
        rear = signed_longitudinal_gap(previous.observations["rear"].track_position,
                                       current.observations["rear"].track_position, track_len)
        front = signed_longitudinal_gap(previous.observations["front"].track_position,
                                        current.observations["front"].track_position, track_len)
        value += front - rear
        gaps.append(value)
    return gaps


def episode_metrics(item: dict, dt: float) -> dict:
    gaps = np.array(item["relative_gap"])
    lead = np.flatnonzero(gaps < -LEAD_MARGIN)
    decisions = item["decisions"]
    targets = [d["target_d"] for d in decisions]
    switches = sum(1 for a, b in zip(targets, targets[1:]) if a != b)
    active = [d for d in decisions if d["gp_mean"] is not None]
    front_d = np.array(item["front_d"])
    errors, covered, switch_errors = [], [], []
    previous_target = None
    for d in decisions:
        step = round(d["t"] / dt)
        horizon = step + round(0.5 / dt)
        if d["gp_mean"] is not None and horizon < len(front_d):
            error = d["gp_mean"] - (front_d[horizon] - front_d[step])
            errors.append(error)
            covered.append(abs(error) <= 2 * d["gp_std"])
            if previous_target is not None and d["target_d"] != previous_target:
                switch_errors.append(error)
        previous_target = d["target_d"]
    return {"first_lead_s": None if len(lead) == 0 else float(lead[0] * dt),
            "final_relative_gap": float(gaps[-1]), "min_relative_gap": float(gaps.min()),
            "collision_steps": item["collision_steps"], "decisions": len(decisions),
            "no_feasible": sum(1 for d in decisions if d["mode"] == "no_feasible_path"),
            "target_switches": switches, "gp_active": len(active),
            "gp_off": dict(Counter(d["gp_off"] for d in decisions if d["gp_off"])),
            "gp_errors": errors, "gp_covered": covered, "gp_switch_errors": switch_errors,
            "wall_seconds": item["wall_seconds"]}


def summarize() -> None:
    runs = {name: json.loads((HERE / f"runs_{name}.json").read_text(encoding="utf-8")) for name in STRATEGIES}
    dt = runs["relative"]["time_step"]
    rows, summary = [], {}
    for name, data in runs.items():
        metrics = [episode_metrics(item, dt) for item in data["results"]]
        for item, m in zip(data["results"], metrics):
            rows.append({"strategy": name, "episode": item["episode"], "front_speed": round(item["front_speed"], 3),
                         "initial_gap": round(item["initial_gap"], 3),
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                            if k not in ("gp_errors", "gp_covered", "gp_switch_errors", "gp_off")},
                         "gp_off": json.dumps(m["gp_off"])})
        errors = np.array([e for m in metrics for e in m["gp_errors"]])
        covered = np.array([c for m in metrics for c in m["gp_covered"]])
        switch_errors = np.array([e for m in metrics for e in m["gp_switch_errors"]])
        leads = [m["first_lead_s"] for m in metrics if m["first_lead_s"] is not None]
        summary[name] = {
            "episodes_with_lead": len(leads), "median_first_lead_s": float(np.median(leads)) if leads else None,
            "episodes_ahead_at_end": sum(1 for m in metrics if m["final_relative_gap"] < 0),
            "collision_steps_total": sum(m["collision_steps"] for m in metrics),
            "episodes_with_collision": sum(1 for m in metrics if m["collision_steps"] > 0),
            "no_feasible_total": sum(m["no_feasible"] for m in metrics),
            "target_switches_mean": float(np.mean([m["target_switches"] for m in metrics])),
            "decisions_total": sum(m["decisions"] for m in metrics),
            "gp_active_total": sum(m["gp_active"] for m in metrics),
            "gp_off_reasons": dict(sum((Counter(m["gp_off"]) for m in metrics), Counter())),
            "online_gp": None if len(errors) == 0 else {
                "n": int(len(errors)), "mae": float(np.mean(np.abs(errors))),
                "coverage_2std": float(np.mean(covered)),
                "switch_decisions": int(len(switch_errors)),
                "switch_mae": None if len(switch_errors) == 0 else float(np.mean(np.abs(switch_errors)))},
            "wall_seconds_mean": float(np.mean([m["wall_seconds"] for m in metrics])),
        }
    with (HERE / "per_episode.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    plot(rows)


def plot(rows: list[dict]) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    surface, ink, muted, grid = "#fcfcfb", "#1f1f1d", "#6b6a63", "#e4e3de"
    colors = {"relative": "#2a78d6", "previous": "#eb6834", "candidate": "#1baf7a"}
    markers = {"relative": "o", "previous": "s", "candidate": "^"}
    figure = Figure(figsize=(11, 4), layout="constrained", facecolor=surface)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, key, label in ((axes[0], "final_relative_gap", "Final gap: front s − attacker s (m)"),
                             (axes[1], "gp_active_fraction", "GP active share of decisions")):
        axis.set_facecolor(surface)
        axis.grid(True, color=grid, linewidth=0.8)
        axis.tick_params(colors=muted)
        for spine in axis.spines.values():
            spine.set_color(grid)
        for offset, name in zip((-0.25, 0.0, 0.25), STRATEGIES):
            items = [row for row in rows if row["strategy"] == name]
            x = np.arange(len(items)) + offset
            y = ([row["gp_active"] / max(row["decisions"], 1) for row in items] if key == "gp_active_fraction"
                 else [row[key] for row in items])
            axis.scatter(x, y, s=36, color=colors[name], marker=markers[name], edgecolors=surface,
                         linewidths=1.5, label=name, zorder=3)
        axis.set_xlabel("Initial condition (episode index)", color=ink)
        axis.set_ylabel(label, color=ink)
    axes[0].axhline(0.0, color=muted, linestyle="--", linewidth=1)
    axes[1].set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=ink)
    figure.suptitle("Paired runs: relative_overtake vs interaction_overtake (previous / candidate)",
                    color=ink, fontsize=11)
    figure.savefig(HERE / "paired_results.png", dpi=130, facecolor=surface)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "summarize"))
    parser.add_argument("--strategy", choices=STRATEGIES)
    args = parser.parse_args()
    if args.mode == "run":
        if args.strategy is None:
            parser.error("run requires --strategy")
        run(args.strategy)
    else:
        summarize()
