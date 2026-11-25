"""rollout_overtake 單場示範：main 的 --interaction-overtake 場景，並以另外兩個策略在同一場作參考。

只代表單一場景，不代表成功率。輸出 summary.json、demo.log（由命令列導出）與圖。
"""

import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

from dataclasses import replace

import numpy as np

import main
from core.track_geometry import signed_longitudinal_gap, track_length, world_to_track
from generate_training_data import _demo_settings
from sim.data_generation import EpisodeScenario, VehicleInitialCondition, run_episode
from strategy.assignment import StrategyAssignment
from strategy.interaction_gp import InteractionGpModel
from training_data import TrainingDataCollector

HERE = Path(__file__).resolve().parent
STRATEGIES = ("rollout_overtake", "interaction_overtake", "relative_overtake")
SECONDS, LEAD_MARGIN = 20.0, 0.6
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1d", "#6b6a63", "#e4e3de"
COLORS = {"rollout_overtake": "#1baf7a", "interaction_overtake": "#eb6834", "relative_overtake": "#2a78d6"}


def build_settings():
    settings, _, _ = _demo_settings(False)
    model = InteractionGpModel.load(main.INTERACTION_GP_MODEL_PATH)
    # 三個策略用相同候選（7 個訓練過的目標）、橫移距離與成本。
    return replace(settings, track=main.create_demo_track("rounded_rectangle"),
                   relative_overtaking_config=main.ROLLOUT_OVERTAKING_CONFIG.relative,
                   interaction_overtaking_config=main.INTERACTION_OVERTAKING_CONFIG,
                   rollout_overtaking_config=main.ROLLOUT_OVERTAKING_CONFIG, interaction_model=model)


def run(name: str, settings) -> dict:
    rear, front = main.INTERACTION_VEHICLE_SPECS
    scenario = EpisodeScenario(f"demo-{name}", (
        VehicleInitialCondition("rear", rear.start_s, 0.0, 0.0, rear.target_speed),
        VehicleInitialCondition("front", front.start_s, 0.0, 0.0, front.target_speed),
    ), {"rear": StrategyAssignment(name), "front": StrategyAssignment("defend", target_vehicle_id="rear")})
    dt = settings.model_config.time_step
    decisions = []

    def observe(step, observation, decision, command):
        snapshot = decision.candidate_snapshot
        if observation.vehicle_id != "rear" or snapshot is None or abs(snapshot.evaluated_at - step * dt) > 1e-9:
            return
        selected = next((c for c in snapshot.candidates if c.selected), None)
        gp = next((item for item in snapshot.gp_predictions if item.delta_d_mean is not None), None)
        decisions.append({"t": step * dt, "target_d": decision.target_d, "mode": decision.mode,
                          "details": decision.details,
                          "selected_states": None if selected is None else selected.predicted_states,
                          "gp_states": None if gp is None else gp.states})

    collector = TrainingDataCollector(dt)
    started = time.perf_counter()
    run_episode(scenario, settings, collector, SECONDS, step_observer=observe)
    return {"frames": collector.episodes[0].frames, "decisions": decisions,
            "wall_seconds": time.perf_counter() - started, "dt": dt,
            "initial_gap": front.start_s - rear.start_s}


def relative_gaps(frames, track, initial_gap):
    length, gaps, value = track_length(track), [initial_gap], initial_gap
    for previous, current in zip(frames, frames[1:]):
        value += (signed_longitudinal_gap(previous.observations["front"].track_position,
                                          current.observations["front"].track_position, length)
                  - signed_longitudinal_gap(previous.observations["rear"].track_position,
                                            current.observations["rear"].track_position, length))
        gaps.append(value)
    return np.array(gaps)


def metrics(result, track) -> dict:
    frames, dt, decisions = result["frames"], result["dt"], result["decisions"]
    gaps = relative_gaps(frames, track, result["initial_gap"])
    lead = np.flatnonzero(gaps < -LEAD_MARGIN)
    targets = [d["target_d"] for d in decisions]
    front_d = np.array([f.observations["front"].track_position.d for f in frames])
    errors = {0.5: [], 1.0: []}
    for d in decisions:
        if d["gp_states"] is None:
            continue
        step = round(d["t"] / dt)
        for horizon in errors:
            index, future = round(horizon / dt), step + round(horizon / dt)
            if future < len(front_d) and index < len(d["gp_states"]):
                predicted = world_to_track(d["gp_states"][index], track).d
                errors[horizon].append(predicted - front_d[future])
    active = sum(1 for d in decisions if d["gp_states"] is not None)
    return {
        "first_lead_s": None if len(lead) == 0 else float(lead[0] * dt),
        "final_gap_m": float(gaps[-1]), "min_gap_m": float(gaps.min()),
        "collision_steps": sum(1 for f in frames if f.collision_pairs),
        "decisions": len(decisions), "target_switches": sum(1 for a, b in zip(targets, targets[1:]) if a != b),
        "gp_active_share": active / max(len(decisions), 1),
        "front_prediction_mae_m": {f"{h:g}s": (None if not e else float(np.mean(np.abs(e))))
                                   for h, e in errors.items()},
        "front_prediction_n": {f"{h:g}s": len(e) for h, e in errors.items()},
        "episode_wall_seconds": result["wall_seconds"],
        "wall_ms_per_rear_decision": 1000.0 * result["wall_seconds"] / max(len(decisions), 1),
    }


def style(axis):
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def plot_snapshots(result, track) -> list[float]:
    """數個決策時刻：上一步軌跡（自車）、推演的對手軌跡、實際對手，橫軸為相對現在的時間。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    frames, dt, decisions = result["frames"], result["dt"], result["decisions"]
    front_d = np.array([f.observations["front"].track_position.d for f in frames])
    candidates = []
    for previous, current in zip(decisions, decisions[1:]):
        step = round(current["t"] / dt)
        if current["gp_states"] is None or previous["selected_states"] is None or step + round(1.0 / dt) >= len(front_d):
            continue
        motion = abs(front_d[step + round(1.0 / dt)] - front_d[step])
        candidates.append((motion, previous, current))
    chosen = []
    for motion, previous, current in sorted(candidates, key=lambda item: -item[0]):
        if all(abs(current["t"] - other[2]["t"]) >= 1.0 for other in chosen):
            chosen.append((motion, previous, current))
        if len(chosen) == 4:
            break
    chosen.sort(key=lambda item: item[2]["t"])
    figure = Figure(figsize=(11, 7), layout="constrained", facecolor=SURFACE)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharey=True).ravel()
    for axis, (_, previous, current) in zip(axes, chosen):
        style(axis)
        step = round(current["t"] / dt)
        lag = current["t"] - previous["t"]
        plan_d = [world_to_track(state, track).d for state in previous["selected_states"]]
        plan_t = np.arange(len(plan_d)) * dt - lag
        rollout_d = [world_to_track(state, track).d for state in current["gp_states"]]
        rollout_t = np.arange(len(rollout_d)) * dt
        actual_t = np.arange(0, round(1.0 / dt) + 1) * dt
        actual = front_d[step:step + len(actual_t)]
        axis.plot(plan_t, plan_d, color=INK, linewidth=2, label="attacker: previous-step trajectory")
        axis.plot(rollout_t, rollout_d, color=COLORS["rollout_overtake"], linewidth=2, label="defender: GP rollout")
        axis.plot(actual_t, actual, color="#6250d6", linewidth=2, linestyle="--", label="defender: actual")
        axis.axvline(0.0, color=MUTED, linewidth=1)
        axis.set_title(f"t = {current['t']:.2f} s, attacker target {current['target_d']:+.2f} m", color=INK, fontsize=10)
        axis.set_xlabel("Time from now (s)", color=INK)
        axis.set_ylabel("Lateral offset d (m)", color=INK)
    axes[0].legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    figure.suptitle("rollout_overtake: previous-step trajectory, GP rollout of the defender, and what the defender did",
                    color=INK, fontsize=11)
    figure.savefig(HERE / "rollout_snapshots.png", dpi=130, facecolor=SURFACE)
    return [item[2]["t"] for item in chosen]


def plot_timeseries(results, track) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(11, 6), layout="constrained", facecolor=SURFACE)
    FigureCanvasAgg(figure)
    top, bottom = figure.subplots(2, 1, sharex=True)
    for axis in (top, bottom):
        style(axis)
    for name in STRATEGIES:
        result = results[name]
        gaps = relative_gaps(result["frames"], track, result["initial_gap"])
        top.plot(np.arange(len(gaps)) * result["dt"], gaps, color=COLORS[name], linewidth=2, label=name)
    top.axhline(0.0, color=MUTED, linestyle="--", linewidth=1)
    top.set_ylabel("Gap: defender s − attacker s (m)", color=INK)
    top.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    result = results["rollout_overtake"]
    frames, dt = result["frames"], result["dt"]
    t = np.arange(len(frames)) * dt
    bottom.plot(t, [f.observations["rear"].track_position.d for f in frames], color=INK, linewidth=2, label="attacker d")
    bottom.plot(t, [f.observations["front"].track_position.d for f in frames], color="#6250d6", linewidth=2,
                label="defender d")
    bottom.step([d["t"] for d in result["decisions"]], [d["target_d"] for d in result["decisions"]], where="post",
                color=MUTED, linestyle="--", linewidth=1.2, label="attacker target d")
    bottom.set_ylabel("rollout_overtake: d (m)", color=INK)
    bottom.set_xlabel("Time (s)", color=INK)
    bottom.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8, ncols=3, labelcolor=INK)
    figure.savefig(HERE / "timeseries.png", dpi=130, facecolor=SURFACE)


def main_demo() -> None:
    settings = build_settings()
    results = {}
    for name in STRATEGIES:
        results[name] = run(name, settings)
        print(f"{name}: {len(results[name]['decisions'])} decisions in {results[name]['wall_seconds']:.1f} s", flush=True)
    summary = {name: metrics(result, settings.track) for name, result in results.items()}
    rollout = results["rollout_overtake"]["decisions"]
    reasons = {}
    for d in rollout:
        for text in d["details"]:
            if text.startswith("gp=off("):
                reasons[text] = reasons.get(text, 0) + 1
            if " hold@" in text:
                key = "hold(" + text.split("hold@")[1].split("(", 1)[1]
                reasons[key] = reasons.get(key, 0) + 1
    summary["rollout_overtake"]["gp_off_and_hold_reasons"] = reasons
    summary["scene"] = {"vehicles": [list(spec) for spec in main.INTERACTION_VEHICLE_SPECS], "seconds": SECONDS,
                        "model": main.INTERACTION_GP_MODEL_PATH,
                        "rollout_step": main.ROLLOUT_OVERTAKING_CONFIG.rollout_step,
                        "band_sigma": main.ROLLOUT_OVERTAKING_CONFIG.band_sigma,
                        "deviation_weight": main.ROLLOUT_OVERTAKING_CONFIG.relative.deviation_weight,
                        "decision_period": main.ROLLOUT_OVERTAKING_CONFIG.relative.decision_period}
    summary["snapshot_times_s"] = plot_snapshots(results["rollout_overtake"], settings.track)
    plot_timeseries(results, settings.track)
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_demo()
