"""paper_rollout_overtake 單場示範：main 的 --interaction-overtake 場景，與 k=0、rollout_overtake、relative_overtake 在同一場比較。

只代表單一場景，不代表成功率。輸出 summary.json、兩張圖；終端機輸出導到 demo.log。
策略物件經工廠包裝取得，只讀取它的推演結果與計時，不改變決策。
"""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

from dataclasses import replace  # noqa: E402
from unittest.mock import patch  # noqa: E402

import numpy as np  # noqa: E402

import main  # noqa: E402
import sim.data_generation as data_generation  # noqa: E402
from core.track_geometry import signed_longitudinal_gap, track_length, world_to_track  # noqa: E402
from generate_training_data import _demo_settings  # noqa: E402
from sim.data_generation import EpisodeScenario, VehicleInitialCondition, run_episode  # noqa: E402
from strategy.assignment import StrategyAssignment  # noqa: E402
from strategy.interaction_gp import InteractionGpModel  # noqa: E402
from strategy.paper_gp import PaperGpModel  # noqa: E402
from training_data import TrainingDataCollector  # noqa: E402

HERE = Path(__file__).resolve().parent
SECONDS, LEAD_MARGIN, HORIZONS, ERROR_FROM = 20.0, 0.6, (0.5, 1.0), 0.5
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1d", "#6b6a63", "#e4e3de"
DEFENDER = "#6250d6"   # 類別色第 7 格（violet）：防守車實際軌跡
# 名稱：(策略, 圖例, 類別色)。依固定順序取第 1～4 格；relative_overtake 與 rollout_overtake 沿用上一份示範的顏色。
RUNS = {
    "relative_overtake": ("relative_overtake", "relative_overtake (constant speed)", "#2a78d6"),
    "paper_k2": ("paper_rollout_overtake", "paper_rollout_overtake, k=2", "#eb6834"),
    "rollout_overtake": ("rollout_overtake", "rollout_overtake (v2)", "#1baf7a"),
    "paper_k0": ("paper_rollout_overtake", "paper_rollout_overtake, k=0 (mean only)", "#eda100"),
}


def build_settings():
    settings, _, _ = _demo_settings(False)   # relative_overtake 與 defend 沿用 main 的設定
    started = time.perf_counter()
    paper = PaperGpModel.load(main.PAPER_GP_MODEL_PATH)
    load_seconds = time.perf_counter() - started
    settings = replace(settings, track=main.create_demo_track("rounded_rectangle"),
                       rollout_overtaking_config=main.ROLLOUT_OVERTAKING_CONFIG,
                       interaction_model=InteractionGpModel.load(main.INTERACTION_GP_MODEL_PATH),
                       paper_rollout_overtaking_config=main.PAPER_ROLLOUT_OVERTAKING_CONFIG, paper_model=paper)
    return settings, load_seconds


def settings_for(name: str, settings):
    if name == "paper_k0":
        return replace(settings, paper_rollout_overtaking_config=replace(
            settings.paper_rollout_overtaking_config, ellipse_sigma=0.0))
    return settings


def run(name: str, settings) -> dict:
    strategy_name = RUNS[name][0]
    rear, front = main.INTERACTION_VEHICLE_SPECS
    scenario = EpisodeScenario(f"demo-{name}", (
        VehicleInitialCondition("rear", rear.start_s, 0.0, 0.0, rear.target_speed),
        VehicleInitialCondition("front", front.start_s, 0.0, 0.0, front.target_speed),
    ), {"rear": StrategyAssignment(strategy_name), "front": StrategyAssignment("defend", target_vehicle_id="rear")})
    dt = settings.model_config.time_step
    created, durations, decisions = {}, [], []
    original = data_generation.create_strategy_factories

    def capturing(*args, **kwargs):
        def wrap(factory):
            def build(vehicle_id):
                strategy = factory(vehicle_id)
                if vehicle_id == "rear":
                    created["rear"], update = strategy, strategy.update

                    def timed(*call_args, **call_kwargs):
                        started = time.perf_counter()
                        try:
                            return update(*call_args, **call_kwargs)
                        finally:
                            durations.append(time.perf_counter() - started)
                    strategy.update = timed
                return strategy
            return build
        return {key: wrap(factory) for key, factory in original(*args, **kwargs).items()}

    def observe(step, observation, decision, command):
        snapshot = decision.candidate_snapshot
        if observation.vehicle_id != "rear" or snapshot is None or abs(snapshot.evaluated_at - step * dt) > 1e-9:
            return
        strategy = created["rear"]
        selected = next((c for c in snapshot.candidates if c.selected), None)
        prediction = strategy.last_target_prediction
        rollout = strategy.last_rollout if strategy_name == "paper_rollout_overtake" else None
        if strategy_name == "paper_rollout_overtake":
            active = rollout is not None
        elif strategy_name == "rollout_overtake":
            active = strategy.last_gp_query is not None and strategy.last_gp_query.active
        else:
            active = False
        decisions.append({
            "t": step * dt, "step": step, "target_d": decision.target_d, "mode": decision.mode,
            "speed_limit": decision.speed_limit,
            "feasible": sum(1 for candidate in snapshot.candidates if candidate.feasible),
            "details": decision.details, "seconds": durations[-1], "gp_active": active, "rollout": rollout,
            "reason": getattr(strategy, "last_reason", ""),
            "selected_states": None if selected is None else selected.predicted_states,
            "prediction": None if prediction is None else (
                np.array(prediction.times), np.array(prediction.progress), np.array(prediction.offsets)),
        })

    collector = TrainingDataCollector(dt)
    started = time.perf_counter()
    with patch.object(data_generation, "create_strategy_factories", capturing):
        run_episode(scenario, settings, collector, SECONDS, step_observer=observe)
    return {"frames": collector.episodes[0].frames, "decisions": decisions, "dt": dt,
            "wall_seconds": time.perf_counter() - started, "initial_gap": front.start_s - rear.start_s}


def accumulate(frames, track, vehicle_id) -> np.ndarray:
    """相對第 0 幀的累積縱向進度，跨起終點不跳變。"""
    length, values = track_length(track), [0.0]
    for previous, current in zip(frames, frames[1:]):
        values.append(values[-1] + signed_longitudinal_gap(previous.observations[vehicle_id].track_position,
                                                           current.observations[vehicle_id].track_position, length))
    return np.array(values)


def relative_gaps(result, track) -> np.ndarray:
    frames = result["frames"]
    return result["initial_gap"] + accumulate(frames, track, "front") - accumulate(frames, track, "rear")


def prediction_errors(result, track, only_gp: bool) -> dict:
    """決策時的前車預測與模擬中實際 h 秒後的位置比較；只用 t ≥ ERROR_FROM 的決策。"""
    frames, dt = result["frames"], result["dt"]
    progress, front_d = accumulate(frames, track, "front"), np.array(
        [f.observations["front"].track_position.d for f in frames])
    out = {}
    for horizon in HORIZONS:
        s_err, d_err, s_cover, d_cover = [], [], [], []
        for item in result["decisions"]:
            future = item["step"] + round(horizon / dt)
            if item["t"] < ERROR_FROM - 1e-9 or future >= len(frames) or item["prediction"] is None:
                continue
            if only_gp and not item["gp_active"]:
                continue
            times, predicted_progress, predicted_d = item["prediction"]
            actual_s = progress[future] - progress[item["step"]]
            s_error = float(np.interp(horizon, times, predicted_progress) - actual_s)
            d_error = float(np.interp(horizon, times, predicted_d) - front_d[future])
            s_err.append(s_error)
            d_err.append(d_error)
            rollout = item["rollout"]
            if rollout is not None:
                nodes = np.asarray(rollout.node_times)
                s_cover.append(abs(s_error) <= 2.0 * float(np.interp(horizon, nodes, rollout.std_s)))
                d_cover.append(abs(d_error) <= 2.0 * float(np.interp(horizon, nodes, rollout.std_d)))
        key = f"{horizon:g}s"
        out[key] = {"n": len(s_err),
                    "s_mae_m": None if not s_err else float(np.mean(np.abs(s_err))),
                    "s_bias_m": None if not s_err else float(np.mean(s_err)),
                    "d_mae_m": None if not d_err else float(np.mean(np.abs(d_err)))}
        if s_cover:
            out[key].update({"s_coverage_2std": float(np.mean(s_cover)), "d_coverage_2std": float(np.mean(d_cover))})
    return out


def collision_intervals(result, track) -> list[dict]:
    """連續有碰撞矩形重疊的時段：起訖時間，以及時段內車距最小時兩車的 d。"""
    frames, dt = result["frames"], result["dt"]
    gaps = relative_gaps(result, track)
    hits = [index for index, frame in enumerate(frames) if frame.collision_pairs]
    intervals, start = [], None
    for index, step in enumerate(hits):
        start = step if start is None else start
        if index + 1 == len(hits) or hits[index + 1] != step + 1:
            closest = start + int(np.argmin(np.abs(gaps[start:step + 1])))
            observations = frames[closest].observations
            intervals.append({"start_s": round(start * dt, 2), "end_s": round(step * dt, 2),
                              "steps": step - start + 1, "gap_m": round(float(gaps[closest]), 3),
                              "rear_d": round(observations["rear"].track_position.d, 3),
                              "front_d": round(observations["front"].track_position.d, 3)})
            start = None
    return intervals


def metrics(name: str, result, track) -> dict:
    frames, decisions = result["frames"], result["decisions"]
    gaps = relative_gaps(result, track)
    lead = np.flatnonzero(gaps < -LEAD_MARGIN)
    targets = [item["target_d"] for item in decisions]
    seconds = np.array([item["seconds"] for item in decisions])
    first_lead = None if len(lead) == 0 else float(lead[0] * result["dt"])
    before = [item["feasible"] for item in decisions if first_lead is None or item["t"] < first_lead]
    summary = {
        "strategy": RUNS[name][0],
        "first_lead_s": first_lead,
        "final_gap_m": float(gaps[-1]), "min_gap_m": float(gaps.min()),
        "collision_steps": sum(1 for f in frames if f.collision_pairs),
        "collision_intervals": collision_intervals(result, track),
        "decisions": len(decisions),
        "target_switches": sum(1 for a, b in zip(targets, targets[1:]) if a != b),
        "no_feasible_path_decisions": sum(1 for item in decisions if item["mode"] == "no_feasible_path"),
        # 未領先前（從未領先則為整場）每次決策的可行候選數；候選共 7 個橫向 × 3～4 種速度。
        "feasible_candidates_before_lead": {"mean": float(np.mean(before)), "min": int(np.min(before))},
        "gp_active_share": sum(item["gp_active"] for item in decisions) / max(len(decisions), 1),
        "prediction_all_decisions": prediction_errors(result, track, only_gp=False),
        "episode_wall_seconds": result["wall_seconds"],
        "decision_ms": {"mean": float(1000 * seconds.mean()), "median": float(1000 * np.median(seconds)),
                        "max": float(1000 * seconds.max())},
    }
    if RUNS[name][0] != "relative_overtake":
        summary["prediction_gp_active"] = prediction_errors(result, track, only_gp=True)
    if RUNS[name][0] == "paper_rollout_overtake":
        active = [item for item in decisions if item["rollout"] is not None]
        held = [item["rollout"].held_count for item in active]
        summary["paper"] = {
            "fallback_reasons": dict(Counter(item["reason"] for item in decisions if item["rollout"] is None)),
            "decisions_with_constant_speed_samples": sum(1 for count in held if count),
            "constant_speed_sample_share": float(np.sum(held) / max(len(held) * main.PAPER_ROLLOUT_OVERTAKING_CONFIG.samples, 1)),
            "exit_reasons": dict(Counter(reason for item in active for reason in item["rollout"].exit_reasons if reason)),
            "std_s_end_m": {"median": float(np.median([item["rollout"].std_s[-1] for item in active])),
                            "max": float(np.max([item["rollout"].std_s[-1] for item in active]))},
            "std_d_end_m": {"median": float(np.median([item["rollout"].std_d[-1] for item in active])),
                            "max": float(np.max([item["rollout"].std_d[-1] for item in active]))},
            "active_decision_ms_mean": float(1000 * np.mean([item["seconds"] for item in active])),
        }
    return summary


def style(axis):
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def plot_snapshots(result, track) -> list[float]:
    """4 個決策時刻：上一步計畫（自車）、取樣推演的樣本、平均 ±2σ、防守車實際；上排橫向、下排縱向。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    frames, dt, decisions = result["frames"], result["dt"], result["decisions"]
    progress = accumulate(frames, track, "front")
    front_d = np.array([f.observations["front"].track_position.d for f in frames])
    horizon_steps = round(1.0 / dt)
    candidates = []
    for previous, current in zip(decisions, decisions[1:]):
        step = current["step"]
        if current["rollout"] is None or previous["selected_states"] is None or step + horizon_steps >= len(frames):
            continue
        candidates.append((abs(front_d[step + horizon_steps] - front_d[step]), previous, current))
    chosen = []
    for motion, previous, current in sorted(candidates, key=lambda item: -item[0]):
        if all(abs(current["t"] - other[2]["t"]) >= 1.0 for other in chosen):
            chosen.append((motion, previous, current))
        if len(chosen) == 4:
            break
    chosen.sort(key=lambda item: item[2]["t"])
    color = RUNS["paper_k2"][2]
    figure = Figure(figsize=(14, 6.6), layout="constrained", facecolor=SURFACE)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, len(chosen), sharey="row", squeeze=False)
    for column, (_, previous, current) in enumerate(chosen):
        top, bottom = axes[0, column], axes[1, column]
        for axis in (top, bottom):
            style(axis)
            axis.axvline(0.0, color=MUTED, linewidth=1)
        step, rollout = current["step"], current["rollout"]
        nodes = np.asarray(rollout.node_times)
        actual_t = np.arange(horizon_steps + 1) * dt
        speed = frames[step].observations["front"].state.velocity
        lag = current["t"] - previous["t"]
        plan_d = [world_to_track(state, track).d for state in previous["selected_states"]]
        top.plot(np.arange(len(plan_d)) * dt - lag, plan_d, color=INK, linewidth=2, label="attacker: previous plan")
        for row in rollout.offsets:
            top.plot(nodes, row, color=color, linewidth=0.8, alpha=0.35)
        mean_d, band_d = rollout.mean_d, 2.0 * rollout.std_d
        top.fill_between(nodes, mean_d - band_d, mean_d + band_d, color=color, alpha=0.18, linewidth=0)
        top.plot(nodes, mean_d, color=color, linewidth=2, label="defender: GP samples, mean ± 2σ")
        top.plot(actual_t, front_d[step:step + horizon_steps + 1], color=DEFENDER, linewidth=2, linestyle="--",
                 label="defender: actual")
        top.set_title(f"t = {current['t']:.2f} s, attacker target {current['target_d']:+.2f} m", color=INK, fontsize=10)
        # 縱向：相對「前車以目前車速等速前進」的差，讓 σ_s 與誤差看得見。
        samples_s = rollout.progress - speed * nodes
        mean_s, band_s = rollout.mean_progress - speed * nodes, 2.0 * rollout.std_s
        for row in samples_s:
            bottom.plot(nodes, row, color=color, linewidth=0.8, alpha=0.35)
        bottom.fill_between(nodes, mean_s - band_s, mean_s + band_s, color=color, alpha=0.18, linewidth=0)
        bottom.plot(nodes, mean_s, color=color, linewidth=2)
        bottom.plot(actual_t, progress[step:step + horizon_steps + 1] - progress[step] - speed * actual_t,
                    color=DEFENDER, linewidth=2, linestyle="--")
        bottom.set_xlabel("Time from now (s)", color=INK)
    axes[0, 0].set_ylabel("Lateral offset d (m)", color=INK)
    axes[1, 0].set_ylabel("Defender s − constant-speed s (m)", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=9, labelcolor=INK)
    figure.suptitle("paper_rollout_overtake (k=2): 25 sampled one-step GP rollouts along the attacker's previous plan",
                    color=INK, fontsize=11)
    figure.savefig(HERE / "rollout_snapshots.png", dpi=130, facecolor=SURFACE)
    return [item[2]["t"] for item in chosen]


def plot_timeseries(results, track) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(11, 8.4), layout="constrained", facecolor=SURFACE)
    FigureCanvasAgg(figure)
    top, middle, bottom = figure.subplots(3, 1, sharex=True)
    for axis in (top, middle, bottom):
        style(axis)
    ends = []
    for name, (_, label, color) in RUNS.items():
        result = results[name]
        gaps = relative_gaps(result, track)
        t = np.arange(len(gaps)) * result["dt"]
        top.plot(t, gaps, color=color, linewidth=2, label=label)
        ends.append((float(gaps[-1]), name, float(t[-1])))
    # 線尾直接標名稱（文字用墨色），黃色等低對比色不只靠顏色辨識；終點相近時往上錯開，避免重疊。
    span = max(value for value, _, _ in ends) - min(value for value, _, _ in ends)
    spacing, placed = max(0.08 * span, 0.5), []
    for value, name, end in sorted(ends):
        y = value if not placed or value - placed[-1] >= spacing else placed[-1] + spacing
        placed.append(y)
        top.annotate(name, (end, value), xytext=(end + 0.3, y), textcoords="data", va="center",
                     color=INK, fontsize=8, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6})
    top.set_xlim(right=t[-1] + 3.2)
    top.axhline(0.0, color=MUTED, linestyle="--", linewidth=1)
    top.set_ylabel("Gap: defender s − attacker s (m)", color=INK)
    top.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    result = results["paper_k2"]
    frames, dt = result["frames"], result["dt"]
    t = np.arange(len(frames)) * dt
    middle.plot(t, [f.observations["rear"].track_position.d for f in frames], color=INK, linewidth=2,
                label="attacker d")
    middle.plot(t, [f.observations["front"].track_position.d for f in frames], color=DEFENDER, linewidth=2,
                label="defender d")
    middle.step([d["t"] for d in result["decisions"]], [d["target_d"] for d in result["decisions"]], where="post",
                color=MUTED, linestyle="--", linewidth=1.2, label="attacker target d")
    middle.set_ylabel("paper k=2: d (m)", color=INK)
    middle.legend(loc="lower left", frameon=False, fontsize=8, ncols=3, labelcolor=INK)
    active = [d for d in result["decisions"] if d["rollout"] is not None]
    color = RUNS["paper_k2"][2]
    times = [d["t"] for d in active]
    std_s = [d["rollout"].std_s[-1] for d in active]
    std_d = [d["rollout"].std_d[-1] for d in active]
    bottom.plot(times, std_s, color=color, linewidth=1.5, label="σ_s at +1 s (along track)")
    bottom.plot(times, std_d, color=color, linewidth=1.5, linestyle="--", label="σ_d at +1 s (lateral)")
    bottom.set_ylabel("paper k=2: sample std (m)", color=INK)
    bottom.set_xlabel("Time (s)", color=INK)
    bottom.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK)
    figure.savefig(HERE / "timeseries.png", dpi=130, facecolor=SURFACE)


def main_demo() -> None:
    settings, load_seconds = build_settings()
    print(f"paper GP loaded in {load_seconds:.1f} s", flush=True)
    results = {}
    for name in RUNS:
        results[name] = run(name, settings_for(name, settings))
        print(f"{name}: {len(results[name]['decisions'])} decisions in {results[name]['wall_seconds']:.1f} s",
              flush=True)
    summary = {name: metrics(name, result, settings.track) for name, result in results.items()}
    # k=2 與 k=0 只差在橢圓、亂數序列相同：第一次決策不同的時刻，就是橢圓第一次改變選擇的時刻。
    pairs = zip(results["paper_k2"]["decisions"], results["paper_k0"]["decisions"])
    different = next(((a, b) for a, b in pairs
                      if (a["target_d"], a["speed_limit"]) != (b["target_d"], b["speed_limit"])), None)
    summary["paper_k2_vs_k0"] = None if different is None else {
        "first_different_decision_s": different[0]["t"],
        **{key: {"target_d": item["target_d"], "speed_limit": item["speed_limit"],
                 "feasible_candidates": item["feasible"]} for key, item in zip(("k2", "k0"), different)},
    }
    compact = PROJECT_ROOT / main.PAPER_GP_MODEL_PATH
    config = main.PAPER_ROLLOUT_OVERTAKING_CONFIG
    summary["scene"] = {
        "vehicles": [list(spec) for spec in main.INTERACTION_VEHICLE_SPECS], "seconds": SECONDS,
        "paper_model": main.PAPER_GP_MODEL_PATH,
        "paper_model_sha256": hashlib.sha256(compact.read_bytes()).hexdigest(),
        "paper_model_load_seconds": load_seconds, "interaction_model": main.INTERACTION_GP_MODEL_PATH,
        "samples": config.samples, "ellipse_sigma": config.ellipse_sigma, "seed": config.seed,
        "candidate_offsets": list(config.relative.candidate_offsets),
        "rollout_overtake_candidate_offsets": list(main.ROLLOUT_OVERTAKING_CONFIG.relative.candidate_offsets),
        "decision_period": config.relative.decision_period, "prediction_horizon": config.relative.prediction_horizon,
        "error_from_seconds": ERROR_FROM,
    }
    summary["snapshot_times_s"] = plot_snapshots(results["paper_k2"], settings.track)
    plot_timeseries(results, settings.track)
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_demo()
