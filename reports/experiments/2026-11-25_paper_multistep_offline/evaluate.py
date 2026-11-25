"""論文格式 GP 的離線多步推演評估：自車未來用實際紀錄，量 1 s 取樣推演的誤差與 ±2σ 涵蓋率。

閉迴路中推演依的是「上一步計畫」；這裡把計畫換成錄下來的自車實際軌跡，排除計畫錯位，只看模型與推演本身。
推演直接呼叫 PaperRolloutOvertakingStrategy._rollout，與線上使用同一份程式。

比較：
- sampled：25 個樣本、從 N(μ, σ²) 抽樣（線上用的作法）。
- mean：只回饋 GP 平均（σ 設為 0、1 個樣本）。
- constant：定速、d 不變（線上 GP 不可用時的退回）。

執行：python reports/experiments/2026-11-25_paper_multistep_offline/evaluate.py
需要本機的 close_range_20261124 與 history_gp_20261124 原始資料。
"""

import csv
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

import main  # noqa: E402
from core.pure_pursuit import PurePursuitConfig  # noqa: E402
from core.track_geometry import signed_longitudinal_gap, track_length  # noqa: E402
from strategy.bicycle_prediction import BicyclePrediction, BicyclePredictionConfig  # noqa: E402
from strategy.interaction_features import LateralHistory  # noqa: E402
from strategy.paper_gp import PaperGpModel  # noqa: E402
from strategy.paper_rollout_overtaking import (  # noqa: E402
    PaperRolloutOvertakingConfig, PaperRolloutOvertakingStrategy,
)
from strategy.rollout_overtaking import PreviousPlan  # noqa: E402
from training_data.io import load_recordings  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = {"close": PROJECT_ROOT / "data/training/close_range_20261124",
        "old": PROJECT_ROOT / "data/training/history_gp_20261124"}
SPEED_SUMMARY = PROJECT_ROOT / "reports/experiments/2026-11-24_speed_amplitude_probe/summary.json"
DT, STEP_FRAMES, HORIZON_FRAMES, HISTORY_FRAMES = 0.02, 5, 50, 20   # 0.1 s 一步、1.0 s 時域、0.4 s 歷史
REPORT_NODES = (5, 10)                                             # +0.5 s、+1.0 s
METHODS = ("sampled", "mean", "constant")


def train_eval_module():
    """沿用論文格式 GP 第一版的場次切分函式（同 v2）。"""
    path = PROJECT_ROOT / "reports/experiments/2026-11-25_paper_format_gp/train_eval.py"
    spec = importlib.util.spec_from_file_location("paper_train_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ZeroStd:
    """只回傳 GP 平均（標準差 0），用來做「只回饋平均」的推演。"""

    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, X, return_std=False):
        mean, std = self.predictor.predict(X, return_std=True)
        return mean, np.zeros_like(std)


def strategies(model: PaperGpModel) -> dict:
    controller = PurePursuitConfig(main.LOOKAHEAD_DISTANCE, 7.0, main.SPEED_GAIN, main.MAX_ACCELERATION,
                                   main.MODEL_CONFIG.max_steering_angle, max_deceleration=main.MAX_DECELERATION)
    prediction = BicyclePredictionConfig(main.MODEL_CONFIG, controller, main.OFFSET_PATH_CONFIG)
    configs = {"rear": prediction, "front": prediction}
    base = main.PAPER_ROLLOUT_OVERTAKING_CONFIG
    zero = replace(model, predictors=tuple(ZeroStd(p) for p in model.predictors))
    return {
        "sampled": PaperRolloutOvertakingStrategy(base, prediction, configs, model),
        "mean": PaperRolloutOvertakingStrategy(
            PaperRolloutOvertakingConfig(base.relative, samples=1, ellipse_sigma=0.0, seed=base.seed),
            prediction, configs, zero),
    }


def moments(episodes, track, wanted: set[str], every_frames: int, gap_range):
    """可評估的時刻：前 0.4 s 歷史與後 1.0 s 都在、窗內沒有碰撞，且起點車距在 gap_range 內。"""
    length = track_length(track)
    for episode in episodes:
        if episode.episode_id not in wanted:
            continue
        frames = episode.frames
        for step in range(HISTORY_FRAMES, len(frames) - HORIZON_FRAMES, every_frames):
            window = frames[step - HISTORY_FRAMES:step + HORIZON_FRAMES + 1]
            if any(f.collision_pairs or not {"rear", "front"} <= set(f.observations) for f in window):
                continue
            rear, front = frames[step].observations["rear"], frames[step].observations["front"]
            gap = signed_longitudinal_gap(rear.track_position, front.track_position, length)
            if gap_range is None or gap_range[0] <= gap <= gap_range[1]:
                yield episode.episode_id, frames, step, gap


def cumulative(frames, vehicle_id, start, count, length) -> np.ndarray:
    values = [0.0]
    for index in range(start, start + count):
        values.append(values[-1] + signed_longitudinal_gap(frames[index].observations[vehicle_id].track_position,
                                                           frames[index + 1].observations[vehicle_id].track_position,
                                                           length))
    return np.array(values)


def evaluate_moment(models, frames, step, track) -> dict:
    length = track_length(track)
    now = step * DT
    rear, front = frames[step].observations["rear"], frames[step].observations["front"]
    # 自車未來用實際紀錄，包成「上一步計畫」交給線上的推演程式。
    rear_progress = cumulative(frames, "rear", step, HORIZON_FRAMES, length)
    future = frames[step:step + HORIZON_FRAMES + 1]
    plan = PreviousPlan(now, BicyclePrediction(
        tuple(f.observations["rear"].state for f in future), tuple(float(p) for p in rear_progress),
        tuple(round(k * DT, 10) for k in range(HORIZON_FRAMES + 1)),
        tuple(f.observations["rear"].track_position.d for f in future)), 0.0)
    nodes = np.arange(0, HORIZON_FRAMES + 1, STEP_FRAMES)
    front_progress = cumulative(frames, "front", step, HORIZON_FRAMES, length)[nodes]
    front_d = np.array([frames[step + k].observations["front"].track_position.d for k in nodes])
    result = {"actual_s": front_progress, "actual_d": front_d}
    for name, strategy in models.items():
        strategy.history = LateralHistory(strategy.history.window)
        for index in range(max(0, step - HISTORY_FRAMES - 5), step + 1):
            frame = frames[index]
            strategy.history.record(index * DT, (frame.observations["rear"], frame.observations["front"]))
        rollout, reason = strategy._rollout(rear, front, track, now, plan)
        result[name] = rollout
        result[f"{name}_reason"] = reason
    speed = max(0.0, front.state.velocity)
    result["constant_s"] = speed * nodes * DT
    result["constant_d"] = np.full(len(nodes), front.track_position.d)
    return result


def summarize(results: list[dict]) -> dict:
    """GP 第 0 步可用的時刻才比較（三種方法用同一批時刻）。"""
    used = [r for r in results if r["sampled"] is not None and r["mean"] is not None]
    out = {"moments": len(results), "used": len(used),
           "fallback_reasons": {}}
    for r in results:
        if r["sampled"] is None:
            out["fallback_reasons"][r["sampled_reason"]] = out["fallback_reasons"].get(r["sampled_reason"], 0) + 1
    if not used:
        return out
    for node in REPORT_NODES:
        key, entry = f"{node * STEP_FRAMES * DT:g}s", {}
        for method in METHODS:
            if method == "constant":
                s = np.array([r["constant_s"][node] for r in used])
                d = np.array([r["constant_d"][node] for r in used])
            else:
                s = np.array([r[method].mean_progress[node] for r in used])
                d = np.array([r[method].mean_d[node] for r in used])
            s_err = s - np.array([r["actual_s"][node] for r in used])
            d_err = d - np.array([r["actual_d"][node] for r in used])
            entry[method] = {"s_mae_m": float(np.mean(np.abs(s_err))), "s_bias_m": float(np.mean(s_err)),
                             "d_mae_m": float(np.mean(np.abs(d_err))),
                             "d_p90_abs_m": float(np.percentile(np.abs(d_err), 90))}
            if method == "sampled":
                std_s = np.array([r["sampled"].std_s[node] for r in used])
                std_d = np.array([r["sampled"].std_d[node] for r in used])
                entry[method].update({
                    "s_coverage_2std": float(np.mean(np.abs(s_err) <= 2 * std_s)),
                    "d_coverage_2std": float(np.mean(np.abs(d_err) <= 2 * std_d)),
                    "std_s_median_m": float(np.median(std_s)), "std_d_median_m": float(np.median(std_d)),
                    # 實際值是否落在 25 個樣本的最小～最大之間。
                    "d_within_sample_range": float(np.mean([
                        r["sampled"].offsets[:, node].min() <= r["actual_d"][node] <= r["sampled"].offsets[:, node].max()
                        for r in used])),
                })
        out[key] = entry
    # σ 隨推演步數的成長：獨立誤差的隨機漫步約為 √k 倍。
    std_d = np.array([r["sampled"].std_d for r in used])
    std_s = np.array([r["sampled"].std_s for r in used])
    out["std_by_node"] = {"seconds": [round(k * STEP_FRAMES * DT, 1) for k in range(std_d.shape[1])],
                          "std_d_median_m": [float(v) for v in np.median(std_d, axis=0)],
                          "std_s_median_m": [float(v) for v in np.median(std_s, axis=0)]}
    held = [sum(1 for reason in r["sampled"].exit_reasons if reason) for r in used]
    samples = len(used[0]["sampled"].exit_reasons)
    out["samples_switched_to_constant_share"] = float(np.sum(held) / (len(used) * samples))
    reasons = {}
    for r in used:
        for reason in r["sampled"].exit_reasons:
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
    out["exit_reasons"] = dict(sorted(reasons.items(), key=lambda item: -item[1]))
    return out


def plot(summaries: dict, results: dict) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    surface, ink, muted, grid = "#fcfcfb", "#1f1f1d", "#6b6a63", "#e4e3de"
    colors = {"constant": "#2a78d6", "mean": "#eb6834", "sampled": "#1baf7a"}
    labels = {"constant": "constant speed, constant d", "mean": "GP, mean only", "sampled": "GP, 25 samples (mean)"}
    used = [r for r in results["close"] if r["sampled"] is not None and r["mean"] is not None]
    seconds = np.arange(HORIZON_FRAMES // STEP_FRAMES + 1) * STEP_FRAMES * DT
    figure = Figure(figsize=(13, 4.3), layout="constrained", facecolor=surface)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    for axis in axes:
        axis.set_facecolor(surface)
        axis.grid(True, color=grid, linewidth=0.8)
        axis.tick_params(colors=muted)
        for spine in axis.spines.values():
            spine.set_color(grid)
        axis.set_xlabel("Prediction time (s)", color=ink)
    for method in METHODS:
        if method == "constant":
            s = np.array([r["constant_s"] for r in used])
            d = np.array([r["constant_d"] for r in used])
        else:
            s = np.array([r[method].mean_progress for r in used])
            d = np.array([r[method].mean_d for r in used])
        actual_s = np.array([r["actual_s"] for r in used])
        actual_d = np.array([r["actual_d"] for r in used])
        axes[0].plot(seconds, np.mean(np.abs(d - actual_d), axis=0), color=colors[method], linewidth=2,
                     marker="o", markersize=4, label=labels[method])
        axes[1].plot(seconds, np.mean(np.abs(s - actual_s), axis=0), color=colors[method], linewidth=2,
                     marker="o", markersize=4, label=labels[method])
    d_err = np.array([np.abs(r["sampled"].mean_d - r["actual_d"]) for r in used])
    std_d = np.array([r["sampled"].std_d for r in used])
    axes[2].plot(seconds, np.median(2 * std_d, axis=0), color=colors["sampled"], linewidth=2, marker="o",
                 markersize=4, label="half-width of ±2σ band (median)")
    axes[2].plot(seconds, np.median(d_err, axis=0), color=colors["sampled"], linewidth=2, linestyle="--",
                 marker="o", markersize=4, label="actual error |mean − actual| (median)")
    axes[0].set_ylabel("Lateral error, MAE (m)", color=ink)
    axes[1].set_ylabel("Along-track error, MAE (m)", color=ink)
    axes[2].set_ylabel("Lateral, 25-sample rollout (m)", color=ink)
    axes[0].set_title("Where will the defender be sideways?", color=ink, fontsize=10)
    axes[1].set_title("How far along the track?", color=ink, fontsize=10)
    axes[2].set_title("Uncertainty band vs actual error", color=ink, fontsize=10)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=ink)
    axes[2].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=ink)
    figure.suptitle(f"Close-range test episodes, 0 ≤ gap ≤ 2 m ({len(used)} moments); attacker's future taken "
                    "from the recording", color=ink, fontsize=11)
    figure.savefig(HERE / "multistep.png", dpi=130, facecolor=surface)


def write_rows(name: str, results: list[dict], writer) -> None:
    for r in results:
        row = {"set": name, "episode": r["episode"], "time_s": r["time_s"], "gap_m": round(r["gap"], 4),
               "fallback": r["sampled_reason"]}
        if r["sampled"] is not None and r["mean"] is not None:
            for node in REPORT_NODES:
                tag = f"{node * STEP_FRAMES * DT:g}s"
                row.update({f"actual_d_{tag}": r["actual_d"][node], f"sampled_d_{tag}": r["sampled"].mean_d[node],
                            f"std_d_{tag}": r["sampled"].std_d[node], f"mean_d_{tag}": r["mean"].mean_d[node],
                            f"constant_d_{tag}": r["constant_d"][node], f"actual_s_{tag}": r["actual_s"][node],
                            f"sampled_s_{tag}": r["sampled"].mean_progress[node],
                            f"std_s_{tag}": r["sampled"].std_s[node], f"mean_s_{tag}": r["mean"].mean_progress[node],
                            f"constant_s_{tag}": r["constant_s"][node]})
        writer.writerow({key: (round(value, 5) if isinstance(value, float) else value) for key, value in row.items()})


def main_evaluate() -> None:
    started = time.perf_counter()
    module = train_eval_module()
    model = PaperGpModel.load(main.PAPER_GP_MODEL_PATH)
    models = strategies(model)
    speed_summary = json.loads(SPEED_SUMMARY.read_text(encoding="utf-8"))["splits"]
    sets = {}
    track_close, _, close_episodes = load_recordings(DATA["close"])
    metadata = json.loads((DATA["close"] / "metadata.json").read_text(encoding="utf-8"))
    _, close_test = module.close_split({item["id"]: item for item in metadata["run"]["scenarios"]})
    # 近距離測試 12 場，起點車距 0～2 m（論文的評估範圍），每 0.1 s 一個時刻。
    sets["close"] = (track_close, list(moments(close_episodes, track_close, set(close_test), STEP_FRAMES, (0.0, 2.0))))
    track_old, _, old_episodes = load_recordings(DATA["old"])
    heldout = set(speed_summary["old"]["validation"] + speed_summary["old"]["test"])
    # 試探資料保留 8 場（車距較遠、攻方左右試探），每 0.5 s 一個時刻。
    sets["old"] = (track_old, list(moments(old_episodes, track_old, heldout, 25, None)))
    results, summaries = {}, {}
    for name, (track, items) in sets.items():
        print(f"{name}: {len(items)} moments", flush=True)
        results[name] = []
        for index, (episode, frames, step, gap) in enumerate(items):
            result = evaluate_moment(models, frames, step, track)
            result.update({"episode": episode, "time_s": round(step * DT, 2), "gap": gap})
            results[name].append(result)
            if (index + 1) % 100 == 0:
                print(f"  {index + 1}/{len(items)} ({time.perf_counter() - started:.0f} s)", flush=True)
        summaries[name] = summarize(results[name])
    summary = {"sets": summaries, "model": main.PAPER_GP_MODEL_PATH,
               "samples": main.PAPER_ROLLOUT_OVERTAKING_CONFIG.samples,
               "seed": main.PAPER_ROLLOUT_OVERTAKING_CONFIG.seed,
               "close_test_episodes": sorted(close_test), "old_heldout_episodes": sorted(heldout),
               "wall_seconds": time.perf_counter() - started}
    plot(summaries, results)
    with (HERE / "moments.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["set", "episode", "time_s", "gap_m", "fallback"] + [
            f"{kind}_{tag}" for tag in ("0.5s", "1s")
            for kind in ("actual_d", "sampled_d", "std_d", "mean_d", "constant_d",
                         "actual_s", "sampled_s", "std_s", "mean_s", "constant_s")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in results:
            write_rows(name, results[name], writer)
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_evaluate()
