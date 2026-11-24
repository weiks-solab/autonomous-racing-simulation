"""載入 v1、v2 精簡檔，在近距離測試 12 場上畫評估圖；不重新擬合。

輸出：evaluation_scatter.png（實際對預測）、calibration.png（±kσ 實際涵蓋率）、timeseries_<場次>.png。
只用車距 −2～3 m 的樣本（v2 的訓練範圍）。
"""

import json
import math
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from strategy.interaction_gp import InteractionGpModel
from training_data.interaction import HISTORY_ACTION_FEATURES
from training_data.io import load_dataset

HERE = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data/training/close_range_20261124"
MODELS = (  # 顏色跟著模型走，與 gap_bins.png 相同
    ("v1", "models/compact/interaction_gp_v1_m1_30.npz", "#eb6834"),
    ("v2", "models/compact/interaction_gp_v2_close.npz", "#1baf7a"),
)
GAP_RANGE = (-2.0, 3.0)
SURFACE, INK, MUTED, GRID, NEUTRAL = "#fcfcfb", "#1f1f1d", "#6b6a63", "#e4e3de", "#b9b8b0"
COLUMN = {name: i for i, name in enumerate(HISTORY_ACTION_FEATURES)}


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def new_figure(width, height, rows=1, cols=1, **kwargs):
    figure = Figure(figsize=(width, height), layout="constrained", facecolor=SURFACE)
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, cols, squeeze=False, **kwargs)


def main() -> None:
    dataset = load_dataset(DATA)
    test = json.loads((HERE / "offline_summary.json").read_text(encoding="utf-8"))["close_test"]
    ids = np.array([s.episode_id for s in dataset.samples])
    times = np.array([s.time_seconds for s in dataset.samples])
    X, y = dataset.X, dataset.Y[:, 0]
    gap = X[:, COLUMN["gap"]]
    mask = np.isin(ids, test) & (gap >= GAP_RANGE[0]) & (gap <= GAP_RANGE[1])
    predictions = {name: InteractionGpModel.load(PROJECT_ROOT / path).predict(X) for name, path, _ in MODELS}
    scatter(y[mask], gap[mask], {k: (m[mask], s[mask]) for k, (m, s) in predictions.items()})
    calibration(y[mask], {k: (m[mask], s[mask]) for k, (m, s) in predictions.items()})
    # 測試場中近距離樣本最多的一場。
    counts = {episode: int(np.sum(mask & (ids == episode))) for episode in test}
    episode = max(counts, key=counts.get)
    timeseries(episode, ids == episode, times, X, y, predictions)


def scatter(y, gap, predictions) -> None:
    figure, axes = new_figure(9, 4.4, 1, 2, sharex=True, sharey=True)
    limit = 1.05 * max(np.abs(y).max(), *(np.abs(m).max() for m, _ in predictions.values()))
    near = gap < 1.6   # 0 以下為並排或剛超越
    for axis, (name, _, color) in zip(axes[0], MODELS):
        mean, _ = predictions[name]
        style(axis)
        axis.plot([-limit, limit], [-limit, limit], "--", color=MUTED, linewidth=1)
        axis.scatter(y[~near], mean[~near], s=10, color=NEUTRAL, linewidths=0, label="gap 1.6–3 m")
        axis.scatter(y[near], mean[near], s=10, color=color, linewidths=0, label="gap −2–1.6 m")
        error = mean - y
        axis.set_title(f"{name}: MAE {np.mean(np.abs(error)):.3f} m (gap <1.6 m: {np.mean(np.abs(error[near])):.3f})",
                       color=INK, fontsize=10)
        axis.set_xlabel("Actual front Δd in 0.5 s (m)", color=INK)
        axis.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    axes[0][0].set_ylabel("GP mean (m)", color=INK)
    axes[0][0].set_xlim(-limit, limit)
    axes[0][0].set_ylim(-limit, limit)
    figure.suptitle(f"Close-range test episodes (12), samples with gap −2–3 m (n={len(y)})", color=INK, fontsize=11)
    figure.savefig(HERE / "evaluation_scatter.png", dpi=130, facecolor=SURFACE)


def calibration(y, predictions) -> None:
    """常態假設下 ±kσ 應涵蓋 erf(k/√2)；曲線低於對角線表示區間太窄。"""
    k = np.linspace(0.05, 3.5, 70)
    nominal = np.array([math.erf(value / math.sqrt(2)) for value in k])
    figure, axes = new_figure(5.6, 4.6)
    axis = axes[0][0]
    style(axis)
    axis.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
    for name, _, color in MODELS:
        mean, std = predictions[name]
        z = np.abs(mean - y) / std
        axis.plot(nominal, [np.mean(z <= value) for value in k], color=color, linewidth=2,
                  label=f"{name} (±2σ: {np.mean(z <= 2):.1%}, mean width {np.mean(4 * std):.2f} m)")
    axis.set_xlabel("Nominal coverage of ±kσ (Gaussian)", color=INK)
    axis.set_ylabel("Observed coverage", color=INK)
    axis.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    axis.set_title("Calibration, close-range test (gap −2–3 m)", color=INK, fontsize=10)
    figure.savefig(HERE / "calibration.png", dpi=130, facecolor=SURFACE)


def timeseries(episode, rows, times, X, y, predictions) -> None:
    order = np.argsort(times[rows])
    t = times[rows][order]
    pick = lambda name: X[rows][order, COLUMN[name]]
    figure, axes = new_figure(11, 9, 4, 1, sharex=True)
    top = axes[0][0]
    style(top)
    top.step(t, pick("d_target"), where="post", color=MUTED, linestyle="--", linewidth=1.5, label="attacker target d")
    top.plot(t, pick("rear_d"), color=INK, linewidth=2, label="attacker d")
    top.plot(t, pick("front_d"), color="#6250d6", linewidth=2, label="defender d")
    top.set_ylabel("Lateral offset d (m)", color=INK)
    top.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8, ncols=3, labelcolor=INK)
    top.set_title(f"{episode}: front speed ≈ {pick('front_speed').mean():.1f} m/s (left is +d)",
                  color=INK, fontsize=10, loc="left")
    gap_axis = axes[1][0]
    style(gap_axis)
    gap_axis.plot(t, pick("gap"), color=INK, linewidth=2)
    gap_axis.axhspan(GAP_RANGE[0], GAP_RANGE[1], color=GRID, alpha=0.6, linewidth=0, label="v2 training gap range")
    gap_axis.axhline(1.56, color=MUTED, linestyle=":", linewidth=1, label="v1 lower limit 1.56 m")
    gap_axis.set_ylabel("Gap (m)", color=INK)
    gap_axis.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8, ncols=2, labelcolor=INK)
    actual = y[rows][order]
    for axis, (name, _, color) in zip((axes[2][0], axes[3][0]), MODELS):
        mean, std = (values[rows][order] for values in predictions[name])
        missed = np.abs(actual - mean) > 2 * std
        style(axis)
        axis.fill_between(t, mean - 2 * std, mean + 2 * std, color=color, alpha=0.2, linewidth=0, label="±2σ")
        axis.plot(t, mean, color=color, linewidth=2, label=f"{name} mean")
        axis.plot(t, actual, color=INK, linewidth=1.5, label="actual")
        axis.scatter(t[missed], actual[missed], s=24, marker="x", color=INK, linewidths=1.5, zorder=3,
                     label=f"outside ±2σ ({missed.mean():.0%})")
        axis.set_ylabel("Front Δd in 0.5 s (m)", color=INK)
        axis.set_ylim(-0.8, 0.8)
        axis.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8, ncols=4, labelcolor=INK)
    axes[3][0].set_xlabel("Time (s)", color=INK)
    figure.savefig(HERE / f"timeseries_{episode}.png", dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    main()
