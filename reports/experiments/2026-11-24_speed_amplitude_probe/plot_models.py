"""載入已保存的 M0／M1-15／M1-30，在新測試 8 場上畫評估圖；不重新擬合。

輸出：evaluation_scatter.png（實際對預測）、calibration.png（±kσ 實際涵蓋率）、
timeseries_<場次>.png（單場預測區間隨時間）。
"""

import json
import math
import os
from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from training_data.interaction import HISTORY_ACTION_FEATURES
from training_data.io import load_dataset

HERE = Path(__file__).resolve().parent
NEW = PROJECT_ROOT / "data/training/history_speed06_20261124"
MODELS = (  # 顏色跟著模型走，與 speed_effect.png 相同（M0 藍、M1-30 橘）
    ("M0", "models/gp/history_20261124_c_action13/model.pkl", "#2a78d6"),
    ("M1-15", "models/gp/speed06_20261124_c_m1_15/model.pkl", "#1baf7a"),
    ("M1-30", "models/gp/speed06_20261124_c_m1_30/model.pkl", "#eb6834"),
)
EPISODES = ("speed06-008", "speed06-007")   # 測試場中最慢與最快的各一場
SURFACE, INK, MUTED, GRID, NEUTRAL = "#fcfcfb", "#1f1f1d", "#6b6a63", "#e4e3de", "#b9b8b0"


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def new_figure(width: float, height: float, rows: int = 1, cols: int = 1, **kwargs):
    figure = Figure(figsize=(width, height), layout="constrained", facecolor=SURFACE)
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, cols, squeeze=False, **kwargs)


def main() -> None:
    dataset = load_dataset(NEW)
    if tuple(dataset.schema.feature_names) != HISTORY_ACTION_FEATURES:
        raise ValueError("unexpected feature order")
    summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
    test = summary["splits"]["new_test"]
    ids = np.array([s.episode_id for s in dataset.samples])
    times = np.array([s.time_seconds for s in dataset.samples])
    mask = np.isin(ids, test)
    X, y = dataset.X[mask], dataset.Y[mask, 0]
    ids, times = ids[mask], times[mask]
    column = {name: i for i, name in enumerate(HISTORY_ACTION_FEATURES)}
    probe = np.isclose(np.abs(X[:, column["d_target"]]), 0.6)
    predictions = {}
    for name, path, _ in MODELS:
        with (PROJECT_ROOT / path).open("rb") as stream:
            bundle = pickle.load(stream)
        if tuple(bundle["feature_names"]) != HISTORY_ACTION_FEATURES:
            raise ValueError(f"{name} feature order differs")
        predictions[name] = bundle["pipeline"].predict(X, return_std=True)
    scatter(y, probe, predictions)
    calibration(y, probe, predictions)
    for episode in EPISODES:
        timeseries(episode, ids, times, X, y, column, predictions)


def scatter(y, probe, predictions) -> None:
    figure, axes = new_figure(12, 4.2, 1, 3, sharex=True, sharey=True)
    limit = 1.05 * max(np.abs(y).max(), *(np.abs(m).max() for m, _ in predictions.values()))
    for axis, (name, _, color) in zip(axes[0], MODELS):
        mean, std = predictions[name]
        style(axis)
        axis.plot([-limit, limit], [-limit, limit], "--", color=MUTED, linewidth=1)
        axis.scatter(y[~probe], mean[~probe], s=10, color=NEUTRAL, linewidths=0, label="center (d_target = 0)")
        axis.scatter(y[probe], mean[probe], s=10, color=color, linewidths=0, label="|d_target| = 0.6 m")
        error = mean - y
        axis.set_title(f"{name}: MAE {np.mean(np.abs(error)):.3f} m, "
                       f"0.6 m probes {np.mean(np.abs(error[probe])):.3f} m", color=INK, fontsize=10)
        axis.set_xlabel("Actual front Δd in 0.5 s (m)", color=INK)
        axis.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    axes[0][0].set_ylabel("GP mean (m)", color=INK)
    axes[0][0].set_xlim(-limit, limit)
    axes[0][0].set_ylim(-limit, limit)
    figure.suptitle("New test episodes (8), all samples 0-30 s", color=INK, fontsize=11)
    figure.savefig(HERE / "evaluation_scatter.png", dpi=130, facecolor=SURFACE)


def calibration(y, probe, predictions) -> None:
    """常態假設下 ±kσ 應涵蓋 erf(k/√2)；曲線低於對角線表示區間太窄。"""
    k = np.linspace(0.05, 3.5, 70)
    nominal = np.array([math.erf(value / math.sqrt(2)) for value in k])
    figure, axes = new_figure(10, 4.4, 1, 2, sharey=True)
    for axis, (title, subset) in zip(axes[0], (("All samples", np.ones_like(probe)), ("|d_target| = 0.6 m only", probe))):
        style(axis)
        axis.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
        for name, _, color in MODELS:
            mean, std = predictions[name]
            z = np.abs(mean[subset] - y[subset]) / std[subset]
            observed = np.array([np.mean(z <= value) for value in k])
            at2 = np.mean(z <= 2)
            axis.plot(nominal, observed, color=color, linewidth=2, label=f"{name} (±2σ: {at2:.1%})")
        axis.axvline(math.erf(2 / math.sqrt(2)), color=GRID, linewidth=1)
        axis.set_title(f"{title}, n={int(subset.sum())}", color=INK, fontsize=10)
        axis.set_xlabel("Nominal coverage of ±kσ (Gaussian)", color=INK)
        axis.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    axes[0][0].set_ylabel("Observed coverage", color=INK)
    figure.suptitle("Interval calibration on new test episodes (below diagonal = intervals too narrow)",
                    color=INK, fontsize=11)
    figure.savefig(HERE / "calibration.png", dpi=130, facecolor=SURFACE)


def timeseries(episode, ids, times, X, y, column, predictions) -> None:
    rows = ids == episode
    t = times[rows]
    order = np.argsort(t)
    t = t[order]
    figure, axes = new_figure(11, 7.2, 3, 1, sharex=True)
    top = axes[0][0]
    style(top)
    top.step(t, X[rows][order, column["d_target"]], where="post", color=MUTED, linestyle="--",
             linewidth=1.5, label="rear target d")
    top.plot(t, X[rows][order, column["rear_d"]], color=INK, linewidth=2, label="rear d")
    top.plot(t, X[rows][order, column["front_d"]], color="#6250d6", linewidth=2, label="front d")
    top.set_ylabel("Lateral offset d (m)", color=INK)
    # 圖例放在圖框上方，避免壓到 ±0.6 m 附近的資料。
    top.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8, ncols=3, labelcolor=INK)
    speed = X[rows][order, column["front_speed"]].mean()
    top.set_title(f"{episode}: front speed ≈ {speed:.1f} m/s (left is +d)", color=INK, fontsize=10, loc="left")
    actual = y[rows][order]
    for axis, name in ((axes[1][0], "M0"), (axes[2][0], "M1-30")):
        color = next(c for n, _, c in MODELS if n == name)
        mean, std = (values[rows][order] for values in predictions[name])
        missed = np.abs(actual - mean) > 2 * std
        style(axis)
        axis.fill_between(t, mean - 2 * std, mean + 2 * std, color=color, alpha=0.2, linewidth=0, label="±2σ")
        axis.plot(t, mean, color=color, linewidth=2, label=f"{name} mean")
        axis.plot(t, actual, color=INK, linewidth=1.5, label="actual")
        axis.scatter(t[missed], actual[missed], s=24, marker="x", color=INK, linewidths=1.5, zorder=3,
                     label=f"outside ±2σ ({missed.mean():.0%})")
        axis.set_ylabel("Front Δd in 0.5 s (m)", color=INK)
        axis.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8, ncols=4, labelcolor=INK)
    axes[2][0].set_xlabel("Time (s)", color=INK)
    figure.savefig(HERE / f"timeseries_{episode}.png", dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    main()
