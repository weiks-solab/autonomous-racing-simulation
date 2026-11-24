"""近距離互動資料訓練 v2，依車距分組比較 v1、v2 與「前車不橫移」基準。

v2 訓練集＝新訓練場中車距 −2～3 m 的樣本（最多 3000 筆）＋ v1（M1-30）訓練樣本隨機抽樣，總數上限 5000。
完整 pkl 存 models/gp/（.gitignore），並匯出可進 Git 的精簡檔。
"""

import csv
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

import numpy as np
import sklearn

from interaction_gp import export, split_groups
from strategy.interaction_gp import InteractionGpModel
from train_gp import TrainingConfig, create_model, prediction_errors
from training_data.interaction import HISTORY_ACTION_FEATURES
from training_data.io import load_dataset

HERE = Path(__file__).resolve().parent
OLD = PROJECT_ROOT / "data/training/history_gp_20261124"
SPEED = PROJECT_ROOT / "data/training/history_speed06_20261124"
CLOSE = PROJECT_ROOT / "data/training/close_range_20261124"
SPEED_SUMMARY = PROJECT_ROOT / "reports/experiments/2026-11-24_speed_amplitude_probe/summary.json"
V1 = PROJECT_ROOT / "models/compact/interaction_gp_v1_m1_30.npz"
V2_DIR = PROJECT_ROOT / "models/gp/close_range_20261124_v2"
V2_COMPACT = PROJECT_ROOT / "models/compact/interaction_gp_v2_close.npz"
CAP_TOTAL, CAP_CLOSE, SEED = 5000, 3000, 42
GAP_RANGE = (-2.0, 3.0)
GAP_BINS = ((-2.0, 0.0), (0.0, 0.8), (0.8, 1.6), (1.6, 3.0))
GAP = HISTORY_ACTION_FEATURES.index("gap")


def load(directory: Path):
    dataset = load_dataset(directory)
    if tuple(dataset.schema.feature_names) != HISTORY_ACTION_FEATURES:
        raise ValueError(f"{directory} is not a 13-feature history dataset")
    ids = np.array([sample.episode_id for sample in dataset.samples])
    scenarios = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))["run"]["scenarios"]
    return dataset.X, dataset.Y[:, 0], ids, {item["id"]: item for item in scenarios}


def close_split(scenarios: dict) -> tuple[list[str], list[str]]:
    """每個「速度區間 × 偏離權重」組合依場次編號：前兩場訓練、最後一場測試。"""
    cells = {}
    for episode in sorted(scenarios):
        design = scenarios[episode]["design"]
        cells.setdefault((tuple(design["speed_stratum"]), design["deviation_weight"]), []).append(episode)
    if any(len(items) < 2 for items in cells.values()):
        raise ValueError("each design cell needs at least two episodes")
    return ([episode for items in cells.values() for episode in items[:-1]],
            [items[-1] for items in cells.values()])


def metrics(actual, mean, std) -> dict:
    if len(actual) == 0:
        return {"n": 0}
    return {**prediction_errors(actual, mean), "n": int(len(actual)),
            "coverage_2std": float(np.mean(np.abs(mean - actual) <= 2 * std)),
            "mean_interval_width_m": float(np.mean(4 * std))}


def main() -> None:
    rng = np.random.default_rng(SEED)
    Xo, yo, io, _ = load(OLD)
    Xs, ys, is_, _ = load(SPEED)
    Xc, yc, ic, close_scenarios = load(CLOSE)
    old_split = {name: set(io[index]) for name, index in split_groups(io, SEED).items()}
    speed_splits = json.loads(SPEED_SUMMARY.read_text(encoding="utf-8"))["splits"]
    close_train, close_test = close_split(close_scenarios)

    # v1 訓練樣本（與 M1-30 相同的場次），作為 v2 的補足來源。
    pool_X = np.vstack((Xo[np.isin(io, list(old_split["train"]))], Xs[np.isin(is_, speed_splits["new_train"])]))
    pool_y = np.concatenate((yo[np.isin(io, list(old_split["train"]))], ys[np.isin(is_, speed_splits["new_train"])]))
    near = np.isin(ic, close_train) & (Xc[:, GAP] >= GAP_RANGE[0]) & (Xc[:, GAP] <= GAP_RANGE[1])
    near_index = np.flatnonzero(near)
    if len(near_index) > CAP_CLOSE:
        near_index = np.sort(rng.choice(near_index, CAP_CLOSE, replace=False))
    fill = min(len(pool_y), CAP_TOTAL - len(near_index))
    pool_index = np.sort(rng.choice(len(pool_y), fill, replace=False))
    X = np.vstack((Xc[near_index], pool_X[pool_index]))
    y = np.concatenate((yc[near_index], pool_y[pool_index]))
    print(f"v2 fit: {len(near_index)} close-range + {fill} v1-pool = {len(y)} samples", flush=True)
    if (V2_DIR / "model.pkl").is_file() and V2_COMPACT.is_file():
        # 已訓練過：沿用保存的模型與擬合資訊，只重做評估（不覆寫模型）。
        with (V2_DIR / "model.pkl").open("rb") as stream:
            saved = pickle.load(stream)
        fit = {key: saved[key] for key in ("fit_count", "close_samples", "pool_samples", "fit_seconds",
                                           "fitted_kernel", "warnings")}
        if fit["fit_count"] != len(y):
            raise ValueError("saved v2 model does not match the current training selection")
        print("  reusing saved v2 model", flush=True)
    else:
        fit = fit_and_save(X, y, close_train, len(near_index), fill)
    evaluate_offline(Xo, yo, io, Xs, ys, is_, Xc, yc, ic, close_scenarios, old_split, speed_splits,
                     close_train, close_test, fit)


def fit_and_save(X, y, close_train, close_count: int, fill: int) -> dict:
    model = create_model(X.shape[1], TrainingConfig(optimizer_restarts=0, seed=SEED))
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    fit = {"fit_count": int(len(y)), "close_samples": int(close_count), "pool_samples": int(fill),
           "fit_seconds": time.perf_counter() - started, "fitted_kernel": str(model.named_steps["gp"].kernel_),
           "warnings": [str(w.message) for w in caught]}
    print(f"  done in {fit['fit_seconds']:.1f} s; warnings={len(fit['warnings'])}", flush=True)
    V2_DIR.mkdir(parents=True, exist_ok=False)
    with (V2_DIR / "model.pkl").open("wb") as stream:
        pickle.dump({"pipeline": model, "feature_names": HISTORY_ACTION_FEATURES, "target": "opponent_delta_d",
                     "label_horizon_seconds": 0.5, "history_segment_seconds": 0.2, "history_seconds": 0.4,
                     "model_kind": "interaction_history_action", "target_unit": "m", "curvature_spacing_m": 0.5,
                     "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                  "sklearn": sklearn.__version__},
                     "training": {"close_train_episodes": close_train, "gap_range": GAP_RANGE,
                                  "cap_close": CAP_CLOSE, "cap_total": CAP_TOTAL, "seed": SEED}, **fit}, stream)
    export(V2_DIR / "model.pkl", V2_COMPACT)
    return fit


def evaluate_offline(Xo, yo, io, Xs, ys, is_, Xc, yc, ic, close_scenarios, old_split, speed_splits,
                     close_train, close_test, fit) -> None:
    models = {"v1": InteractionGpModel.load(V1), "v2": InteractionGpModel.load(V2_COMPACT)}
    sets = {"close_test": (Xc[np.isin(ic, close_test)], yc[np.isin(ic, close_test)], ic[np.isin(ic, close_test)]),
            "old_heldout": (Xo[np.isin(io, list(old_split["validation"] | old_split["test"]))],
                            yo[np.isin(io, list(old_split["validation"] | old_split["test"]))], None),
            "speed06_test": (Xs[np.isin(is_, speed_splits["new_test"])], ys[np.isin(is_, speed_splits["new_test"])], None)}
    predictions = {(name, key): model.predict(data[0]) for name, model in models.items() for key, data in sets.items()}
    summary = {"fit": fit, "close_train": close_train, "close_test": close_test, "sets": {}, "in_range": {}}
    rows = []
    for key, (Xe, ye, ids) in sets.items():
        groups = {"all": np.ones(len(ye), bool)}
        if key == "close_test":
            groups.update({f"gap[{low:g},{high:g})": (Xe[:, GAP] >= low) & (Xe[:, GAP] < high) for low, high in GAP_BINS})
        summary["sets"][key] = {}
        for label, mask in groups.items():
            entry = {"zero": metrics(ye[mask], np.zeros(mask.sum()), np.ones(mask.sum()))}
            for name in models:
                mean, std = predictions[(name, key)]
                entry[name] = metrics(ye[mask], mean[mask], std[mask])
            summary["sets"][key][label] = entry
        for name, model in models.items():
            summary["in_range"].setdefault(name, {})[key] = float(np.mean([model.check(row) is None for row in Xe]))
        if ids is not None:
            for episode in sorted(set(ids)):
                for label, mask in groups.items():
                    selected = mask & (ids == episode)
                    if not selected.any():
                        continue
                    row = {"episode": episode, "group": label, "n": int(selected.sum()),
                           **{f"design_{k}": v for k, v in close_scenarios[episode]["design"].items()}}
                    for name in models:
                        mean, std = predictions[(name, key)]
                        m = metrics(ye[selected], mean[selected], std[selected])
                        row.update({f"{name}_mae": round(m["mae"], 5), f"{name}_cov": round(m["coverage_2std"], 4)})
                    row["zero_mae"] = round(float(np.mean(np.abs(ye[selected]))), 5)
                    rows.append(row)
    (HERE / "offline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
                                      encoding="utf-8")
    with (HERE / "offline_per_episode.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_tables(summary)
    plot(summary)


def print_tables(summary: dict) -> None:
    print("\n| 資料 | 分組 | n | 基準 MAE | v1 MAE | v2 MAE | v1 涵蓋 | v2 涵蓋 | v1 寬度 | v2 寬度 |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, groups in summary["sets"].items():
        for label, entry in groups.items():
            if entry["v1"].get("n", 0) == 0:
                continue
            v1, v2 = entry["v1"], entry["v2"]
            print(f"| {key} | {label} | {v1['n']} | {entry['zero']['mae']:.4f} | {v1['mae']:.4f} | {v2['mae']:.4f} | "
                  f"{v1['coverage_2std']:.1%} | {v2['coverage_2std']:.1%} | "
                  f"{v1['mean_interval_width_m']:.3f} | {v2['mean_interval_width_m']:.3f} |")
    print("\n輸入落在模型訓練範圍內的比例：", json.dumps(summary["in_range"], indent=2))


def plot(summary: dict) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    surface, ink, muted, grid = "#fcfcfb", "#1f1f1d", "#6b6a63", "#e4e3de"
    series = (("zero", "Baseline: front keeps d", "#b9b8b0"), ("v1", "v1 (M1-30)", "#eb6834"),
              ("v2", "v2 (+ close range)", "#1baf7a"))
    groups = [label for label in summary["sets"]["close_test"] if label != "all"]
    figure = Figure(figsize=(11, 4), layout="constrained", facecolor=surface)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    x = np.arange(len(groups))
    for axis, metric, label in ((axes[0], "mae", "MAE of front Δd in 0.5 s (m)"),
                                (axes[1], "coverage_2std", "±2σ coverage")):
        axis.set_facecolor(surface)
        axis.grid(True, axis="y", color=grid, linewidth=0.8)
        axis.tick_params(colors=muted)
        for spine in axis.spines.values():
            spine.set_color(grid)
        shown = [item for item in series if not (metric == "coverage_2std" and item[0] == "zero")]
        width = 0.8 / len(shown)
        for offset, (key, name, color) in enumerate(shown):
            values = [summary["sets"]["close_test"][group][key][metric] for group in groups]
            axis.bar(x + (offset - (len(shown) - 1) / 2) * width, values, width * 0.9, color=color, label=name)
        axis.set_xticks(x, [group.replace("gap", "gap ") + " m" for group in groups], color=ink)
        axis.set_xlabel("Gap = front s − attacker s", color=ink)
        axis.set_ylabel(label, color=ink)
    axes[1].axhline(0.95, color=muted, linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1.05)
    for axis in axes:
        axis.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=ink)
    # 上方留給圖例
    tallest = max(summary["sets"]["close_test"][group][key]["mae"] for group in groups for key in ("zero", "v1", "v2"))
    axes[0].set_ylim(0, tallest * 1.4)
    axes[1].set_ylim(0, 1.25)
    figure.suptitle("Close-range test episodes: prediction error and interval coverage by gap", color=ink, fontsize=11)
    figure.savefig(HERE / "gap_bins.png", dpi=130, facecolor=surface)


if __name__ == "__main__":
    main()
