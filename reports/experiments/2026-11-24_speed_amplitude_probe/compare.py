"""前車速度 × ±0.6 m 試探新場次：比較 M0（現有模型）、M1-15、M1-30，先逐場再合併。

M0 不重訓；M1 以舊 12 場訓練＋新訓練 8 場（前 15 s 或完整 30 s）擬合，模型存在 models/gp/（.gitignore）。
"""

import csv
from dataclasses import dataclass
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

from interaction_gp import split_groups
from train_gp import TrainingConfig, create_model
from training_data.interaction import HISTORY_ACTION_FEATURES
from training_data.io import load_dataset

HERE = Path(__file__).resolve().parent
OLD = PROJECT_ROOT / "data/training/history_gp_20261124"
NEW = PROJECT_ROOT / "data/training/history_speed06_20261124"
M0_PATH = PROJECT_ROOT / "models/gp/history_20261124_c_action13/model.pkl"
M1_PATHS = {"M1-15": PROJECT_ROOT / "models/gp/speed06_20261124_c_m1_15",
            "M1-30": PROJECT_ROOT / "models/gp/speed06_20261124_c_m1_30"}
FIRST_SEGMENT_END = 14.5   # 前 15 s 模擬內、0.5 s 答案仍在 15 s 內的最後樣本時間


@dataclass(frozen=True)
class Data:
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    time: np.ndarray
    target: np.ndarray       # 後車當下目標 d
    scenarios: dict


def load(directory: Path) -> Data:
    dataset = load_dataset(directory)
    if tuple(dataset.schema.feature_names) != HISTORY_ACTION_FEATURES:
        raise ValueError(f"{directory} is not a 13-feature history dataset")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))["run"]
    X = dataset.X
    return Data(X, dataset.Y[:, 0], np.array([s.episode_id for s in dataset.samples]),
                np.array([s.time_seconds for s in dataset.samples]),
                X[:, HISTORY_ACTION_FEATURES.index("d_target")],
                {s["id"]: s for s in metadata["scenarios"]})


def metrics(actual: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    error = mean - actual
    return {"n": int(len(actual)), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "coverage_2std": float(np.mean(np.abs(error) <= 2 * std)),
            "mean_interval_width_m": float(np.mean(4 * std)), "mean_std_m": float(np.mean(std))}


def subsets(data: Data, mask: np.ndarray) -> dict[str, np.ndarray]:
    size = np.abs(data.target)
    return {"all": mask, "probe06": mask & np.isclose(size, 0.6),
            "probe03_045": mask & (np.isclose(size, 0.3) | np.isclose(size, 0.45)),
            "center": mask & np.isclose(size, 0.0)}


def fit(X: np.ndarray, y: np.ndarray) -> tuple[object, dict]:
    model = create_model(X.shape[1], TrainingConfig(optimizer_restarts=0, seed=42))
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    return model, {"fit_count": int(len(y)), "fit_seconds": time.perf_counter() - started,
                   "fitted_kernel": str(model.named_steps["gp"].kernel_),
                   "warnings": [str(w.message) for w in caught]}


def save_model(model, path: Path, info: dict, episodes: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump({"pipeline": model, "feature_names": HISTORY_ACTION_FEATURES, "target": "opponent_delta_d",
                     "label_horizon_seconds": 0.5, "model_kind": "interaction_history_action",
                     "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                  "sklearn": sklearn.__version__},
                     "target_unit": "m", "history_seconds": 0.4, "history_segment_seconds": 0.2,
                     "curvature_spacing_m": 0.5, "training_episodes": episodes, **info}, stream)


def main() -> None:
    old, new = load(OLD), load(NEW)
    old_split = {name: sorted(set(old.ids[index])) for name, index in split_groups(old.ids, 42).items()}
    # 新場次：每個速度區間（先左、靜止）＋（先右、移動）訓練，另外兩種組合測試。
    role = {}
    for episode, scenario in new.scenarios.items():
        design = scenario["design"]
        train = (design["first_side"] == "left") != design["moving_start"]
        role[episode] = "new_train" if train else "new_test"
    new_train = sorted(e for e, r in role.items() if r == "new_train")
    new_test = sorted(e for e, r in role.items() if r == "new_test")
    old_train_mask = np.isin(old.ids, old_split["train"])
    new_train_mask = np.isin(new.ids, new_train)
    first = new.time <= FIRST_SEGMENT_END + 1e-9

    with M0_PATH.open("rb") as stream:
        m0 = pickle.load(stream)
    if tuple(m0["feature_names"]) != HISTORY_ACTION_FEATURES:
        raise ValueError("M0 feature order differs from the dataset")
    models = {"M0": m0["pipeline"]}
    fits = {"M0": {"source": str(M0_PATH.relative_to(PROJECT_ROOT)), "fit_count": int(old_train_mask.sum())}}
    for name, mask in (("M1-15", new_train_mask & first), ("M1-30", new_train_mask)):
        X = np.vstack((old.X[old_train_mask], new.X[mask]))
        y = np.concatenate((old.y[old_train_mask], new.y[mask]))
        print(f"fitting {name}: {len(y)} samples ({int(old_train_mask.sum())} old + {int(mask.sum())} new)", flush=True)
        model, info = fit(X, y)
        save_model(model, M1_PATHS[name], info, old_split["train"] + new_train)
        models[name], fits[name] = model, {**info, "saved": str(M1_PATHS[name].relative_to(PROJECT_ROOT))}
        print(f"  done in {info['fit_seconds']:.1f} s; warnings={len(info['warnings'])}", flush=True)

    evaluations = {
        "new_test_0_15s": (new, np.isin(new.ids, new_test) & first),
        "new_test_15_30s": (new, np.isin(new.ids, new_test) & ~first),
        "old_heldout": (old, np.isin(old.ids, old_split["validation"] + old_split["test"])),
    }
    pooled, rows = {}, []
    for model_name, model in models.items():
        cache = {}
        for data_name, data in (("old", old), ("new", new)):
            cache[data_name] = model.predict(data.X, return_std=True)
        for eval_name, (data, mask) in evaluations.items():
            mean, std = cache["new" if data is new else "old"]
            pooled.setdefault(model_name, {})[eval_name] = {
                key: metrics(data.y[sub], mean[sub], std[sub]) for key, sub in subsets(data, mask).items() if sub.any()}
        # 逐場：新 16 場對 M0 都是未見過；M1 只報新測試場與舊保留場。
        for data_name, data in (("new", new), ("old", old)):
            mean, std = cache[data_name]
            for episode in sorted(set(data.ids)):
                if data is old and episode not in old_split["validation"] + old_split["test"]:
                    continue
                if data is new and model_name != "M0" and role[episode] != "new_test":
                    continue
                scenario = data.scenarios[episode]
                design = scenario.get("design", {})
                for segment, seg_mask in (("0_15s", data.time <= FIRST_SEGMENT_END + 1e-9),
                                          ("15_30s", data.time > FIRST_SEGMENT_END + 1e-9)):
                    for key, sub in subsets(data, (data.ids == episode) & seg_mask).items():
                        if not sub.any():
                            continue
                        rows.append({"model": model_name, "episode_id": episode,
                                     "role": role.get(episode, "old_validation" if episode in old_split["validation"] else "old_test"),
                                     "front_speed": round(scenario["vehicles"][1]["target_speed"], 4),
                                     "first_side": design.get("first_side", ""), "moving_start": design.get("moving_start", ""),
                                     "segment": segment, "subset": key,
                                     **{k: (round(v, 6) if isinstance(v, float) else v)
                                        for k, v in metrics(data.y[sub], mean[sub], std[sub]).items()}})

    with (HERE / "per_episode.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"splits": {"old": old_split, "new_train": new_train, "new_test": new_test},
               "fits": fits, "pooled": pooled, "first_segment_end_s": FIRST_SEGMENT_END,
               "new_sample_count": int(len(new.y)), "seed_split_old": 42}
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_tables(pooled, rows)
    plot(rows)


def print_tables(pooled: dict, rows: list[dict]) -> None:
    print("\n| 模型 | 評估集 | 子集 | n | MAE | RMSE | 涵蓋率 | ±2σ 平均寬度 |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for model_name, evals in pooled.items():
        for eval_name, groups in evals.items():
            for key in ("all", "probe06", "probe03_045", "center"):
                if key in groups:
                    m = groups[key]
                    print(f"| {model_name} | {eval_name} | {key} | {m['n']} | {m['mae']:.4f} | {m['rmse']:.4f} | "
                          f"{m['coverage_2std']:.1%} | {m['mean_interval_width_m']:.4f} |")
    print("\nM0 在新 16 場（0.6 m 試探樣本，0～30 s）依速度區間：每場涵蓋率與 RMSE")
    episodes = {}
    for row in rows:
        if row["model"] == "M0" and row["role"].startswith("new") and row["subset"] == "probe06":
            episodes.setdefault(row["episode_id"], []).append(row)
    for low in (3, 4, 5, 6):
        cells = []
        for episode, items in sorted(episodes.items()):
            speed = items[0]["front_speed"]
            if low <= speed < low + 1:
                n = sum(r["n"] for r in items)
                cov = sum(r["coverage_2std"] * r["n"] for r in items) / n
                rmse = np.sqrt(sum(r["rmse"] ** 2 * r["n"] for r in items) / n)
                cells.append(f"{episode[-3:]}(v={speed:.2f}, n={n}) cov={cov:.2f} rmse={rmse:.3f}")
        print(f"  [{low},{low + 1}) m/s: " + "; ".join(cells))


def episode_values(rows: list[dict], model: str, roles: set[str]) -> list[tuple[float, float, float]]:
    """把 0～15 s 與 15～30 s 合併成每場一點：(前車速度, 涵蓋率, RMSE)，只用 0.6 m 試探樣本。"""
    grouped = {}
    for row in rows:
        if row["model"] == model and row["role"] in roles and row["subset"] == "probe06":
            grouped.setdefault(row["episode_id"], []).append(row)
    points = []
    for items in grouped.values():
        n = sum(r["n"] for r in items)
        points.append((items[0]["front_speed"], sum(r["coverage_2std"] * r["n"] for r in items) / n,
                       float(np.sqrt(sum(r["rmse"] ** 2 * r["n"] for r in items) / n))))
    return sorted(points)


def plot(rows: list[dict]) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    surface, ink, muted, grid = "#fcfcfb", "#1f1f1d", "#6b6a63", "#e4e3de"
    # 圖上文字用英文，避免預設字型缺中文字形。
    series = (("M0: existing model, all 16 new episodes", "M0", {"new_train", "new_test"}, "#2a78d6", "o"),
              ("M1-30: retrained, 8 new test episodes", "M1-30", {"new_test"}, "#eb6834", "s"))
    figure = Figure(figsize=(9, 3.8), layout="constrained", facecolor=surface)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, index, label in ((axes[0], 1, "±2σ coverage"), (axes[1], 2, "RMSE (m)")):
        axis.set_facecolor(surface)
        for name, model, roles, color, marker in series:
            points = episode_values(rows, model, roles)
            axis.scatter([p[0] for p in points], [p[index] for p in points], s=64, color=color, marker=marker,
                         edgecolors=surface, linewidths=2, label=name, zorder=3)
        axis.set_xlabel("Front cruise speed (m/s)", color=ink)
        axis.set_ylabel(label, color=ink)
        axis.grid(True, color=grid, linewidth=0.8)
        axis.tick_params(colors=muted)
        for spine in axis.spines.values():
            spine.set_color(grid)
    axes[0].axhline(0.95, color=muted, linestyle="--", linewidth=1)
    axes[0].text(3.0, 0.955, "95%", color=muted, fontsize=9, va="bottom")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9, labelcolor=ink)
    figure.suptitle("Fixed 0.6 m probes: one point per episode (samples with |d_target| = 0.6 only)",
                    color=ink, fontsize=11)
    figure.savefig(HERE / "speed_effect.png", dpi=130, facecolor=surface)


if __name__ == "__main__":
    main()
