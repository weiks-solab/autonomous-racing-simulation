"""論文格式 GP 第一版：輸入論文 z_k（10 維）＋v2 歷史 4 欄，答案為前車 0.1 s 後的 [Δs, Δd, Δe_ψ, Δv]。

不重新模擬：由三份既有原始紀錄（同一組 defend 設定）重建資料。場次切分與訓練樣本選取規則完全沿用 v2
（reports/experiments/2026-11-24_close_range_gp/train_compare.py），kernel 也同 v2，只改輸入與輸出。
4 個答案各擬合一個獨立 GP，使用同一批訓練樣本。完整 pkl 與逐筆預測存 models/gp/（.gitignore）。

執行：python reports/experiments/2026-11-25_paper_format_gp/train_eval.py
模型已存在時沿用（核對訓練選取一致），只重做評估。
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))

import numpy as np
import sklearn

from interaction_gp import split_groups
from strategy.interaction_features import (
    CURVATURE_LOOKAHEAD, PAPER_FEATURE_UNITS, PAPER_FEATURES, PAPER_TARGET_UNITS, PAPER_TARGETS,
)
from train_gp import TrainingConfig, create_model, prediction_errors
from training_data.interaction import HISTORY_SEGMENT_SECONDS, build_paper_dataset
from training_data.io import load_recordings
from training_data.schema import SamplingConfig, VehiclePair

HERE = Path(__file__).resolve().parent
SOURCES = {
    "old": PROJECT_ROOT / "data/training/history_gp_20261124",
    "speed06": PROJECT_ROOT / "data/training/history_speed06_20261124",
    "close": PROJECT_ROOT / "data/training/close_range_20261124",
}
SPEED_SUMMARY = PROJECT_ROOT / "reports/experiments/2026-11-24_speed_amplitude_probe/summary.json"
MODEL_DIR = PROJECT_ROOT / "models/gp/paper_format_20261125"
CAP_TOTAL, CAP_CLOSE, SEED = 5000, 3000, 42           # 同 v2
GAP_RANGE = (-2.0, 3.0)                               # 同 v2：近距離訓練樣本的車距範圍
GAP_BINS = ((-2.0, 0.0), (0.0, 0.8), (0.8, 1.6), (1.6, 3.0))
PAPER_GAP = (0.0, 2.0)                                # 論文評估：0 ≤ s_ov − s_ev ≤ 2 m
STRAIGHT_CURVATURE = 0.02                             # |曲率| 小於此值視為直線，同既有報告
SAMPLING = SamplingConfig(0.02, 5, 5)                 # 每 0.1 s 取樣，答案為 0.1 s 後
PAIR = VehiclePair("rear", "front")
COLUMN = {name: index for index, name in enumerate(PAPER_FEATURES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(directory: Path):
    track, sampling, episodes = load_recordings(directory)
    if not np.isclose(sampling.time_step, SAMPLING.time_step):
        raise ValueError(f"{directory} time step {sampling.time_step} differs from {SAMPLING.time_step}")
    dataset = build_paper_dataset(episodes, track, PAIR, SAMPLING)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    run = metadata["run"]
    return {"track": track, "X": dataset.X, "Y": dataset.Y,
            "ids": np.array([sample.episode_id for sample in dataset.samples]),
            "steps": np.array([sample.step_index for sample in dataset.samples]),
            "times": np.array([sample.time_seconds for sample in dataset.samples]),
            "skipped": dict(dataset.skipped), "episodes": len(episodes),
            "scenarios": {item["id"]: item for item in run["scenarios"]},
            "defense_config": run["configs"]["defense_config"],
            "raw_sha256": sha256(directory / "raw_states.csv")}


def close_split(scenarios: dict) -> tuple[list[str], list[str]]:
    """同 v2：每個「速度區間 × 偏離權重」組合依場次編號，最後一場測試，其餘訓練。"""
    cells = {}
    for episode in sorted(scenarios):
        design = scenarios[episode]["design"]
        cells.setdefault((tuple(design["speed_stratum"]), design["deviation_weight"]), []).append(episode)
    if any(len(items) < 2 for items in cells.values()):
        raise ValueError("each design cell needs at least two episodes")
    return ([episode for items in cells.values() for episode in items[:-1]],
            [items[-1] for items in cells.values()])


def check_sources(data: dict) -> None:
    """三份資料須為同一賽道與同一組防守設定，否則不能合併成「單一防守策略」的模型。"""
    first = next(iter(data.values()))
    for name, item in data.items():
        same_track = (np.array_equal(item["track"].centerline, first["track"].centerline)
                      and item["track"].closed == first["track"].closed and item["track"].width == first["track"].width)
        if not same_track:
            raise ValueError(f"{name} uses a different track")
        if item["defense_config"] != first["defense_config"]:
            raise ValueError(f"{name} uses a different defense configuration")


def constant_velocity(X: np.ndarray) -> np.ndarray:
    """等速外推基準：Δs=v·Δt，Δd 延續最近 0.2 s 的橫移速度，航向誤差與車速不變。"""
    horizon = SAMPLING.label_horizon_seconds
    return np.column_stack((X[:, COLUMN["front_speed"]] * horizon,
                            X[:, COLUMN["front_dd_late"]] * horizon / HISTORY_SEGMENT_SECONDS,
                            np.zeros(len(X)), np.zeros(len(X))))


def metrics(actual, mean, std=None) -> dict:
    if len(actual) == 0:
        return {"n": 0}
    result = {**prediction_errors(actual, mean), "n": int(len(actual))}
    if std is not None:
        result.update({"coverage_2std": float(np.mean(np.abs(mean - actual) <= 2 * std)),
                       "mean_interval_width": float(np.mean(4 * std))})
    return result


def select_training(data: dict, old_split: dict, speed_splits: dict, close_train: list[str]):
    """同 v2 的抽樣順序：先取近距離樣本（上限 3000），再由 v1 訓練場補到 5000。"""
    rng = np.random.default_rng(SEED)
    old, speed, close = data["old"], data["speed06"], data["close"]
    pool_X = np.vstack((old["X"][np.isin(old["ids"], old_split["train"])],
                        speed["X"][np.isin(speed["ids"], speed_splits["new_train"])]))
    pool_Y = np.vstack((old["Y"][np.isin(old["ids"], old_split["train"])],
                        speed["Y"][np.isin(speed["ids"], speed_splits["new_train"])]))
    gap = close["X"][:, COLUMN["gap"]]
    near_index = np.flatnonzero(np.isin(close["ids"], close_train) & (gap >= GAP_RANGE[0]) & (gap <= GAP_RANGE[1]))
    if len(near_index) > CAP_CLOSE:
        near_index = np.sort(rng.choice(near_index, CAP_CLOSE, replace=False))
    fill = min(len(pool_Y), CAP_TOTAL - len(near_index))
    pool_index = np.sort(rng.choice(len(pool_Y), fill, replace=False))
    X = np.vstack((close["X"][near_index], pool_X[pool_index]))
    Y = np.vstack((close["Y"][near_index], pool_Y[pool_index]))
    return X, Y, {"close_samples": int(len(near_index)), "pool_samples": int(fill), "fit_count": int(len(Y))}


def fit_models(X: np.ndarray, Y: np.ndarray, selection: dict, training: dict) -> dict:
    pipelines, fits = {}, {}
    for column, target in enumerate(PAPER_TARGETS):
        model = create_model(X.shape[1], TrainingConfig(optimizer_restarts=0, seed=SEED))
        print(f"fit {target}: {len(Y)} samples x {X.shape[1]} features", flush=True)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, Y[:, column])
        fits[target] = {"fit_seconds": time.perf_counter() - started,
                        "fitted_kernel": str(model.named_steps["gp"].kernel_),
                        "warnings": [str(w.message) for w in caught]}
        print(f"  done in {fits[target]['fit_seconds']:.1f} s; warnings={len(fits[target]['warnings'])}", flush=True)
        pipelines[target] = model
    bundle = {"pipelines": pipelines, "feature_names": PAPER_FEATURES, "feature_units": PAPER_FEATURE_UNITS,
              "target_names": PAPER_TARGETS, "target_units": PAPER_TARGET_UNITS,
              "label_horizon_seconds": SAMPLING.label_horizon_seconds,
              "history_segment_seconds": HISTORY_SEGMENT_SECONDS, "history_seconds": 2 * HISTORY_SEGMENT_SECONDS,
              "curvature_lookahead_m": CURVATURE_LOOKAHEAD, "curvature_spacing_m": 0.5,
              "model_kind": "paper_format_one_step",
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
              "training": training, **selection, "fits": fits}
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    with (MODEL_DIR / "model.pkl").open("wb") as stream:
        pickle.dump(bundle, stream)
    return bundle


def load_or_fit(X: np.ndarray, Y: np.ndarray, selection: dict, training: dict) -> dict:
    path = MODEL_DIR / "model.pkl"
    if not path.is_file():
        return fit_models(X, Y, selection, training)
    with path.open("rb") as stream:
        bundle = pickle.load(stream)   # 只載入本腳本產生的模型
    if bundle["fit_count"] != selection["fit_count"] or bundle["training"] != training:
        raise ValueError("saved model does not match the current training selection")
    for column, target in enumerate(PAPER_TARGETS):
        gp = bundle["pipelines"][target].named_steps["gp"]
        if not np.allclose(gp.y_train_ * gp._y_train_std + gp._y_train_mean, Y[:, column]):
            raise ValueError(f"saved {target} model was fitted on different samples")
    print("reusing saved model", flush=True)
    return bundle


def predict(bundle: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means, stds = zip(*(bundle["pipelines"][target].predict(X, return_std=True) for target in PAPER_TARGETS))
    return np.column_stack(means), np.column_stack(stds)


def evaluate(bundle: dict, data: dict, sets: dict) -> tuple[dict, list[dict], list[dict]]:
    summary, per_episode, rows = {}, [], []
    for key, (source, episodes) in sets.items():
        item = data[source]
        mask = np.isin(item["ids"], episodes)
        X, Y, ids = item["X"][mask], item["Y"][mask], item["ids"][mask]
        mean, std = predict(bundle, X)
        baseline = constant_velocity(X)
        curvature = np.abs(X[:, COLUMN["front_curvature_0"]])
        groups = {"all": np.ones(len(Y), bool), "straight": curvature < STRAIGHT_CURVATURE,
                  "curve": curvature >= STRAIGHT_CURVATURE}
        gap = X[:, COLUMN["gap"]]
        # 近距離訓練樣本只有 −2～3 m；攻方超過後（gap < −2）在訓練範圍外。
        groups.update({"gap<-2": gap < GAP_RANGE[0], "gap[-2,3)": (gap >= GAP_RANGE[0]) & (gap < GAP_RANGE[1]),
                       "gap>=3": gap >= GAP_RANGE[1]})
        if source == "close":
            groups.update({f"gap[{low:g},{high:g})": (gap >= low) & (gap < high) for low, high in GAP_BINS})
            groups[f"paper_gap[{PAPER_GAP[0]:g},{PAPER_GAP[1]:g}]"] = (gap >= PAPER_GAP[0]) & (gap <= PAPER_GAP[1])
        summary[key] = {"episodes": sorted(set(ids)), "groups": {}}
        for label, selected in groups.items():
            summary[key]["groups"][label] = {target: {
                "gp": metrics(Y[selected, c], mean[selected, c], std[selected, c]),
                "zero": metrics(Y[selected, c], np.zeros(selected.sum())),
                "constant_velocity": metrics(Y[selected, c], baseline[selected, c]),
            } for c, target in enumerate(PAPER_TARGETS)}
        for episode in sorted(set(ids)):
            selected = ids == episode
            row = {"set": key, "episode": episode, "n": int(selected.sum())}
            for c, target in enumerate(PAPER_TARGETS):
                gp = metrics(Y[selected, c], mean[selected, c], std[selected, c])
                row.update({f"{target}_gp_mae": round(gp["mae"], 6),
                            f"{target}_cv_mae": round(prediction_errors(Y[selected, c], baseline[selected, c])["mae"], 6),
                            f"{target}_gp_cov": round(gp["coverage_2std"], 4)})
            per_episode.append(row)
        steps, times = item["steps"][mask], item["times"][mask]
        for i in range(len(Y)):
            row = {"set": key, "episode": ids[i], "step_index": int(steps[i]), "time_s": float(times[i]),
                   **{name: float(X[i, j]) for j, name in enumerate(PAPER_FEATURES)}}
            for c, target in enumerate(PAPER_TARGETS):
                row.update({f"{target}_actual": float(Y[i, c]), f"{target}_mean": float(mean[i, c]),
                            f"{target}_std": float(std[i, c]), f"{target}_cv": float(baseline[i, c])})
            rows.append(row)
    return summary, per_episode, rows


def timing(bundle: dict, X: np.ndarray) -> dict:
    """之後多步取樣推演（Q=25）的參考：4 個 GP 各預測 1 列與 25 列（含 std）的時間，取 5 次中位數。"""
    result = {}
    for count in (1, 25):
        rows = X[:count]
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            predict(bundle, rows)
            samples.append(time.perf_counter() - started)
        result[f"rows_{count}_ms"] = 1000 * float(np.median(samples))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_tables(summary: dict) -> None:
    for target in PAPER_TARGETS:
        print(f"\n### {target}\n")
        print("| 測試集 | 分組 | n | 全 0 MAE | 等速 MAE | GP MAE | GP RMSE | ±2σ 涵蓋 | 平均 4σ 寬 |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for key, item in summary.items():
            for label, entry in item["groups"].items():
                gp = entry[target]["gp"]
                if gp["n"] == 0:
                    continue
                print(f"| {key} | {label} | {gp['n']} | {entry[target]['zero']['mae']:.5f} | "
                      f"{entry[target]['constant_velocity']['mae']:.5f} | {gp['mae']:.5f} | {gp['rmse']:.5f} | "
                      f"{gp['coverage_2std']:.1%} | {gp['mean_interval_width']:.5f} |")


def plot(rows: list[dict]) -> None:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
    # 類別色依參考調色盤固定順序取前三個（三個以內可兩兩區分）。
    series = (("close_test", "Close-range test", "#2a78d6"), ("old_heldout", "Probe held-out", "#eb6834"),
              ("speed06_test", "Probe speed test", "#1baf7a"))
    labels = {"front_delta_s": "Δs (m)", "front_delta_d": "Δd (m)",
              "front_delta_heading_error": "Δ heading error (rad)", "front_delta_speed": "Δv (m/s)"}
    figure = Figure(figsize=(10, 8.5), layout="constrained", facecolor=surface)
    FigureCanvasAgg(figure)
    for axis, target in zip(figure.subplots(2, 2).ravel(), PAPER_TARGETS):
        axis.set_facecolor(surface)
        axis.grid(True, color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=muted, labelcolor=ink)
        for spine in axis.spines.values():
            spine.set_color(grid)
        values = []
        for key, name, color in series:
            selected = [row for row in rows if row["set"] == key]
            actual = [row[f"{target}_actual"] for row in selected]
            predicted = [row[f"{target}_mean"] for row in selected]
            values += actual + predicted
            axis.scatter(actual, predicted, s=10, color=color, alpha=0.45, linewidths=0, label=name)
        limits = (min(values), max(values))
        axis.plot(limits, limits, "--", color=muted, linewidth=1)
        axis.set_xlabel(f"Actual {labels[target]}", color=ink)
        axis.set_ylabel(f"GP mean {labels[target]}", color=ink)
        axis.set_title(labels[target], color=ink, fontsize=10)
    figure.legend(*figure.axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3,
                  frameon=False, labelcolor=ink, markerscale=2)
    figure.suptitle("Front vehicle one-step (0.1 s) prediction on held-out episodes, 14 inputs",
                    color=ink, fontsize=11)
    figure.savefig(HERE / "evaluation.png", dpi=130, facecolor=surface)


def main() -> None:
    data = {name: load_source(path) for name, path in SOURCES.items()}
    check_sources(data)
    for name, item in data.items():
        print(f"{name}: {len(item['ids'])} samples from {item['episodes']} episodes; skipped={item['skipped']}", flush=True)
    old_split = {name: sorted(set(data["old"]["ids"][index]))
                 for name, index in split_groups(data["old"]["ids"], SEED).items()}
    speed_summary = json.loads(SPEED_SUMMARY.read_text(encoding="utf-8"))["splits"]
    if old_split != {name: sorted(speed_summary["old"][name]) for name in old_split}:
        raise ValueError("old split differs from the split recorded for v1/v2")
    speed_splits = {name: speed_summary[name] for name in ("new_train", "new_test")}
    close_train, close_test = close_split(data["close"]["scenarios"])
    training = {"old_train": old_split["train"], "speed06_train": speed_splits["new_train"],
                "close_train": close_train, "gap_range": list(GAP_RANGE), "cap_close": CAP_CLOSE,
                "cap_total": CAP_TOTAL, "seed": SEED}
    X, Y, selection = select_training(data, old_split, speed_splits, close_train)
    print(f"fit set: {selection['close_samples']} close-range + {selection['pool_samples']} probe = "
          f"{selection['fit_count']} samples", flush=True)
    bundle = load_or_fit(X, Y, selection, training)
    sets = {"close_test": ("close", close_test),
            "old_heldout": ("old", old_split["validation"] + old_split["test"]),
            "speed06_test": ("speed06", speed_splits["new_test"])}
    summary, per_episode, rows = evaluate(bundle, data, sets)
    report = {
        "model": str(MODEL_DIR.relative_to(PROJECT_ROOT) / "model.pkl"),
        "model_sha256": sha256(MODEL_DIR / "model.pkl"),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "worktree": "含未提交修改",
        "features": list(PAPER_FEATURES), "feature_units": list(PAPER_FEATURE_UNITS),
        "targets": list(PAPER_TARGETS), "target_units": list(PAPER_TARGET_UNITS),
        "label_horizon_seconds": SAMPLING.label_horizon_seconds,
        "sample_interval_seconds": SAMPLING.sample_stride_steps * SAMPLING.time_step,
        "history_segment_seconds": HISTORY_SEGMENT_SECONDS, "curvature_lookahead_m": CURVATURE_LOOKAHEAD,
        "kernel_setup": "StandardScaler + Constant x ARD RBF + White, normalize_y, no optimizer restart (same as v2)",
        "baselines": {"zero": "all four deltas 0",
                      "constant_velocity": "ds=front_speed*0.1, dd=front_dd_late*0.1/0.2, dheading=0, dv=0"},
        "sources": {name: {"path": str(SOURCES[name].relative_to(PROJECT_ROOT)), "raw_states_sha256": item["raw_sha256"],
                           "episodes": item["episodes"], "samples": int(len(item["ids"])), "skipped": item["skipped"]}
                    for name, item in data.items()},
        "defense_config": data["close"]["defense_config"],
        "training": training, "selection": selection, "fits": bundle["fits"],
        "versions": bundle["versions"], "predict_timing": timing(bundle, X),
        "test_sets": summary,
    }
    (HERE / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(HERE / "per_episode.csv", per_episode)
    write_csv(MODEL_DIR / "predictions.csv", rows)
    print_tables(summary)
    print("\npredict timing:", report["predict_timing"])
    plot(rows)


if __name__ == "__main__":
    main()
