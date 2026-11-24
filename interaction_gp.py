"""試探互動：收集 probe_follow 對 defend 的資料、訓練 0.5 s GP、離線評估。"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import time
from typing import Sequence
import warnings

import numpy as np
import sklearn

from core.track_geometry import track_length
from generate_training_data import _demo_settings
from sim.data_generation import EpisodeScenario, VehicleInitialCondition, run_episode
from strategy.assignment import StrategyAssignment
from strategy.probe_following import ProbeFollowingConfig
from training_data import TrainingDataCollector
from training_data.interaction import (
    ACTION_FEATURES, FEATURES, HISTORY_ACTION_FEATURES, HISTORY_FEATURES, HISTORY_SEGMENT_SECONDS,
    HISTORY_TERMS, build_interaction_dataset,
)
from training_data.io import export_dataset, load_dataset, project_path
from training_data.schema import SamplingConfig, VehiclePair
from train_gp import TrainingConfig, create_model, prediction_errors

SUPPORTED_FEATURES = (FEATURES, ACTION_FEATURES, HISTORY_FEATURES, HISTORY_ACTION_FEATURES)
# 收集時隨機化試探：每場先左或先右，左右交替，每次幅度與整場保持時間隨機。
PROBE_AMPLITUDES = (0.3, 0.45, 0.6)   # 側向試探幅度候選，m
PROBE_HOLD_RANGE = (1.0, 2.5)         # 整場 hold_seconds 的均勻分布範圍，s
PROBE_SIDE_COUNT = 4                  # 一個序列內的側向試探次數，每次之後回中
MOVING_THRESHOLD = 0.02               # 評估子集：後車最近一段 |Δd| 超過此值視為橫移中，m


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def random_probe_config(base: ProbeFollowingConfig, rng: np.random.Generator) -> ProbeFollowingConfig:
    """產生一場的試探序列與保持時間；其他試探跟車參數沿用 base。"""
    first = 1.0 if rng.random() < 0.5 else -1.0
    sequence: list[float] = []
    for number in range(PROBE_SIDE_COUNT):
        side = first if number % 2 == 0 else -first
        sequence += [side * float(rng.choice(PROBE_AMPLITUDES)), 0.0]
    return replace(base, probe_sequence=tuple(sequence),
                   hold_seconds=float(rng.uniform(*PROBE_HOLD_RANGE)))


def collect(output: Path, episodes: int, seconds: float, seed: int,
            feature_set: str = "basic", randomize_probes: bool = False) -> None:
    import main
    if episodes < 5:
        raise ValueError("use at least five episodes for train/validation/test")
    if feature_set not in ("basic", "history"):
        raise ValueError("feature_set must be 'basic' or 'history'")
    settings, _, _ = _demo_settings(False)
    settings = replace(settings, track=main.create_demo_track("rounded_rectangle"))
    dt = settings.model_config.time_step
    if not np.isclose(round(0.5 / dt) * dt, 0.5):
        raise ValueError("time step must divide 0.5 seconds")
    rng = np.random.default_rng(seed)
    # 獨立的試探亂數，原本的起點、車距、速度抽樣在相同 seed 下不變。
    probe_rng = np.random.default_rng([seed, 1])
    collector = TrainingDataCollector(dt)
    actions, scenarios = [], []
    for number in range(episodes):
        # 各場獨立初始狀態，固定防守配置；前半樣本涵蓋移動中起步。
        origin = float(rng.uniform(0, track_length(settings.track)))
        gap, speed = float(rng.uniform(2, 5)), float(rng.uniform(3, 7))
        initial = speed if number % 2 else 0.0
        episode_settings = settings
        if randomize_probes:
            episode_settings = replace(settings, probe_following_config=random_probe_config(
                settings.probe_following_config, probe_rng))
        probe = episode_settings.probe_following_config
        scenario = EpisodeScenario(f"interaction-{number:03d}", (
            VehicleInitialCondition("rear", origin, 0.0, initial, speed + 2),
            VehicleInitialCondition("front", origin + gap, 0.0, initial, speed),
        ), {"rear": StrategyAssignment("probe_follow"),
            "front": StrategyAssignment("defend", target_vehicle_id="rear")})
        scenarios.append({"id": scenario.episode_id, "vehicles": [asdict(v) for v in scenario.vehicles],
                          "probe_sequence": list(probe.sequence), "hold_seconds": probe.hold_seconds})

        def observe(step, observation, decision, command):
            if observation.vehicle_id != "rear":
                return
            actions.append({"episode_id": scenario.episode_id, "step_index": step, "time_s": step * dt,
                            "mode": decision.mode, "target_d": decision.target_d,
                            "speed_limit": decision.speed_limit,
                            "effective_target_speed": min(speed + 2, decision.speed_limit) if decision.speed_limit is not None else speed + 2,
                            "steering": command.steering_angle, "acceleration": command.acceleration})

        run_episode(scenario, episode_settings, collector, seconds, step_observer=observe)
        print(f"collected {number + 1}/{episodes}", flush=True)
    sampling = SamplingConfig(dt, round(0.5 / dt), max(1, round(0.1 / dt)))
    targets = None
    if feature_set == "history":
        # 歷史版同時保存後車當下實際採用的目標，訓練時可用 --feature-subset 決定是否使用。
        targets = {(row["episode_id"], row["step_index"]): row["target_d"] for row in actions}
    dataset = build_interaction_dataset(collector.episodes, settings.track, VehiclePair("rear", "front"),
                                        sampling, targets, feature_set=feature_set)
    metadata = {"seed": seed, "seconds": seconds, "scenarios": scenarios, "feature_set": feature_set,
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "worktree": "含未提交修改",
                "probe_order": ("randomized per episode" if randomize_probes
                                else "left, center, right, center"),
                "configs": {name: asdict(getattr(settings, name)) for name in (
                    "model_config", "geometry", "controller_config", "probe_following_config", "defense_config", "offset_path_config")}}
    if feature_set == "history":
        metadata["history_segment_seconds"] = HISTORY_SEGMENT_SECONDS
    if randomize_probes:
        metadata["probe_randomization"] = {
            "amplitudes_m": PROBE_AMPLITUDES, "hold_seconds_range": PROBE_HOLD_RANGE,
            "side_count": PROBE_SIDE_COUNT, "rng": "numpy default_rng([seed, 1])",
            "note": "每場隨機先左或先右，左右交替；每次側向幅度獨立抽樣；hold_seconds 整場相同。"}
    export_dataset(dataset, collector.episodes, settings.track, output, metadata)
    write_csv(output / "actions.csv", actions)
    print(f"Collected X={dataset.X.shape}; skipped={dict(dataset.skipped)}", flush=True)


def split_groups(ids: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    groups = np.unique(ids)
    if len(groups) < 5:
        raise ValueError("at least five valid episodes required")
    np.random.default_rng(seed).shuffle(groups)
    count = max(1, round(len(groups) * 0.2))
    return {name: np.flatnonzero(np.isin(ids, selected)) for name, selected in (
        ("test", groups[:count]), ("validation", groups[count:2 * count]), ("train", groups[2 * count:]))}


def select_features(
    available: tuple[str, ...], subset: Sequence[str] | None,
) -> tuple[tuple[str, ...], list[int]]:
    """依名稱選取模型輸入欄；None 使用資料中的全部欄位。"""
    if subset is None:
        return available, list(range(len(available)))
    names = tuple(subset)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature subset must be non-empty and contain unique names")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"unknown features {unknown}; available: {list(available)}")
    return names, [available.index(name) for name in names]


def lateral_baseline(X: np.ndarray, column: dict[str, int], segment_seconds: float) -> np.ndarray:
    """延續前車當下橫向速度 0.5 s 的基準；歷史資料改用最近一段的差分速度。"""
    if "front_lateral_speed" in column:
        return X[:, column["front_lateral_speed"]] * 0.5
    return X[:, column["front_dd_late"]] * (0.5 / segment_seconds)


def history_metadata(names: tuple[str, ...], segment_seconds: float) -> dict[str, float]:
    """模型實際用到的歷史長度，供之後線上組特徵時核對。"""
    if any(name in HISTORY_TERMS for name in names):
        return {"history_seconds": 2 * segment_seconds, "history_segment_seconds": segment_seconds}
    return {"history_seconds": 0.1 if "front_lateral_speed" in names else 0.0}


def action_events(data: Path, actions: dict) -> list[dict]:
    """從原始狀態與決策還原目標切換、首次到達及不可行事件。"""
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    rear_id = metadata.get("run", {}).get("pair", {}).get("ego_id", "rear")
    with (data / "raw_states.csv").open(encoding="utf-8") as stream:
        positions = {(row["episode_id"], int(row["step_index"])): float(row["d"])
                     for row in csv.DictReader(stream) if row["vehicle_id"] == rear_id}
    tolerance = metadata.get("run", {}).get("configs", {}).get("probe_following_config", {}).get("lateral_tolerance", 0.1)
    previous, arrived, events = {}, {}, []
    for key, row in actions.items():
        episode = key[0]
        target = float(row["target_d"])
        labels = []
        if previous.get(episode) != target:
            labels.append("target_changed")
            arrived[episode] = False
        if not arrived.get(episode, False) and abs(positions[key] - target) <= tolerance:
            labels.append("first_arrival")
            arrived[episode] = True
        if row["mode"] == "no_feasible_path":
            labels.append("no_feasible_path")
        for label in labels:
            events.append({"episode_id": episode, "time_s": row["time_s"], "event": label,
                           "target_d": target, "actual_d": positions[key]})
        previous[episode] = target
    return events


def train(data: Path, output: Path, seed: int, max_train: int,
          feature_subset: Sequence[str] | None = None) -> None:
    if max_train < 0:
        raise ValueError("max_train must be non-negative; 0 means all training samples")
    dataset = load_dataset(data)
    available = tuple(dataset.schema.feature_names)
    if available not in SUPPORTED_FEATURES or not np.isclose(dataset.sampling.label_horizon_seconds, 0.5):
        raise ValueError("expected interaction features (basic/history, with or without d_target) and 0.5 s horizon")
    names, columns = select_features(available, feature_subset)
    run_metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8")).get("run", {})
    segment = float(run_metadata.get("history_segment_seconds", HISTORY_SEGMENT_SECONDS))
    full, y = dataset.X, dataset.Y[:, 0]
    X = full[:, columns]
    # 基準與評估子集一律用資料中的全部欄位，讓不同輸入組合的子集完全相同。
    column = {name: index for index, name in enumerate(available)}
    lateral = lateral_baseline(full, column, segment)
    with (data / "actions.csv").open(encoding="utf-8") as stream:
        actions = {(row["episode_id"], int(row["step_index"])): row for row in csv.DictReader(stream)}
    modes = np.array([actions.get((sample.episode_id, sample.step_index), {}).get("mode", "unknown")
                      for sample in dataset.samples])
    ids = np.array([sample.episode_id for sample in dataset.samples])
    split = split_groups(ids, seed)
    fit = split["train"]
    if max_train and len(fit) > max_train:
        fit = np.sort(np.random.default_rng(seed).choice(fit, max_train, replace=False))
    model = create_model(len(names), TrainingConfig(optimizer_restarts=0, seed=seed))
    print(f"GP fit: {len(fit)} samples, {len(names)} features; validation/test excluded", flush=True)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X[fit], y[fit])
    report = {"horizon_seconds": 0.5, "features": names, "dataset_features": available,
              "feature_subset": None if feature_subset is None else list(names), "seed": seed,
              "fit_seconds": time.perf_counter() - started,
              "executed_at_utc": datetime.now(timezone.utc).isoformat(),
              "samples_sha256": hashlib.sha256((data / "samples.csv").read_bytes()).hexdigest(),
              "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
              "data_directory": str(data), "versions": {"python": platform.python_version(),
                  "numpy": np.__version__, "sklearn": sklearn.__version__},
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "worktree": "含未提交修改", "training_policy": "fixed hyperparameter setup; validation not used for tuning",
              "sample_count": len(y), "skipped": dict(dataset.skipped),
              "mode_counts": {mode: int(np.sum(modes == mode)) for mode in np.unique(modes)},
              "feature_ranges": {name: [float(X[:, i].min()), float(X[:, i].max())] for i, name in enumerate(names)},
              "fit_count": len(fit), "fit_sample_indices": fit.tolist(),
              "fitted_kernel": str(model.named_steps["gp"].kernel_),
              "warnings": [str(w.message) for w in caught], "splits": {}}
    if "d_target" in names:
        values, counts = np.unique(X[:, names.index("d_target")], return_counts=True)
        report["d_target_counts"] = {str(float(value)): int(count) for value, count in zip(values, counts)}
        report["action_semantics"] = "當下實際採用的候選目標；未執行候選無反事實答案；未固定未來 0.5 s 的目標。"
    if "front_dd_late" in column and "front_lateral_speed" not in column:
        report["lateral_baseline"] = f"front_dd_late / {segment:g} s × 0.5 s"
    predictions = []
    for name in ("validation", "test"):
        index = split[name]
        mean, std = model.predict(X[index], return_std=True)
        actual = y[index]
        metrics = {"gp": prediction_errors(actual, mean),
                   "zero": prediction_errors(actual, np.zeros_like(actual)),
                   "training_mean": prediction_errors(actual, np.full_like(actual, np.mean(y[fit]))),
                   "constant_lateral_speed": prediction_errors(actual, lateral[index])}
        curvature, rear_d = full[index, column["front_curvature"]], full[index, column["rear_d"]]
        subsets = {"straight": np.abs(curvature) < 0.02, "curve": np.abs(curvature) >= 0.02,
                   "rear_left": rear_d > 0.2, "rear_right": rear_d < -0.2,
                   "rear_center": np.abs(rear_d) <= 0.2}
        if "rear_dd_late" in column:
            moving = np.abs(full[index, column["rear_dd_late"]]) > MOVING_THRESHOLD
            subsets.update({"rear_moving": moving, "rear_holding": ~moving})
        subsets.update({mode: modes[index] == mode for mode in np.unique(modes)})
        report["splits"][name] = {"episodes": sorted(set(ids[index])), "count": len(index), "metrics": metrics,
            "coverage_2std": float(np.mean(np.abs(actual - mean) <= 2 * std)),
            "subsets": {key: {"count": int(mask.sum()), "gp": prediction_errors(actual[mask], mean[mask])}
                        for key, mask in subsets.items() if mask.any()}}
        for i, mu, sigma in zip(index, mean, std):
            predictions.append({"split": name, "episode_id": ids[i], "time_s": dataset.samples[i].time_seconds,
                                "actual": y[i], "mean": mu, "std": sigma,
                                "constant_lateral_speed": lateral[i],
                                **{feature: float(value) for feature, value in zip(names, X[i])}})
    report["splits"]["train"] = {"episodes": sorted(set(ids[split["train"]])), "count": len(split["train"])}
    history_kind = available in (HISTORY_FEATURES, HISTORY_ACTION_FEATURES)
    model_kind = ("interaction_history" if history_kind else "interaction") + ("_action" if "d_target" in names else "")
    units = {spec.name: spec.unit for spec in dataset.feature_specs}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "model.pkl").open("wb") as stream:
        pickle.dump({"pipeline": model, "feature_names": names, "target": "opponent_delta_d",
                     "label_horizon_seconds": 0.5, "model_kind": model_kind,
                     "versions": report["versions"], "feature_units": [units[name] for name in names],
                     "target_unit": "m", **history_metadata(names, segment), "curvature_spacing_m": 0.5}, stream)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "predictions.csv", predictions)
    events = action_events(data, actions)
    if events:
        write_csv(output / "events.csv", events)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(6, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    rows = [row for row in predictions if row["split"] == "test"]
    axis.scatter([row["actual"] for row in rows], [row["mean"] for row in rows], s=10)
    limits = [min(row["actual"] for row in rows), max(row["actual"] for row in rows)]
    axis.plot(limits, limits, "--", color="gray")
    axis.set(xlabel="Actual delta d (m)", ylabel="GP mean (m)",
             title=f"Held-out episodes, horizon 0.5 s, {len(names)} inputs")
    figure.savefig(output / "evaluation.png", dpi=130)
    print(json.dumps(report["splits"]["test"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "train"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--data")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=800, help="0 使用全部訓練樣本")
    parser.add_argument("--features", choices=("basic", "history"), default="basic",
                        help="collect：basic 為原七輸入；history 為雙車狀態＋0.4 s 互動歷史＋d_target")
    parser.add_argument("--randomize-probes", action="store_true",
                        help="collect：每場隨機試探順序、幅度與保持時間")
    parser.add_argument("--feature-subset", nargs="+", metavar="NAME",
                        help="train：只用這些輸入欄位（依名稱），例如消融比較")
    args = parser.parse_args()
    output = project_path(args.output)
    if output.exists():
        parser.error("output already exists")
    os.environ.setdefault("MPLCONFIGDIR", str(project_path(".cache/matplotlib")))
    if args.mode == "collect":
        collect(output, args.episodes, args.seconds, args.seed, args.features, args.randomize_probes)
    else:
        if args.data is None:
            parser.error("train requires --data")
        train(project_path(args.data), output, args.seed, args.max_train, args.feature_subset)
