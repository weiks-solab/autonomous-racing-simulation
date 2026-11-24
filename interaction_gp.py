"""試探互動：收集 probe_follow 對 defend 的資料、訓練 0.5 s GP、離線評估。"""

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
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


def check_amplitudes(amplitudes: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in amplitudes)
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("probe amplitudes must be positive and finite")
    return values


def random_probe_config(
    base: ProbeFollowingConfig, rng: np.random.Generator,
    amplitudes: Sequence[float] = PROBE_AMPLITUDES, first_side: float | None = None,
) -> ProbeFollowingConfig:
    """產生一場的試探序列與保持時間；first_side 為 None 時隨機先左或先右（+1 左、-1 右）。"""
    choices = check_amplitudes(amplitudes)
    if first_side is None:
        first = 1.0 if rng.random() < 0.5 else -1.0
    elif first_side in (1.0, -1.0):
        first = float(first_side)
    else:
        raise ValueError("first_side must be 1, -1 or None")
    sequence: list[float] = []
    for number in range(PROBE_SIDE_COUNT):
        side = first if number % 2 == 0 else -first
        sequence += [side * float(rng.choice(choices)), 0.0]
    return replace(base, probe_sequence=tuple(sequence),
                   hold_seconds=float(rng.uniform(*PROBE_HOLD_RANGE)))


def speed_strata(edges: Sequence[float]) -> tuple[tuple[float, float], ...]:
    """前車巡航速度區間，例如邊界 3 4 5 6 7 得到四個 1 m/s 區間。"""
    values = [float(value) for value in edges]
    if (len(values) < 2 or any(not math.isfinite(value) or value <= 0.0 for value in values)
            or any(high <= low for low, high in zip(values, values[1:]))):
        raise ValueError("speed strata need at least two increasing positive edges")
    return tuple(zip(values[:-1], values[1:]))


def stratified_design(number: int, strata_count: int) -> tuple[int, float, bool]:
    """速度區間輪流分配；同區間依序為（先左、靜止）（先右、靜止）（先左、移動）（先右、移動）。"""
    if number < 0 or strata_count < 1:
        raise ValueError("episode number must be non-negative and strata_count positive")
    stratum, block = number % strata_count, number // strata_count
    return stratum, 1.0 if block % 2 == 0 else -1.0, (block // 2) % 2 == 1


@dataclass(frozen=True)
class AttackerCollection:
    """攻方實際超車收集：relative_overtake 對 defend，涵蓋貼近、並排與超越。"""

    deviation_weights: tuple[float, ...] = (0.0, 1.0, 3.0)   # 每場輪流使用，保守與積極的攻方都有
    speed_advantage: tuple[float, float] = (0.5, 1.5)        # 攻方巡航＝前車＋U(範圍)，m/s
    initial_gap: tuple[float, float] = (1.5, 4.0)            # 初始車距 U(範圍)，m

    def __post_init__(self) -> None:
        if not self.deviation_weights or any(not math.isfinite(w) or w < 0.0 for w in self.deviation_weights):
            raise ValueError("deviation weights must be non-negative and finite")
        for name in ("speed_advantage", "initial_gap"):
            low, high = getattr(self, name)
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low <= high):
                raise ValueError(f"{name} must be a finite range with 0 <= low <= high")


def collect(output: Path, episodes: int, seconds: float, seed: int,
            feature_set: str = "basic", randomize_probes: bool = False,
            strata: Sequence[tuple[float, float]] | None = None,
            probe_amplitudes: Sequence[float] = PROBE_AMPLITUDES,
            episode_prefix: str = "interaction",
            attacker: AttackerCollection | None = None) -> None:
    import main
    if episodes < 5:
        raise ValueError("use at least five episodes for train/validation/test")
    if feature_set not in ("basic", "history"):
        raise ValueError("feature_set must be 'basic' or 'history'")
    amplitudes = check_amplitudes(probe_amplitudes)
    if amplitudes != PROBE_AMPLITUDES and not randomize_probes:
        raise ValueError("probe amplitudes only apply with randomize_probes")
    if attacker is not None:
        if feature_set != "history" or randomize_probes:
            raise ValueError("attacker collection needs history features and no probe randomization")
        if strata is not None and episodes % (len(strata) * len(attacker.deviation_weights)):
            raise ValueError("episodes must be a multiple of strata x deviation weights for a balanced design")
    elif strata is not None:
        if not randomize_probes:
            raise ValueError("speed strata require randomize_probes")
        if episodes % (4 * len(strata)):
            raise ValueError("episodes must be a multiple of 4 x strata for a balanced design")
    if re.fullmatch(r"[A-Za-z0-9_]+", episode_prefix) is None:
        raise ValueError("episode_prefix may contain only letters, digits and underscores")
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
        design = None
        extra = {}
        if attacker is not None:
            # 速度區間依場次輪流；每輪完所有區間換下一個偏離中心線權重。兩車移動中起步。
            stratum = None if strata is None else strata[number % len(strata)]
            weight = attacker.deviation_weights[(number // (1 if strata is None else len(strata)))
                                                % len(attacker.deviation_weights)]
            gap = float(rng.uniform(*attacker.initial_gap))
            speed = float(rng.uniform(*(stratum or (3.0, 7.0))))
            rear_speed = speed + float(rng.uniform(*attacker.speed_advantage))
            rear_initial, front_initial, rear_strategy = rear_speed, speed, "relative_overtake"
            # 攻方候選與橫移距離同 interaction_overtake（7 個訓練過的目標、3.0 m），只換 deviation_weight。
            episode_settings = replace(settings, relative_overtaking_config=replace(
                main.INTERACTION_OVERTAKING_CONFIG.relative, deviation_weight=weight))
            design = {"speed_stratum": None if stratum is None else list(stratum),
                      "deviation_weight": weight, "speed_advantage": rear_speed - speed}
        else:
            if strata is None:
                gap, speed = float(rng.uniform(2, 5)), float(rng.uniform(3, 7))
                initial, first_side = (speed if number % 2 else 0.0), None
            else:
                # 平衡設計：速度區間、先行側與起步狀態依場次編號分配，避免彼此混在一起。
                stratum, first_side, moving = stratified_design(number, len(strata))
                gap, speed = float(rng.uniform(2, 5)), float(rng.uniform(*strata[stratum]))
                initial = speed if moving else 0.0
                design = {"speed_stratum": list(strata[stratum]),
                          "first_side": "left" if first_side > 0 else "right", "moving_start": moving}
            episode_settings = settings
            if randomize_probes:
                episode_settings = replace(settings, probe_following_config=random_probe_config(
                    settings.probe_following_config, probe_rng, amplitudes, first_side))
            probe = episode_settings.probe_following_config
            rear_speed, rear_initial, front_initial, rear_strategy = speed + 2, initial, initial, "probe_follow"
            extra = {"probe_sequence": list(probe.sequence), "hold_seconds": probe.hold_seconds}
        scenario = EpisodeScenario(f"{episode_prefix}-{number:03d}", (
            VehicleInitialCondition("rear", origin, 0.0, rear_initial, rear_speed),
            VehicleInitialCondition("front", origin + gap, 0.0, front_initial, speed),
        ), {"rear": StrategyAssignment(rear_strategy),
            "front": StrategyAssignment("defend", target_vehicle_id="rear")})
        scenarios.append({"id": scenario.episode_id, "vehicles": [asdict(v) for v in scenario.vehicles],
                          **extra, **({} if design is None else {"design": design})})

        def observe(step, observation, decision, command):
            if observation.vehicle_id != "rear":
                return
            actions.append({"episode_id": scenario.episode_id, "step_index": step, "time_s": step * dt,
                            "mode": decision.mode, "target_d": decision.target_d,
                            "speed_limit": decision.speed_limit,
                            "effective_target_speed": min(rear_speed, decision.speed_limit) if decision.speed_limit is not None else rear_speed,
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
                "probe_order": ("not used (relative_overtake attacker)" if attacker is not None
                                else "randomized per episode" if randomize_probes
                                else "left, center, right, center"),
                "configs": {name: asdict(getattr(settings, name)) for name in (
                    "model_config", "geometry", "controller_config", "probe_following_config", "defense_config", "offset_path_config")}}
    if feature_set == "history":
        metadata["history_segment_seconds"] = HISTORY_SEGMENT_SECONDS
    if randomize_probes:
        metadata["probe_randomization"] = {
            "amplitudes_m": amplitudes, "hold_seconds_range": PROBE_HOLD_RANGE,
            "side_count": PROBE_SIDE_COUNT, "rng": "numpy default_rng([seed, 1])",
            "note": ("先行側依平衡設計，" if strata is not None else "每場隨機先左或先右，")
                    + "左右交替；每次側向幅度獨立抽樣；hold_seconds 整場相同。"}
    if attacker is not None:
        metadata["rear_strategy"] = "relative_overtake"
        metadata["attacker"] = {**asdict(attacker),
                                "base_config": asdict(main.INTERACTION_OVERTAKING_CONFIG.relative),
                                "design_rule": "速度區間依場次輪流；每輪完所有區間換下一個 deviation_weight；兩車移動中起步"}
    if strata is not None:
        metadata["speed_strata"] = [list(stratum) for stratum in strata]
        if attacker is None:
            metadata["design_rule"] = "速度區間依場次輪流；同區間依序為（先左、靜止）（先右、靜止）（先左、移動）（先右、移動）"
    if episode_prefix != "interaction":
        metadata["episode_prefix"] = episode_prefix
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


def export(model: Path, output: Path) -> None:
    """由 pkl 產生可放進 Git 的精簡檔，並確認兩者在訓練輸入上的預測一致。"""
    from strategy.interaction_gp import InteractionGpModel, compact_arrays
    if output.suffix != ".npz":
        raise ValueError("compact model output must end with .npz")
    source = model.read_bytes()
    bundle = pickle.loads(source)   # 只處理自己訓練、可信任的模型
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **compact_arrays(bundle, hashlib.sha256(source).hexdigest()))
    original, compact = InteractionGpModel.load(model), InteractionGpModel.load(output)
    gp = bundle["pipeline"].named_steps["gp"]
    inputs = bundle["pipeline"].named_steps["scaler"].inverse_transform(gp.X_train_)
    rows = inputs[np.linspace(0, len(inputs) - 1, min(500, len(inputs))).astype(int)]
    (mean_a, std_a), (mean_b, std_b) = original.predict(rows), compact.predict(rows)
    difference = float(max(np.abs(mean_a - mean_b).max(), np.abs(std_a - std_b).max()))
    same_ranges = (np.array_equal(original.lower, compact.lower) and np.array_equal(original.upper, compact.upper)
                   and original.trained_targets == compact.trained_targets)
    report = {"source": str(model), "source_sha256": hashlib.sha256(source).hexdigest(),
              "compact": str(output), "compact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "compact_bytes": output.stat().st_size, "checked_rows": len(rows),
              "max_prediction_difference": difference, "same_input_ranges": same_ranges}
    print(json.dumps(report, indent=2), flush=True)
    if difference > 1e-9 or not same_ranges:
        output.unlink()
        raise ValueError("compact model does not reproduce the original predictions; output removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "train", "export"))
    parser.add_argument("--model", help="export：要轉成精簡檔的 model.pkl")
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
    parser.add_argument("--speed-strata", nargs="+", type=float, metavar="EDGE",
                        help="collect：前車巡航速度區間邊界（m/s），例如 3 4 5 6 7；平衡分配場次，需 --randomize-probes")
    parser.add_argument("--probe-amplitudes", nargs="+", type=float, default=list(PROBE_AMPLITUDES), metavar="M",
                        help="collect：側向試探幅度候選（m），需 --randomize-probes")
    parser.add_argument("--episode-prefix", default="interaction",
                        help="collect：場次 ID 前綴，合併不同資料集時避免重複")
    parser.add_argument("--rear-strategy", choices=("probe_follow", "relative_overtake"), default="probe_follow",
                        help="collect：後車策略；relative_overtake 為攻方實際超車收集（需 --features history）")
    parser.add_argument("--deviation-weights", nargs="+", type=float, default=[0.0, 1.0, 3.0],
                        help="collect（relative_overtake）：每場輪流使用的偏離中心線權重")
    parser.add_argument("--speed-advantage", nargs=2, type=float, default=[0.5, 1.5], metavar=("LOW", "HIGH"),
                        help="collect（relative_overtake）：攻方比前車快的範圍，m/s")
    parser.add_argument("--initial-gap", nargs=2, type=float, default=[1.5, 4.0], metavar=("LOW", "HIGH"),
                        help="collect（relative_overtake）：初始車距範圍，m")
    args = parser.parse_args()
    output = project_path(args.output)
    if output.exists():
        parser.error("output already exists")
    os.environ.setdefault("MPLCONFIGDIR", str(project_path(".cache/matplotlib")))
    if args.mode == "collect":
        attacker = None if args.rear_strategy == "probe_follow" else AttackerCollection(
            tuple(args.deviation_weights), tuple(args.speed_advantage), tuple(args.initial_gap))
        collect(output, args.episodes, args.seconds, args.seed, args.features, args.randomize_probes,
                None if args.speed_strata is None else speed_strata(args.speed_strata),
                tuple(args.probe_amplitudes), args.episode_prefix, attacker)
    elif args.mode == "export":
        if args.model is None:
            parser.error("export requires --model")
        export(project_path(args.model), output)
    else:
        if args.data is None:
            parser.error("train requires --data")
        train(project_path(args.data), output, args.seed, args.max_train, args.feature_subset)
