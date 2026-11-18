"""核對已完成模型的欄位、標籤、切分、縮放與保存後預測；不重新擬合。"""

import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from train_gp import prediction_errors
from training_data import load_dataset
from training_data.interaction import ACTION_FEATURES


def verify() -> None:
    directory = Path(__file__).resolve().parent
    data = PROJECT_ROOT / "data/training/action_gp_20261118_current8"
    output = PROJECT_ROOT / "models/gp/action_gp_20261118_current8"
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    dataset = load_dataset(data)
    with (output / "model.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    assert tuple(bundle["feature_names"]) == ACTION_FEATURES
    assert bundle["label_horizon_seconds"] == dataset.sampling.label_horizon_seconds == 0.5
    groups = {name: set(info["episodes"]) for name, info in report["splits"].items()}
    assert {name: len(ids) for name, ids in groups.items()} == {"train": 12, "validation": 4, "test": 4}
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
                or groups["validation"] & groups["test"])
    fit = np.asarray(report["fit_sample_indices"])
    train_rows = [i for i, sample in enumerate(dataset.samples) if sample.episode_id in groups["train"]]
    np.testing.assert_array_equal(fit, train_rows)
    np.testing.assert_allclose(bundle["pipeline"].named_steps["scaler"].mean_, dataset.X[fit].mean(axis=0))
    with (data / "actions.csv").open(encoding="utf-8", newline="") as stream:
        actions = {(row["episode_id"], int(row["step_index"])): float(row["target_d"])
                   for row in csv.DictReader(stream)}
    with (data / "raw_states.csv").open(encoding="utf-8", newline="") as stream:
        front = {(row["episode_id"], int(row["step_index"])): (float(row["time_s"]), float(row["d"]))
                 for row in csv.DictReader(stream) if row["vehicle_id"] == "vehicle-2"}
    changed = 0
    for sample in dataset.samples:
        key = sample.episode_id, sample.step_index
        future_key = sample.episode_id, sample.step_index + dataset.sampling.label_horizon_steps
        np.testing.assert_allclose(front[future_key][0] - front[key][0], 0.5, atol=1e-12)
        np.testing.assert_allclose(sample.targets[0], front[future_key][1] - front[key][1], atol=1e-12)
        assert sample.features[-1] == actions[key]
        changed += any(actions[(sample.episode_id, step)] != actions[key]
                       for step in range(sample.step_index + 1, future_key[1]))
    metrics = {}
    for name in ("validation", "test"):
        rows = [i for i, sample in enumerate(dataset.samples) if sample.episode_id in groups[name]]
        mean, std = bundle["pipeline"].predict(dataset.X[rows], return_std=True)
        assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std >= 0)
        metrics[name] = prediction_errors(dataset.Y[rows, 0], mean)
        for key, value in metrics[name].items():
            np.testing.assert_allclose(value, report["splits"][name]["metrics"]["gp"][key], atol=1e-12)
    audit = {
        "feature_order": ACTION_FEATURES, "horizon_seconds": 0.5,
        "checked_samples": len(dataset.samples), "fit_count": len(fit),
        "all_training_samples_used": True, "episode_counts": {key: len(value) for key, value in groups.items()},
        "split_disjoint": True, "scaler_train_only": True, "raw_label_checks_passed": True,
        "executed_d_target_checks_passed": True, "reloaded_model_metrics": metrics,
        "samples_with_action_change_in_next_0p5s": changed,
        "action_change_fraction": changed / len(dataset.samples),
        "role_front_is_behind_samples": int(np.sum(dataset.X[:, 0] < 0)),
        "model_sha256": hashlib.sha256((output / "model.pkl").read_bytes()).hexdigest(),
        "meaning": "d_target 是當下已執行候選；下一個 0.5 s 仍可能改目標。未驗證未執行候選的反事實預測。",
    }
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
