"""涵蓋率診斷：驗證／測試的 z 分數、子集、各場涵蓋率、殘差自相關與切分條件；只讀檔，不擬合。"""

import csv
import json
from pathlib import Path
import re

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT_ROOT / "reports/experiments/2026-11-24_history_gp"   # 已提交的 predictions／report 副本
DATA = PROJECT_ROOT / "data/training/history_gp_20261124"              # 原始樣本只在本機


def stats(a: np.ndarray, m: np.ndarray, s: np.ndarray) -> dict:
    z = (a - m) / s
    e = m - a
    return dict(n=len(a), cov=np.mean(np.abs(z) <= 2), k95=np.quantile(np.abs(z), 0.95),
                zstd=z.std(), rmse=np.sqrt(np.mean(e ** 2)), mean_std=s.mean(),
                kurt=np.mean(((e - e.mean()) / e.std()) ** 4))


def fmt(values: dict) -> str:
    return " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in values.items())


def main() -> None:
    samples = {}
    with (DATA / "samples.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            samples[(row["episode_id"], round(float(row["time_s"]), 4))] = row
    for name in ("a_basic8", "b_history12", "c_action13"):
        report = json.loads((RESULTS / name / "report.json").read_text(encoding="utf-8"))
        with (RESULTS / name / "predictions.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        noise = float(re.search(r"WhiteKernel\(noise_level=([0-9.e-]+)\)", report["fitted_kernel"]).group(1))
        train_eps = set(report["splits"]["train"]["episodes"])
        y_train = np.array([float(r["opponent_delta_d"]) for k, r in samples.items() if k[0] in train_eps])
        # normalize_y：雜訊變異數以訓練答案標準差換回公尺。
        print(f"== {name}: noise std (m) = {np.sqrt(noise) * y_train.std():.4f}; train y std = {y_train.std():.4f}")
        for split in ("validation", "test"):
            rs = [r for r in rows if r["split"] == split]
            a, m, s = (np.array([float(r[k]) for r in rs]) for k in ("actual", "mean", "std"))
            feats = [samples[(r["episode_id"], round(float(r["time_s"]), 4))] for r in rs]
            moving = np.array([abs(float(f["rear_dd_late"])) > 0.02 for f in feats])
            big = np.abs(a) > 0.1
            print(f"  {split}: " + fmt(stats(a, m, s)))
            for label, mask in (("rear_moving", moving), ("rear_holding", ~moving),
                                ("|actual|>0.1", big), ("|actual|<=0.1", ~big)):
                print(f"     {label:14s} " + fmt(stats(a[mask], m[mask], s[mask])))
            per, lag1, lag5 = [], [], []
            for episode in sorted({r["episode_id"] for r in rs}):
                index = np.array([i for i, r in enumerate(rs) if r["episode_id"] == episode])
                index = index[np.argsort([float(rs[i]["time_s"]) for i in index])]
                per.append(f"{episode[-3:]}:{np.mean(np.abs((a[index] - m[index]) / s[index]) <= 2):.2f}")
                e = m[index] - a[index]
                lag1.append(np.corrcoef(e[:-1], e[1:])[0, 1])
                lag5.append(np.corrcoef(e[:-5], e[5:])[0, 1])
            print(f"     per-episode coverage {per}; residual autocorr lag0.1s={np.mean(lag1):.2f} lag0.5s={np.mean(lag5):.2f}")
    print("== split conditions (seed 42 episode split)")
    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))["run"]
    groups = np.unique([s["id"] for s in metadata["scenarios"]])
    np.random.default_rng(42).shuffle(groups)
    split = {"test": set(groups[:4]), "validation": set(groups[4:8]), "train": set(groups[8:])}
    for name, ids in split.items():
        rows = []
        for scenario in metadata["scenarios"]:
            if scenario["id"] in ids:
                v, seq = scenario["vehicles"], scenario["probe_sequence"]
                rows.append((scenario["id"][-3:], round(v[1]["target_speed"], 2), round(v[1]["start_s"] - v[0]["start_s"], 2),
                             "move" if v[0]["initial_speed"] > 0 else "stop", round(scenario["hold_seconds"], 2),
                             "L" if seq[0] > 0 else "R", max(abs(x) for x in seq)))
        speeds = [r[1] for r in rows]
        print(f"{name:10s} front speed {min(speeds):.2f}-{max(speeds):.2f} mean {np.mean(speeds):.2f}")
        for row in sorted(rows, key=lambda r: r[1]):
            print("   ", row)


if __name__ == "__main__":
    main()
