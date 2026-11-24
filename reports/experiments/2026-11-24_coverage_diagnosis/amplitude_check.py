"""依後車當下 |d_target| 與前車速度重算模型 c 的涵蓋率；只讀已提交的 predictions.csv。"""

import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS = PROJECT_ROOT / "reports/experiments/2026-11-24_history_gp/c_action13/predictions.csv"


def arr(rows, key):
    return np.array([float(r[key]) for r in rows])


def cov(rows):
    a, m, s = arr(rows, "actual"), arr(rows, "mean"), arr(rows, "std")
    return np.mean(np.abs(a - m) <= 2 * s), np.sqrt(np.mean((a - m) ** 2)), s.mean()


def main() -> None:
    with PREDICTIONS.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = (("0.30/0.45", lambda v: abs(abs(v) - 0.3) < 1e-9 or abs(abs(v) - 0.45) < 1e-9),
              ("0.60", lambda v: abs(abs(v) - 0.6) < 1e-9), ("0 (center)", lambda v: abs(v) < 1e-9))
    for split in ("validation", "test"):
        for label, test in groups:
            rs = [r for r in rows if r["split"] == split and test(float(r["d_target"]))]
            c, rmse, sd = cov(rs)
            eps = sorted({r["episode_id"][-3:] for r in rs})
            print(f"{split:10s} |d_target|={label:10s} n={len(rs):4d} cov={c:.3f} rmse={rmse:.4f} meanstd={sd:.4f} eps={eps}")
    print("--- |d_target|=0.6, val+test, by front_speed")
    big = [r for r in rows if abs(abs(float(r["d_target"])) - 0.6) < 1e-9]
    for lo, hi in ((0, 4.5), (4.5, 5.5), (5.5, 9)):
        rs = [r for r in big if lo <= float(r["front_speed"]) < hi]
        if not rs:
            print(f"speed [{lo},{hi}) n=0")
            continue
        c, rmse, sd = cov(rs)
        per = {}
        for r in rs:
            per.setdefault(r["episode_id"][-3:], []).append(r)
        detail = " ".join(f"{e}:{cov(v)[0]:.2f}(n={len(v)})" for e, v in sorted(per.items()))
        print(f"speed [{lo},{hi}) n={len(rs):3d} cov={c:.3f} rmse={rmse:.4f} meanstd={sd:.4f} | per-episode {detail}")


if __name__ == "__main__":
    main()
