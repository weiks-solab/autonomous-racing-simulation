"""模型 c 的學習曲線（每場秒數、訓練場數）與 20 場 5 折交叉驗證；只在記憶體擬合，不保存模型。"""

import csv
from pathlib import Path
import sys
import time
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from train_gp import TrainingConfig, create_model

DATA = PROJECT_ROOT / "data/training/history_gp_20261124"   # 原始樣本只在本機
NAMES = ["gap", "rear_d", "rear_heading_error", "rear_speed", "front_d", "front_heading_error",
         "front_speed", "front_curvature", "rear_dd_early", "front_dd_early", "rear_dd_late",
         "front_dd_late", "d_target"]

with (DATA / "samples.csv").open(encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
X = np.array([[float(r[n]) for n in NAMES] for r in rows])
y = np.array([float(r["opponent_delta_d"]) for r in rows])
ids = np.array([r["episode_id"] for r in rows])
t = np.array([float(r["time_s"]) for r in rows])
groups = np.unique(ids)
np.random.default_rng(42).shuffle(groups)        # 與 interaction_gp.split_groups 相同的切分
test_eps, val_eps, train_eps = groups[:4], groups[4:8], groups[8:]
C, B = list(range(13)), list(range(12))


def fit_eval(train_mask, cols, evals):
    model = create_model(len(cols), TrainingConfig(optimizer_restarts=0, seed=42))
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X[train_mask][:, cols], y[train_mask])
    out = {"n": int(train_mask.sum()), "sec": time.perf_counter() - start}
    for name, eps in evals:
        mask = np.isin(ids, eps)
        mu, sd = model.predict(X[mask][:, cols], return_std=True)
        e = mu - y[mask]
        out[name] = (np.mean(np.abs(e)), np.mean(np.abs(e) <= 2 * sd), sd.mean(), np.sqrt(np.mean(e ** 2)))
    return out, model


def show(label, out):
    parts = [f"{label:28s} n={out['n']:5d} fit={out['sec']:5.1f}s"]
    for name in ("val", "test"):
        if name in out:
            mae, cov, sd, rmse = out[name]
            parts.append(f"{name}: MAE={mae:.4f} RMSE={rmse:.4f} cov={cov:.3f} meanstd={sd:.4f}")
    print(" | ".join(parts), flush=True)


def main() -> None:
    E = (("val", val_eps), ("test", test_eps))
    print("== model c (13 inputs): sanity, all 12 train episodes, full 15 s", flush=True)
    show("c 12ep x 15s", fit_eval(np.isin(ids, train_eps), C, E)[0])
    print("== duration: 12 train episodes truncated to T seconds (samples with time+0.5<=T)", flush=True)
    for T in (5.0, 7.5, 10.0):
        show(f"c 12ep x {T:g}s", fit_eval(np.isin(ids, train_eps) & (t + 0.5 <= T + 1e-9), C, E)[0])
    show("b 12ep x 7.5s", fit_eval(np.isin(ids, train_eps) & (t + 0.5 <= 7.5 + 1e-9), B, E)[0])
    show("b 12ep x 15s", fit_eval(np.isin(ids, train_eps), B, E)[0])
    print("== episode count: k random train episodes x 15 s, 3 draws each", flush=True)
    for k in (3, 6, 9):
        for draw in range(3):
            chosen = np.random.default_rng(100 + draw).choice(train_eps, k, replace=False)
            show(f"c {k}ep x 15s draw{draw}", fit_eval(np.isin(ids, chosen), C, E)[0])
    print("== grouped 5-fold CV over all 20 episodes (model c, 16 train eps each)", flush=True)
    all_eps = np.unique(ids)
    order = all_eps.copy()
    np.random.default_rng(7).shuffle(order)
    pooled = []
    for fold in range(5):
        held = order[fold * 4:(fold + 1) * 4]
        out, model = fit_eval(~np.isin(ids, held), C, (("test", held),))
        show(f"c CV fold{fold}", out)
        for episode in held:
            mask = ids == episode
            mu, sd = model.predict(X[mask][:, C], return_std=True)
            for e, s, fs in zip(mu - y[mask], sd, X[mask][:, 6]):
                pooled.append((episode, e, s, fs))
    ep_arr = np.array([p[0] for p in pooled])
    e = np.array([p[1] for p in pooled])
    s = np.array([p[2] for p in pooled])
    fs = np.array([p[3] for p in pooled])
    print(f"CV pooled: n={len(e)} MAE={np.mean(np.abs(e)):.4f} cov={np.mean(np.abs(e) <= 2 * s):.3f} "
          f"k95={np.quantile(np.abs(e / s), 0.95):.2f}", flush=True)
    covs = {ep: np.mean(np.abs(e[ep_arr == ep]) <= 2 * s[ep_arr == ep]) for ep in all_eps}
    print("per-episode CV coverage:", " ".join(f"{k[-3:]}:{v:.2f}" for k, v in sorted(covs.items(), key=lambda kv: kv[1])), flush=True)
    print(f"episodes with coverage >= 0.95: {sum(v >= 0.95 for v in covs.values())}/20", flush=True)
    for lo, hi in ((3.0, 4.5), (4.5, 5.5), (5.5, 7.5)):
        m = (fs >= lo) & (fs < hi)
        print(f"front_speed [{lo},{hi}) n={m.sum():4d} RMSE={np.sqrt(np.mean(e[m] ** 2)):.4f} "
              f"meanstd={s[m].mean():.4f} cov={np.mean(np.abs(e[m]) <= 2 * s[m]):.3f}", flush=True)


if __name__ == "__main__":
    main()
