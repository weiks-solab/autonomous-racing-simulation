"""Train one GP target from a collected training-data folder."""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import platform
import time
from typing import Any, Sequence
from uuid import uuid4
import warnings

import numpy as np
from numpy.typing import NDArray
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training_data import FieldSpec, TrainingDataset, load_dataset
from training_data.io import PROJECT_ROOT, project_path
from training_data.schema import require_integer


# ===== 常用設定：命令列可選資料、答案、測試比例及 seed =====
@dataclass(frozen=True)
class TrainingConfig:
    target: str = "opponent_delta_d"
    test_fraction: float = 0.2       # 保留多少比例的整場模擬作測試。
    seed: int = 42                  # 固定資料切分與參數搜尋，方便比較。
    length_scale: float = 1.0        # 標準化後，每個輸入的相似程度初始尺度。
    noise_level: float = 0.02        # 標準化答案的雜訊變異數初始值。
    optimizer_restarts: int = 2      # 額外嘗試幾組參數初始值。

    def __post_init__(self) -> None:
        if not math.isfinite(self.test_fraction) or not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction 必須介於 0 與 1 之間。")
        require_integer("seed", self.seed)
        require_integer("optimizer_restarts", self.optimizer_restarts)
        if not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed 必須介於 0 與 2**32 - 1 之間。")
        if not math.isfinite(self.length_scale) or not 0.05 <= self.length_scale <= 20.0:
            raise ValueError("length_scale 必須介於 0.05 與 20 之間。")
        if not math.isfinite(self.noise_level) or not 1e-6 <= self.noise_level <= 1.0:
            raise ValueError("noise_level 必須介於 1e-6 與 1 之間。")


@dataclass(frozen=True)
class TrainingResult:
    model: Pipeline
    target_spec: FieldSpec
    train_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    metrics: dict[str, dict[str, float]]
    fit_seconds: float
    fit_warnings: tuple[str, ...]


def select_target(dataset: TrainingDataset, target: str) -> tuple[FieldSpec, NDArray[np.float64]]:
    if not dataset.samples:
        raise ValueError("資料集沒有有效樣本。")
    if target not in dataset.schema.target_names:
        available = ", ".join(dataset.schema.target_names)
        raise ValueError(f"找不到答案欄位 {target}；可用欄位：{available}")
    index = dataset.schema.target_names.index(target)
    return dataset.target_specs[index], dataset.Y[:, index]


def split_episodes(
    dataset: TrainingDataset, config: TrainingConfig,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """同一場模擬的所有時間點與車輛配對，必須留在同一邊。"""
    groups = np.asarray([sample.episode_id for sample in dataset.samples])
    count = len(np.unique(groups))
    if count < 2:
        raise ValueError("至少需要 2 個有有效樣本的 episode，才能分開訓練與測試。")
    if math.ceil(count * config.test_fraction) >= count:
        raise ValueError("測試比例太高，請降低 --test-fraction，保留至少一場訓練資料。")
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_fraction, random_state=config.seed)
    return next(splitter.split(np.zeros((len(groups), 1)), groups=groups))


def create_model(feature_count: int, config: TrainingConfig) -> Pipeline:
    """各欄先標準化，再分別學習其 RBF 尺度。"""
    require_integer("feature_count", feature_count, 1)
    kernel = (
        ConstantKernel(1.0, (0.01, 100.0))
        * RBF(np.full(feature_count, config.length_scale), (0.05, 20.0))
        + WhiteKernel(config.noise_level, (1e-6, 1.0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True,
        n_restarts_optimizer=config.optimizer_restarts, random_state=config.seed,
    )
    return Pipeline([("scaler", StandardScaler()), ("gp", gp)])


def prediction_errors(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> dict[str, float]:
    difference = predicted - actual
    return {"mae": float(np.mean(np.abs(difference))), "rmse": float(np.sqrt(np.mean(difference ** 2)))}


def train_model(dataset: TrainingDataset, config: TrainingConfig = TrainingConfig()) -> TrainingResult:
    """測試集只用來評分，不參與縮放器與 GP 的擬合。"""
    target_spec, y = select_target(dataset, config.target)
    train_indices, test_indices = split_episodes(dataset, config)
    X = dataset.X
    model = create_model(X.shape[1], config)
    started_at = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X[train_indices], y[train_indices])
    fit_seconds = time.perf_counter() - started_at
    mean, std = model.predict(X[test_indices], return_std=True)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std < 0.0):
        raise ValueError("模型產生了無效的預測值或標準差。")
    actual = y[test_indices]
    metrics = {
        "gp": prediction_errors(actual, mean),
        "always_zero": prediction_errors(actual, np.zeros_like(actual)),
    }
    return TrainingResult(
        model, target_spec, train_indices, test_indices, mean, std, metrics,
        fit_seconds, tuple(str(item.message) for item in caught),
    )


def build_report(
    data_directory: Path, dataset: TrainingDataset, result: TrainingResult, config: TrainingConfig,
) -> dict[str, Any]:
    def episode_ids(indices: NDArray[np.int64]) -> list[str]:
        return sorted({dataset.samples[index].episode_id for index in indices})

    gp = result.model.named_steps["gp"]
    scaler = result.model.named_steps["scaler"]
    fitted = gp.kernel_
    return {
        "format_version": 1,
        "data_directory": str(project_path(data_directory).relative_to(PROJECT_ROOT)),
        "feature_specs": [asdict(spec) for spec in dataset.feature_specs],
        "target_spec": asdict(result.target_spec),
        "sampling": asdict(dataset.sampling),
        "label_horizon_seconds": dataset.sampling.label_horizon_seconds,
        "config": asdict(config),
        "train_sample_count": len(result.train_indices),
        "test_sample_count": len(result.test_indices),
        "train_episode_ids": episode_ids(result.train_indices),
        "test_episode_ids": episode_ids(result.test_indices),
        "metrics": result.metrics,
        "initial_kernel": str(gp.kernel),
        "fitted_kernel": str(fitted),
        "learned_parameters": {
            "signal_variance": float(fitted.k1.k1.constant_value),
            "length_scales": np.atleast_1d(fitted.k1.k2.length_scale).tolist(),
            "noise_variance": float(fitted.k2.noise_level),
        },
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "fit_seconds": result.fit_seconds,
        "fit_warnings": list(result.fit_warnings),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }


def save_predictions(path: Path, dataset: TrainingDataset, result: TrainingResult) -> None:
    """保留測試列的 ID、輸入與不確定性，方便對照原始資料。"""
    _, y = select_target(dataset, result.target_spec.name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "episode_id", "step_index", "time_s", "ego_id", "opponent_id",
            *(f"feature_{name}" for name in dataset.schema.feature_names),
            "actual", "predicted_mean", "predicted_std", "error",
        ))
        for index, mean, std in zip(result.test_indices, result.mean, result.std):
            sample = dataset.samples[index]
            writer.writerow((
                sample.episode_id, sample.step_index, sample.time_seconds,
                sample.pair.ego_id, sample.pair.opponent_id, *sample.features,
                y[index], mean, std, mean - y[index],
            ))


def save_plot(path: Path, dataset: TrainingDataset, result: TrainingResult) -> None:
    """直接寫入圖片，不開啟視窗。"""
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    _, y = select_target(dataset, result.target_spec.name)
    actual = y[result.test_indices]
    low = float(min(np.min(actual), np.min(result.mean)))
    high = float(max(np.max(actual), np.max(result.mean)))
    padding = max((high - low) * 0.05, abs(high) * 0.01, 1e-6)
    limits = (low - padding, high + padding)
    figure = Figure(figsize=(6.0, 6.0), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.scatter(actual, result.mean, s=16, alpha=0.65, label="Held-out predictions")
    axes.plot(limits, limits, "--", color="gray", label="Perfect prediction")
    axes.set(xlim=limits, ylim=limits, aspect="equal",
             xlabel=f"Actual {result.target_spec.name} ({result.target_spec.unit})",
             ylabel=f"Predicted ({result.target_spec.unit})", title="GP on held-out episodes")
    axes.grid(alpha=0.25)
    axes.legend()
    figure.savefig(path, dpi=150)


def save_results(
    output_directory: str | Path, data_directory: str | Path,
    dataset: TrainingDataset, result: TrainingResult, config: TrainingConfig,
) -> Path:
    output = project_path(output_directory)
    report = build_report(Path(data_directory), dataset, result, config)
    report_json = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    # 保存的是接受測試評估的模型，沒有再用測試集重新訓練。
    model_bundle = {
        "format_version": 1,
        "pipeline": result.model,
        "feature_names": dataset.schema.feature_names,
        "feature_specs": report["feature_specs"],
        "target_spec": report["target_spec"],
        "label_horizon_seconds": dataset.sampling.label_horizon_seconds,
        "versions": report["versions"],
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "model.pkl").open("wb") as stream:
        pickle.dump(model_bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (output / "report.json").write_text(report_json + "\n", encoding="utf-8")
    save_predictions(output / "predictions.csv", dataset, result)
    save_plot(output / "evaluation.png", dataset, result)
    return output


def print_result(output: Path, dataset: TrainingDataset, result: TrainingResult) -> None:
    unit = result.target_spec.unit
    metrics = result.metrics
    print(f"輸入欄位：{', '.join(dataset.schema.feature_names)}")
    print(f"答案：{result.target_spec.name}；預測時間：{dataset.sampling.label_horizon_seconds:g} 秒")
    print(f"訓練 {len(result.train_indices)} 筆；測試 {len(result.test_indices)} 筆")
    print(f"GP 測試 MAE：{metrics['gp']['mae']:.6g} {unit}")
    print(f"GP 測試 RMSE：{metrics['gp']['rmse']:.6g} {unit}")
    print(f"固定預測 0 的 MAE：{metrics['always_zero']['mae']:.6g} {unit}")
    if metrics["gp"]["mae"] >= metrics["always_zero"]["mae"]:
        print("本次 GP 的 MAE 未優於固定 0 基準，請先檢查資料情境與輸入資訊。")
    if result.fit_warnings:
        print(f"有 {len(result.fit_warnings)} 則調參提示，詳見 report.json 的 fit_warnings。")
    print(f"學到的 kernel：{result.model.named_steps['gp'].kernel_}")
    print(f"已儲存：{output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainingConfig()
    parser.add_argument("--data", required=True, type=Path, help="收集程式輸出的資料夾")
    parser.add_argument("--target", default=defaults.target, help="這次要學習的答案欄位")
    parser.add_argument("--test-fraction", type=float, default=defaults.test_fraction, help="測試 episode 比例")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output", type=Path, help="專案內尚不存在的輸出資料夾")
    args = parser.parse_args(argv)
    try:
        config = TrainingConfig(target=args.target, test_fraction=args.test_fraction, seed=args.seed)
        data_directory = project_path(args.data)
        dataset = load_dataset(data_directory)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        output = project_path(args.output if args.output is not None else Path("models/gp") / run_id)
        if output.exists():
            raise ValueError("輸出資料夾已存在，請使用新的資料夾，避免覆寫模型。")
        print(f"開始訓練：{len(dataset.samples)} 筆資料，答案 {config.target}", flush=True)
        result = train_model(dataset, config)
        save_results(output, data_directory, dataset, result, config)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print_result(output, dataset, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
