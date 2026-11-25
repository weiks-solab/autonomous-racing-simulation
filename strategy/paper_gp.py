"""載入論文格式一步 GP（14 維輸入、4 個輸出），檢查輸入是否在訓練範圍內，並一次預測多列。

4 個輸出依 PAPER_TARGETS 的順序：前車 0.1 s 後的 Δs、Δd、Δe_ψ、Δv，各自是一個 StandardScaler＋GP。
訓練見 reports/experiments/2026-11-25_paper_format_gp/train_eval.py；精簡檔由 interaction_gp.py export 產生。
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from strategy.interaction_features import CURVATURE_LOOKAHEAD, PAPER_FEATURES, PAPER_TARGETS
from strategy.interaction_gp import PROJECT_ROOT, RANGE_TOLERANCE, pipeline_arrays, rebuild_pipeline

PAPER_MODEL_KIND = "paper_format_one_step"
PAPER_COMPACT_FORMAT = "paper_gp_compact_v1"
CURVATURE_SPACING = 0.5  # centerline_curvature 的取點間距（m），線上特徵與訓練相同
REAR_SPEED = PAPER_FEATURES.index("rear_speed")
FRONT_SPEED = PAPER_FEATURES.index("front_speed")


def check_bundle(saved: dict) -> None:
    """欄位、答案、時域、歷史段與曲率設定都要與線上特徵一致，否則不能使用。"""
    if (saved.get("model_kind") != PAPER_MODEL_KIND
            or tuple(saved.get("feature_names", ())) != PAPER_FEATURES
            or tuple(saved.get("target_names", ())) != PAPER_TARGETS
            or set(saved.get("pipelines", {})) != set(PAPER_TARGETS)
            or not math.isclose(saved.get("label_horizon_seconds", math.nan), 0.1)
            or not math.isclose(saved.get("history_segment_seconds", math.nan), 0.2)
            or not math.isclose(saved.get("curvature_lookahead_m", math.nan), CURVATURE_LOOKAHEAD)
            or not math.isclose(saved.get("curvature_spacing_m", math.nan), CURVATURE_SPACING)):
        raise ValueError("模型必須是論文格式一步 GP：14 維輸入，預測前車 0.1 s 後的 [Δs, Δd, Δe_ψ, Δv]，"
                         "歷史段長 0.2 s，曲率前看與取點間距和線上特徵相同。")


def training_inputs(pipeline: Any) -> NDArray[np.float64]:
    """GP 內存的是已標準化的訓練輸入，還原成原始單位。"""
    return pipeline.named_steps["scaler"].inverse_transform(pipeline.named_steps["gp"].X_train_)


@dataclass(frozen=True)
class PaperGpModel:
    """predictors 依 PAPER_TARGETS 順序，各需提供 predict(X, return_std=True)；範圍取自訓練輸入。"""

    predictors: tuple[Any, ...]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    relative_speed_range: tuple[float, float]   # 後車速度減前車速度，m/s
    horizon_seconds: float = 0.1
    segment_seconds: float = 0.2
    curvature_lookahead: float = CURVATURE_LOOKAHEAD

    @classmethod
    def load(cls, path: str | Path) -> "PaperGpModel":
        path = (PROJECT_ROOT / path).resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError("paper-format GP model must remain inside this repository")
        if not path.is_file():
            raise FileNotFoundError(
                f"論文格式 GP 模型不存在：{path}；請修改 PAPER_GP_MODEL_PATH 或先訓練模型。")
        if path.suffix == ".npz":
            saved = load_paper_compact(path)
        else:
            # 只載入自己訓練、可信任的 pickle；不在模擬時重新訓練。
            with path.open("rb") as stream:
                saved = pickle.load(stream)
        check_bundle(saved)
        pipelines = tuple(saved["pipelines"][target] for target in PAPER_TARGETS)
        inputs = [training_inputs(pipeline) for pipeline in pipelines]
        if any(item.shape != inputs[0].shape or not np.allclose(item, inputs[0], rtol=0.0, atol=1e-9)
               for item in inputs[1:]):
            raise ValueError("the four paper-format GPs must share the same training inputs")
        return cls.from_training_inputs(pipelines, inputs[0], saved["label_horizon_seconds"],
                                        saved["history_segment_seconds"], saved["curvature_lookahead_m"])

    @classmethod
    def from_training_inputs(
        cls, predictors: Sequence[Any], inputs: ArrayLike, horizon_seconds: float = 0.1,
        segment_seconds: float = 0.2, curvature_lookahead: float = CURVATURE_LOOKAHEAD,
    ) -> "PaperGpModel":
        inputs = np.asarray(inputs, dtype=float)
        if len(predictors) != len(PAPER_TARGETS):
            raise ValueError("paper-format GP needs one predictor per target")
        if inputs.ndim != 2 or inputs.shape[1] != len(PAPER_FEATURES) or len(inputs) == 0:
            raise ValueError("training inputs must be a non-empty (n, 14) array")
        relative = inputs[:, REAR_SPEED] - inputs[:, FRONT_SPEED]
        return cls(tuple(predictors), inputs.min(axis=0), inputs.max(axis=0),
                   (float(relative.min()), float(relative.max())), float(horizon_seconds),
                   float(segment_seconds), float(curvature_lookahead))

    def check_rows(self, rows: ArrayLike) -> list[str | None]:
        """逐列回傳 None（可使用）或超出訓練範圍的欄位名稱；relative_speed 為後車減前車速度。"""
        X = np.atleast_2d(np.asarray(rows, dtype=float))
        if X.ndim != 2 or X.shape[1] != len(PAPER_FEATURES):
            raise ValueError("rows must have 14 columns in PAPER_FEATURES order")
        finite = np.all(np.isfinite(X), axis=1)
        outside = (X < self.lower - RANGE_TOLERANCE) | (X > self.upper + RANGE_TOLERANCE)
        relative = X[:, REAR_SPEED] - X[:, FRONT_SPEED]
        low, high = self.relative_speed_range
        relative_outside = (relative < low - RANGE_TOLERANCE) | (relative > high + RANGE_TOLERANCE)
        reasons: list[str | None] = []
        for index in range(len(X)):
            if not finite[index]:
                reasons.append("non_finite")
            elif outside[index].any():
                reasons.append(PAPER_FEATURES[int(np.argmax(outside[index]))])
            elif relative_outside[index]:
                reasons.append("relative_speed")
            else:
                reasons.append(None)
        return reasons

    def predict(self, rows: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """回傳 (n, 4) 的平均與標準差，欄位依 PAPER_TARGETS（m、m、rad、m/s）。"""
        X = np.atleast_2d(np.asarray(rows, dtype=float))
        means, stds = [], []
        for predictor in self.predictors:
            mean, std = predictor.predict(X, return_std=True)
            means.append(np.asarray(mean, dtype=float).reshape(-1))
            stds.append(np.asarray(std, dtype=float).reshape(-1))
        mean, std = np.column_stack(means), np.column_stack(stds)
        if (mean.shape != (len(X), len(PAPER_TARGETS)) or std.shape != mean.shape
                or not np.all(np.isfinite(mean) & np.isfinite(std))):
            raise ValueError("paper-format GP must return one finite mean and std per row and target")
        return mean, std


def paper_compact_arrays(bundle: dict, source_sha256: str = "") -> dict[str, NDArray]:
    """把 pkl 內容轉成精簡格式：4 個 GP 各自加前綴，只存重建所需的數字。"""
    check_bundle(bundle)
    pipelines = bundle["pipelines"]
    metadata = {
        "format": PAPER_COMPACT_FORMAT, "model_kind": bundle["model_kind"],
        "feature_names": list(bundle["feature_names"]), "target_names": list(bundle["target_names"]),
        "feature_units": list(bundle.get("feature_units", ())), "target_units": list(bundle.get("target_units", ())),
        "label_horizon_seconds": bundle["label_horizon_seconds"],
        "history_segment_seconds": bundle["history_segment_seconds"],
        "curvature_lookahead_m": bundle["curvature_lookahead_m"], "curvature_spacing_m": bundle["curvature_spacing_m"],
        "kernels": {target: str(pipelines[target].named_steps["gp"].kernel_) for target in PAPER_TARGETS},
        "source_sha256": source_sha256, "versions": bundle.get("versions", {}),
    }
    arrays: dict[str, NDArray] = {"metadata": np.array(json.dumps(metadata, ensure_ascii=False))}
    for target in PAPER_TARGETS:
        arrays.update({f"{target}__{key}": value for key, value in pipeline_arrays(pipelines[target]).items()})
    return arrays


def load_paper_compact(path: Path) -> dict:
    """讀精簡檔（不經 pickle），以固定超參數重建 4 個 Pipeline。"""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(str(arrays["metadata"]))
    if metadata.get("format") != PAPER_COMPACT_FORMAT:
        raise ValueError(f"unsupported paper-format compact model: {metadata.get('format')!r}")
    pipelines = {}
    for target in PAPER_TARGETS:
        prefix = f"{target}__"
        pipelines[target] = rebuild_pipeline(
            {key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)})
    return {"pipelines": pipelines, "model_kind": metadata["model_kind"],
            "feature_names": tuple(metadata["feature_names"]), "target_names": tuple(metadata["target_names"]),
            "label_horizon_seconds": metadata["label_horizon_seconds"],
            "history_segment_seconds": metadata["history_segment_seconds"],
            "curvature_lookahead_m": metadata["curvature_lookahead_m"],
            "curvature_spacing_m": metadata["curvature_spacing_m"], "compact_metadata": metadata}
