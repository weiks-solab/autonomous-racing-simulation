"""載入互動歷史 GP（13 維），判斷輸入是否在訓練範圍內，並把預測的 d 組成前車軌跡。"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from core.track import Track
from core.track_geometry import track_length, track_to_world_points
from core.vehicle_state import VehicleState
from strategy.bicycle_prediction import BicyclePrediction
from strategy.interaction_features import INTERACTION_FEATURES
from strategy.observations import VehicleObservation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPACT_FORMAT = "interaction_gp_compact_v1"
REAR_SPEED = INTERACTION_FEATURES.index("rear_speed")
FRONT_SPEED = INTERACTION_FEATURES.index("front_speed")
D_TARGET = INTERACTION_FEATURES.index("d_target")
RANGE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class InteractionGpModel:
    """predictor 需提供 predict(X, return_std=True)；範圍取自模型的訓練輸入。"""

    predictor: Any
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    relative_speed_range: tuple[float, float]   # 後車速度減前車速度，m/s
    trained_targets: tuple[float, ...]          # 訓練中出現過的 d_target，m
    horizon_seconds: float = 0.5
    segment_seconds: float = 0.2

    @classmethod
    def load(cls, path: str | Path) -> "InteractionGpModel":
        path = (PROJECT_ROOT / path).resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError("interaction GP model must remain inside this repository")
        if not path.is_file():
            raise FileNotFoundError(
                f"互動 GP 模型不存在：{path}；請修改 INTERACTION_GP_MODEL_PATH 或先訓練模型。")
        if path.suffix == ".npz":
            saved = load_compact(path)
        else:
            # 只載入自己訓練、可信任的 pickle；不在模擬時重新訓練。
            with path.open("rb") as stream:
                saved = pickle.load(stream)
        if (tuple(saved.get("feature_names", ())) != INTERACTION_FEATURES
                or saved.get("target") != "opponent_delta_d"
                or not math.isclose(saved.get("label_horizon_seconds", math.nan), 0.5)
                or not math.isclose(saved.get("history_segment_seconds", math.nan), 0.2)):
            raise ValueError("模型必須是 13 維互動歷史 GP：預測前車 0.5 s 的 Δd，歷史段長 0.2 s。")
        pipeline = saved["pipeline"]
        # 訓練輸入存在 GP 內（已標準化），還原成原始單位以取得範圍。
        inputs = pipeline.named_steps["scaler"].inverse_transform(pipeline.named_steps["gp"].X_train_)
        return cls.from_training_inputs(pipeline, inputs, saved["label_horizon_seconds"],
                                        saved["history_segment_seconds"])

    @classmethod
    def from_training_inputs(
        cls, predictor: Any, inputs: NDArray[np.float64],
        horizon_seconds: float = 0.5, segment_seconds: float = 0.2,
    ) -> "InteractionGpModel":
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 2 or inputs.shape[1] != len(INTERACTION_FEATURES) or len(inputs) == 0:
            raise ValueError("training inputs must be a non-empty (n, 13) array")
        relative = inputs[:, REAR_SPEED] - inputs[:, FRONT_SPEED]
        targets = tuple(sorted({round(float(value), 6) for value in inputs[:, D_TARGET]}))
        return cls(predictor, inputs.min(axis=0), inputs.max(axis=0),
                   (float(relative.min()), float(relative.max())), targets,
                   float(horizon_seconds), float(segment_seconds))

    def is_trained_target(self, value: float) -> bool:
        return any(math.isclose(value, target, abs_tol=RANGE_TOLERANCE) for target in self.trained_targets)

    def check(self, features: Sequence[float]) -> str | None:
        """可使用時回傳 None；否則回傳超出訓練範圍的欄位名稱。"""
        x = np.asarray(features, dtype=float)
        if x.shape != (len(INTERACTION_FEATURES),) or not np.all(np.isfinite(x)):
            return "non_finite"
        for index, name in enumerate(INTERACTION_FEATURES):
            if index == D_TARGET:
                continue
            if not self.lower[index] - RANGE_TOLERANCE <= x[index] <= self.upper[index] + RANGE_TOLERANCE:
                return name
        low, high = self.relative_speed_range
        if not low - RANGE_TOLERANCE <= x[REAR_SPEED] - x[FRONT_SPEED] <= high + RANGE_TOLERANCE:
            return "relative_speed"
        if not self.is_trained_target(float(x[D_TARGET])):
            return "d_target"
        return None

    def predict(self, features: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """回傳每列的 Δd 平均與標準差（m）。"""
        X = np.atleast_2d(np.asarray(features, dtype=float))
        mean, std = self.predictor.predict(X, return_std=True)
        mean, std = np.asarray(mean, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1)
        if mean.shape != (len(X),) or std.shape != (len(X),) or not np.all(np.isfinite(mean) & np.isfinite(std)):
            raise ValueError("interaction GP must return one finite mean and std per row")
        return mean, std


def compact_arrays(bundle: dict, source_sha256: str = "") -> dict[str, NDArray]:
    """把已訓練的 pkl 內容轉成精簡格式：只存重建所需的數字，不存 n×n 分解矩陣。

    重建時用相同的訓練輸入、已標準化答案與 kernel 數值重新計算分解，預測與原模型一致。
    """
    pipeline = bundle["pipeline"]
    scaler, gp = pipeline.named_steps["scaler"], pipeline.named_steps["gp"]
    kernel = gp.kernel_
    metadata = {
        "format": COMPACT_FORMAT, "feature_names": list(bundle["feature_names"]), "target": bundle["target"],
        "label_horizon_seconds": bundle["label_horizon_seconds"],
        "history_segment_seconds": bundle["history_segment_seconds"],
        "kernel": str(kernel), "source_sha256": source_sha256,
        "versions": bundle.get("versions", {}),
    }
    return {
        "metadata": np.array(json.dumps(metadata, ensure_ascii=False)),
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "scaler_var": scaler.var_,
        "scaler_samples": np.asarray(scaler.n_samples_seen_),
        "x_train": gp.X_train_, "y_train": gp.y_train_,
        "y_mean": np.asarray(gp._y_train_mean), "y_std": np.asarray(gp._y_train_std),
        "constant": np.asarray(kernel.k1.k1.constant_value), "length_scale": np.asarray(kernel.k1.k2.length_scale),
        "noise": np.asarray(kernel.k2.noise_level), "alpha": np.asarray(gp.alpha),
    }


def load_compact(path: Path) -> dict:
    """讀精簡檔（不經 pickle），以固定超參數重建 StandardScaler＋GP Pipeline。"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(str(arrays["metadata"]))
    if metadata.get("format") != COMPACT_FORMAT:
        raise ValueError(f"unsupported compact model format: {metadata.get('format')!r}")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = arrays["scaler_mean"], arrays["scaler_scale"], arrays["scaler_var"]
    scaler.n_samples_seen_ = arrays["scaler_samples"]
    scaler.n_features_in_ = len(scaler.mean_)
    kernel = (ConstantKernel(float(arrays["constant"]), "fixed")
              * RBF(arrays["length_scale"], "fixed") + WhiteKernel(float(arrays["noise"]), "fixed"))
    # 以已標準化的答案擬合（不再標準化），再放回原本的平均與標準差，計算步驟與原模型相同。
    gp = GaussianProcessRegressor(kernel=kernel, alpha=float(arrays["alpha"]), optimizer=None, normalize_y=False)
    gp.fit(arrays["x_train"], arrays["y_train"])
    gp._y_train_mean, gp._y_train_std = arrays["y_mean"], arrays["y_std"]
    return {"pipeline": Pipeline([("scaler", scaler), ("gp", gp)]),
            "feature_names": tuple(metadata["feature_names"]), "target": metadata["target"],
            "label_horizon_seconds": metadata["label_horizon_seconds"],
            "history_segment_seconds": metadata["history_segment_seconds"], "compact_metadata": metadata}


def lateral_profile(times: NDArray[np.float64], d0: float, delta: float, horizon: float) -> NDArray[np.float64]:
    """0～horizon 由 d0 線性移到 d0+delta，之後維持。"""
    return d0 + delta * np.minimum(np.asarray(times, dtype=float) / horizon, 1.0)


def predict_profile_trajectory(
    vehicle: VehicleObservation, track: Track, offsets: NDArray[np.float64],
    speed: float, horizon: float, time_step: float,
) -> BicyclePrediction:
    """s(t)=s0+speed·t，d(t) 由 offsets 給定（與時間格點同長）；時間格點與自行車預測相同。"""
    if not track.closed:
        raise ValueError("profile prediction requires a closed track")
    if any(not math.isfinite(value) or value <= 0.0 for value in (horizon, time_step)):
        raise ValueError("prediction horizon and time_step must be positive and finite")
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("speed must be non-negative and finite")
    times = np.minimum(np.arange(math.ceil(horizon / time_step) + 1) * time_step, horizon)
    d = np.asarray(offsets, dtype=float)
    if d.shape != times.shape or not np.all(np.isfinite(d)):
        raise ValueError("offsets must give one finite d per prediction time")
    progress = speed * times
    s = (vehicle.track_position.s + progress) % track_length(track)
    points = track_to_world_points(s, d, track)
    if speed == 0.0:
        yaws = np.full(len(times), vehicle.state.yaw)
    else:
        # 航向取往下一點的方向；最後一點沿用前一段。
        directions = np.diff(points, axis=0)
        yaws = np.arctan2(directions[:, 1], directions[:, 0])
        yaws = np.append(yaws, yaws[-1])
    # t=0 保留觀測狀態，不讓預測隱藏目前已存在的重疊。
    states = [vehicle.state, *(VehicleState(float(x), float(y), float(yaw), vehicle.state.velocity)
                               for (x, y), yaw in zip(points[1:], yaws[1:]))]
    return BicyclePrediction(tuple(states), tuple(float(p) for p in progress),
                             tuple(float(t) for t in times), (vehicle.track_position.d, *(float(v) for v in d[1:])))
