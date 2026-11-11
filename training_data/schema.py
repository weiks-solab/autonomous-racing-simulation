"""Value objects for simulator-independent training data generation."""

from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from strategy.observations import VehicleObservation


SAMPLE_KEY_COLUMNS = ("episode_id", "step_index", "time_s", "ego_id", "opponent_id")
DEFAULT_FEATURES = ("relative_s", "relative_d", "relative_speed", "opponent_d")
DEFAULT_TARGETS = ("opponent_delta_d",)
TargetTiming = Literal["current", "future"]


def require_integer(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def require_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class VehiclePair:
    ego_id: str
    opponent_id: str

    def __post_init__(self) -> None:
        require_id("ego_id", self.ego_id)
        require_id("opponent_id", self.opponent_id)
        if self.ego_id == self.opponent_id:
            raise ValueError("ego and opponent must be different vehicles")


@dataclass(frozen=True)
class SamplingConfig:
    time_step: float = 0.02
    label_horizon_steps: int = 1
    sample_stride_steps: int = 1
    reject_collisions: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_step) or self.time_step <= 0.0:
            raise ValueError("time_step must be positive and finite")
        require_integer("label_horizon_steps", self.label_horizon_steps)
        require_integer("sample_stride_steps", self.sample_stride_steps, 1)
        if not isinstance(self.reject_collisions, bool):
            raise ValueError("reject_collisions must be a bool")

    @property
    def label_horizon_seconds(self) -> float:
        return self.time_step * self.label_horizon_steps

    @property
    def target_timing(self) -> TargetTiming:
        return "current" if self.label_horizon_steps == 0 else "future"


@dataclass(frozen=True)
class DatasetSchema:
    feature_names: tuple[str, ...] = DEFAULT_FEATURES
    target_names: tuple[str, ...] = DEFAULT_TARGETS

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "target_names", tuple(self.target_names))
        if not self.feature_names or not self.target_names:
            raise ValueError("at least one feature and target are required")
        names = self.feature_names + self.target_names
        if len(set(names)) != len(names):
            raise ValueError("feature and target names must be unique")
        for name in names:
            validate_field_name(name)


def validate_field_name(name: str) -> None:
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError("field names must be lowercase identifiers")
    if name in SAMPLE_KEY_COLUMNS:
        raise ValueError(f"field name is reserved: {name}")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    unit: str
    description: str

    def __post_init__(self) -> None:
        validate_field_name(self.name)
        require_id("unit", self.unit)
        require_id("description", self.description)


@dataclass(frozen=True)
class FrameSnapshot:
    """All observations belong to the same simulation step."""

    episode_id: str
    step_index: int
    time_seconds: float
    observations: Mapping[str, VehicleObservation]
    collision_pairs: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        require_id("episode_id", self.episode_id)
        require_integer("step_index", self.step_index)
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0.0:
            raise ValueError("time_seconds must be non-negative and finite")
        observations = dict(self.observations)
        if not observations or any(key != item.vehicle_id for key, item in observations.items()):
            raise ValueError("observation keys must match non-empty vehicle observations")
        collisions = set()
        for pair in self.collision_pairs:
            if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(observations):
                raise ValueError("collisions must reference two observed vehicles")
            collisions.add(tuple(sorted(pair)))
        object.__setattr__(self, "observations", MappingProxyType(observations))
        object.__setattr__(self, "collision_pairs", frozenset(collisions))


@dataclass(frozen=True)
class RecordedEpisode:
    episode_id: str
    frames: tuple[FrameSnapshot, ...]

    def __post_init__(self) -> None:
        require_id("episode_id", self.episode_id)
        object.__setattr__(self, "frames", tuple(self.frames))
        if not self.frames:
            raise ValueError("an episode must contain its initial frame")
        for index, frame in enumerate(self.frames):
            if frame.episode_id != self.episode_id or frame.step_index != index:
                raise ValueError("episode frames must be consecutive, starting at step zero")


@dataclass(frozen=True)
class TrainingSample:
    episode_id: str
    step_index: int
    time_seconds: float
    pair: VehiclePair
    features: tuple[float, ...]
    targets: tuple[float, ...]


@dataclass(frozen=True)
class TrainingDataset:
    schema: DatasetSchema
    sampling: SamplingConfig
    samples: tuple[TrainingSample, ...]
    feature_specs: tuple[FieldSpec, ...]
    target_specs: tuple[FieldSpec, ...]
    skipped: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if tuple(item.name for item in self.feature_specs) != self.schema.feature_names:
            raise ValueError("feature specifications must match schema order")
        if tuple(item.name for item in self.target_specs) != self.schema.target_names:
            raise ValueError("target specifications must match schema order")
        for sample in self.samples:
            if len(sample.features) != len(self.feature_specs) or len(sample.targets) != len(self.target_specs):
                raise ValueError("sample dimensions must match schema")
            if not all(math.isfinite(value) for value in (*sample.features, *sample.targets)):
                raise ValueError("sample values must be finite")
            require_id("episode_id", sample.episode_id)
            require_integer("step_index", sample.step_index)
            if not math.isclose(sample.time_seconds, sample.step_index * self.sampling.time_step,
                                rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("sample time must match step_index and time_step")
        object.__setattr__(self, "skipped", MappingProxyType(dict(self.skipped)))

    @property
    def X(self) -> NDArray[np.float64]:
        return np.asarray([sample.features for sample in self.samples], dtype=float).reshape(
            len(self.samples), len(self.schema.feature_names)
        )

    @property
    def Y(self) -> NDArray[np.float64]:
        return np.asarray([sample.targets for sample in self.samples], dtype=float).reshape(
            len(self.samples), len(self.schema.target_names)
        )
