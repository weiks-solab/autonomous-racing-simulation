"""Ordered, extensible scalar features and targets; no estimator dependency."""

from dataclasses import dataclass
from typing import Callable, Mapping, TypeVar

from core.track import Track
from strategy.observations import VehicleObservation
from training_data.schema import DatasetSchema, FieldSpec, FrameSnapshot, TargetTiming, VehiclePair


Definition = TypeVar("Definition")


@dataclass(frozen=True)
class FeatureContext:
    """Feature extractors deliberately have no access to the future frame."""

    current: FrameSnapshot
    pair: VehiclePair
    track: Track

    @property
    def ego(self) -> VehicleObservation:
        return self.current.observations[self.pair.ego_id]

    @property
    def opponent(self) -> VehicleObservation:
        return self.current.observations[self.pair.opponent_id]


@dataclass(frozen=True)
class TargetContext:
    current: FrameSnapshot
    future: FrameSnapshot
    pair: VehiclePair
    track: Track

    @property
    def ego(self) -> VehicleObservation:
        return self.current.observations[self.pair.ego_id]

    @property
    def opponent(self) -> VehicleObservation:
        return self.current.observations[self.pair.opponent_id]

    @property
    def future_opponent(self) -> VehicleObservation:
        return self.future.observations[self.pair.opponent_id]

    @property
    def future_ego(self) -> VehicleObservation:
        return self.future.observations[self.pair.ego_id]


@dataclass(frozen=True)
class InputField:
    name: str
    unit: str
    description: str
    value: Callable[[FeatureContext], float]


@dataclass(frozen=True)
class OutputField:
    name: str
    unit: str
    description: str
    value: Callable[[TargetContext], float]
    timing: TargetTiming = "future"


class ExtractorRegistry:
    """Each registry owns its definitions; extending one does not change others."""

    def __init__(self) -> None:
        self._features: dict[str, tuple[FieldSpec, Callable[[FeatureContext], float]]] = {}
        self._targets: dict[str, tuple[FieldSpec, Callable[[TargetContext], float]]] = {}
        self._target_timings: dict[str, TargetTiming] = {}

    def register_feature(self, spec: FieldSpec, extractor: Callable[[FeatureContext], float]) -> None:
        if spec.name in self._features or not callable(extractor):
            raise ValueError("feature must have a unique name and callable extractor")
        self._features[spec.name] = (spec, extractor)

    def register_target(
        self, spec: FieldSpec, extractor: Callable[[TargetContext], float],
        *, timing: TargetTiming = "future",
    ) -> None:
        if spec.name in self._targets or not callable(extractor):
            raise ValueError("target must have a unique name and callable extractor")
        if timing not in ("current", "future"):
            raise ValueError("答案的 timing 只能是 'current'（當下）或 'future'（未來）。")
        self._targets[spec.name] = (spec, extractor)
        self._target_timings[spec.name] = timing

    def target_timing(self, schema: DatasetSchema) -> TargetTiming:
        self.resolve_targets(schema)
        timings = {self._target_timings[name] for name in schema.target_names}
        if len(timings) != 1:
            raise ValueError("當下與未來的答案請分開收集，不能共用同一份時間設定。")
        return next(iter(timings))

    def validate_horizon(self, schema: DatasetSchema, steps: int) -> None:
        timing = self.target_timing(schema)
        if timing == "current" and steps != 0:
            raise ValueError("當下答案必須使用 --horizon-steps 0，或省略此選項讓程式自動選擇。")
        if timing == "future" and steps < 1:
            raise ValueError("未來答案必須使用至少 1 個 horizon step，不能使用 0。")

    def resolve_features(
        self, schema: DatasetSchema,
    ) -> tuple[tuple[FieldSpec, Callable[[FeatureContext], float]], ...]:
        return self._resolve(schema.feature_names, self._features)

    def resolve_targets(
        self, schema: DatasetSchema,
    ) -> tuple[tuple[FieldSpec, Callable[[TargetContext], float]], ...]:
        return self._resolve(schema.target_names, self._targets)

    @staticmethod
    def _resolve(names: tuple[str, ...], definitions: Mapping[str, Definition]) -> tuple[Definition, ...]:
        unknown = set(names) - set(definitions)
        if unknown:
            raise ValueError(f"unregistered data fields: {', '.join(sorted(unknown))}")
        return tuple(definitions[name] for name in names)


def create_default_registry() -> ExtractorRegistry:
    """The command-line collector also uses this one editable field list."""
    from training_data.fields import INPUT_FIELDS, OUTPUT_FIELDS

    registry = ExtractorRegistry()
    for item in INPUT_FIELDS:
        registry.register_feature(FieldSpec(item.name, item.unit, item.description), item.value)
    for item in OUTPUT_FIELDS:
        registry.register_target(
            FieldSpec(item.name, item.unit, item.description), item.value, timing=item.timing,
        )
    return registry
