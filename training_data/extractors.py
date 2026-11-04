"""Ordered, extensible scalar features and targets; no estimator dependency."""

from dataclasses import dataclass
from typing import Callable, Mapping, TypeVar

from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length
from strategy.observations import VehicleObservation
from training_data.schema import DatasetSchema, FieldSpec, FrameSnapshot, VehiclePair


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
    def opponent(self) -> VehicleObservation:
        return self.current.observations[self.pair.opponent_id]

    @property
    def future_opponent(self) -> VehicleObservation:
        return self.future.observations[self.pair.opponent_id]


class ExtractorRegistry:
    """Each registry owns its definitions; extending one does not change others."""

    def __init__(self) -> None:
        self._features: dict[str, tuple[FieldSpec, Callable[[FeatureContext], float]]] = {}
        self._targets: dict[str, tuple[FieldSpec, Callable[[TargetContext], float]]] = {}

    def register_feature(self, spec: FieldSpec, extractor: Callable[[FeatureContext], float]) -> None:
        if spec.name in self._features or not callable(extractor):
            raise ValueError("feature must have a unique name and callable extractor")
        self._features[spec.name] = (spec, extractor)

    def register_target(self, spec: FieldSpec, extractor: Callable[[TargetContext], float]) -> None:
        if spec.name in self._targets or not callable(extractor):
            raise ValueError("target must have a unique name and callable extractor")
        self._targets[spec.name] = (spec, extractor)

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


def _relative_s(context: FeatureContext) -> float:
    ego = context.ego.track_position
    opponent = context.opponent.track_position
    if context.track.closed:
        return signed_longitudinal_gap(opponent, ego, track_length(context.track))
    return ego.s - opponent.s


def create_default_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    features = (
        ("relative_s", "m", "Ego s minus opponent s; shortest signed gap on a closed track.", _relative_s),
        ("relative_d", "m", "Ego d minus opponent d; positive means ego is to the left.",
         lambda ctx: ctx.ego.track_position.d - ctx.opponent.track_position.d),
        ("relative_speed", "m/s", "Ego scalar vehicle speed minus opponent scalar vehicle speed.",
         lambda ctx: ctx.ego.state.velocity - ctx.opponent.state.velocity),
        ("opponent_d", "m", "Current opponent d relative to the shared centerline; left is positive.",
         lambda ctx: ctx.opponent.track_position.d),
        ("ego_speed", "m/s", "Current ego scalar vehicle speed.", lambda ctx: ctx.ego.state.velocity),
        ("opponent_speed", "m/s", "Current opponent scalar vehicle speed.",
         lambda ctx: ctx.opponent.state.velocity),
    )
    for name, unit, description, extractor in features:
        registry.register_feature(FieldSpec(name, unit, description), extractor)
    targets = (
        ("opponent_delta_d", "m", "Opponent d at t + horizon minus opponent d at t.",
         lambda ctx: ctx.future_opponent.track_position.d - ctx.opponent.track_position.d),
        ("opponent_future_d", "m", "Opponent absolute track-relative d at t + horizon.",
         lambda ctx: ctx.future_opponent.track_position.d),
        ("opponent_delta_speed", "m/s", "Opponent scalar speed at t + horizon minus its speed at t.",
         lambda ctx: ctx.future_opponent.state.velocity - ctx.opponent.state.velocity),
    )
    for name, unit, description, extractor in targets:
        registry.register_target(FieldSpec(name, unit, description), extractor)
    return registry
