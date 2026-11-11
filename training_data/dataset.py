"""Turn synchronized episodes into paired current/future training samples."""

from collections import Counter
import math
from typing import Iterable

from core.track import Track
from training_data.extractors import (
    ExtractorRegistry,
    FeatureContext,
    TargetContext,
    create_default_registry,
)
from training_data.schema import (
    DatasetSchema,
    FrameSnapshot,
    RecordedEpisode,
    SamplingConfig,
    TrainingDataset,
    TrainingSample,
    VehiclePair,
)


def _interval_rejection(
    frames: tuple[FrameSnapshot, ...], pair: VehiclePair, reject_collisions: bool,
) -> str | None:
    ids = {pair.ego_id, pair.opponent_id}
    if any(not ids <= set(frame.observations) for frame in frames):
        return "missing_vehicle"
    if reject_collisions and any(
        ids.intersection(collision)
        for frame in frames for collision in frame.collision_pairs
    ):
        # 與第三台車碰撞也會影響這組配對，整個標籤區間都要檢查。
        return "collision"
    return None


def build_dataset(
    episodes: Iterable[RecordedEpisode],
    track: Track,
    pairs: Iterable[VehiclePair],
    sampling: SamplingConfig = SamplingConfig(),
    schema: DatasetSchema = DatasetSchema(),
    registry: ExtractorRegistry | None = None,
) -> TrainingDataset:
    """Align by integer steps, never across episodes or changing opponent IDs."""
    pairs = tuple(pairs)
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("vehicle pairs must be non-empty and unique")
    registry = create_default_registry() if registry is None else registry
    features = registry.resolve_features(schema)
    targets = registry.resolve_targets(schema)
    registry.validate_horizon(schema, sampling.label_horizon_steps)
    samples: list[TrainingSample] = []
    skipped: Counter[str] = Counter()
    episode_ids: set[str] = set()
    for episode in episodes:
        if episode.episode_id in episode_ids:
            raise ValueError("episode IDs must be unique")
        episode_ids.add(episode.episode_id)
        for frame in episode.frames:
            if not math.isclose(frame.time_seconds, frame.step_index * sampling.time_step,
                                rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("recorded times do not match the dataset time_step")
        for index in range(0, len(episode.frames), sampling.sample_stride_steps):
            future_index = index + sampling.label_horizon_steps
            if future_index >= len(episode.frames):
                skipped["insufficient_future"] += len(pairs)
                continue
            interval = episode.frames[index:future_index + 1]
            for pair in pairs:
                reason = _interval_rejection(interval, pair, sampling.reject_collisions)
                if reason is not None:
                    skipped[reason] += 1
                    continue
                current, future = interval[0], interval[-1]
                feature_context = FeatureContext(current, pair, track)
                target_context = TargetContext(current, future, pair, track)
                x = tuple(float(extractor(feature_context)) for _, extractor in features)
                y = tuple(float(extractor(target_context)) for _, extractor in targets)
                if not all(math.isfinite(value) for value in (*x, *y)):
                    skipped["non_finite"] += 1
                    continue
                samples.append(TrainingSample(
                    episode.episode_id, current.step_index, current.time_seconds, pair, x, y,
                ))
    return TrainingDataset(
        schema, sampling, tuple(samples),
        tuple(spec for spec, _ in features), tuple(spec for spec, _ in targets), skipped,
    )
