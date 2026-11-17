"""由過去與當下觀測建立互動特徵，以未來觀測建立答案。"""

from collections import Counter
import math

import numpy as np

from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length, track_to_world
from training_data.schema import (
    DatasetSchema, FieldSpec, RecordedEpisode, SamplingConfig, TrainingDataset,
    TrainingSample, VehiclePair,
)

FEATURES = ("gap", "rear_d", "front_d", "relative_speed", "front_speed",
            "front_lateral_speed", "front_curvature")
UNITS = ("m", "m", "m", "m/s", "m/s", "m/s", "1/m")
DESCRIPTIONS = ("前車 s 減後車 s，短距有號車距", "後車當下 d", "前車當下 d",
                "後車車身速度減前車車身速度", "前車車身速度",
                "前車 d 的過去 0.1 s 差分速度", "前車所在中心線曲率，左彎為正")


def curvature(track: Track, s: float, spacing: float = 0.5) -> float:
    """用中心線前後各 0.5 m 的三點估計有號曲率，左彎為正。"""
    a, b, c = (np.asarray(track_to_world(s + offset, 0.0, track))
               for offset in (-spacing, 0.0, spacing))
    u, v = b - a, c - a
    denominator = np.linalg.norm(u) * np.linalg.norm(v) * np.linalg.norm(c - b)
    return 0.0 if denominator < 1e-12 else float(2 * (u[0] * v[1] - u[1] * v[0]) / denominator)


def features(episode: RecordedEpisode, index: int, pair: VehiclePair, track: Track) -> tuple[float, ...]:
    """橫向速度只用最近至多 0.1 s 的過去差分，禁止未來差分。"""
    frame = episode.frames[index]
    rear, front = (frame.observations[key] for key in (pair.ego_id, pair.opponent_id))
    previous_index = index
    while previous_index > 0 and frame.time_seconds - episode.frames[previous_index].time_seconds < 0.1 - 1e-9:
        previous_index -= 1
    previous = episode.frames[previous_index]
    elapsed = frame.time_seconds - previous.time_seconds
    lateral_speed = ((front.track_position.d - previous.observations[pair.opponent_id].track_position.d) / elapsed
                     if elapsed > 0 else 0.0)
    gap = signed_longitudinal_gap(rear.track_position, front.track_position, track_length(track))
    return (gap, rear.track_position.d, front.track_position.d,
            rear.state.velocity - front.state.velocity, front.state.velocity,
            lateral_speed, curvature(track, front.track_position.s))


def build_interaction_dataset(
    episodes: tuple[RecordedEpisode, ...], track: Track, pair: VehiclePair,
    sampling: SamplingConfig,
) -> TrainingDataset:
    """標籤窗不得跨 episode 或碰撞；第一筆沒有歷史速度，略過。"""
    if sampling.label_horizon_steps < 1:
        raise ValueError("interaction prediction requires a future horizon")
    samples, skipped = [], Counter()
    for episode in episodes:
        for index in range(0, len(episode.frames), sampling.sample_stride_steps):
            end = index + sampling.label_horizon_steps
            if index == 0:
                skipped["no_history"] += 1
                continue
            if end >= len(episode.frames):
                skipped["insufficient_future"] += 1
                continue
            interval = episode.frames[index:end + 1]
            if any(not {pair.ego_id, pair.opponent_id} <= set(frame.observations) for frame in interval):
                skipped["missing_vehicle"] += 1
                continue
            if any({pair.ego_id, pair.opponent_id}.intersection(collision)
                   for frame in interval for collision in frame.collision_pairs):
                skipped["collision"] += 1
                continue
            current, future = interval[0], interval[-1]
            if not math.isclose(future.time_seconds - current.time_seconds, sampling.label_horizon_seconds):
                raise ValueError("frame timing does not match label horizon")
            x = features(episode, index, pair, track)
            y = future.observations[pair.opponent_id].track_position.d - current.observations[pair.opponent_id].track_position.d
            if not all(math.isfinite(value) for value in (*x, y)):
                skipped["non_finite"] += 1
                continue
            samples.append(TrainingSample(episode.episode_id, current.step_index, current.time_seconds, pair, x, (y,)))
    return TrainingDataset(
        DatasetSchema(FEATURES, ("opponent_delta_d",)), sampling, tuple(samples),
        tuple(FieldSpec(name, unit, description) for name, unit, description in zip(FEATURES, UNITS, DESCRIPTIONS)),
        (FieldSpec("opponent_delta_d", "m", "前車未來 d 減當下 d"),), skipped,
    )
