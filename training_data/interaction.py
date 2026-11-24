"""由過去與當下觀測建立互動特徵，以未來觀測建立答案。"""

from collections import Counter
import math
from typing import Literal, Mapping

from core.track import Track
from core.track_geometry import signed_longitudinal_gap, track_length
# 特徵計算與線上策略共用，確保訓練與使用時算法一致。
from strategy.interaction_features import (
    centerline_curvature as curvature, heading_error, history_state_features,
)
from training_data.schema import (
    DatasetSchema, FieldSpec, RecordedEpisode, SamplingConfig, TrainingDataset,
    TrainingSample, VehiclePair,
)

FEATURES = ("gap", "rear_d", "front_d", "relative_speed", "front_speed",
            "front_lateral_speed", "front_curvature")
ACTION_FEATURES = FEATURES + ("d_target",)
UNITS = ("m", "m", "m", "m/s", "m/s", "m/s", "1/m")
DESCRIPTIONS = ("前車 s 減後車 s，短距有號車距", "後車當下 d", "前車當下 d",
                "後車車身速度減前車車身速度", "前車車身速度",
                "前車 d 的過去 0.1 s 差分速度", "前車所在中心線曲率，左彎為正")

# 互動歷史版：雙車各自的 d、車頭角誤差、車速，加上過去兩段的橫移量。
HISTORY_SEGMENT_SECONDS = 0.2
HISTORY_BASE_FEATURES = ("gap", "rear_d", "rear_heading_error", "rear_speed",
                         "front_d", "front_heading_error", "front_speed", "front_curvature")
HISTORY_TERMS = ("rear_dd_early", "front_dd_early", "rear_dd_late", "front_dd_late")
HISTORY_FEATURES = HISTORY_BASE_FEATURES + HISTORY_TERMS
HISTORY_ACTION_FEATURES = HISTORY_FEATURES + ("d_target",)
HISTORY_UNITS = ("m", "m", "rad", "m/s", "m", "rad", "m/s", "1/m", "m", "m", "m", "m")
FeatureSet = Literal["basic", "history"]


def history_descriptions(segment_seconds: float) -> tuple[str, ...]:
    early = f"t-{2 * segment_seconds:g} 至 t-{segment_seconds:g} s"
    late = f"t-{segment_seconds:g} 至 t s"
    return ("前車 s 減後車 s，短距有號車距", "後車當下 d，左正",
            "後車車頭角減中心線切線角，左轉為正", "後車車身速度",
            "前車當下 d，左正", "前車車頭角減中心線切線角，左轉為正", "前車車身速度",
            "前車所在中心線曲率，左彎為正",
            f"後車 d 在 {early} 的變化", f"前車 d 在 {early} 的變化",
            f"後車 d 在 {late} 的變化", f"前車 d 在 {late} 的變化")


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


def history_segment_steps(segment_seconds: float, time_step: float) -> int:
    """歷史每段的步數；必須是模擬步長的正整數倍。"""
    if not math.isfinite(segment_seconds) or segment_seconds <= 0.0:
        raise ValueError("history segment must be positive and finite")
    steps = round(segment_seconds / time_step)
    if steps < 1 or not math.isclose(steps * time_step, segment_seconds, abs_tol=1e-9):
        raise ValueError("history segment must be a whole number of time steps")
    return steps


def history_features(
    episode: RecordedEpisode, index: int, pair: VehiclePair, track: Track, segment_steps: int,
) -> tuple[float, ...]:
    """只讀 t-2段、t-1段與 t 三幀；歷史四項依序為後車前段、前車前段、後車後段、前車後段。"""
    if segment_steps < 1 or index < 2 * segment_steps:
        raise ValueError("not enough history before this frame")
    past = (episode.frames[index - 2 * segment_steps], episode.frames[index - segment_steps])
    rear_past, front_past = (tuple(frame.observations[key].track_position.d for frame in past)
                             for key in (pair.ego_id, pair.opponent_id))
    current = episode.frames[index]
    return history_state_features(current.observations[pair.ego_id], current.observations[pair.opponent_id],
                                  rear_past, front_past, track)


def build_interaction_dataset(
    episodes: tuple[RecordedEpisode, ...], track: Track, pair: VehiclePair,
    sampling: SamplingConfig,
    action_targets: Mapping[tuple[str, int], float] | None = None,
    feature_set: FeatureSet = "basic",
    history_segment_seconds: float = HISTORY_SEGMENT_SECONDS,
) -> TrainingDataset:
    """標籤窗不得跨 episode 或碰撞；basic 第一筆沒有歷史速度，history 需完整歷史窗。"""
    if sampling.label_horizon_steps < 1:
        raise ValueError("interaction prediction requires a future horizon")
    if feature_set not in ("basic", "history"):
        raise ValueError("feature_set must be 'basic' or 'history'")
    history_steps = 0
    if feature_set == "history":
        history_steps = history_segment_steps(history_segment_seconds, sampling.time_step)
    samples, skipped = [], Counter()
    for episode in episodes:
        for index in range(0, len(episode.frames), sampling.sample_stride_steps):
            end = index + sampling.label_horizon_steps
            if index == 0 or index < 2 * history_steps:
                skipped["no_history" if feature_set == "basic" else "insufficient_history"] += 1
                continue
            if end >= len(episode.frames):
                skipped["insufficient_future"] += 1
                continue
            # history 的檢查範圍包含歷史窗，避免歷史橫移來自碰撞中的狀態。
            interval = episode.frames[index - 2 * history_steps:end + 1]
            if any(not {pair.ego_id, pair.opponent_id} <= set(frame.observations) for frame in interval):
                skipped["missing_vehicle"] += 1
                continue
            if any({pair.ego_id, pair.opponent_id}.intersection(collision)
                   for frame in interval for collision in frame.collision_pairs):
                skipped["collision"] += 1
                continue
            current, future = episode.frames[index], interval[-1]
            if not math.isclose(future.time_seconds - current.time_seconds, sampling.label_horizon_seconds):
                raise ValueError("frame timing does not match label horizon")
            x = (features(episode, index, pair, track) if feature_set == "basic"
                 else history_features(episode, index, pair, track, history_steps))
            if action_targets is not None:
                # 使用此時真正執行的候選，不把未執行候選配上同一個未來答案。
                key = (episode.episode_id, current.step_index)
                if key not in action_targets:
                    raise ValueError(f"missing executed d_target: {key}")
                x += (float(action_targets[key]),)
            y = future.observations[pair.opponent_id].track_position.d - current.observations[pair.opponent_id].track_position.d
            if not all(math.isfinite(value) for value in (*x, y)):
                skipped["non_finite"] += 1
                continue
            samples.append(TrainingSample(episode.episode_id, current.step_index, current.time_seconds, pair, x, (y,)))
    if feature_set == "basic":
        base, units, descriptions = FEATURES, UNITS, DESCRIPTIONS
    else:
        base, units = HISTORY_FEATURES, HISTORY_UNITS
        descriptions = history_descriptions(history_steps * sampling.time_step)
    names = base if action_targets is None else base + ("d_target",)
    specs = tuple(FieldSpec(name, unit, description) for name, unit, description in zip(base, units, descriptions))
    if action_targets is not None:
        specs += (FieldSpec("d_target", "m", "後車當下實際採用的候選目標 d；後續決策仍可更新"),)
    return TrainingDataset(
        DatasetSchema(names, ("opponent_delta_d",)), sampling, tuple(samples), specs,
        (FieldSpec("opponent_delta_d", "m", "前車未來 d 減當下 d"),), skipped,
    )
