"""CSV/JSON bundles, including raw observations for later feature rebuilding."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.track import Track
from core.track_geometry import TrackPosition
from core.vehicle_geometry import CollisionConfig, VehicleGeometry
from core.vehicle_state import VehicleState
from strategy.observations import VehicleObservation
from strategy.offset_path import LateralManeuver
from training_data.collector import TrainingDataCollector
from training_data.schema import (
    SAMPLE_KEY_COLUMNS,
    DatasetSchema,
    FieldSpec,
    RecordedEpisode,
    SamplingConfig,
    TrainingDataset,
    TrainingSample,
    VehiclePair,
)


FORMAT_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_COLUMNS = (
    "episode_id", "step_index", "time_s", "vehicle_id", "x", "y", "yaw", "velocity",
    "s", "d", "centerline_index", "length", "width", "rear_axle_to_center",
    "maneuver_start_s", "maneuver_start_d", "maneuver_target_d", "maneuver_transition_length",
    "collision_pairs",
    "collision_length", "collision_width",
)


def project_path(path: str | Path) -> Path:
    """Keep generated files and replay reads within this repository."""
    path = Path(path)
    resolved = (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError("training data paths must remain inside this repository")
    return resolved


def _write_raw(path: Path, episodes: Iterable[RecordedEpisode]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(RAW_COLUMNS)
        for episode in episodes:
            for frame in episode.frames:
                collisions = json.dumps(sorted(frame.collision_pairs))
                for vehicle_id, observation in frame.observations.items():
                    state = observation.state
                    position = observation.track_position
                    geometry = observation.geometry
                    collision = geometry.collision
                    maneuver = observation.maneuver
                    maneuver_values = ("", "", "", "") if maneuver is None else (
                        maneuver.start_s, maneuver.start_d, maneuver.target_d, maneuver.transition_length,
                    )
                    writer.writerow((
                        episode.episode_id, frame.step_index, frame.time_seconds, vehicle_id,
                        state.x, state.y, state.yaw, state.velocity,
                        position.s, position.d, position.centerline_index,
                        geometry.length, geometry.width, geometry.rear_axle_to_center,
                        *maneuver_values, collisions,
                        "" if collision is None else collision.length,
                        "" if collision is None else collision.width,
                    ))


def export_dataset(
    dataset: TrainingDataset,
    episodes: Iterable[RecordedEpisode],
    track: Track,
    output_directory: str | Path,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    """Create a new bundle directory; existing runs are never overwritten."""
    episodes = tuple(episodes)
    output = project_path(output_directory)
    metadata = {
        "format_version": FORMAT_VERSION,
        "schema": asdict(dataset.schema),
        "feature_specs": [asdict(spec) for spec in dataset.feature_specs],
        "target_specs": [asdict(spec) for spec in dataset.target_specs],
        "dtype": "float64",
        "sampling": asdict(dataset.sampling),
        "target_timing": dataset.sampling.target_timing,
        "label_horizon_seconds": dataset.sampling.label_horizon_seconds,
        "reference_point": "rear axle",
        "coordinate_convention": "s follows centerline order; d is positive left; relative = ego - opponent",
        "track": {"centerline": track.centerline.tolist(), "closed": track.closed, "width": track.width},
        "episodes": [{"episode_id": ep.episode_id, "frame_count": len(ep.frames)} for ep in episodes],
        "sample_count": len(dataset.samples),
        "skipped": dict(dataset.skipped),
        "run": {} if run_metadata is None else run_metadata,
    }
    # 檔案建立前先驗證 metadata，避免序列化失敗留下半份資料。
    metadata_json = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((*SAMPLE_KEY_COLUMNS, *dataset.schema.feature_names, *dataset.schema.target_names))
        for sample in dataset.samples:
            writer.writerow((
                sample.episode_id, sample.step_index, sample.time_seconds,
                sample.pair.ego_id, sample.pair.opponent_id, *sample.features, *sample.targets,
            ))
    _write_raw(output / "raw_states.csv", episodes)
    (output / "metadata.json").write_text(metadata_json + "\n", encoding="utf-8")
    return output


def _read_metadata(directory: Path) -> dict[str, Any]:
    with project_path(directory / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported training data format version")
    timing = SamplingConfig(**metadata["sampling"]).target_timing
    if metadata.get("target_timing", timing) != timing:
        raise ValueError("metadata 的答案時間與 sampling 設定不一致。")
    return metadata


def load_dataset(directory: str | Path) -> TrainingDataset:
    directory = project_path(directory)
    metadata = _read_metadata(directory)
    schema = DatasetSchema(**metadata["schema"])
    sampling = SamplingConfig(**metadata["sampling"])
    expected = [*SAMPLE_KEY_COLUMNS, *schema.feature_names, *schema.target_names]
    samples = []
    with project_path(directory / "samples.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected:
            raise ValueError("sample columns do not match saved schema order")
        for row in reader:
            samples.append(TrainingSample(
                row["episode_id"], int(row["step_index"]), float(row["time_s"]),
                VehiclePair(row["ego_id"], row["opponent_id"]),
                tuple(float(row[name]) for name in schema.feature_names),
                tuple(float(row[name]) for name in schema.target_names),
            ))
    if len(samples) != metadata["sample_count"]:
        raise ValueError("sample count does not match metadata")
    return TrainingDataset(
        schema, sampling, tuple(samples),
        tuple(FieldSpec(**spec) for spec in metadata["feature_specs"]),
        tuple(FieldSpec(**spec) for spec in metadata["target_specs"]), metadata["skipped"],
    )


def _read_observation(row: dict[str, str]) -> VehicleObservation:
    maneuver = None
    if row["maneuver_start_s"]:
        maneuver = LateralManeuver(*(float(row[name]) for name in RAW_COLUMNS[14:18]))
    collision = None
    if row.get("collision_length") or row.get("collision_width"):
        collision = CollisionConfig(float(row["collision_length"]), float(row["collision_width"]))
    return VehicleObservation(
        row["vehicle_id"],
        VehicleState(*(float(row[name]) for name in ("x", "y", "yaw", "velocity"))),
        TrackPosition(float(row["s"]), float(row["d"]), int(row["centerline_index"])),
        VehicleGeometry(*(float(row[name]) for name in ("length", "width", "rear_axle_to_center")),
                        collision=collision),
        maneuver,
    )


def load_recordings(directory: str | Path) -> tuple[Track, SamplingConfig, tuple[RecordedEpisode, ...]]:
    """Rebuild observations from raw CSV, without running the simulator again."""
    directory = project_path(directory)
    metadata = _read_metadata(directory)
    track_info = metadata["track"]
    track = Track(np.asarray(track_info["centerline"], dtype=float), track_info["closed"], track_info["width"])
    sampling = SamplingConfig(**metadata["sampling"])
    frames: dict[tuple[str, int], dict[str, Any]] = {}
    with project_path(directory / "raw_states.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        # 舊資料沒有可選的碰撞尺寸欄位，仍可讀取。
        if reader.fieldnames not in (list(RAW_COLUMNS), list(RAW_COLUMNS[:-2])):
            raise ValueError("raw state columns do not match the bundle format")
        for row in reader:
            key = (row["episode_id"], int(row["step_index"]))
            time_seconds = float(row["time_s"])
            collisions = frozenset(tuple(pair) for pair in json.loads(row["collision_pairs"]))
            frame = frames.setdefault(key, {"time": time_seconds, "collisions": collisions, "observations": {}})
            if frame["time"] != time_seconds or frame["collisions"] != collisions:
                raise ValueError("raw rows for one frame must share time and collision data")
            if row["vehicle_id"] in frame["observations"]:
                raise ValueError("duplicate vehicle in a raw frame")
            frame["observations"][row["vehicle_id"]] = _read_observation(row)
    collector = TrainingDataCollector(sampling.time_step)
    for (episode_id, index), frame in frames.items():
        collector.record(episode_id, index, frame["time"], frame["observations"], frame["collisions"])
    actual = [{"episode_id": ep.episode_id, "frame_count": len(ep.frames)} for ep in collector.episodes]
    if actual != metadata["episodes"]:
        raise ValueError("raw episode lengths do not match metadata")
    return track, sampling, collector.episodes
