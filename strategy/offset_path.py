"""Smooth track-relative transitions converted into temporary paths."""

from dataclasses import dataclass
import math

import numpy as np

from core.track import Track
from core.track_geometry import TrackPosition, track_length, track_to_world


@dataclass(frozen=True)
class LateralManeuver:
    """A lateral transition fixed to one origin on the base track."""

    start_s: float
    start_d: float
    target_d: float
    transition_length: float

    def __post_init__(self) -> None:
        values = (
            self.start_s,
            self.start_d,
            self.target_d,
            self.transition_length,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("lateral maneuver values must be finite")
        if self.start_s < 0.0:
            raise ValueError("start_s must be non-negative")
        if self.transition_length <= 0.0:
            raise ValueError("transition_length must be positive")


@dataclass(frozen=True)
class OffsetPathConfig:
    sample_spacing: float = 0.2
    path_length: float = 18.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_spacing) or self.sample_spacing <= 0.0:
            raise ValueError("sample_spacing must be positive and finite")
        if not math.isfinite(self.path_length) or self.path_length <= 0.0:
            raise ValueError("path_length must be positive and finite")


def smoothstep(xi: float) -> float:
    """Cubic transition with zero slope at both clamped endpoints."""
    if not math.isfinite(xi):
        raise ValueError("xi must be finite")
    clamped = min(max(xi, 0.0), 1.0)
    return 3.0 * clamped**2 - 2.0 * clamped**3


def lateral_offset_at_distance(
    forward_distance: float,
    d_start: float,
    d_target: float,
    transition_length: float,
) -> float:
    """Evaluate the prescribed smooth lateral profile."""
    values = (forward_distance, d_start, d_target, transition_length)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("lateral profile values must be finite")
    if forward_distance < 0.0:
        raise ValueError("forward_distance must be non-negative")
    if transition_length <= 0.0:
        raise ValueError("transition_length must be positive")
    return d_start + (d_target - d_start) * smoothstep(
        forward_distance / transition_length
    )


def wrapped_forward_distance(
    start_s: float,
    current_s: float,
    centerline_length: float,
) -> float:
    """Return forward progress from a fixed origin on a closed track."""
    values = (start_s, current_s, centerline_length)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("track distances must be finite")
    if start_s < 0.0 or current_s < 0.0:
        raise ValueError("track positions must be non-negative")
    if centerline_length <= 0.0:
        raise ValueError("centerline_length must be positive")
    return (current_s - start_s) % centerline_length


def generate_offset_track(
    base_track: Track,
    current_position: TrackPosition,
    maneuver: LateralManeuver,
    config: OffsetPathConfig,
) -> Track:
    """Generate an open path from progress along a fixed lateral maneuver."""
    if not base_track.closed:
        raise ValueError("offset path generation requires a closed base track")
    base_length = track_length(base_track)
    if config.path_length >= base_length:
        raise ValueError("temporary path must be shorter than one base-track lap")
    if maneuver.transition_length >= base_length:
        raise ValueError("transition must be shorter than one base-track lap")
    if config.path_length < maneuver.transition_length:
        raise ValueError("path_length must cover the complete transition")

    progress = wrapped_forward_distance(
        maneuver.start_s,
        current_position.s,
        base_length,
    )

    distances = np.arange(0.0, config.path_length, config.sample_spacing)
    distances = np.append(distances, config.path_length)
    points = np.array(
        [
            track_to_world(
                current_position.s + float(distance),
                lateral_offset_at_distance(
                    progress + float(distance),
                    maneuver.start_d,
                    maneuver.target_d,
                    maneuver.transition_length,
                ),
                base_track,
            )
            for distance in distances
        ]
    )
    return Track(centerline=points, closed=False, width=base_track.width)
