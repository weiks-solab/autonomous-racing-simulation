"""Track-relative geometry for discretized centerlines."""

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
from numpy.typing import NDArray

from core.track import Track
from core.vehicle_state import VehicleState


@dataclass(frozen=True)
class TrackPosition:
    """Position relative to the ordered track centerline.

    ``s`` is wrapped longitudinal distance from ``centerline[0]`` in the
    centerline index direction. ``d`` is positive to the left of that
    direction and negative to the right. ``centerline_index`` is the start
    index of the segment onto which the world position was projected.
    """

    s: float
    d: float
    centerline_index: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.s) or self.s < 0.0:
            raise ValueError("s must be non-negative and finite")
        if not math.isfinite(self.d):
            raise ValueError("d must be finite")
        if isinstance(self.centerline_index, bool) or not isinstance(
            self.centerline_index, (int, np.integer)
        ):
            raise TypeError("centerline_index must be an integer")
        if self.centerline_index < 0:
            raise ValueError("centerline_index must be non-negative")


@dataclass(frozen=True)
class RelativeTrackPosition:
    """Wrapped pairwise separation from a reference to another position."""

    longitudinal_gap: float
    lateral_gap: float

    @property
    def is_ahead(self) -> bool:
        return self.longitudinal_gap > 0.0

    @property
    def is_behind(self) -> bool:
        return self.longitudinal_gap < 0.0


class TrackBoundaryStatus(Enum):
    INSIDE = "inside"
    OUTSIDE_LEFT = "outside_left"
    OUTSIDE_RIGHT = "outside_right"


def _segments(
    track: Track,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    points = track.centerline
    if track.closed:
        starts = points
        ends = np.roll(points, -1, axis=0)
    else:
        starts = points[:-1]
        ends = points[1:]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    return starts, vectors, lengths


def track_length(track: Track) -> float:
    """Return centerline polyline length in metres."""
    return float(np.sum(_segments(track)[2]))


def world_to_track(state: VehicleState, track: Track) -> TrackPosition:
    """Project a rear-axle reference point onto the nearest track segment."""
    starts, vectors, lengths = _segments(track)
    point = np.array((state.x, state.y), dtype=float)
    offsets = point - starts
    parameters = np.einsum("ij,ij->i", offsets, vectors) / (lengths * lengths)
    parameters = np.clip(parameters, 0.0, 1.0)
    projections = starts + parameters[:, np.newaxis] * vectors
    differences = point - projections
    squared_distances = np.einsum("ij,ij->i", differences, differences)
    segment_index = int(np.argmin(squared_distances))

    cumulative_lengths = np.concatenate(([0.0], np.cumsum(lengths)))
    s = float(
        cumulative_lengths[segment_index]
        + parameters[segment_index] * lengths[segment_index]
    )
    total_length = float(cumulative_lengths[-1])
    if track.closed:
        s %= total_length

    tangent = vectors[segment_index] / lengths[segment_index]
    left_normal = np.array((-tangent[1], tangent[0]))
    d = float(np.dot(differences[segment_index], left_normal))
    return TrackPosition(s=s, d=d, centerline_index=segment_index)


def signed_longitudinal_gap(
    reference: TrackPosition,
    other: TrackPosition,
    centerline_length: float,
) -> float:
    """Return the shortest wrapped gap from reference to other.

    The result lies in ``[-L/2, L/2)``. Positive means ``other`` is ahead in
    the centerline direction. Exactly half a lap is inherently ambiguous and
    maps to ``-L/2`` by this interval convention.
    """
    if not math.isfinite(centerline_length) or centerline_length <= 0.0:
        raise ValueError("centerline_length must be positive and finite")
    half_length = centerline_length / 2.0
    return (other.s - reference.s + half_length) % centerline_length - half_length


def relative_track_position(
    reference: TrackPosition,
    other: TrackPosition,
    centerline_length: float,
) -> RelativeTrackPosition:
    """Compare any two track positions without assigning vehicle roles."""
    return RelativeTrackPosition(
        longitudinal_gap=signed_longitudinal_gap(
            reference, other, centerline_length
        ),
        lateral_gap=other.d - reference.d,
    )


def classify_reference_point_boundary(
    position: TrackPosition,
    track: Track,
) -> TrackBoundaryStatus:
    """Classify only the state reference point, not the whole vehicle body."""
    half_width = track.width / 2.0
    if position.d > half_width:
        return TrackBoundaryStatus.OUTSIDE_LEFT
    if position.d < -half_width:
        return TrackBoundaryStatus.OUTSIDE_RIGHT
    return TrackBoundaryStatus.INSIDE


def track_boundaries(
    track: Track,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate sampled left and right boundaries at the configured width."""
    _, vectors, lengths = _segments(track)
    unit_tangents = vectors / lengths[:, np.newaxis]
    if track.closed:
        incoming = np.roll(unit_tangents, 1, axis=0)
        outgoing = unit_tangents
    else:
        incoming = np.vstack((unit_tangents[0], unit_tangents))
        outgoing = np.vstack((unit_tangents, unit_tangents[-1]))

    vertex_tangents = incoming + outgoing
    tangent_norms = np.linalg.norm(vertex_tangents, axis=1)
    degenerate = tangent_norms < 1e-12
    vertex_tangents[degenerate] = outgoing[degenerate]
    tangent_norms[degenerate] = 1.0
    vertex_tangents /= tangent_norms[:, np.newaxis]
    left_normals = np.column_stack(
        (-vertex_tangents[:, 1], vertex_tangents[:, 0])
    )
    offset = left_normals * (track.width / 2.0)
    return track.centerline + offset, track.centerline - offset
