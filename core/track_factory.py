"""Build interchangeable tracks without depending on the simulation runtime."""

from dataclasses import dataclass
import math

import numpy as np

from core.track import Track


@dataclass(frozen=True)
class TrackConfig:
    """All dimensions are centreline dimensions in metres, excluding road width."""

    name: str = "rounded_rectangle"
    point_count: int = 500
    width: float = 2.0
    rectangle_length: float = 20.0
    rectangle_height: float = 12.0
    corner_radius: float = 3.0
    semi_major_axis: float = 10.0
    semi_minor_axis: float = 5.0


def create_track(config: TrackConfig) -> Track:
    """Select a track shape while keeping the same Track interface."""
    if config.name == "oval":
        return Track.create_oval(
            config.semi_major_axis, config.semi_minor_axis,
            config.point_count, config.width,
        )
    if config.name == "rounded_rectangle":
        return _rounded_rectangle(config)
    raise ValueError(f"unknown track name {config.name!r}; use 'oval' or 'rounded_rectangle'")


def _rounded_rectangle(config: TrackConfig) -> Track:
    """Sample four tangent-connected straights and quarter circles by arc length.

    The track is centred on the origin and runs counterclockwise, starting
    at the left end of the bottom straight. The closing point is not repeated.
    """
    dimensions = (config.rectangle_length, config.rectangle_height,
                  config.corner_radius, config.width)
    if any(not math.isfinite(value) or value <= 0.0 for value in dimensions):
        raise ValueError("rectangle dimensions, radius and track width must be positive and finite")
    if isinstance(config.point_count, bool) or not isinstance(config.point_count, int):
        raise TypeError("point_count must be an integer")
    if config.point_count < 16:
        raise ValueError("rounded rectangle point_count must be at least 16")
    radius = config.corner_radius
    if 2.0 * radius >= min(config.rectangle_length, config.rectangle_height):
        raise ValueError("corner diameter must be smaller than both rectangle dimensions")
    if radius <= config.width / 2.0:
        raise ValueError("corner radius must exceed half the track width")

    half_length = config.rectangle_length / 2.0
    half_height = config.rectangle_height / 2.0
    a, b = half_length - radius, half_height - radius
    straight_starts = np.array(((-a, -half_height), (half_length, -b),
                                (a, half_height), (-half_length, b)))
    directions = np.array(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))
    corner_centres = np.array(((a, -b), (a, b), (-a, b), (-a, -b)))
    quarter_circle = math.pi * radius / 2.0
    lengths = np.array((2.0 * a, quarter_circle, 2.0 * b, quarter_circle) * 2)
    boundaries = np.concatenate(([0.0], np.cumsum(lengths)))
    distances = np.linspace(0.0, boundaries[-1], config.point_count, endpoint=False)
    points = np.empty((config.point_count, 2), dtype=float)

    for side in range(4):
        start, bend, end = boundaries[2 * side:2 * side + 3]
        straight = (distances >= start) & (distances < bend)
        points[straight] = (straight_starts[side]
                            + (distances[straight] - start)[:, np.newaxis] * directions[side])
        arc = (distances >= bend) & (distances < end)
        angles = (distances[arc] - bend) / radius + (side - 1) * math.pi / 2.0
        points[arc] = corner_centres[side] + radius * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
    return Track(points, closed=True, width=config.width)
