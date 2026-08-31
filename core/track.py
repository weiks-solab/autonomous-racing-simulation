"""Closed centerline track representation and geometry queries."""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Track:
    centerline: NDArray[np.float64]
    closed: bool = True
    width: float = 2.0

    def __post_init__(self) -> None:
        points = np.asarray(self.centerline, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("centerline must have shape (N, 2)")
        minimum_points = 3 if self.closed else 2
        if len(points) < minimum_points:
            raise ValueError(f"centerline must contain at least {minimum_points} points")
        if not np.all(np.isfinite(points)):
            raise ValueError("centerline points must be finite")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0.0):
            raise ValueError("adjacent centerline points must be distinct")
        if self.closed and np.linalg.norm(points[-1] - points[0]) == 0.0:
            raise ValueError("closed centerline must not repeat its first point")
        if not math.isfinite(self.width) or self.width <= 0.0:
            raise ValueError("track width must be positive and finite")
        points = points.copy()
        points.setflags(write=False)
        object.__setattr__(self, "centerline", points)

    @classmethod
    def create_oval(
        cls,
        semi_major_axis: float,
        semi_minor_axis: float,
        point_count: int,
        width: float = 2.0,
    ) -> "Track":
        """Create a true ellipse sampled uniformly by parameter angle."""
        if not math.isfinite(semi_major_axis) or semi_major_axis <= 0.0:
            raise ValueError("semi_major_axis must be positive and finite")
        if not math.isfinite(semi_minor_axis) or semi_minor_axis <= 0.0:
            raise ValueError("semi_minor_axis must be positive and finite")
        if isinstance(point_count, bool) or not isinstance(point_count, int):
            raise TypeError("point_count must be an integer")
        if point_count < 3:
            raise ValueError("point_count must be at least 3")
        angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
        points = np.column_stack(
            (semi_major_axis * np.cos(angles), semi_minor_axis * np.sin(angles))
        )
        return cls(points, closed=True, width=width)

    def point(self, index: int) -> NDArray[np.float64]:
        return self.centerline[self._normalize_index(index)].copy()

    def nearest_point_index(self, x: float, y: float) -> int:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("query position must be finite")
        offsets = self.centerline - np.array((x, y))
        return int(np.argmin(np.einsum("ij,ij->i", offsets, offsets)))

    def forward_point_index(self, start_index: int, distance: float) -> int:
        """Find the first sampled point at least distance forward by arc length."""
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("distance must be non-negative and finite")
        current = self._normalize_index(start_index)
        if distance == 0.0:
            return current

        travelled = 0.0
        maximum_segments = len(self.centerline) if self.closed else len(self.centerline) - 1
        for _ in range(maximum_segments):
            next_index = current + 1
            if next_index == len(self.centerline):
                if not self.closed:
                    return current
                next_index = 0
            travelled += float(
                np.linalg.norm(self.centerline[next_index] - self.centerline[current])
            )
            current = next_index
            if travelled >= distance:
                return current
        return current

    def _normalize_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("index must be an integer")
        if self.closed:
            return int(index) % len(self.centerline)
        if not 0 <= index < len(self.centerline):
            raise IndexError("track index out of range")
        return int(index)
