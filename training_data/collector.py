"""Record synchronized observations without controlling the simulation."""

import math
from typing import Iterable, Mapping

from strategy.observations import VehicleObservation
from training_data.schema import FrameSnapshot, RecordedEpisode, SamplingConfig


class TrainingDataCollector:
    def __init__(self, time_step: float) -> None:
        self.time_step = SamplingConfig(time_step=time_step).time_step
        self._episodes: dict[str, list[FrameSnapshot]] = {}

    def record(
        self,
        episode_id: str,
        step_index: int,
        time_seconds: float,
        observations: Mapping[str, VehicleObservation],
        collision_pairs: Iterable[tuple[str, str]] = (),
    ) -> None:
        """Record the initial state, then one frame after all vehicles advance."""
        frame = FrameSnapshot(
            episode_id, step_index, time_seconds, observations, frozenset(collision_pairs),
        )
        expected_index = len(self._episodes.get(episode_id, ()))
        if step_index != expected_index:
            raise ValueError("record consecutive frames starting at zero for each episode")
        if not math.isclose(time_seconds, step_index * self.time_step, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("frame time must match step_index and time_step")
        self._episodes.setdefault(episode_id, []).append(frame)

    @property
    def episodes(self) -> tuple[RecordedEpisode, ...]:
        return tuple(RecordedEpisode(name, tuple(frames)) for name, frames in self._episodes.items())
