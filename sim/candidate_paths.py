"""將決策快照轉成可畫出的參考曲線，不重新評估策略。"""

import numpy as np
from numpy.typing import NDArray

from core.track import Track
from core.track_geometry import track_length, track_to_world
from strategy.base import CandidatePath, CandidateSnapshot
from strategy.offset_path import lateral_offset_at_distance, wrapped_forward_distance


def candidate_path_points(
    snapshot: CandidateSnapshot,
    candidate: CandidatePath,
    track: Track,
) -> NDArray[np.float64]:
    """優先使用策略保存的預測狀態；未提供時才依幾何距離取樣。"""
    if candidate.predicted_states:
        return np.array([(state.x, state.y) for state in candidate.predicted_states])
    maneuver = candidate.maneuver
    progress = wrapped_forward_distance(
        maneuver.start_s, snapshot.position.s, track_length(track),
    )
    offsets = [
        lateral_offset_at_distance(
            progress + distance, maneuver.start_d,
            maneuver.target_d, maneuver.transition_length,
        )
        for distance in snapshot.distances
    ]
    offsets[0] = snapshot.position.d
    return np.array([
        track_to_world(snapshot.position.s + distance, offset, track)
        for distance, offset in zip(snapshot.distances, offsets)
    ])


def candidate_path_label(candidate: CandidatePath) -> str:
    """顯示各策略提供的成本或可行性，不加入額外評估。"""
    label = f"{candidate.name + ' ' if candidate.name else ''}d={candidate.maneuver.target_d:+.2f}"
    if not candidate.feasible:
        label += f" {candidate.reason}"
    elif candidate.cost_terms:
        label += f" J={sum(value for _, value in candidate.cost_terms):.2f}"
        label += " " + " ".join(f"{name}={value:.2f}" for name, value in candidate.cost_terms)
    elif candidate.race_cost is not None and candidate.blocking_cost is not None:
        label += f" J={candidate.race_cost + candidate.blocking_cost:.2f}"
    else:
        label += " clear" if candidate.feasible else f" {candidate.reason}"
    return label + (" *" if candidate.selected else "")
