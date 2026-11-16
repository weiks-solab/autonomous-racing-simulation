"""離線取樣整圈 GP 平均路線，與實際時間預測分開顯示。"""

from dataclasses import dataclass

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from core.track import Track
from core.track_geometry import track_boundaries, track_length, track_to_world_points
from strategy.gp_prediction import GpRoute


@dataclass(frozen=True)
class GpRouteSamples:
    s: NDArray[np.float64]
    d: NDArray[np.float64]
    points: NDArray[np.float64]


def sample_gp_route(track: Track, route: GpRoute, point_count: int = 1000) -> GpRouteSamples:
    """沿中心線 s 等距取樣；不需要模擬車輛或假設速度。"""
    if not track.closed:
        raise ValueError("full-lap GP visualization requires a closed track")
    if point_count < 3:
        raise ValueError("point_count must be at least 3")
    s = np.linspace(0.0, track_length(track), point_count, endpoint=False)
    d = route.predict_d(s)
    return GpRouteSamples(s, d, track_to_world_points(s, d, track))


def create_gp_route_figure(track: Track, samples: GpRouteSamples) -> Figure:
    """同一張圖並列世界座標路線與 s→d，僅顯示 GP 平均值。"""
    figure = Figure(figsize=(12, 5.5), layout="constrained")
    map_axes, sd_axes = figure.subplots(1, 2, gridspec_kw={"width_ratios": (1.3, 1)})
    left, right = track_boundaries(track)
    for boundary, label in ((left, "track boundary"), (right, "_nolegend_")):
        closed = np.vstack((boundary, boundary[0]))
        map_axes.plot(closed[:, 0], closed[:, 1], color="black", linewidth=1, label=label)
    centerline = np.vstack((track.centerline, track.centerline[0]))
    map_axes.plot(centerline[:, 0], centerline[:, 1], "--", color="0.6", linewidth=1, label="centerline")
    # 只為畫出整圈而接回起點；沒有替 GP 加上週期性或平滑修正。
    closed = np.vstack((samples.points, samples.points[0]))
    map_axes.plot(closed[:, 0], closed[:, 1], color="tab:purple", linewidth=2, label="GP mean route")
    map_axes.plot(*samples.points[0], "o", color="tab:purple", label="s = 0")
    map_axes.set(xlabel="x (m)", ylabel="y (m)", title="Full-lap GP route", aspect="equal")
    map_axes.legend(loc="center", fontsize=8)
    sd_axes.plot(samples.s, samples.d, color="tab:purple", label="GP mean d(s)")
    sd_axes.axhline(0, color="0.6", linestyle="--", linewidth=1)
    sd_axes.set(xlabel="s (m)", ylabel="d (m)", title="Same-time s to d mapping")
    sd_axes.legend(fontsize=8)
    for axes in (map_axes, sd_axes):
        axes.grid(alpha=0.2)
    return figure
