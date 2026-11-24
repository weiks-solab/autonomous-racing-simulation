"""Load the fixed-width F1TENTH CSV format using only NumPy."""

import math
from pathlib import Path

import numpy as np

from core.track import Track


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_ROOT = "external/f1tenth_racetracks"


def project_file(path: str | Path) -> Path:
    """Resolve a repository path without allowing traversal outside it."""
    resolved = (PROJECT_ROOT / path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError("map and output paths must remain inside this repository")
    return resolved


def available_maps(map_root: str | Path = DEFAULT_MAP_ROOT) -> list[str]:
    """列出 map_root 下含有 *_centerline.csv 的地圖資料夾，供錯誤訊息提示可選名稱。"""
    root = project_file(map_root)
    if not root.is_dir():
        return []
    return sorted(
        folder.name for folder in root.iterdir()
        if folder.is_dir() and any(folder.glob("*_centerline.csv"))
    )


def find_centerline(name: str, map_root: str | Path = DEFAULT_MAP_ROOT) -> Path:
    # 資料夾名稱與檔名前綴不一定相同，例如 Mexico City / MexicoCity。
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("track name must be a single directory name")
    folder = project_file(project_file(map_root) / name)
    matches = sorted(folder.glob("*_centerline.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected one *_centerline.csv in {folder}, found {len(matches)}")
    return project_file(matches[0])


def load_centerline(path: str | Path, sample_spacing: float = 0.4) -> Track:
    """Resample a closed x/y/right/left CSV, retaining its start and direction.

    This first version requires constant symmetric half widths. Rejecting
    other widths avoids silently turning a narrow section into drivable road.
    """
    if not math.isfinite(sample_spacing) or sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be positive and finite")
    path = project_file(path)
    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2, encoding="utf-8-sig")
    if data.ndim != 2 or data.shape[1] != 4 or len(data) < 3:
        raise ValueError("centerline CSV must contain at least three x,y,right,left rows")
    if not np.all(np.isfinite(data)) or np.any(data[:, 2:] <= 0.0):
        raise ValueError("centerline values must be finite and half widths positive")
    if not np.allclose(data[:, 2:], data[0, 2], rtol=0.0, atol=1e-8):
        raise ValueError("only constant symmetric track widths are supported")

    # 移除連續重複點與重複終點；Track 自己負責最後一段閉合。
    points = data[:, :2]
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]]
    if len(points) > 1 and np.linalg.norm(points[-1] - points[0]) < 1e-9:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("centerline must contain at least three distinct points")
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    # 不把明顯斷開的開放路徑，補成一條穿越場地的閉合直線。
    if lengths[-1] > 3.0 * np.median(lengths[:-1]):
        raise ValueError("centerline closing gap is too large for a closed track")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    count = max(3, math.ceil(float(cumulative[-1]) / sample_spacing))
    distances = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    loop = np.vstack((points, points[0]))
    resampled = np.column_stack([
        np.interp(distances, cumulative, loop[:, axis]) for axis in range(2)
    ])
    return Track(resampled, closed=True, width=float(2.0 * data[0, 2]))
