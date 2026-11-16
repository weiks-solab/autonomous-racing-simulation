"""Display the flat map YAML/PNG format supplied with F1TENTH racetracks."""

import ast
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib.image as mpimg
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.transforms import Affine2D
import numpy as np
from numpy.typing import NDArray

from core.track_loader import project_file


@dataclass(frozen=True)
class MapBackground:
    yaml_file: Path
    image_file: Path
    resolution: float
    origin: tuple[float, float, float]
    pixels: NDArray

    def draw(self, axes: Axes) -> AxesImage:
        height, width = self.pixels.shape[:2]
        x, y, yaw = self.origin
        transform = Affine2D().rotate(yaw).translate(x, y) + axes.transData
        # 影像第 0 列在上方；YAML 原點位於影像左下角，單位為公尺。
        return axes.imshow(
            self.pixels, origin="upper", cmap="gray", vmin=0.0, vmax=1.0,
            extent=(0.0, width * self.resolution, 0.0, height * self.resolution),
            transform=transform, interpolation="nearest", zorder=-10,
        )


def load_map_background(path: str | Path) -> MapBackground:
    """Read scalar image/resolution and an inline [x,y,yaw] origin.

    This intentionally supports the downloaded flat map metadata, not general
    YAML (no tags, anchors or nested mappings); no additional dependency is needed.
    Occupancy thresholds are not used because the raster is display-only.
    """
    path = project_file(path)
    fields: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, separator, value = line.partition(":")
        if not separator or key.strip() in fields:
            raise ValueError(f"invalid or duplicate map YAML field: {raw}")
        fields[key.strip()] = value.strip()
    if not {"image", "resolution", "origin"} <= fields.keys():
        raise ValueError("map YAML requires image, resolution and origin")
    image_name = fields["image"]
    if image_name.startswith(("'", '"')):
        image_name = ast.literal_eval(image_name)
    if not isinstance(image_name, str) or not image_name:
        raise ValueError("map image must be a non-empty filename")
    image_path = project_file(path.parent / image_name)
    resolution = float(fields["resolution"])
    origin_values = ast.literal_eval(fields["origin"])
    if not isinstance(origin_values, (tuple, list)) or len(origin_values) != 3:
        raise ValueError("map origin must contain [x, y, yaw]")
    origin = tuple(float(value) for value in origin_values)
    if not math.isfinite(resolution) or resolution <= 0.0 or not all(map(math.isfinite, origin)):
        raise ValueError("map resolution must be positive and origin must be finite")
    pixels = mpimg.imread(image_path)
    if pixels.ndim not in (2, 3) or min(pixels.shape[:2]) == 0:
        raise ValueError("map image must be a non-empty raster")
    return MapBackground(path, image_path, resolution, origin, pixels)
