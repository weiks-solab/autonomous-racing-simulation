"""輸出整圈 GP 路線 PNG、取樣 CSV 與設定 JSON，不執行模擬或訓練。"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from strategy.gp_prediction import GpRoute, PROJECT_ROOT
from training_data.io import project_path


def main(argv: Sequence[str] | None = None) -> int:
    date = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", default="vehicle-2", help="main.py 中 GP_MODEL_PATHS 的對手 ID")
    parser.add_argument("--model", type=Path, help="直接指定專案內 model.pkl，覆蓋對手 ID 的模型設定")
    parser.add_argument("--points", type=int, default=1000, help="整圈等距 s 取樣數")
    parser.add_argument("--output", type=Path,
                        default=Path(f"reports/experiments/{date}_gp_visualization/gp_route.png"))
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import main as demo
    from sim.gp_visualization import create_gp_route_figure, sample_gp_route

    try:
        if args.model is None and args.opponent not in demo.GP_MODEL_PATHS:
            raise ValueError("對手未設定 GP_MODEL_PATHS，請指定 --model。")
        model_path = project_path(args.model or demo.GP_MODEL_PATHS[args.opponent])
        output = project_path(args.output)
        if output.suffix.lower() != ".png":
            raise ValueError("--output must end with .png")
        track = demo.create_demo_track()
        samples = sample_gp_route(track, GpRoute.load(model_path), args.points)
        figure = create_gp_route_figure(track, samples)
        FigureCanvasAgg(figure)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("s_m", "d_m", "x_m", "y_m"))
            writer.writerows((s, d, x, y) for s, d, (x, y) in zip(samples.s, samples.d, samples.points))
        metadata = {
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "point_count": len(samples.s), "meaning": "GP 平均路線，非時間預測或實際行駛軌跡。",
            "track": {"centerline": track.centerline.tolist(), "width": track.width, "closed": track.closed},
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(f"GP 整圈路線：{output.relative_to(PROJECT_ROOT)}")
    print("同目錄已保存取樣 CSV 與設定 JSON；沒有執行模擬或訓練。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
