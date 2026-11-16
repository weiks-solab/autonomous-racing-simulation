"""用實際 TkAgg show 方法與模擬視窗，驗證 pause 的額外重繪請求。"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends._backend_tk import FigureManagerTk

canvas = Mock()
canvas.figure.stale = False
manager = SimpleNamespace(_shown=True, canvas=canvas)
with mpl.rc_context({"figure.raise_window": False}), \
     patch("matplotlib.backends._backend_tk._restore_foreground_window_at_end", nullcontext), \
     patch("matplotlib._pylab_helpers.Gcf.get_active", return_value=manager), \
     patch("matplotlib.pyplot.show", side_effect=lambda **kwargs: FigureManagerTk.show(manager)):
    plt.pause(0.001)
assert canvas.draw_idle.call_count == 1
result = {"date": "2026-11-16", "scope": "實際 pyplot.pause 及 TkAgg show 程式；視窗與 canvas 使用 mock，未開 GUI。",
          "figure_stale_before_pause": False, "window_already_shown": True,
          "draw_idle_requests": canvas.draw_idle.call_count,
          "event_loop_calls": canvas.start_event_loop.call_count}
(Path(__file__).parent / "tk_pause_trace.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Clean figure + TkAgg pause: one additional draw_idle request.")
