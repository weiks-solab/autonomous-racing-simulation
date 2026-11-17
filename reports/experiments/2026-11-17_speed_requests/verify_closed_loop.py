"""執行一次雙車閉迴路驗證，保存實際決策、控制與碰撞結果。"""

from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".cache" / "matplotlib")

from tests.test_speed_requests import SpeedCandidateIntegrationTests


if __name__ == "__main__":
    case = SpeedCandidateIntegrationTests("test_closed_loop_replanning_slows_without_collision")
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(case)
    summary = {
        "executed_at_taipei": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True,
        ).strip(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip(),
        "worktree": "含未提交修改",
        "python": platform.python_version(),
        "passed": result.wasSuccessful(),
        "closed_loop": getattr(case, "closed_loop_result", None),
    }
    (OUTPUT / "closed_loop_unittest.log").write_text(log.getvalue(), encoding="utf-8")
    (OUTPUT / "closed_loop_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(log.getvalue(), end="")
    if summary["closed_loop"] is not None:
        measured = summary["closed_loop"]
        print(f"replanning_count_including_initial={measured['replanning_count_including_initial']}")
        print("first_lower_speed_step=" + json.dumps(measured["first_lower_speed_step"]))
        print("collision_frames=" + json.dumps(measured["collision_frames"]))
    sys.exit(0 if result.wasSuccessful() else 1)
