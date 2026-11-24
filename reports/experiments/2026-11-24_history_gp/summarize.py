"""彙整三組消融的 report.json，寫出 summary.json 並印出 Markdown 表；不重新擬合。"""

import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = {
    "a_basic8": "models/gp/history_20261124_a_basic8",
    "b_history12": "models/gp/history_20261124_b_history12",
    "c_action13": "models/gp/history_20261124_c_action13",
}
SUBSETS = ("straight", "curve", "rear_moving", "rear_holding", "rear_left", "rear_right")


def main() -> None:
    summary = {}
    for name, directory in RUNS.items():
        source = ROOT / directory
        report = json.loads((source / "report.json").read_text(encoding="utf-8"))
        # 小型報告與圖表複製進證據目錄；model.pkl 留在本機（.gitignore）。
        target = HERE / name
        target.mkdir(exist_ok=True)
        for file in ("report.json", "predictions.csv", "evaluation.png"):
            shutil.copy2(source / file, target / file)
        summary[name] = {
            "features": report["features"], "fit_count": report["fit_count"],
            "fit_seconds": report["fit_seconds"], "fitted_kernel": report["fitted_kernel"],
            "warnings": report["warnings"],
            **{split: {"count": report["splits"][split]["count"],
                       "metrics": report["splits"][split]["metrics"],
                       "coverage_2std": report["splits"][split]["coverage_2std"],
                       "subsets": {key: report["splits"][split]["subsets"][key]
                                   for key in SUBSETS if key in report["splits"][split]["subsets"]}}
               for split in ("validation", "test")},
        }
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("| 組 | 輸入數 | 驗證 MAE | 驗證 RMSE | 測試 MAE | 測試 RMSE | 測試 ±2σ 涵蓋 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, item in summary.items():
        validation, test = item["validation"], item["test"]
        print(f"| {name} | {len(item['features'])} | {validation['metrics']['gp']['mae']:.6f} | "
              f"{validation['metrics']['gp']['rmse']:.6f} | {test['metrics']['gp']['mae']:.6f} | "
              f"{test['metrics']['gp']['rmse']:.6f} | {test['coverage_2std']:.2%} |")
    first = next(iter(summary.values()))
    for baseline in ("zero", "training_mean", "constant_lateral_speed"):
        print(f"| 基準 {baseline} | - | {first['validation']['metrics'][baseline]['mae']:.6f} | "
              f"{first['validation']['metrics'][baseline]['rmse']:.6f} | "
              f"{first['test']['metrics'][baseline]['mae']:.6f} | {first['test']['metrics'][baseline]['rmse']:.6f} | - |")
    print("\n測試子集 GP MAE（m）")
    print("| 組 | " + " | ".join(SUBSETS) + " |")
    print("| --- |" + " ---: |" * len(SUBSETS))
    for name, item in summary.items():
        cells = [f"{item['test']['subsets'][key]['gp']['mae']:.4f} (n={item['test']['subsets'][key]['count']})"
                 if key in item["test"]["subsets"] else "-" for key in SUBSETS]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
