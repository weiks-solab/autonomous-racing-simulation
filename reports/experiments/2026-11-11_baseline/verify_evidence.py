"""以標準函式庫核對封存證據，不執行模擬或訓練。從專案根目錄執行。"""

import csv
import hashlib
import json
import math
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def errors(actual: list[float], predicted: list[float]) -> dict[str, float]:
    differences = [prediction - value for value, prediction in zip(actual, predicted)]
    return {
        "mae": sum(abs(value) for value in differences) / len(differences),
        "rmse": math.sqrt(sum(value ** 2 for value in differences) / len(differences)),
    }


def verify() -> None:
    manifest = read_json(DIRECTORY / "source_manifest.json")
    for item in manifest["files"]:
        path = ROOT / item["archived"]
        assert path.stat().st_size == item["bytes"], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path
    checked_rows = 0
    for summary in read_json(DIRECTORY / "summary.json"):
        folder = DIRECTORY / summary["archive"]
        report = read_json(folder / "report.json")
        metadata = read_json(folder / "metadata.json")
        with (folder / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == report["test_sample_count"] == summary["test_sample_count"]
        assert metadata["sample_count"] == report["train_sample_count"] + len(rows)
        assert report["sampling"] == metadata["sampling"]
        assert not set(report["train_episode_ids"]) & set(report["test_episode_ids"])
        assert {row["episode_id"] for row in rows} == set(report["test_episode_ids"])
        actual = [float(row["actual"]) for row in rows]
        means = [float(row["predicted_mean"]) for row in rows]
        measured = {"gp": errors(actual, means), "always_zero": errors(actual, [0.0] * len(rows))}
        if "training_mean" in report["metrics"]:
            aggregation = summary["training_target_aggregation_from_local_samples"]
            assert aggregation["count"] == report["train_sample_count"]
            measured["training_mean"] = errors(actual, [aggregation["mean"]] * len(rows))
        for name, values in measured.items():
            for metric, value in values.items():
                assert math.isclose(value, report["metrics"][name][metric], rel_tol=1e-10, abs_tol=1e-12)
        for row, value, mean in zip(rows, actual, means):
            assert math.isclose(float(row["error"]), mean - value, abs_tol=1e-12)
            assert float(row["predicted_std"]) >= 0
        checked_rows += len(rows)
        print(f"PASS {summary['archive']}: {len(rows)} predictions; metrics match report")
    print(f"PASS {len(manifest['files'])} archived file hashes; {checked_rows} prediction rows")
    print("Local raw-data checks are recorded in summary.json; original CSVs/models are not bundled.")


if __name__ == "__main__":
    verify()
