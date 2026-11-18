"""沿用目前 main 設定，收集八輸入資料並以 12/4/4 場訓練、驗證與測試。"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

import numpy as np

from generate_training_data import _demo_settings
from interaction_gp import split_groups, train, write_csv
from sim.data_generation import EpisodeScenario, VehicleInitialCondition, run_episode, simulation_step_count
from core.track_geometry import track_length
from training_data import TrainingDataCollector
from training_data.interaction import build_interaction_dataset
from training_data.io import PROJECT_ROOT, export_dataset, project_path
from training_data.schema import SamplingConfig, VehiclePair


def collect_current(data: Path, pair: VehiclePair, seconds: float = 25.0, seed: int = 42) -> None:
    """保持車距、巡航速度與策略；各場只隨機平移起點，避免重複同一條資料。"""
    settings, specs, assignments = _demo_settings(False)
    main_hash = hashlib.sha256((PROJECT_ROOT / "main.py").read_bytes()).hexdigest()
    ids = {spec[0] for spec in specs}
    if not {pair.ego_id, pair.opponent_id} <= ids:
        raise ValueError("--pair must name vehicles in main.VEHICLE_SPECS")
    dt = settings.model_config.time_step
    simulation_step_count(seconds, dt)
    if not np.isclose(round(0.5 / dt) * dt, 0.5) or not np.isclose(round(0.1 / dt) * dt, 0.1):
        raise ValueError("time_step must divide 0.5 s and 0.1 s")
    collector = TrainingDataCollector(dt)
    rng = np.random.default_rng(seed)
    scenarios, actions = [], []
    rear_speed = next(speed for key, _, speed in specs if key == pair.ego_id)
    for number in range(20):
        origin = float(rng.uniform(0.0, track_length(settings.track)))
        scenario = EpisodeScenario(f"action-{number:03d}", tuple(
            VehicleInitialCondition(key, (start + origin) % track_length(settings.track), 0.0, 0.0, speed)
            for key, start, speed in specs
        ), assignments)
        scenarios.append({"id": scenario.episode_id, "vehicles": [asdict(v) for v in scenario.vehicles]})

        def observe(step, observation, decision, command):
            if observation.vehicle_id == pair.ego_id:
                actions.append({"episode_id": scenario.episode_id, "step_index": step, "time_s": step * dt,
                                "mode": decision.mode, "target_d": decision.target_d,
                                "speed_limit": decision.speed_limit,
                                "effective_target_speed": min(rear_speed, decision.speed_limit)
                                if decision.speed_limit is not None else rear_speed,
                                "steering": command.steering_angle, "acceleration": command.acceleration})

        run_episode(scenario, settings, collector, seconds, step_observer=observe)
        print(f"collected {number + 1}/20 episodes ({seconds:g}s each)", flush=True)
    targets = {(row["episode_id"], row["step_index"]): row["target_d"] for row in actions}
    sampling = SamplingConfig(dt, round(0.5 / dt), round(0.1 / dt))
    dataset = build_interaction_dataset(collector.episodes, settings.track, pair, sampling, targets)
    valid_episodes = {sample.episode_id for sample in dataset.samples}
    metadata = {
        "seed": seed, "seconds": seconds, "pair": asdict(pair), "vehicle_specs": specs,
        "assignments": {key: asdict(value) for key, value in assignments.items()}, "scenarios": scenarios,
        "initialization": "沿用 main 車距、目標速度與策略，d=0、靜止起步，各場僅隨機平移賽道起點。",
        "action_semantics": "後車當下實際執行的候選目標 d；後續策略可以重新決策。",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree": "含未提交修改",
        "main_sha256": main_hash,
        "configs": {name: asdict(getattr(settings, name)) for name in (
            "model_config", "geometry", "controller_config", "overtaking_config", "defense_config",
            "relative_overtaking_config", "cost_overtaking_config", "progress_config", "probe_following_config",
            "offset_path_config")},
    }
    export_dataset(dataset, collector.episodes, settings.track, data, metadata)
    write_csv(data / "actions.csv", actions)
    if len(valid_episodes) != 20:
        raise ValueError(f"only {len(valid_episodes)}/20 episodes have valid samples; saved data, cannot claim 12/4/4")
    split = split_groups(np.array([sample.episode_id for sample in dataset.samples]), seed)
    print(f"Collected X={dataset.X.shape}, Y={dataset.Y.shape}; skipped={dict(dataset.skipped)}", flush=True)
    print("Episode split: train=12, validation=4, test=4", flush=True)
    assert sorted(len({dataset.samples[i].episode_id for i in indices}) for indices in split.values()) == [4, 4, 12]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="新資料目錄")
    parser.add_argument("--output", required=True, type=Path, help="新模型目錄")
    parser.add_argument("--pair", nargs=2, default=("vehicle-1", "vehicle-2"), metavar=("REAR", "FRONT"))
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collect-only", action="store_true", help="只收集，之後用 interaction_gp.py train")
    args = parser.parse_args(argv)
    data, output = project_path(args.data), project_path(args.output)
    if data.exists() or output.exists():
        parser.error("data/output directory already exists; use new paths")
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
    collect_current(data, VehiclePair(*args.pair), args.seconds, args.seed)
    if not args.collect_only:
        train(data, output, args.seed, max_train=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
