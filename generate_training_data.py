"""Generate training CSVs with existing simulation settings, without a window."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from core.pure_pursuit import PurePursuitConfig
from core.track import Track
from sim.data_generation import (
    EpisodeScenario,
    RandomizationConfig,
    SimulationSettings,
    randomized_scenarios,
    run_episode,
    simulation_step_count,
)
from strategy.assignment import StrategyAssignment
from training_data import (
    DatasetSchema,
    SamplingConfig,
    TrainingDataCollector,
    VehiclePair,
    build_dataset,
    create_default_registry,
    export_dataset,
)
from training_data.io import PROJECT_ROOT, project_path
from training_data.schema import DEFAULT_FEATURES, DEFAULT_TARGETS


def _demo_settings(
    no_defense: bool,
) -> tuple[SimulationSettings, tuple[tuple[str, float, float], ...], dict[str, StrategyAssignment]]:
    # 只讀取既有設定；import 不會執行 main.py 的模擬入口。
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import main as demo

    settings = SimulationSettings(
        track=Track.create_oval(
            demo.TRACK_SEMI_MAJOR_AXIS, demo.TRACK_SEMI_MINOR_AXIS,
            demo.TRACK_POINT_COUNT, demo.TRACK_WIDTH,
        ),
        model_config=demo.MODEL_CONFIG,
        geometry=demo.VEHICLE_GEOMETRY,
        controller_config=PurePursuitConfig(
            demo.LOOKAHEAD_DISTANCE, 0.0, demo.SPEED_GAIN,
            demo.MAX_ACCELERATION, demo.MODEL_CONFIG.max_steering_angle,
        ),
        overtaking_config=demo.OVERTAKING_CONFIG,
        defense_config=demo.DEFENSE_CONFIG,
        cost_overtaking_config=demo.COST_OVERTAKING_CONFIG,
        offset_path_config=demo.OFFSET_PATH_CONFIG,
        defense_enabled=demo.DEFENSE_ENABLED and not no_defense,
    )
    return settings, demo.VEHICLE_SPECS, dict(demo.STRATEGY_ASSIGNMENTS)


def _select_pairs(
    requested: list[list[str]] | None,
    vehicle_ids: Sequence[str],
    assignments: dict[str, StrategyAssignment],
) -> tuple[VehiclePair, ...]:
    if requested:
        pairs = tuple(VehiclePair(ego, opponent) for ego, opponent in requested)
    else:
        pairs = tuple(
            VehiclePair(assignment.target_vehicle_id, vehicle_id)
            for vehicle_id, assignment in assignments.items()
            if assignment.name == "defend" and assignment.target_vehicle_id is not None
        )
        if not pairs:
            pairs = tuple(
                VehiclePair(ego_id, other_id)
                for ego_id, assignment in assignments.items()
                if assignment.name in ("overtake", "cost_overtake")
                for other_id in vehicle_ids if other_id != ego_id
            )
    if not pairs:
        raise ValueError("no training pairs selected; use --pair EGO_ID OPPONENT_ID")
    if len(set(pairs)) != len(pairs):
        raise ValueError("training pairs must be unique")
    if any({pair.ego_id, pair.opponent_id} - set(vehicle_ids) for pair in pairs):
        raise ValueError("training pairs reference unknown vehicle IDs")
    return pairs


def _run_metadata(
    settings: SimulationSettings, scenarios: tuple[EpisodeScenario, ...],
    pairs: tuple[VehiclePair, ...], seed: int, seconds: float, randomization: RandomizationConfig,
) -> dict:
    config_names = (
        "model_config", "geometry", "controller_config", "overtaking_config",
        "defense_config", "cost_overtaking_config", "offset_path_config",
    )
    return {
        "seed": seed,
        "seconds_per_episode": seconds,
        "defense_enabled": settings.defense_enabled,
        "pairs": [asdict(pair) for pair in pairs],
        "randomization": asdict(randomization),
        "simulation_configs": {name: asdict(getattr(settings, name)) for name in config_names},
        "offset_path_rule": "length is at least each assigned strategy's transition length",
        "scenarios": [
            {
                "episode_id": scenario.episode_id,
                "vehicles": [asdict(vehicle) for vehicle in scenario.vehicles],
                "assignments": {key: asdict(value) for key, value in scenario.assignments.items()},
            }
            for scenario in scenarios
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=4.0, help="duration of each episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-steps", type=int, default=1, help="future label offset in model steps")
    parser.add_argument("--stride-steps", type=int, default=1, help="steps between input samples")
    parser.add_argument("--pair", nargs=2, action="append", metavar=("EGO_ID", "OPPONENT_ID"))
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--no-defense", action="store_true")
    parser.add_argument("--keep-collisions", action="store_true", help="retain collision intervals")
    parser.add_argument("--output", type=Path, help="new output directory inside this repository")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        schema = DatasetSchema(tuple(args.features), tuple(args.targets))
        registry = create_default_registry()
        registry.resolve_features(schema)
        registry.resolve_targets(schema)
        settings, vehicle_specs, assignments = _demo_settings(args.no_defense)
        sampling = SamplingConfig(
            settings.model_config.time_step, args.horizon_steps, args.stride_steps,
            reject_collisions=not args.keep_collisions,
        )
        if simulation_step_count(args.seconds, sampling.time_step) < sampling.label_horizon_steps:
            raise ValueError("episode duration must cover the label horizon")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        output = project_path(args.output if args.output is not None else Path("data/training") / run_id)
        if output.exists():
            raise ValueError("output directory already exists; choose a new run directory")
        pairs = _select_pairs(args.pair, [spec[0] for spec in vehicle_specs], assignments)
        randomization = RandomizationConfig()
        scenarios = randomized_scenarios(
            settings, vehicle_specs, assignments, args.episodes, args.seed, randomization,
        )
        collector = TrainingDataCollector(sampling.time_step)
        for index, scenario in enumerate(scenarios, 1):
            run_episode(scenario, settings, collector, args.seconds)
            print(f"Recorded episode {index}/{len(scenarios)}: {scenario.episode_id}", flush=True)
        dataset = build_dataset(collector.episodes, settings.track, pairs, sampling, schema, registry)
        if not dataset.samples:
            raise ValueError("no valid samples; increase duration or review selected pairs and collisions")
        export_dataset(
            dataset, collector.episodes, settings.track, output,
            _run_metadata(settings, scenarios, pairs, args.seed, args.seconds, randomization),
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Saved: {output}")
    print(f"X: {dataset.X.shape}, Y: {dataset.Y.shape}, label horizon: {sampling.label_horizon_seconds:g} s")
    print(f"Skipped samples: {dict(dataset.skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
