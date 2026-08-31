"""Assemble and run the Milestone 4 rule-based overtaking simulation."""

import argparse
import math

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.track_geometry import (
    TrackPosition,
    track_length,
    track_to_world,
    world_to_track,
)
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import VehicleState
from sim.simulated_vehicle import (
    SimulatedVehicle,
    find_vehicle_collisions,
    validate_unique_vehicle_ids,
)
from sim.visualizer import SimulationVisualizer, VisualizationConfig
from strategy.observations import VehicleObservation, create_vehicle_observation
from strategy.offset_path import OffsetPathConfig, generate_offset_track
from strategy.overtaking import (
    OvertakingConfig,
    OvertakingDecision,
    OvertakingMode,
    OvertakingSide,
    RuleBasedOvertakingStrategy,
)


def create_display_options(realtime: bool) -> VisualizationConfig:
    """Choose which elements are visible in the simulation figure."""
    return VisualizationConfig(
        realtime=realtime,

        # Track
        show_centerline=True,
        show_track_boundaries=True,

        # Vehicles
        show_vehicle_markers=False,
        show_vehicle_footprints=True,
        show_vehicle_heading=False,
        show_collision_status=True,
        show_strategy_status=True,
        show_temporary_paths=True,

        # Controller and history information
        show_trajectories=False,
        show_pursuit_targets=True,
        show_track_positions=True,
    )


def _create_vehicle(
    vehicle_id: str,
    start_s: float,
    target_speed: float,
    model_config: VehicleModelConfig,
    track: Track,
) -> SimulatedVehicle:
    point = track_to_world(start_s, 0.0, track)
    point_ahead = track_to_world(start_s + 0.05, 0.0, track)
    tangent_yaw = math.atan2(
        point_ahead[1] - point[1],
        point_ahead[0] - point[0],
    )
    controller_config = PurePursuitConfig(
        lookahead_distance=3,
        target_speed=target_speed,
        speed_gain=1.5,
        max_acceleration=3.0,
        max_steering_angle=model_config.max_steering_angle,
    )
    return SimulatedVehicle(
        vehicle_id=vehicle_id,
        state=VehicleState(
            x=float(point[0]),
            y=float(point[1]),
            yaw=tangent_yaw,
            velocity=0.0,
        ),
        model=KinematicBicycleModel(model_config),
        controller=PurePursuitController(controller_config),
        geometry=VehicleGeometry(
            length=0.55,
            width=0.30,
            rear_axle_to_center=0.12,
        ),
    )


def _visualize_vehicles(
    visualizer: SimulationVisualizer,
    vehicles: list[SimulatedVehicle],
    track: Track,
    collision_pairs: set[tuple[str, str]],
    ego_id: str,
    decision: OvertakingDecision,
    temporary_track: Track | None,
) -> None:
    track_positions: dict[str, TrackPosition] = {
        vehicle.vehicle_id: world_to_track(vehicle.state, track)
        for vehicle in vehicles
    }
    visualizer.update_vehicles(
        states={vehicle.vehicle_id: vehicle.state for vehicle in vehicles},
        trajectories={
            vehicle.vehicle_id: vehicle.trajectory for vehicle in vehicles
        },
        targets={
            vehicle.vehicle_id: vehicle.current_target for vehicle in vehicles
        },
        geometries={
            vehicle.vehicle_id: vehicle.geometry for vehicle in vehicles
        },
        track_positions=track_positions,
        collision_pairs=collision_pairs,
        strategy_labels={ego_id: decision.display_text},
        temporary_paths=(
            {ego_id: temporary_track} if temporary_track is not None else None
        ),
    )


def _create_observations(
    vehicles: list[SimulatedVehicle],
    track: Track,
) -> dict[str, VehicleObservation]:
    return {
        vehicle.vehicle_id: create_vehicle_observation(
            vehicle.vehicle_id,
            vehicle.state,
            vehicle.geometry,
            track,
        )
        for vehicle in vehicles
    }


def run_simulation(realtime: bool = False) -> None:
    track = Track.create_oval(
        semi_major_axis=10.0,
        semi_minor_axis=5.0,
        point_count=500,
        width=2.0,
    )
    model_config = VehicleModelConfig(
        wheelbase=0.33,
        time_step=0.02,
        max_steering_angle=math.radians(50.0),
        min_velocity=0.0,
        max_velocity=10.0,
    )
    ego_id = "vehicle-1"
    base_length = track_length(track)
    vehicle_specs = (
        (ego_id, 0.0, 10.0),
        ("vehicle-2", 3.0, 3.0),
        ("vehicle-3", base_length - 20.0, 4.0),
    )
    vehicles = [
        _create_vehicle(vehicle_id, start_s, target_speed, model_config, track)
        for vehicle_id, start_s, target_speed in vehicle_specs
    ]
    validate_unique_vehicle_ids(vehicles)
    visualizer = SimulationVisualizer(
        track,
        create_display_options(realtime),
    )
    strategy = RuleBasedOvertakingStrategy(OvertakingConfig())
    offset_path_config = OffsetPathConfig(sample_spacing=0.2, path_length=18.0)
    decision = OvertakingDecision(
        mode=OvertakingMode.FOLLOW,
        target_d=0.0,
        target_vehicle_id=None,
        side=None,
        reason="simulation initialized",
    )
    temporary_track: Track | None = None
    transition_events = [(0.0, OvertakingMode.FOLLOW)]
    selected_side: OvertakingSide | None = None
    selected_candidate_summary = "not evaluated"

    # Twenty-one seconds covers two laps without reaching the third vehicle.
    step_count = int(21.0 / model_config.time_step)
    detected_collisions: set[tuple[str, str]] = set()
    current_collisions: set[tuple[str, str]] = set()
    for step_index in range(step_count):
        observations = _create_observations(vehicles, track)
        ego_observation = observations[ego_id]
        decision = strategy.update(
            ego_observation,
            tuple(
                observation
                for vehicle_id, observation in observations.items()
                if vehicle_id != ego_id
            ),
            track,
        )
        if decision.mode is not transition_events[-1][1]:
            transition_events.append(
                (step_index * model_config.time_step, decision.mode)
            )
        if decision.mode is OvertakingMode.OVERTAKE:
            selected_side = decision.side
            if strategy.last_candidate_evaluations:
                selected_candidate_summary = ", ".join(
                    f"{evaluation.side.name}={evaluation.reason}"
                    for evaluation in strategy.last_candidate_evaluations
                )

        if decision.maneuver is not None:
            temporary_track = generate_offset_track(
                track,
                ego_observation.track_position,
                decision.maneuver,
                offset_path_config,
            )
        else:
            temporary_track = None

        for vehicle in vehicles:
            tracking_track = (
                temporary_track
                if vehicle.vehicle_id == ego_id and temporary_track is not None
                else track
            )
            vehicle.step(tracking_track)
        current_collisions = find_vehicle_collisions(vehicles)
        detected_collisions.update(current_collisions)
        if realtime:
            _visualize_vehicles(
                visualizer,
                vehicles,
                track,
                current_collisions,
                ego_id,
                decision,
                temporary_track,
            )

    if not realtime:
        _visualize_vehicles(
            visualizer,
            vehicles,
            track,
            current_collisions,
            ego_id,
            decision,
            temporary_track,
        )
    print(
        "Strategy transitions: "
        + " -> ".join(
            f"{time_seconds:.2f}s {mode.name}"
            for time_seconds, mode in transition_events
        )
    )
    print(
        "Selected overtaking side: "
        + (selected_side.name if selected_side is not None else "none")
    )
    print(f"Candidate evaluations: {selected_candidate_summary}")
    final_ego_position = world_to_track(
        next(vehicle.state for vehicle in vehicles if vehicle.vehicle_id == ego_id),
        track,
    )
    print(f"Final ego d: {final_ego_position.d:+.3f} m")
    transitioned_modes = [mode for _, mode in transition_events]
    print(
        "Pass completed: "
        + ("yes" if OvertakingMode.RETURN in transitioned_modes else "no")
    )
    print(
        "Return completed: "
        + (
            "yes"
            if OvertakingMode.RETURN in transitioned_modes
            and transition_events[-1][1] is OvertakingMode.FOLLOW
            else "no"
        )
    )
    if detected_collisions:
        formatted_pairs = ", ".join(
            f"{first}/{second}" for first, second in sorted(detected_collisions)
        )
        print(f"Detected collision pairs: {formatted_pairs}")
    else:
        print("Detected collision pairs: none")
    visualizer.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="animate the simulation as it runs",
    )
    arguments = parser.parse_args()
    run_simulation(realtime=arguments.realtime)
