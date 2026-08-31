"""Assemble and run the Milestone 3 N-vehicle simulation."""

import argparse
import math

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.track_geometry import TrackPosition, world_to_track
from core.vehicle_geometry import VehicleGeometry
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import VehicleState
from sim.simulated_vehicle import (
    SimulatedVehicle,
    find_vehicle_collisions,
    validate_unique_vehicle_ids,
)
from sim.visualizer import SimulationVisualizer, VisualizationConfig


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

        # Controller and history information
        show_trajectories=False,
        show_pursuit_targets=True,
        show_track_positions=True,
    )


def _create_vehicle(
    vehicle_id: str,
    ellipse_angle: float,
    target_speed: float,
    model_config: VehicleModelConfig,
) -> SimulatedVehicle:
    semi_major_axis = 10.0
    semi_minor_axis = 5.0
    x = semi_major_axis * math.cos(ellipse_angle)
    y = semi_minor_axis * math.sin(ellipse_angle)
    tangent_yaw = math.atan2(
        semi_minor_axis * math.cos(ellipse_angle),
        -semi_major_axis * math.sin(ellipse_angle),
    )
    controller_config = PurePursuitConfig(
        lookahead_distance=6.0,
        target_speed=target_speed,
        speed_gain=1.5,
        max_acceleration=3.0,
        max_steering_angle=model_config.max_steering_angle,
    )
    return SimulatedVehicle(
        vehicle_id=vehicle_id,
        state=VehicleState(x=x, y=y, yaw=tangent_yaw, velocity=0.0),
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
    )


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
    vehicle_specs = (
        ("vehicle-1", 0.0, 15),
        ("vehicle-2", 2.0 * math.pi / 3.0, 10),
        ("vehicle-3", 4.0 * math.pi / 3.0, 7.0),
    )
    vehicles = [
        _create_vehicle(vehicle_id, angle, target_speed, model_config)
        for vehicle_id, angle, target_speed in vehicle_specs
    ]
    validate_unique_vehicle_ids(vehicles)
    visualizer = SimulationVisualizer(
        track,
        create_display_options(realtime),
    )
    # Forty seconds comfortably covers more than two laps at the target speed.
    step_count = int(40.0 / model_config.time_step)
    detected_collisions: set[tuple[str, str]] = set()
    current_collisions: set[tuple[str, str]] = set()
    for _ in range(step_count):
        for vehicle in vehicles:
            vehicle.step(track)
        current_collisions = find_vehicle_collisions(vehicles)
        detected_collisions.update(current_collisions)
        if realtime:
            _visualize_vehicles(
                visualizer,
                vehicles,
                track,
                current_collisions,
            )

    if not realtime:
        _visualize_vehicles(
            visualizer,
            vehicles,
            track,
            current_collisions,
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
