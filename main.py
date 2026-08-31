"""Assemble and run the Milestone 2 N-vehicle simulation."""

import argparse
import math

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import VehicleState
from sim.simulated_vehicle import SimulatedVehicle, validate_unique_vehicle_ids
from sim.visualizer import SimulationVisualizer, VisualizationConfig


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
        lookahead_distance=3.0,
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
    )


def _visualize_vehicles(
    visualizer: SimulationVisualizer,
    vehicles: list[SimulatedVehicle],
) -> None:
    visualizer.update_vehicles(
        states={vehicle.vehicle_id: vehicle.state for vehicle in vehicles},
        trajectories={
            vehicle.vehicle_id: vehicle.trajectory for vehicle in vehicles
        },
        targets={
            vehicle.vehicle_id: vehicle.current_target for vehicle in vehicles
        },
    )


def run_simulation(realtime: bool = False) -> None:
    track = Track.create_oval(
        semi_major_axis=10.0,
        semi_minor_axis=5.0,
        point_count=500,
    )
    model_config = VehicleModelConfig(
        wheelbase=0.33,
        time_step=0.02,
        max_steering_angle=math.radians(50.0),
        min_velocity=0.0,
        max_velocity=10.0,
    )
    vehicle_specs = (
        ("vehicle-1", 0.0, 9.0),
        ("vehicle-2", 0.0 * math.pi / 3.0, 8.5),
        ("vehicle-3", 0.0 * math.pi / 3.0, 7.0),
    )
    vehicles = [
        _create_vehicle(vehicle_id, angle, target_speed, model_config)
        for vehicle_id, angle, target_speed in vehicle_specs
    ]
    validate_unique_vehicle_ids(vehicles)
    visualizer = SimulationVisualizer(
        track,
        VisualizationConfig(realtime=realtime),
    )
    # Forty seconds comfortably covers more than two laps at the target speed.
    step_count = int(40.0 / model_config.time_step)
    for _ in range(step_count):
        for vehicle in vehicles:
            vehicle.step(track)
        if realtime:
            _visualize_vehicles(visualizer, vehicles)

    if not realtime:
        _visualize_vehicles(visualizer, vehicles)
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
