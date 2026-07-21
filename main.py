"""Assemble and run the Milestone 1 single-vehicle simulation."""

import argparse
import math

from core.pure_pursuit import PurePursuitConfig, PurePursuitController
from core.track import Track
from core.vehicle_model import KinematicBicycleModel, VehicleModelConfig
from core.vehicle_state import VehicleState
from sim.visualizer import SimulationVisualizer, VisualizationConfig


def run_simulation(realtime: bool = False) -> None:
    track = Track.create_oval(
        semi_major_axis=10.0,
        semi_minor_axis=5.0,
        point_count=500,
    )
    model_config = VehicleModelConfig(
        wheelbase=0.33,
        time_step=0.02,
        max_steering_angle=math.radians(30.0),
        min_velocity=0.0,
        max_velocity=5.0,
    )
    controller_config = PurePursuitConfig(
        lookahead_distance=1.2,
        target_speed=4.0,
        speed_gain=1.5,
        max_acceleration=3.0,
        max_steering_angle=model_config.max_steering_angle,
    )
    model = KinematicBicycleModel(model_config)
    controller = PurePursuitController(controller_config)
    visualizer = SimulationVisualizer(
        track,
        VisualizationConfig(realtime=realtime),
    )
    state = VehicleState(x=10.0, y=0.0, yaw=math.pi / 2.0, velocity=0.0)
    trajectory = [state]
    last_target = None

    # Forty seconds comfortably covers more than two laps at the target speed.
    step_count = int(40.0 / model_config.time_step)
    for _ in range(step_count):
        command, last_target = controller.compute_command(
            state, track, model_config.wheelbase
        )
        state = model.step(state, command)
        trajectory.append(state)
        if realtime:
            visualizer.update(state, trajectory, last_target)

    if not realtime:
        visualizer.update(state, trajectory, last_target)
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
