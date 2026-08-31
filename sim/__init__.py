"""Simulation presentation components."""

from sim.simulated_vehicle import (
    SimulatedVehicle,
    find_vehicle_collisions,
    validate_unique_vehicle_ids,
)
from sim.visualizer import SimulationVisualizer, VisualizationConfig

__all__ = [
    "SimulatedVehicle",
    "SimulationVisualizer",
    "VisualizationConfig",
    "find_vehicle_collisions",
    "validate_unique_vehicle_ids",
]
