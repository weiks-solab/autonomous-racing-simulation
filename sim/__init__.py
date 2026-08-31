"""Simulation presentation components."""

from sim.simulated_vehicle import SimulatedVehicle, validate_unique_vehicle_ids
from sim.visualizer import SimulationVisualizer, VisualizationConfig

__all__ = [
    "SimulatedVehicle",
    "SimulationVisualizer",
    "VisualizationConfig",
    "validate_unique_vehicle_ids",
]
