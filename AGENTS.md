# Racing Overtake Simulator

## Project goal

Build a lightweight Python 2D racing simulator for studying
multi-vehicle overtaking and defense algorithms.

The simulator must eventually support an arbitrary number of vehicles
and allow the algorithm core to be reused in ROS2 Humble on a real
F1TENTH-style vehicle.

## Current milestone: Milestone 4 plus a cost-based defense extension

Provide a simple rule-based overtaking strategy for vehicles assigned to attack.
Strategy code must live in `strategy/` and emit high-level lateral targets,
never steering commands. Pure Pursuit continues to provide low-level control.

Assign strategies per vehicle through STRATEGY_ASSIGNMENTS in main.py.
Keep strategy implementations separate in strategy/ and use a shared update
interface with explicit observations, track data and simulation time.
Each assigned vehicle must own independent strategy state, even when multiple
vehicles use the same strategy type. The default scene has one attacker,
one defender and one centerline follower.
Vehicles assigned to defend may choose defensive lateral targets by comparing
a small set of smooth paths with a lateral-alignment cost.
Unassigned vehicles continue independently following the shared centerline.
Disabling defense must preserve the passive-opponent overtaking mode.
Preserve arbitrary N-vehicle support and all Milestone 1-3 behavior.

Do not implement game theory, Nash, Stackelberg,
reinforcement learning, MPC, ROS2, race rules, collision response, dynamic
vehicle physics, sensors, or multi-agent strategic optimization.

Keep tunable demo parameters together near the top of main.py. Use short,
plain comments to explain the defense decisions and cost calculation.

## Technical requirements

- Use Python 3.
- Use type hints.
- Use dataclasses for vehicle state and control commands.
- Use NumPy for numerical calculations.
- Use Matplotlib for visualization.
- Keep functions small and readable.
- Add docstrings where the purpose is not obvious.
- Handle invalid inputs when reasonable.
- Do not add heavy dependencies.

## Architecture

Create this structure:

racing_overtake_sim/
├── AGENTS.md
├── README.md
├── requirements.txt
├── main.py
├── core/
│   ├── __init__.py
│   ├── vehicle_state.py
│   ├── vehicle_model.py
│   ├── track.py
│   └── pure_pursuit.py
├── sim/
│   ├── __init__.py
│   └── visualizer.py
└── tests/
    ├── __init__.py
    └── test_vehicle_model.py

## Architecture rules

- core/ must not import ROS2, Matplotlib, Pygame, or simulator-specific objects.
- sim/ may import Matplotlib.
- main.py should only assemble and run the simulation.
- Do not pass the entire simulator into controllers or planners.
- Algorithms should receive explicit vehicle states and track data.
- Do not write code specifically for exactly two vehicles.
- Preserve the possibility of supporting N vehicles later.

## Vehicle state

VehicleState should contain:

- x in metres
- y in metres
- yaw in radians
- velocity in metres per second

## Vehicle model

Use a rear-axle kinematic bicycle model with:

- steering angle in radians
- acceleration in metres per second squared
- configurable wheelbase
- configurable simulation time step
- steering and velocity limits

## Pure Pursuit

Pure Pursuit should:

- locate the nearest track point
- search forward for a lookahead point
- work across the end of a closed path
- calculate steering angle
- limit steering angle
- regulate toward a configurable target speed

## Definition of done

The task is complete when:

1. `python main.py` runs without errors.
2. One vehicle follows an oval track for at least two laps.
3. A Matplotlib figure shows the centerline and vehicle trajectory.
4. The vehicle remains reasonably close to the centerline.
5. Tests pass using `python -m unittest`.
6. README.md explains installation and execution.

# Workspace Rules

This repository is the only workspace.

Never access files outside this repository.

Never modify files outside this project.

Do not assume the existence of other projects.

All generated files must remain inside this repository.
