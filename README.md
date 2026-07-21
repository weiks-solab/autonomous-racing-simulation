# Racing Overtake Simulator — Milestone 1

A lightweight Python 2D simulator. The current milestone contains one vehicle
following a closed elliptical centerline with Pure Pursuit and a rear-axle
kinematic bicycle model.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The simulation runs for 40 seconds (more than two laps at the configured target
speed), then displays the elliptical centerline and driven trajectory.

For headless verification:

```bash
MPLBACKEND=Agg python main.py
```

## Tests

```bash
python -m unittest
```

The `core` package is independent of visualization and simulator-specific
objects. Matplotlib usage is isolated in `sim/visualizer.py`.
