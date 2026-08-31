# Racing Overtake Simulator — Milestone 2

A lightweight Python 2D simulator. The current milestone demonstrates three
independently simulated vehicles following the same elliptical centerline.
Each vehicle has its own state, bicycle model, Pure Pursuit controller, target
speed, and trajectory. The framework supports an arbitrary vehicle count.

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

The simulation runs for 40 seconds, then displays the elliptical centerline and
all three trajectories. Use `python main.py --realtime` for live animation.

For headless verification:

```bash
MPLBACKEND=Agg python main.py
```

## Tests

```bash
python -m unittest
```

The `core` package is independent of visualization and simulator-specific
objects. Matplotlib usage is isolated in `sim/visualizer.py`. Milestone 2 does
not model interaction or collision between vehicles.
