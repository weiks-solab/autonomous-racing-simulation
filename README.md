# Racing Overtake Simulator — Milestone 3

A lightweight Python 2D simulator. The current milestone demonstrates three
independently simulated vehicles following the same elliptical centerline.
Each vehicle has its own state, bicycle model, Pure Pursuit controller, target
speed, and trajectory. The framework supports an arbitrary vehicle count.

Milestone 3 adds reusable track-relative coordinates, constant-width track
boundaries, oriented vehicle footprints, and collision reporting. Collision
detection is informational only and never changes vehicle motion.

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
The plot also shows both track boundaries and each vehicle's rectangular body.

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
not model interaction between vehicles. Milestone 3 can report rectangular
body overlap, but does not avoid or resolve collisions.

## Track-relative convention

- `s` starts at `centerline[0]`, increases in centerline index order, and wraps
  into `[0, track_length)`.
- `d` is positive to the left when looking in the track direction and negative
  to the right.
- Boundary classification applies only to the rear-axle reference point, not
  the complete vehicle footprint.
