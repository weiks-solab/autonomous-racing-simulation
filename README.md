# Racing Overtake Simulator — Milestone 4

A lightweight Python 2D simulator. The current milestone demonstrates three
independently simulated vehicles following the same elliptical centerline.
Each vehicle has its own state, bicycle model, Pure Pursuit controller, target
speed, and trajectory. The framework supports an arbitrary vehicle count.

Milestone 3 adds reusable track-relative coordinates, constant-width track
boundaries, oriented vehicle footprints, and collision reporting. Collision
detection is informational only and never changes vehicle motion.

Milestone 4 designates one ego vehicle to make a deterministic rule-based
overtaking decision. Strategy outputs a lateral target, an open temporary path
is generated with a smoothstep transition, and the existing Pure Pursuit
controller tracks that path. Other vehicles continue following the centerline.

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
In realtime mode it can also show the ego strategy state and temporary path.

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

The Milestone 4 candidate check is a short-horizon geometric prediction in
track coordinates. Opponents keep constant speed and lateral offset; it does
not integrate bicycle dynamics or perform collision response.

## Track-relative convention

- `s` starts at `centerline[0]`, increases in centerline index order, and wraps
  into `[0, track_length)`.
- `d` is positive to the left when looking in the track direction and negative
  to the right.
- Boundary classification applies only to the rear-axle reference point, not
  the complete vehicle footprint.
