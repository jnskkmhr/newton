# Rover wheel assets

Both standalone MJCF files define the complete rover: a free chassis, four hinge
wheel joints, visual rotation markers, and four velocity actuators.

- `rover_flat_wheel.xml`: a convex 24-sided tire mesh per wheel.
- `rover_grouser_wheel.xml`: a convex 24-sided core plus twelve separate tapered
  convex lug meshes per wheel. Each lug is an eight-vertex mesh. The protruding
  tread is not represented by a primitive or a single convex hull of the wheel.

The separate convex pieces retain the gaps between lugs. Making a single convex
hull around the complete wheel would fill these gaps and remove the grouser
contact geometry. All pieces are rigidly attached to their wheel body; no extra
joints or dynamic bodies are introduced. Mesh vertices and faces are embedded
in the XML, so no external mesh files or asset downloads are required.

| Setting | Value |
|---|---:|
| Nominal outer wheel radius | Flat: 0.24 m; grouser: 0.27 m |
| Track width | 0.72 m |
| Wheel width | 0.14 m |
| Chassis mass | 30 kg |
| Wheel mass | 2.026704 kg each |
| Total rover mass | 38.106817 kg |
| Velocity actuator gain | 20 N m s/rad |
| Motor torque limit | 30 N m per wheel |

The grouser core radius is 0.19 m and lugs protrude approximately 0.08 m. Each
lug tapers from 0.036 m tangential width at its base to 0.016 m at its tip. The
wheel-radius metadata is updated so differential-drive commands use the new
outer radius. The two
assets use the same explicit approximate inertias (based on the original flat
wheel), so selecting grousers changes contact geometry without changing mass,
inertia, or motor strength. Friction settings are also matched. The colored
sidewall markers are visual only.

The default is `grouser`. Select either asset at startup:

```bash
uv run --extra examples -m newton.examples moving_terrain_mpm_rover --wheel-type grouser
uv run --extra examples -m newton.examples moving_terrain_mpm_rover --wheel-type flat
```

These examples retain moving-patch terrain, differential-drive commands, two-way
coupling, local GPU height persistence. The grouser version has more
collision shapes and can cost more to simulate. Additional traction is not
guaranteed: the current MPM voxel width is comparable to the grouser height,
and steep-slope behavior depends on sand friction, slip, loading, and motor torque.
A grid-resolution study is needed before treating the geometry as a validated
traction model.

## Validation

```bash
uv run --no-sync python -m unittest newton.tests.test_mpm_beta_rover -v
```

The local tests check both MJCF assets load with a free chassis, four velocity
actuators, and the expected differential-drive wheel directions. The beta example
currently supports flat underlying ground. It does not establish a traction
advantage for grouser wheels or validate slope climbing.
