# Moving terrain MPM beta

The `example_*.py` files are the scene entry points. Shared implementation modules
and controllers live in `impl/`. Documentation, images, and rover assets remain
in their existing locations.

All beta scenes (box, shovel, rover, and G1) follow the body tracked by the terrain
patch. The shared base preserves OpenGL orbit and zoom as the body moves. Other
viewers use the reference offset `(2.5, -2.5, 1.0)` m, pitch −25°, and yaw 125°.
Camera tracking changes only the view, not the terrain or physics.

Run the mouse-dragged box scene (the existing command name is retained):

```bash
uv run -m newton.examples moving_terrain_mpm_box
```

Run the four-wheel rover with differential drive:

```bash
uv run -m newton.examples moving_terrain_mpm_rover
uv run -m newton.examples moving_terrain_mpm_rover --wheel-type flat
uv run -m newton.examples moving_terrain_mpm_rover --drive-speed 0.4 --turn-rate 0.3
```

The default rover uses the local grouser-wheel MJCF asset. I/K commands forward
and reverse, J/L turns, and O stops automatic driving. `--manual` starts with
keyboard control. The example reads the viewer's existing key state; it does not
register keyboard hooks. Wheel actuators drive a free chassis, and all five rover
bodies participate in two-way coupling. The terrain follows the chassis by name.

![Rover on beta moving terrain](../../../docs/images/examples/example_moving_terrain_mpm_rover.jpg)

The box, rover, and G1 inherit directly from `MovingTerrainExample` in
`impl/base_example.py`. The base owns terrain, coupling, rendering, and shared
options. Each example supplies its own bodies and controls. The rover uses
`impl/controller/differential_drive.py`.
Its wheel geometry and drive controls are unchanged.

The box is a dynamic 75 kg rigid body measuring 0.5 × 0.7 × 0.5 m. It starts
2 m above the local ground and falls onto the sand. **Right-click drag the box**
to move it; no automatic object controller is enabled. Gravity, contact, and
two-way MPM coupling remain active. Enable **Show fixed boundary particles** to
show the boundary terrain in orange. The shovel geometry remains available in
`impl/shovel.py` for a separate scene.

![Box on simulated terrain](../../../docs/images/examples/example_moving_terrain_mpm_box.jpg)

The example uses `MovingTerrainMPMWrapper` from `impl/moving_terrain_mpm.py` and imports
no classes from `mpm_moving_patch_alpha`. The local `impl/terrain_generator.py` supplies flat,
slope, and sinusoidal ground to both examples.

The terrain wrapper provides:

- Dynamic particles near the tracked body retain their state.
- Boundary particles remain `ACTIVE` with zero simulation mass and velocity. A
  separate snapshot preserves their positions, physical mass, velocity, APIC
  gradient, elastic deformation, plastic state, stress, rendering transform,
  radius, flags, and height-color reference fraction.
- Visited terrain outside the particle patch is frozen in GPU particle-state
  storage. Returning terrain restores those particles without resampling them.
  Previously empty columns stay empty, and transported material keeps its position.

Material density and the `material_parameters` dictionary (friction, Young's
modulus and Poisson's ratio) are defined in `impl/base_example.py` and
passed to the wrapper. The rover inherits those settings. The wrapper creates
particles using the supplied material values; it does not choose its own material.

Particle jitter uses the same `jitter` parameter as Newton's
`ModelBuilder.add_particle_grid`: it is a distance in meters, and each coordinate
receives `(random - 0.5) * jitter`. The example sets `jitter = 2.0 * radius`, giving
offsets up to ±radius, just like the reference coupled example. Set `jitter = 0.0`
to disable it. The rover inherits the same setting. The fixed seeds keep initialization and first visits reproducible. Revisiting saved
terrain restores its exact positions and adds no jitter. Full-state caching requires
jitter no larger than the smallest particle spacing so first-visit sampling stays
inside its assigned terrain column.

`MovingTerrainMPMConfig` specifies the moving terrain size, boundary thickness,
background size, centered background origin and background height spacing.
The wrapper resolves the tracked body by name and derives particle count from
these dimensions, sand depth, voxel size and samples per voxel axis.

The patch center follows the tracked body in increments of one background sample, with
sample-aligned bounds and a one-sample deadband to avoid repeated shifts from
small body oscillations. Background coordinates do not move. Leaving the finite
background bounds raises an error before modifying particles or heights.

The box selects `material = MATERIAL_PRESETS["sand"]` in its
`setup_terrain()` method. Change the key to `"snow"` or `"clay"` to use those
presets. Physical fields become `mpm:` entries in `material_parameters`;
`density` is passed separately to the wrapper and `label` is display metadata.
Poisson's ratio remains 0.3. Rover and G1 retain the base material settings.

## Physics substeps

`--sub-step` sets the number of coupled physics steps per rendered frame (default
1). The base currently sets `fps = 50`, so `--sub-step 4` runs coupled physics at
200 Hz and rendering at 50 Hz:

```bash
uv run -m newton.examples moving_terrain_mpm_box --sub-step 4
```

`frame_dt = 1 / fps` and `sim_dt = frame_dt / sub_step`. Each physics substep
updates the terrain, reapplies forces, performs collision detection, and advances
both coupled solvers. The viewer renders after all substeps complete. These are
simulation-time rates; achieved wall-clock FPS depends on computation cost.

`--rigid-substeps` remains MuJoCo's additional substeps inside each coupled step.
G1 policy inference stays at 50 Hz, with joint targets held between policy updates;
its physics rate must be an integer multiple of 50 Hz.

## Override terrain and solver setup

Each child example can override two methods in
[`MovingTerrainExample`](impl/base_example.py):

- `setup_terrain(args, *, tracked_body)` defines `self.terrain_gen` and
  `self.mpm_wrapper`. It runs before ground and particle allocation. Set material
  parameters, density, jitter, voxel size, particle samples, sand depth, and the
  moving/background terrain sizes in the wrapper definition here.
- `setup_solver(args, *, rigid_bodies, mpm_config=None)` defines `self.solver`
  after model finalization. It contains the MPM configuration and two-way coupled
  solver definition. Pass a complete custom MPM config to the base method to
  retain its coupling setup, or override the method to change coupling as well.

For example, a child can change material friction and supply its own solver config:

```python
from newton.solvers import SolverImplicitMPM


def setup_terrain(self, args, *, tracked_body: str):
    super().setup_terrain(args, tracked_body=tracked_body)
    # No particles have been allocated yet.
    self.mpm_wrapper.material_parameters["mpm:friction"] = 0.7


def setup_solver(self, args, *, rigid_bodies: list[int], mpm_config=None):
    config = SolverImplicitMPM.Config()
    config.voxel_size = self.mpm_wrapper.voxel_size
    config.grid_type = "sparse"
    config.max_active_cell_count = 131072
    config.max_iterations = 80
    super().setup_solver(args, rigid_bodies=rigid_bodies, mpm_config=config)
```

For different terrain dimensions, construct `MovingTerrainMPMConfig` and
`MovingTerrainMPMWrapper` in the child's `setup_terrain()`, following the base
method's definition. The base initializer handles allocation, model finalization,
state creation, wrapper binding, controls, and rendering afterward. The default
solver derives voxel size and capacity from the actual wrapper, including its
sand depth. A supplied MPM config is used unchanged.

## G1 locomotion policy

```bash
uv run -m newton.examples moving_terrain_mpm_g1 --config config/g1.yaml
uv run -m newton.examples moving_terrain_mpm_g1 --auto-forward --command 0.2 0 0
uv run -m newton.examples moving_terrain_mpm_g1 --terrain-type slope --slope-angle 10
```

![G1 on beta moving terrain](../../../docs/images/examples/example_moving_terrain_mpm_g1.jpg)

The camera follows the G1 pelvis. OpenGL preserves the user's orbit and zoom
while tracking its translation. Other viewers use the reference offset
`(2.5, -2.5, 1.0)` m from the pelvis, with pitch −25° and yaw 125°.

The visuals match the reference G1 scene: green arrows show commanded XY velocity
(rotated by the robot's yaw), and blue arrows show measured world XY velocity.
Both use 0.4 m of arrow length per m/s, ignore vertical velocity, and are drawn
above the robot. They update each rendered frame from the current simulation state.
Sand uses RGB `(0.7, 0.6, 0.4)`, including boundary particles when shown. The ground
uses the same seeded brown texture at 32 texels per meter. These helpers live in
`impl/scene_visuals.py`; beta does not import the older example. Optional slope
coloring still overrides the sand palette. The box and rover keep their existing
appearance.


G1 uses the same 29-joint TorchScript policy contract as the earlier G1 example.
The policy and its history helpers are local to `impl/controller/`. Observation
history is ordered by term: body angular velocity, projected gravity, velocity
and heading command, joint position error, joint velocity, and previous action.
Joint names from the YAML file map policy outputs to model coordinates; the
configured stiffness, damping, armature, and action scale are retained.

The policy updates at 50 Hz while the coupled scene runs at `fps * sub_step` Hz. Policy targets
are held between updates. MuJoCo advances the articulated robot and exchanges
forces with MPM through the same two-way proxy coupling as the other beta scenes.
The terrain follows the pelvis and uses Ver1.0 caching by default. Boundary
particles remain active and fixed.

Hold I/K for forward/backward, J/L for sideways motion, and U/O for yaw. Commands
increase while keys are held; releasing a key retains the command. Backspace
stops it. `--auto-forward` uses `--command VX VY WZ` after the initial settling
interval. Keyboard input overrides automatic walking.

`--config` defaults to the repository's `config/g1.yaml`. `--policy` and `--asset`
override the configured TorchScript file and MJCF/URDF robot. The asset lookup
uses the sibling `newton-assets/unitree_g1` directory when present, otherwise the
Newton asset downloader. The policy file must already be available. Torch and
PyYAML are required, as in the original G1 example.

Short flat-ground and slope runs check finite coupled states and policy execution;
they do not establish long-duration balance on deformable terrain.

## Terrain height persistence

Ver1.0 measured-height caching is the default for the box, rover, and G1. Each
background XY column stores a height estimated from particle positions and a
separate reference material volume per column area. Height controls restored
particle Z; reference material volume controls mass and radius. Vertical
compression therefore changes the saved height without removing material.

The height estimate uses mean height above local ground, corrected by the original
particle sampling fractions. It assumes a uniformly filled column. It preserves
uniform vertical compression and accounts for initial jitter, but does not preserve
individual particle positions, overhangs, or full material history. Returning
particles may settle again because stress and deformation reset.

Both background arrays stay on the GPU. Column sums run only when the terrain
patch moves. Ver1.0 does not allocate the full-state cache or its extra particle
slots. `--height-only-cache` explicitly selects this default.

```bash
# Ver1.0: measured-height cache
uv run -m newton.examples moving_terrain_mpm_box --color-by-height
# Ver2.0: optional, more expensive full-state cache
uv run -m newton.examples moving_terrain_mpm_box --full-state-cache --color-by-height
```

`MovingTerrainMPMConfig.preserve_particle_state=True` selects Ver2.0 in code.
Ver2.0 preserves XYZ positions and material history in GPU particle records,
including a separate boundary snapshot and 50% extra simulation slots. Its
storage grows with visited terrain and it performs more copying and allocation.
See [the cache design history](changelog/particle_state_cache.md) for the fields,
transitions, costs, and limitations of each version.

Both modes run without outer CUDA graph capture. They update MPM particle
properties through the public solver API; `reset(flags=0)` clears numerical warm
starts without resetting physical particle state.

## Validation

```bash
uv run --no-sync python -m unittest newton.tests.test_mpm_beta_particle_state_cache newton.tests.test_mpm_moving_terrain_beta newton.tests.test_mpm_beta_rover -v
uv run -m newton.examples moving_terrain_mpm_box --viewer null --num-frames 300 \
    --voxel-size 0.1 --particles-per-cell 2 --test
```

The tests cover exact compressed-position and material-state revisits, boundary
snapshot restoration after overwritten fields, empty columns, repeated visits
without duplicates, and the measured-height reconstruction. The example's `test_final()` checks finite simulation state and boundary
mass/flags. These checks do not establish physical convergence against a larger
fully simulated bed.

## Generated terrain

Both beta examples accept the same terrain options:

```bash
uv run -m newton.examples moving_terrain_mpm_rover --terrain-type slope --slope-angle 20
uv run -m newton.examples moving_terrain_mpm_box --terrain-type sinusoidal --wave-amplitude 0.3 --wave-wavelength 4
```

Flat ground remains the default. `--wave-phase` sets the sinusoidal phase in degrees.
The example creates the generator and passes it to the wrapper:

```python
terrain_gen = TerrainGenerator(type="slope", params={"angle_degrees": 20.0})
# Pass terrain_gen=terrain_gen to MovingTerrainMPMWrapper(...).
```

The wrapper samples the ground once at `background_height_spacing`. The static
collision mesh, initial particles, and recycled particles use those same samples.
All supported profiles vary along X and remain constant along Y; linear interpolation
between ground samples therefore matches the mesh triangles. `sand_depth` is vertical
thickness, not thickness measured perpendicular to a slope.

The underlying ground samples remain unchanged. Full-state mode caches the exact
particles above that ground. Measured-height mode additionally updates a sand
surface height array and reconstructs particles with default material history.
The rover starts parallel to the ground slope under its chassis. This does not guarantee
clearance across its entire footprint on sharply curved terrain. Steep slopes may also
cause the sand to flow; the generator does not change the material's friction.

## Particle rendering detail

Both beta examples compact particles into separate rendering arrays before calling
GL or RTX. Zero-radius particles (hidden boundary particles or inactive particles)
are excluded from the submitted arrays. Physics still uses the full simulated and
boundary terrain. Hiding all particles also skips selection and compaction.

`--particle-lod-pixels` controls distance-based level of detail (LOD):

```bash
# Keep every visible grain, but exclude hidden particles from viewer updates.
uv run -m newton.examples moving_terrain_mpm_rover --viewer rtx --particle-lod-pixels 0
# Reduce distant particle detail more aggressively.
uv run -m newton.examples moving_terrain_mpm_rover --viewer rtx --particle-lod-pixels 4
```

The default is 2 pixels. The GL side panel exposes the same setting and shows the
submitted particle count. Larger values keep fewer distant particles. The selection
uses approximate screen size from camera distance, image height, and field of view.
Particles that already appear larger than the target retain full detail. Without a
camera, only visibility compaction is applied.

Distant particles use a stable sample based on particle ID and larger render spheres
to preserve approximate projected area. This can change silhouettes or produce small
pops as the camera moves; set the value to 0 when inspecting individual grains. This
is not a sand-surface reconstruction, and buried or off-screen particles are not
explicitly culled. Particle positions, physical radii, mass, and material state remain
unchanged. The implementation is local to `impl/particle_rendering.py` and shared by the
beta box, rover, and G1 examples.

Rendering buffers are allocated once and reused. GPU compaction scans all simulation
particles; only a draw count is read back to the CPU. RTX still transfers the selected
particle attributes through its existing update path. Compaction therefore reduces
viewer work but adds a small synchronization and does not guarantee a particular FPS
improvement. It also uses additional GPU storage for rendering buffers.

## Terrain slope particle colors

Enable **Color particles by terrain slope** in the GL or RTX sidebar, or run:

```bash
uv run -m newton.examples moving_terrain_mpm_box --color-by-slope --slope-color-range 1.0
```

The rover accepts the same options. Flat terrain is blue. Increasing steepness
blends toward red. The diagnostic uses the gradient magnitude:

```text
slope = sqrt((partial z / partial x)^2 + (partial z / partial y)^2)
```

Unlike adding the two derivatives, this does not cancel opposing X/Y slopes.
`--slope-color-range` is the dimensionless slope at full red, in meters of rise
per meter horizontally. The default 1.0 corresponds to a 45-degree slope.
The earlier `--color-by-height` and `--height-color-range` flags remain aliases,
but now select slope coloring and use slope units.

Each rendered frame measures the live particles, including visible boundary
terrain. It estimates one surface height per XY column using the same mean-based
height reconstruction as Ver1.0, then takes finite differences at
`background_height_spacing`. The gradient includes the underlying ground slope.
Missing neighboring columns use one-sided differences; isolated columns are blue
because their slope cannot be estimated. Empty space is not treated as zero height.

This shows steepness, not how far a particle has sunk. A flat-bottomed depression
is blue while its sloped sides become red. Particle noise and the approximate
column reconstruction can affect the estimate; it is not a fitted surface normal.

The diagnostic is optional and off by default. Enabling it adds three GPU kernels
per rendered frame and three small arrays covering the moving and boundary
terrain. It does not read particle positions back to the CPU or modify the terrain
cache, particle positions, or material state. Particle compaction and LOD preserve
the selected colors.
