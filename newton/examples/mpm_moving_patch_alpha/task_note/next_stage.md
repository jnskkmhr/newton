## Scaling to rough terrain 
* Current implementation resets particles at reference z position (which is initial particle q)
* But for rough terrain setup where object keeps moving over rough terrain, there will be rough-shaped hard ground and MPM particles spawened on that
* This requires sampling z height efficiently over rough terrain, wondering what is the good approach. And this method needs to scale to hundreds of parallel environment. I think there will be shared hard (rough) ground and MPM-grid + robots will be spawened over rough terrain per environment. 
* You can refer to newton/examples/mpm/example_mpm_snow_ball.py for sampling particle position at reset from height field. 
* Can you make one example file called `example_mujoco_mpm_coupled_solver_moving_patch_slope.py` where you simulate slope terrain with slope angle as parameters (let's do 30deg slope as default). In this case, height field is static mesh with 30deg slope. You could probably also render this static height field mesh.


## Clean modular framework

I want to integrate this moving patch method to main MPM codebase or write wrapper class around MPM so that it can be easily integrated to later integraiton to IsaacLab
I think the main part is 
* warp kernel (_recycle_terrain) that reset particles in buffer region (not rendered)
* adaptive particle rendering kernel (_particle_render_radii) 
* (optional) _record_window_center and _window_outline for moving window rendering

I would add wrapper class and add methods: 
* reset_particle_background
* render_active_particle


I would make code like

```python 

self.mpm_wrapper = MovingPatchMPMWrapper(mpm, ....)

# inside main file simulate function, 
self.mpm_wrapper.reset_particle_background()

# inside render function, 
self.mpm.render_active_particle(..)
self.viewer.log_points(
        "/sand",
        points=render_state.particle_q,
        radii=self.particle_render_radii,
        colors=self.particle_render_colors,
        hidden=not self.viewer.show_particles,
    )
```
## Implemented: sampled terrain and reusable patch management

The first rough-terrain stage is implemented in
[`moving_patch_mpm.py`](../impl/moving_patch_mpm.py) and
[`example_mujoco_mpm_coupled_solver_moving_patch_rover.py`](../example_mujoco_mpm_coupled_solver_moving_patch_rover.py).
The existing flat moving-patch example now delegates patch management to the
same wrapper, so the rover and articulated subclasses reuse it too. MPM and
proxy-coupling solver algorithms are unchanged.

Run the dynamically driven rover on a 30-degree uphill slope:

```bash
uv run --extra examples -m newton.examples mujoco_mpm_coupled_solver_moving_patch_rover --terrain-type slope --slope-angle 30
```

The angle is in degrees; positive angles rise along world +X. Zero and negative
angles are supported. The rover is initially aligned with the slope. It retains
the flat scene's terrain sampling, material parameters, differential wheel drive,
two-way coupling, camera tracking, and optional video recording:

```bash
uv run --extra examples -m newton.examples mujoco_mpm_coupled_solver_moving_patch_rover \
  --terrain-type slope --slope-angle 30 --turn-rate 0 --drive-speed 0.5 \
  --headless --video slope.mp4 --video-duration 5
```

The gray terrain is a stationary watertight heightfield collision mesh, not a
moving floor. No horizontal ground plane is added. The blue optional outline
marks the simulation window at its reference sand-top height; it does not trace
the deformed sand surface. The visible region still has a sharp cutoff, with no
fade zone.

### Height sampling and recycling

`HeightFieldTerrain` uploads a regular array of ground heights once. Samples use
`heights[i, j] = height(x_i, y_j)`, matching `newton.Mesh.create_heightfield`.
The GPU sampler uses the same pair of triangles per cell as that mesh. This
avoids the surface discrepancy that bilinear interpolation can introduce on a
nonplanar cell. Lookup has constant cost and needs no per-particle mesh raycast.
The CPU equivalent is used only for initial particle construction and diagnostics.

Initial sand positions are the original jittered particle bed plus the sampled
ground height. On each coupled step, only particles outside the outer XY window
are wrapped. Their new height is:

```text
new_z = ground_height(new_x, new_y)
      + reference_z - ground_height(reference_x, reference_y)
```

Thus each recycled slot retains its original vertical layer offset above the
local terrain, rather than its original world Z. This is vertical bed thickness,
not thickness measured normal to the surface. Retained particles preserve their
world position and material history. Escaped particles reset velocity, APIC
gradient, elastic strain, plastic determinant, stress, and rendering deformation.
The plastic determinant is restored from each particle's configured initial
value, including custom values.

Background particles still inside the full simulation window are not reset and
remain physically simulated even when hidden.

### Wrapper interface

The wrapper is example-local and does not own or step the MPM solver. Pass the
state containing world particle positions/velocities separately from the state
containing MPM history. In proxy coupling these belong to different states:

```python
from newton.examples.mpm_moving_patch_alpha.impl.moving_patch_mpm import MovingPatchMPMWrapper

self.mpm_wrapper = MovingPatchMPMWrapper(
    self.model,
    self.state_0,
    self.solver.entry_state("mpm", phase="input"),
    bed_lo=self.bed_lo,
    bed_hi=self.bed_hi,
    shift_step=self.shift_step,
    terrain=self.terrain_heightfield,  # None means flat ground at Z=0.
    body_index=0,
)

# Before the coupled physics step:
self.mpm_wrapper.reset_particle_background()

# When rendering, before viewer.log_points(...):
radii = self.mpm_wrapper.render_active_particle(
    positions=render_state.particle_q,
    visible_ratio=self.visible_ratio,
    show_buffer=self.show_buffer,
)

# Optional simulation-window visualization:
self.mpm_wrapper.update_window_outline()
```

`render_active_particle` computes rendering radii; it does not draw or modify
physical particle radii. For a standalone in-place MPM solver, the same state can
be supplied for both state arguments. The wrapper binds those arrays at
construction; recreate it if states are reallocated or ownership changes.

### Scaling and limits

A shared heightfield can serve multiple wrappers on the same device in the same
world coordinate system. Per-step sampling and recycling run on the GPU and are
compatible with CUDA graph capture. The number of particles scanned is bounded
by the patch allocation, not the full terrain area.

This is not yet a batched implementation or an IsaacLab integration. Each wrapper
launches its own kernels, and each MPM simulation retains its own solver and
particle allocations. Hundreds of environments still need a benchmark and likely
batched particle/environment indexing, per-environment state views, and shared
or tiled terrain collision data. Increasing full-terrain mesh resolution also
increases terrain storage and collision setup cost, independently of patch size.

The slope demo has a finite ground mesh and explicitly stops if the complete
patch leaves its sampled domain. Its bounds guard copies one rigid pose to the
CPU per step; that demo guard should be batched or moved to a GPU status buffer
for large environment counts. Height lookup itself clamps at sample edges and
does not provide terrain outside the physical mesh. General rough heightfields
can use the same sampler, but overhangs, terrain streaming, persistent tracks,
and force continuity across recycled boundaries remain outside this stage.

At the current material friction of `tan(30 degrees)`, the default slope is near
the sand's friction angle. In the tested run the rover and sand slide downhill
despite an uphill wheel command. The example demonstrates terrain following and
coupling, not a validated hill-climbing controller.

### Validation

```bash
uv run --extra dev -m newton.tests.test_mpm_moving_patch
uv run --extra dev -m newton.tests.test_mpm_moving_patch_heightfield
```

The tests cover flat-window behavior, CPU/GPU triangle sampling, multiple window
shifts under CUDA graph replay, preservation of retained particles, complete
reset of recycled state with custom initial plastic determinants, render-only
visibility, and terrain-relative window outlines. A rendered 400-step default
slope run remained finite, stayed above the sampled ground within the test
tolerance, and performed four window shifts.

## Procedural terrain generator

[`TerrainGenerator`](../impl/terrain_generator.py) now supplies the heights used by
`_emit_ground()`. It supports `flat`, `slope`, and `sinusoidal` profiles, all
constant along Y. The constructor accepts a terrain type and parameter mapping:

```python
self.terrain_gen = TerrainGenerator(
    type="sinusoidal",
    params={"amplitude": 0.3, "wavelength": 4.0, "phase_degrees": 0.0},
)

# Inside _emit_ground(), with the existing X/Y sampling coordinates:
heights = self.terrain_gen.sample_height(x[:, None], y[None, :])
terrain = HeightFieldTerrain(
    heights,
    origin=(x[0], y[0], 0.0),
    spacing=(x[1] - x[0], y[1] - y[0]),
)
```

The rest of the heightfield mesh, checkerboard texture, initial particle bed, and
moving-patch wrapper uses these same samples. The procedural function runs only
when constructing the terrain; per-step sampling remains in the GPU heightfield
wrapper. The rover's initial height and tilt are computed from the sampled ground
near its starting position instead of using a slope-specific formula.

| Type | Parameters and defaults | Height function |
|---|---|---|
| `flat` | `height=0` | `height` |
| `slope` | `angle_degrees=30`, `height=0` | `height + tan(angle) * x` |
| `sinusoidal` | `amplitude=0.3`, `wavelength=4`, `phase_degrees=0`, `height=0` | `height + amplitude * sin(2*pi*x/wavelength + phase)` |

Lengths and heights are in meters; angle and phase parameters are in degrees.
Amplitude is the offset from the mean, so peak-to-trough height is twice the
amplitude. The generator accepts scalars, arrays, and broadcastable coordinate
grids, and rejects unknown or invalid parameters. `TerrainType` enum values may
also be used instead of strings.

The unified rover example defaults to flat terrain; select slope or waves explicitly:

```bash
uv run --extra examples -m newton.examples mujoco_mpm_coupled_solver_moving_patch_rover \
  --terrain-type slope --slope-angle 30

uv run --extra examples -m newton.examples mujoco_mpm_coupled_solver_moving_patch_rover \
  --terrain-type sinusoidal --wave-amplitude 0.3 --wave-wavelength 4 --wave-phase 0

uv run --extra examples -m newton.examples mujoco_mpm_coupled_solver_moving_patch_rover \
  --terrain-type flat
```

All terrain choices support `--wheel-type`, camera tracking, and recording. The
terrain grid spacing remains defined in `_emit_ground()`: short wavelengths need
finer terrain samples. The collision surface and recycling use the triangulated
sampled surface, not the continuous analytic function between samples. Strong
curvature or steep waves may require finer MPM resolution and a more capable
controller; selecting a profile does not guarantee traversability.

Generator tests:

```bash
uv run --extra dev -m newton.tests.test_mpm_terrain_generator
```


## Shared terrain for policy robots

Rover, ANYmal, and G1 now use `TerrainMovingPatchExample` in
[`terrain_moving_patch.py`](../impl/terrain_moving_patch.py). All three expose the
same `--terrain-type flat|slope|sinusoidal`, `--slope-angle`, and wave parameters.
The sampled mesh, initial sand placement, and GPU recycling share one heightfield.
Flat remains the default. The patch stops at the finite ground mesh boundary.

ANYmal and G1 retain their policy joint poses and configured root orientations;
their starting height is offset by the sampled terrain height at the root.
They are not tilted to the local slope like the rover. Existing policy inference,
control rates, and two-way coupling are unchanged. Terrain support does not imply
the pretrained policies can traverse steep slopes or deep sand.

```bash
uv run --no-sync -m newton.examples mujoco_mpm_coupled_solver_moving_patch_anymal \
  --terrain-type slope --slope-angle 10
uv run --no-sync -m newton.examples mujoco_mpm_coupled_solver_moving_patch_g1 \
  --config config/g1.yaml --terrain-type sinusoidal --wave-amplitude 0.15 --wave-wavelength 4
```

### Incremental keyboard commands (ANYmal and G1)

Hold I/K to increase/decrease forward velocity, J/L for sideways velocity, and
U/O for yaw rate. Commands accumulate from the current value and persist on
release. Rates are 1 m/s² forward, 0.5 m/s² sideways, and 1 rad/s² yaw, scaled
by simulation time. Backspace zeros all commands; Space remains viewer pause.
Keyboard input takes over from automatic walking without clearing the current
command. Re-enabling automatic walking selects its configured command again.

### Recording policy robots

ANYmal, G1, and rover share OpenGL recording. Pass `--video output.mp4`
and optionally `--video-duration 10` (simulated seconds), `--video-speed 1`
(playback speed), and `--headless` for offscreen rendering. Recording duration
supersedes `--num-frames`, prints progress, and exits and saves automatically.
Ctrl+C saves the frames captured so far. Camera tracking and velocity arrows
are included. Recording requires `--viewer gl`; RTX frame capture is not supported.

```bash
uv run --no-sync -m newton.examples mujoco_mpm_coupled_solver_moving_patch_anymal \
  --viewer gl --video anymal.mp4 --video-duration 10
uv run --no-sync -m newton.examples mujoco_mpm_coupled_solver_moving_patch_g1 \
  --viewer gl --video g1.mp4 --video-duration 10
```

## G1 mimic jump on moving-patch sand

`example_mujoco_mpm_coupled_solver_moving_patch_g1_jump.py` uses the reference
`example_g1_jump.py` mimic observation contract: 29 reference joints, one phase,
body angular velocity, projected gravity, 29 joint-position offsets, 29 joint
velocities, and 29 previous actions (123 inputs total). The policy outputs
29 position-target offsets, scaled by the YAML configuration.

The supplied `config/g1_mimic.yaml`, `config/motion.csv`, and its selected
`config/policy_irsvxnpg.pt` are copied from the reference checkout. Joint order,
gains, scales, motion indexing, and final-frame/phase holding follow that source.
The robot starts at its configured pose lifted by the sand-bed thickness and
sampled terrain height. All original robot particle-contact shapes are retained.

Policy and video frames run at 60 Hz; each frame advances four coupled physics
steps at 240 Hz, with one rigid substep by default. The existing moving-patch
MPM material and proxy coupling settings are retained. This is an adaptation of
the policy to the moving bed, not a reproduction of the source's fixed launch
platform, retaining walls, PIC transfer, or extra landing projection. Successful
execution does not by itself establish the original jump distance or stability.
Keyboard velocity commands do not apply to a reference-motion policy.

```bash
uv run --no-sync -m newton.examples mujoco_mpm_coupled_solver_moving_patch_g1_jump
uv run --no-sync -m newton.examples mujoco_mpm_coupled_solver_moving_patch_g1_jump \
  --headless --video g1_jump.mp4 --video-duration 3.5
```

Use `--config`, `--motion`, `--policy`, or `--asset` to override the inputs.
Terrain selection uses the same `--terrain-type` and slope/wave parameters as
other robot examples. The default run covers the loaded clip plus one final held frame (210 frames,
3.5 simulated seconds for the supplied 209-row CSV).
After the motion ends its last target is held; the simulation is never reset.

Validation: an extended 810-frame run completed with finite coupled state and
terrain-relative ground checks passing, including the held-reference interval.
However, the robot falls early when starting directly on this deep sand bed.
The imported policy has not been retuned, and a clean jump is not demonstrated.
