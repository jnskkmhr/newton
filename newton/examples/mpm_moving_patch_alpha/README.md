# Alpha: moving patch terrain

See the [implementation explanation and figures](task_note/moving_patch_terrain.md) for:

- The physically simulated window, visible sand, and buffer.
- Particle recycling when the window follows the robot.
- The setup, two-way coupling, and rendering sequence.

## Directory layout

Examples remain at the top level. Shared implementation code follows the beta layout:

```text
mpm_moving_patch_alpha/
├── example_*.py
├── impl/
│   ├── base_example.py
│   ├── moving_terrain_mpm.py
│   ├── moving_patch_mpm.py
│   ├── terrain_moving_patch.py
│   ├── terrain_scene.py
│   ├── terrain_generator.py
│   ├── controller/
│   └── utils/
├── assets/
├── benchmark/
├── dynamic_sparse/
├── sparse_capacity_sweep/
├── doc/
└── task_note/
```

The directory was renamed from `mpm_moving_patch`. Existing short example commands are unchanged.

## Terrain appearance and collision

The **background visual mesh** is drawn at the generated ground height plus
`sand_depth`, matching the initial MPM sand surface. It has no collision geometry
in the physics model. Four surface strips leave an opening over the simulated
terrain, so the visual mesh does not hide tracks or depressions. Enabling boundary
particle rendering expands this opening to show the boundary too.

Use the sidebar checkbox **Show boundary particles** to toggle them during a run.
They are hidden by default. Start with `--show-boundary-particles` to show them,
or `--no-show-boundary-particles` to explicitly hide them. This option is shared
by the box, rover, ANYmal, G1, and G1 jump examples and changes rendering only.


A separate, **hidden local collider mesh** sits at the ground height beneath the
particles. It covers simulated terrain plus boundary terrain, with extra space
on every side. The margin is at least two MPM voxels, one patch shift, and one
ground sample spacing; the mesh dimensions are rounded outward to the ground grid.
It follows the patch and resamples the original ground on the GPU, supporting flat,
slope, and sinusoidal terrain. Its vertex count stays fixed. The same updated Warp
mesh is used by MPM and Newton's rigid collision pipeline; MuJoCo consumes Newton
contacts (`use_mujoco_contacts=False`). Mesh velocities remain zero because moving
the sampling window does not move the physical ground.

**Boundary particles** keep `ParticleFlags.ACTIVE`, their radius, and their material
volume, but have zero mass and inverse mass. Their velocity and APIC velocity gradient
are zeroed, making the boundary kinematic. Particles entering simulated terrain
recover their original mass and fresh material state. Particles already in simulated
terrain retain their physical state. No terrain-history cache is created.

The current example runs without outer CUDA graph capture: changing boundary masses
requires the MPM particle-property refresh API, which reads small status/extrema
values on the CPU. Particle resets, collider vertex updates, mesh refits, and visual
surface sampling run on the GPU. This change does not reduce the number of ACTIVE
particles in MPM. Keep the particle bed within the configured background extent.

## Physics substeps

The moving-patch examples use the same defaults as beta and beta2: 50 rendering
frames per second and `--sub-step 2`, giving 100 coupled physics steps per second.
`sim_dt = frame_dt / sub_step`. For example, `--sub-step 4` gives 200 Hz physics.
Each physics substep updates terrain, applies forces, detects contacts, and steps
both MPM and rigid bodies. Rendering and video capture run once per frame.
`--rigid-substeps` separately controls rigid-solver substeps inside each coupled step.

```bash
uv run -m newton.examples mujoco_mpm_coupled_solver_moving_patch --sub-step 2
```

G1 and ANYmal walking policies stay at 50 Hz as physics substeps change. G1 jump
keeps its 60 Hz reference/rendering rate and defaults to `--sub-step 4` (240 Hz
physics). The standalone full-bed benchmark examples do not use this shared base.

## Shared terrain API

Alpha uses `MovingTerrainMPMConfig` and `MovingTerrainMPMWrapper` from
[`impl/moving_terrain_mpm.py`](impl/moving_terrain_mpm.py), with the same constructor
and lifecycle names as beta/beta2. Alpha **discards departed terrain history**.
Returning to a previously disturbed area generates fresh particles above the original
ground profile. There is no saved-height or particle-state cache.

The box and articulated robot examples share
[`MovingTerrainExample`](impl/base_example.py). Override `setup_terrain()` for
material parameters, terrain dimensions, and particle sampling, or `setup_solver()`
for the MPM and two-way coupling settings. The rough-terrain robot base adds terrain
textures, robot placement, camera following, and video recording. The fixed and
full-bed sparse/adaptive-rendering benchmark scenes remain full-bed simulations;
they do not use a moving-terrain wrapper.

```python
from newton.examples.mpm_moving_patch_alpha.impl.moving_terrain_mpm import (
    MovingTerrainMPMConfig,
    MovingTerrainMPMWrapper,
)
from newton.examples.mpm_moving_patch_alpha.impl.terrain_generator import TerrainGenerator


def setup_terrain(self, args, *, tracked_body: str):
    self.terrain_gen = TerrainGenerator(type="flat")
    self.mpm_wrapper = MovingTerrainMPMWrapper(
        config=MovingTerrainMPMConfig(
            moving_terrain_size=(2.0, 2.0),
            boundary_terrain_size=0.5,
            background_terrain_size=(40.0, 16.0),
            background_terrain_origin=(0.0, 0.0),
            background_height_spacing=0.5,
            shift_step=8.0 * args.voxel_size,
        ),
        tracked_body=tracked_body,
        terrain_gen=self.terrain_gen,
        voxel_size=args.voxel_size,
        particles_per_cell=args.particles_per_cell,
        sand_depth=0.5,
        density=3000.0,
        jitter=args.voxel_size / args.particles_per_cell,
        material_parameters={
            "mpm:friction": 0.75,
            "mpm:young_modulus": 40.0e6,
            "mpm:poisson_ratio": 0.3,
        },
    )
```

`tracked_body` is an exact body name, not an integer index. The shared initializer
calls `add_ground(builder)` and `allocate_particle_storage(builder)`, finalizes the
model and coupled solver, then calls
`bind(model=model, state=state, coupled_solver=solver)`. Before each physics step,
`update_terrain_layers()` recycles escaped particles. `render_active_particle()`
selects the visible region without changing physical particle radii.

| Parameter or behavior | Alpha | Beta / beta2 |
| --- | --- | --- |
| `moving_terrain_size: tuple[float, float]` | Full X/Y size of normally visible simulated terrain [m] | Full X/Y size of dynamic simulated terrain [m] |
| `boundary_terrain_size: float` | Zero-mass boundary width on each side [m] | Same |
| `background_terrain_size: tuple[float, float]` | Visual background X/Y size [m] | Static ground mesh X/Y size [m] |
| `background_terrain_origin: tuple[float, float]` | Center of the visual background [m] | Center of the static ground mesh [m] |
| `background_height_spacing: float` | Original ground sample spacing [m] | Also controls saved-height sampling |
| `shift_step: float` | Existing alpha patch-center quantization [m] | Patch shifts follow background sample spacing |
| `density`, `material_parameters`, `voxel_size`, `particles_per_cell`, `jitter` | Same constructor names and units | Same |
| Departed terrain | Discarded; fresh particles on return | Deformation preserved by the selected terrain cache |

The full particle bed has X/Y size
`moving_terrain_size + 2 * boundary_terrain_size`. `visible_ratio` defaults to `1.0`
and scales the visible moving terrain per axis; `show_buffer=True` includes the
boundary. `bed_lo` and `bed_hi` are derived internal bounds, not additional user
configuration. Alpha retains its original vertical-endpoint particle sampling.
Its boundary is now kinematic, matching beta/beta2's zero-mass convention.

Particles that remain inside simulated terrain keep their velocity, deformation,
and stress. Recycled particles reset velocity, APIC velocity gradient, elastic strain,
plastic volume ratio, stress, and rendering transform; the plastic volume ratio
returns to its configured initial value. Ground sampling during recycling and these
resets run on the GPU. Boundary mass changes use an uncaptured property refresh.
The original ground samples are generated once during setup and are never updated
with sand deformation. The background visual surface therefore returns to its
original shape after the patch leaves an area.

All moving-patch commands keep their names and now accept `--voxel-size` and
`--particles-per-cell`. Existing flat, slope, sinusoidal terrain, policy control,
rover wheels, and recording options remain available.

## Adaptive rendering benchmark

`example_mujoco_mpm_coupled_solver_adaptive_rendering.py` keeps the original
coupled scene's physics and sparse-grid settings, but draws only a central XY
region of width `(bed_hi - bed_lo) * visible_ratio`. All heights are included.
There is no fading, particle recycling, or change to simulation particle count.
The region stays centered on the bed rather than following a rigid body.

```bash
uv run --extra examples -m newton.examples mujoco_mpm_coupled_solver_adaptive_rendering
```

The default 3 m by 3 m bed has a 1.875 m by 1.875 m visible region. The ratio is
per side, not area: a uniformly populated bed initially shows approximately
`visible_ratio**2` of its particles. Set `self.visible_ratio = 1.0` in the example for the full bed
footprint (particles outside those XY bounds remain hidden), or
`self.visible_ratio = 0.0` to skip sand drawing and compaction entirely.

Visible particles are compacted on the GPU and only that subset is submitted to
the viewer. The viewer's default unfiltered particle drawing is disabled.
Compaction and the host readback of the visible count add rendering overhead;
include both when measuring total frame time. Ratio zero still renders rigid
bodies, ground, and the UI, so it removes sand rendering rather than all rendering.
Use `--viewer null` for a separate headless simulation baseline. Keep physics,
particle counts, camera, viewport, impulses, and timing warmup consistent when
comparing cases. This example does not add a fixed-grid or moving-patch mode;
those remain separate scenes/configurations for the comparison.

## Autonomous dynamic box tracking

The original, adaptive-rendering, and moving-patch box scenes share
[`object_controller.py`](impl/controller/object_controller.py). Edit its `ObjectController`
settings to change the common reference motion or controller gains.

The box holds its initial horizontal position and orientation for 1 s, then
smoothly accelerates over 1 s toward a straight world +X reference at 0.2 m/s.
Horizontal PD forces are limited to 5000 N and orientation torques to 2000 N m.
There is no commanded vertical force: gravity, sinking, and sand reactions remain
fully dynamic. The controller adds forces before the coupled solver step; it never
sets body poses or velocities. Each box tracks from its own initial position.

The clock advances on the GPU with every physics frame, including CUDA-graph
replay. Mouse forces can still perturb tracking. Articulated robot subclasses keep
their policy controller and do not receive this box controller.

For benchmarking, use the same controller settings and simulated duration in all
three scenes. Exclude settling and acceleration from steady-motion timing. The
reference does not stop at the bed edge; choose a duration that keeps the box away
from full-terrain boundaries. Actual tracking can differ because sand forces remain
active; a common reference does not guarantee identical actual trajectories.

## Dynamic sparse allocation

```bash
uv run --extra examples -m newton.examples mujoco_mpm_coupled_solver_sparse
```

`example_mujoco_mpm_coupled_solver_sparse.py` is a separate full-bed scene with
`grid_type="sparse"`, `grid_padding=0`, and `max_active_cell_count=-1`.
It allocates the sparse grid from particle positions every step and does not
capture an outer CUDA graph. It retains the original scene's terrain sampling,
shared dynamic box controller, and two-way MuJoCo-MPM coupling. There is no
moving-patch recycling or partial rendering.

This is a different execution path from capacity-bounded, graph-captured sparse
MPM; label it separately in benchmark comparisons.
