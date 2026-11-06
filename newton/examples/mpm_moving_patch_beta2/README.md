# Beta2: moving MPM terrain with GPU-only height memory

Beta2 simulates a local particle bed and saves departed terrain heights on the
GPU. There is no CPU terrain cache, no height-window paging, and no runtime
height-array download or upload.

```bash
uv run -m newton.examples moving_terrain_mpm_box_beta2
uv run -m newton.examples moving_terrain_mpm_box_beta2 --color-by-height
uv run -m newton.examples moving_terrain_mpm_box_beta2 --terrain-type slope --slope-angle 20
```

Right-click drag the dynamic box; the camera follows it. MuJoCo and implicit MPM
exchange forces in both directions. Physics runs at `50 * sub_step` Hz.
The default `sub_step=2` matches beta: two physics steps per rendered frame.

## Robot and shovel examples

```bash
uv run -m newton.examples moving_terrain_mpm_rover_beta2
uv run -m newton.examples moving_terrain_mpm_rover_beta2 --wheel-type grouser --terrain-type slope --slope-angle 10
uv run -m newton.examples moving_terrain_mpm_shovel_beta2
uv run -m newton.examples moving_terrain_mpm_g1_beta2 --config config/g1.yaml
```

These scenes share `impl/base_example.py` and beta2's GPU height cache. Controllers,
rover assets, shovel geometry, slope coloring, and scene visuals are local to beta2.
The original beta examples remain unchanged. All scenes retain two-way coupling,
static background ground, camera following, compact particle rendering, and optional
`--color-by-slope` visualization. Terrain choices are `flat` and `slope`.

- **Rover:** drives forward automatically; use `--manual` for keyboard control.
  I/K drives forward/backward, J/L turns, and O stops automatic driving.
- **Shovel:** right-click drag with the mouse; no autonomous object controller.
- **G1:** uses the existing G1 configuration, TorchScript policy, and robot asset.
  I/K, J/L, and U/O increment forward, sideways, and yaw commands; Backspace stops.
  `--auto-forward --command 1 0 0` enables automatic walking. Green arrows show
  commanded velocity and blue arrows measured XY velocity. The policy runs at 50 Hz.

MPM material settings remain in each scene's `setup_terrain()` or the shared base.
The shovel preserves beta's sand preset; it is not material-matched to the rover/G1.
Physics runs at `50 * sub_step` Hz, independently of the 50 Hz rendered frames.

## What is simulated and rendered

| Terrain | Representation | Behavior |
| --- | --- | --- |
| Simulated terrain | Dynamic MPM particles around the tracked body | Simulated and rendered |
| Boundary terrain | Zero-mass MPM particles around simulated terrain | Remains `ACTIVE` to provide support; hidden unless enabled in the sidebar |
| Background terrain | Saved heights and reference material volumes in GPU tiles | Neither simulated nor rendered |

The brown surface below the particles is **supporting ground**, separate from
these three sand layers. It is a static heightfield mesh with collision enabled,
covering `background_terrain_size` and centered at `background_terrain_origin`.
The box example uses a 40 × 40 m background centered at `(0, 0)`, matching beta.
It supplies the generated flat or sloped bottom surface and does not move or
display saved sand deformation. Keep the particle bed inside this finite mesh;
there is no supporting ground beyond its edges.

## Terrain update

1. Compute the patch center from the tracked body on the GPU.
2. Check whether the patch moved or dynamic sand crossed its boundary. These
   particle-departure checks scan only local particles, not the background world.
   Sand can cross a boundary even when the robot is stationary.
3. Before freezing or recycling particles, measure column heights and reference
   volumes and write them directly into GPU background tiles.
4. Recycled particles read those GPU tiles. Missing samples use the initial
   terrain height. Saved empty columns remain empty. Retained simulated particles
   keep their physical state.
5. Update changed particle properties and advance the coupled physics solvers.

```text
local MPM particles -> saved heights/volumes on GPU -> reconstructed particles
```

A column's sand thickness is the sum of particle heights above ground divided by
the sum of their initial vertical fractions. This keeps beta's measured-height
reconstruction and its sampling/jitter correction. Reference volume is saved
separately so compression does not erase material mass.

Height memory does not preserve stresses, voids, overhangs, or exact particle
arrangements. Reconstructed particles receive fresh material history, including
the configured initial `particle_Jp`, and may settle again. Terrain outside the
simulated region does not continue flowing or settling.

## GPU storage and configuration

A GPU hash table maps world tile coordinates to saved tiles. The allocation stays
fixed as the robot moves; there is no rectangular allocation for saved sand heights and no
CPU tile dictionary. Only visited samples occupy valid cache entries.

```python
config = MovingTerrainMPMConfig(
    moving_terrain_size=(2.0, 2.0),  # tuple[float, float]: simulated X/Y size [m]
    boundary_terrain_size=0.5,  # float: support band on each side [m]
    background_terrain_size=(40.0, 40.0),  # tuple[float, float]: supporting mesh X/Y size [m]
    background_terrain_origin=(0.0, 0.0),  # tuple[float, float]: supporting mesh center [m]
    background_tile_size=2.0,  # float: saved tile side length [m]
    background_height_spacing=0.05,  # float: sample spacing [m]
    background_tile_capacity=1024,  # int: maximum saved tiles on the GPU
)
```

The default height cache reserves approximately **14.07 MiB**, including heights,
reference volumes, validity, and tile keys. It raises an error if capacity is
exhausted, before recycling particles. It never evicts saved terrain or spills
to the CPU. Increase `background_tile_capacity` to retain more terrain. History
is kept only for the lifetime of the simulation. Origin rebasing is not implemented.

Python still launches simulation kernels and reads small status counters for
control flow and errors. The targeted MPM property refresh also reads four scalar
material extrema. **Terrain heights and reference volumes remain on the GPU.**
This is not an entirely CPU-free or CUDA-graph-captured simulation loop.

Material parameters remain in `setup_terrain()` and solver settings in
`setup_solver()`. Jitter uses the same distance units as `add_particle_grid`.
The wrapper allocates particles before model finalization, binds to the coupled
MPM entry, and updates terrain before each physics step.

## Performance scope

Beta2 currently simulates the same local particle bed as beta. The default bed
contains **972,000 particles**, including boundary particles. GPU-only terrain
memory removes CPU height paging; the targeted particle-property notification
avoids full material-array readbacks. Neither change reduces the number of
particles and grid cells in the MPM solve.

Separate regions around wheels/feet and cheaper boundary support are not yet
implemented. Those would target the actual MPM workload. Current support is one
tracked body, one world, and flat or sloped ground. Wave-shaped ground colliders,
full physical-state history, and the discard-history option are not implemented.
Height preservation is always enabled.

## Validation

Tests cover retained particle state, revisiting compressed terrain, empty columns,
stationary material outflow, negative coordinates, hash collisions, concurrent
writes, and capacity exhaustion without eviction. A transfer-guard test rejects
height downloads/uploads during terrain updates. A real coupled test travels to
between positive and negative coordinates while checking fixed allocations and a stationary ground mesh.

```bash
uv run --no-sync python -m unittest \
  newton.tests.test_mpm_moving_terrain_beta2 \
  newton.tests.test_mpm_beta2_height_cache \
  newton.tests.test_mpm_beta2_coupled \
  newton.tests.test_implicit_mpm_particle_properties
```

See [benchmark/README.md](benchmark/README.md) for a short matched timing check.
Force continuity, boundary-size sensitivity, and long-duration terrain-memory
accuracy still need broader validation against a larger fully simulated bed.
