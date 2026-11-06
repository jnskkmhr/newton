# Moving terrain cache design history

## Motivation

The moving-terrain examples must preserve disturbed sand when the shovel or robot
leaves an area and later returns. The beta wrapper's Ver0.0 volume-based height cache did not
reliably preserve settled or compacted terrain. Returning particles could appear
higher than the particles that previously occupied the same location and then
settle again.

There were two separate causes:

1. The saved height was calculated as `ground height + sum(reference particle
   volumes) / column area`. It did not measure the actual particle surface. A column
   could settle vertically without changing its saved height if its particles
   remained in the same XY column.
2. Reconstructed particles received default material state. Their elastic
   deformation, plastic state, and stress were lost. Boundary-to-dynamic transitions
   also reset that history.

There are two ways to address these losses. First, measure height from actual
particle positions while keeping a small height cache. Second, preserve the full
particle arrangement and material state. The full-state prototype was implemented
before testing the simpler measured-height change and proved expensive.

The versions below distinguish these designs rather than release order:

| Version | Design | Current status |
| --- | --- | --- |
| Ver0.0 | Volume-based height cache | Replaced by Ver1.0 |
| Ver1.0 | Measured-height cache | Implemented; current default |
| Ver2.0 | Full state cache | Implemented prototype; opt in with `--full-state-cache` |

All implemented changes are local to the beta examples; the core implicit MPM
solver is unchanged.

## Ver0.0: Volume-based height cache

Ver0.0 stores one estimated sand height per background XY column. When terrain
returns, the wrapper generates particle positions by stretching the original
vertical sampling to the saved sand thickness. It resets velocity, APIC gradient,
elastic deformation, plastic state, stress, and the rendering transform to their
initial values.

The saved height comes from particle reference volume divided by column area,
not from the actual surface. This formula was chosen as a simple bulk-volume
estimate: reference volume has units of cubic meters, and dividing by column area
gives a thickness. It reconstructs an undisturbed uniformly sampled bed without
using the highest particle, which could be airborne. However, it assumes that
reference volume represents the column's occupied volume. That assumption fails
for the vertical compression this diagnostic needs to preserve. Therefore, vertical settling without lateral particle
transport can be lost. The restored particles also need to establish their material
state again under gravity and contact loads.

This version uses relatively little background storage and no full-state snapshots
or extra simulation slots. It has been replaced by Ver1.0; `--height-only-cache` now selects Ver1.0.

## Ver1.0: Measured-height cache (implemented, default)

Ver1.0 measures particle heights above the local ground in each background XY
column. It reconstructs a uniformly filled column when particles return. It does
not store individual particle positions or material history.

A mean particle height is not a surface height. In a uniformly filled column,
the mean is approximately half the sand thickness. We correct for the original
sampling positions, including jitter, using each particle's existing `fraction`
(initial height above ground divided by initial sand depth):

```text
saved sand thickness = sum(max(particle_z - local_ground_z, 0)) / sum(fraction)
saved height = ground height + saved sand thickness
restored particle_z = local_ground_z + fraction * saved sand thickness
```

For uniform sampling, this is approximately twice the mean height above ground.
The sampling correction prevents an untouched jittered bed from changing height
and preserves uniform vertical compression. Empty columns save zero thickness.
This is a mean-based column reconstruction, not a measurement of the highest
particle or a fitted surface. It replaces the earlier proposed upper-particle
estimator with a cheaper sum per column.

Height and amount of sand are separate. A second float per background column,
`background_reference_depth`, stores summed reference particle volume divided by
column area. Recycling uses this value for particle radius and mass, and uses
measured height for particle Z. Compressing a column therefore does not by itself
remove mass. Exact mass conservation under arbitrary lateral transport and
resampling is not guaranteed.

Both arrays stay on the GPU. Temporary sums cover only the moving terrain and
boundary terrain. Heights are saved just before the patch center moves; there is
no height reduction while the patch center stays unchanged. There are no full-state
snapshots, per-tile particle allocations, or extra particle slots in this mode.

On return, velocity, velocity gradient, elastic strain, plastic state, stress,
and rendering deformation reset to their initial values. Further settling is
still possible. A single reconstructed column cannot preserve overhangs, separate
layers, or arbitrary particle arrangements. Airborne particles and nonuniform
vertical distributions can bias the estimated height. Ver1.0 fixes the lost
uniform-compression case; it does not guarantee identical terrain after further
physics steps.

## Ver2.0: Full state cache (implemented prototype)

Ver2.0 replaces height-based reconstruction with saved particle positions and
material history. It also saves a separate boundary snapshot before particles
become fixed, so later restoration does not use state overwritten during boundary
simulation. The goal is to preserve the previously deformed particle arrangement
at restoration, including vertical compression that Ver0.0 loses.

This is an optional comparison mode (`--full-state-cache`). It improves shape memory in the restoration tests,
but adds substantial storage and update overhead. The details below describe the
implemented version; the proposed performance optimizations remain future work.

### Terrain behavior

| Terrain region | Simulation behavior | Saved state |
| --- | --- | --- |
| Simulated terrain | Dynamic MPM particles | Current particle state remains live. |
| Boundary terrain | `ACTIVE` particles with zero simulation mass, velocity, and APIC gradient | A separate snapshot preserves the state from before the particle became fixed. |
| Background terrain | No MPM solve | Visited particles remain frozen in GPU storage until their terrain returns. |

APIC is the particle-to-grid transfer method that uses a particle velocity gradient
in addition to its velocity. Restoring both preserves the transfer state.

The boundary snapshot is necessary because a zero-mass MPM step can overwrite
fields such as stress. Saving those overwritten fields when a particle finally
leaves the patch would already be too late.

### Cached fields

Each saved particle contains the following eleven fields:

| Field | Purpose |
| --- | --- |
| `particle_q` | Actual XYZ position, including settled particle arrangement |
| `particle_qd` | Velocity when the terrain was frozen |
| `mpm:particle_qd_grad` | Velocity gradient for APIC transfer |
| `mpm:particle_elastic_strain` | Elastic deformation state |
| `mpm:particle_Jp` | Plastic history used by the material model |
| `mpm:particle_stress` | Stress tensor |
| `mpm:particle_transform` | Deformation used for rendering |
| `particle_mass` | Physical mass, before applying zero boundary mass |
| `particle_radius` | Radius used to derive reference particle volume |
| `particle_flags` | Particle activity flags |
| `fraction` | Original vertical fraction of sand depth, used by height-change coloring |

Material parameters remain homogeneous and are supplied by the example. They are
not separately cached per background particle. Solver warm starts are not cached.

### Particle transitions

1. **Dynamic to boundary:** copy the full particle state into the boundary snapshot,
   then impose zero mass, velocity, and APIC gradient on its simulation copy.
2. **Remaining in the boundary:** preserve the snapshot. The current implementation
   copies it back to the simulation arrays each update, then reapplies the fixed
   boundary conditions.
3. **Boundary to dynamic:** restore the saved physical state, including mass,
   velocity, material history, and the rendering transform. Do not reset it to
   default values.
4. **Leaving the entire particle patch:** store the particle snapshot in a tile at
   its world XY location and release its simulation slot. A slot is an existing
   entry in the model's fixed-size particle arrays.
5. **Revisiting a tile:** restore its saved particles into available simulation
   slots and remove those records from background storage. Positions are not
   resampled and jitter is not reapplied.
6. **Entering never-visited terrain:** generate new particles with the configured
   terrain, material, and jitter settings. Add any transported particles that were
   already cached there.

Tiles contain variable particle counts. Previously emptied tiles remain empty;
previously visited terrain is not automatically filled with fresh particles.
Particles are grouped by world location, not by recycled simulation slot number.

### Storage and solver integration

[`impl/particle_state_cache.py`](../impl/particle_state_cache.py) stores particle attributes
in Warp arrays on the GPU. Python tracks tile keys, visited tiles, and available
simulation slots. Each tile's horizontal side length is
`background_height_spacing`; in full-state mode this controls storage grouping,
not a reduction of the saved shape to one height.

Only outgoing indices, tile keys, and small status/count values are read back to
the CPU by the cache. The current implementation groups outgoing records with
NumPy, allocates GPU arrays per tile batch, and uploads restoration slot indices.
Thus, GPU-resident particle data does not imply a CPU-free update path.

The wrapper refreshes the MPM solver's mass, inverse mass, radius, and flag arrays
when particles change roles. It calls `notify_model_changed(MODEL_PROPERTIES)` and
`reset(flags=0)` through the existing public API. The latter clears numerical warm
starts without resetting physical particle state. Material refresh can also
transfer metadata to the CPU.

The simulation allocation contains the initial particle count plus **50% inactive
spare slots** to accommodate nonuniform returning particle counts. These slots add
no material until used. If an accumulation exceeds the available slots, restoration
raises an error instead of discarding cached material. GPU cache storage grows with
visited terrain and currently has no eviction policy.

`background_heights` is used by the Ver1.0 measured-height path. It is not the
source of restored geometry in full-state mode.

### Cost and current limitations

This implementation established restoration correctness, but was not performance
validated before becoming the default. It can be substantially more expensive than
the height-only implementation.

- Every simulation slot has a full boundary-snapshot allocation, including slots
  that are currently dynamic or inactive.
- Particle arrays are 50% larger because of the reserve. Inactive slots do not add
  active material, but still enlarge storage and particle-wide operations.
- Boundary particles copy all eleven fields every update, even when the tracked
  object is stationary.
- Moving the patch creates many small per-tile allocations and kernel launches.
- CPU grouping, slot management, and GPU-to-CPU synchronization add overhead.
- Solver property refresh operates on the enlarged particle arrays.

The eleven saved fields require **188 bytes per particle**, excluding allocation
and Python bookkeeping overhead. At the default initial count of 972,000 particles,
full-state mode allocates 1,458,000 simulation slots. Their separate snapshot alone
occupies approximately 274 MB; this excludes enlarged model, solver, rendering, and
background-cache storage.

The background is frozen, not simulated while absent. Restored velocities can
resume motion. Changed robot loading, neighboring support, fixed-boundary effects,
and floating-point differences can still cause deformation after restoration.
Exact saved-state restoration does not guarantee static equilibrium afterward.
The tracking deadband can also select a slightly different patch center on return;
shape comparisons must compare the same world region.

Potential optimizations are **not implemented**: copy boundary state only when
needed for transitions, batch records into reusable GPU buffers, reduce per-tile
allocation and CPU grouping, and revisit the spare-slot allocation strategy.
Reducing the saved fields would be a separate accuracy tradeoff requiring tests.

## Selecting and comparing modes

Ver1.0 is the default for both `moving_terrain_mpm_box` (the shovel) and
`moving_terrain_mpm_rover`. `--height-only-cache` explicitly selects that same
mode. Ver2.0 requires `--full-state-cache`. In code, set
`MovingTerrainMPMConfig.preserve_particle_state=True` only for Ver2.0.

```bash
# Ver1.0: measured-height reconstruction, no full-state cache
uv run -m newton.examples moving_terrain_mpm_box --color-by-height

# Ver2.0: full particle-state restoration, more expensive
uv run -m newton.examples moving_terrain_mpm_box --full-state-cache --color-by-height
```

Keep rendering settings identical when comparing. `--particle-lod-pixels 0`
renders every visible particle for close shape inspection and may lower FPS.

The earlier `--height-only-cache` wiring bug was fixed before Ver1.0: the flag
had been accepted without changing the wrapper configuration. Regression tests
now check the default and both explicit mode flags through the actual example
constructor.

## Validation

[`test_mpm_beta_particle_state_cache.py`](../../../tests/test_mpm_beta_particle_state_cache.py)
checks compressed-position and material-state restoration, boundary snapshots after
simulated solver overwrites, empty tiles, repeated visits without duplication,
capacity errors without discarding saved records, and CLI-to-wrapper mode selection.

A controlled vertical-compression test restores particle heights with zero error in
full-state mode; Ver1.0 also preserves uniform compression within the sampling tolerance. This isolates cache
behavior and is not a physical soil-compaction benchmark.

A coupled GPU leave-and-return check restored all eleven fields exactly at the same
grid-aligned patch center, preserved particle count, synchronized solver properties,
and continued with finite simulation state. Shovel default-resolution and rover
slope smoke tests also completed. No FPS improvement or runtime-overhead bound has
been established for the full-state cache.
