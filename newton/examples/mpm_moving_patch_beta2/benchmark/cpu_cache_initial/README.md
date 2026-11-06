> Historical measurements of the removed CPU-backed cache. These do not describe
> the current GPU-only implementation. The old reproduction commands below no
> longer reproduce this implementation; see the parent directory for current results.

# Initial beta2 validation and timings

These are short implementation checks, not a large-scene scaling sweep or proof
of physical equivalence. Each version ran in a fresh process on the same GPU,
with no competing compute process observed. The recorded pair ran beta2 first,
then beta. Step standard deviation describes variation within one run; it is not
a confidence interval or an estimate of between-run variation.

## Matched setup

Both cases use a 75 kg dynamic box, a 2 × 2 m simulated region with a 0.5 m
boundary on each side, 0.5 m sand depth, 0.05 m voxels, and 3 particles per axis
per cell: **972,000 particles**. Density is 2,500 kg/m³, Young's modulus 40 MPa,
Poisson's ratio 0.3, and friction tan(50°). Jitter is 0.05/3 m. Both use the same
sparse-grid capacity, two-way proxy coupling, 200 Hz coupled steps, and four
MuJoCo substeps per coupled step.

A force controller tracks forward motion at 0.4 m/s after a short settling
period. The box responds dynamically and can tilt or roll. There are 150 warm-up
steps followed by 300 measured steps. Timing includes terrain maintenance,
controller forces, rigid collision detection, coupled physics, and completion
synchronization. Rendering is disabled. Both cases use the same static ground
plane to isolate terrain maintenance from beta2's following-ground implementation.

Beta allocates a 40 × 40 m background height map. Beta2 allocates its local GPU
height window. The measured route stays inside that window: **these timings do
not include a background page transfer**. Long-distance paging and reversals are
covered by a separate real coupled test, not by this timing result.

## Results

| Version | Physics step [ms], mean ± std | Steady GPU memory [GiB], mean ± std | Sampled peak GPU memory [GiB] | Background GPU arrays [MiB] |
| --- | ---: | ---: | ---: | ---: |
| beta | 38.37 ± 3.06 | 4.203 ± 0.527 | 4.807 | 7.343 |
| beta2 | 37.35 ± 0.90 | 4.166 ± 0.508 | 4.998 | 0.391 |

GPU memory is sampled through NVML at 20 Hz and subtracts device usage before
CUDA setup. The sampled peak includes initialization and may miss shorter peaks.
It is device-wide measurement, so desktop GPU activity can affect small differences.
Background-array sizes are exact allocation sizes; beta2 includes its dirty-column
mask. GPU memory for particles, rendering buffers, and the MPM solver remains.

The isolated full particle/material notification took
**6.158 ± 0.496 ms**, while the targeted notification took
**0.122 ± 0.003 ms** on the same beta2 solver.
These are 30-call microbenchmarks after five warm-up calls, outside the physics
timing interval. They exclude copying global particle properties into the coupled
view and resetting warm starts, which remain part of wrapper maintenance.

The targeted notification is substantially cheaper. The full physics-step
improvement in this pair is modest, and sampled peak memory is higher for beta2.
This result does not establish an overall peak-memory reduction or a statistically
established speedup. The architectural change is that background GPU allocation
no longer grows with a configured world size. Saved terrain tiles consume CPU
memory as the robot visits more terrain.

## Correctness checks

- CPU transition tests restore a compressed central region after positive,
  negative, and diagonal travel; preserve retained particle state and custom
  initial plastic state; retain empty columns; and save stationary material outflow.
- A real coupled CUDA test travels through several locations including
  (1,024 m, 1,024 m), returns to the origin, checks finite states and supporting
  ground positions, and verifies fixed particle/background allocation sizes.
- The targeted notification matches the original full notification's derived
  arrays and material extrema, including inactive/proxy/collider exclusions.
- The default-resolution flat scene and a sloped scene pass example final-state
  checks. OpenGL headless rendering and frame capture were exercised.

These checks do not replace force/sinkage comparisons against a larger fully
simulated bed or long-duration terrain-memory validation. Separate wheel/foot
regions and reduced boundary-particle work are not implemented.

## Reproduce

```bash
uv run --no-sync python newton/examples/mpm_moving_patch_beta2/benchmark/compare.py \
  --mode beta2 --output /tmp/beta2.json
uv run --no-sync python newton/examples/mpm_moving_patch_beta2/benchmark/compare.py \
  --mode beta --output /tmp/beta.json
```

Use an otherwise idle GPU. Raw measurements are in [beta.json](beta.json) and
[beta2.json](beta2.json). Environment and physics-source fingerprints are recorded
in [environment.json](environment.json).
