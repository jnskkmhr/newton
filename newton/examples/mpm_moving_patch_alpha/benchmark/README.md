Current alpha box, rover, and G1 physics/VRAM measurements are in [physics_vram/README.md](physics_vram/README.md). The results below are retained from the earlier benchmark.

# MPM benchmark

**Latest rerun:** [dynamic sparse allocation with mean +/- standard-deviation
line plots](../dynamic_sparse/README.md). Rendering overhead is excluded from that
rerun. The results below preserve the earlier bounded-sparse benchmark.

This directory contains the benchmark runner, raw measurements, summaries, and
four comparison figures. The runs use an RTX 5090; exact hardware and driver
information is recorded in `environment.json`.

- [Fixed versus sparse](01_grid_comparison.png)
- [Particle rendering overhead](02_rendering_overhead.png)
- [Terrain-size scaling](03_terrain_scaling.png)
- [Moving-patch overhead](04_moving_patch_overhead.png)
- [Numerical results and validation](RESULTS.md)
- [Machine-readable summary](summary.csv)

## Reproduce

From the repository root:

```bash
uv run --extra examples newton/examples/mpm_moving_patch_alpha/benchmark/run_benchmarks.py
uv run --extra examples newton/examples/mpm_moving_patch_alpha/benchmark/plot_results.py
```

The runner resumes existing results. Move the `raw` directory aside before a
fresh complete rerun. Use `--only sparse_3m` to select a case, or `--repeats`,
`--warmup`, and `--steps` to change the measurement budget. Do not mix results
from different settings in the same `raw` directory. The saved figure labels
assume the default 200 warmup and 200 measured steps.

## Cases

| Criterion | Implementation | Controlled change |
|---|---|---|
| Fixed vs sparse | Original coupled example | Grid type, with identical 3 m bed and particle count |
| Rendering overhead | Adaptive-rendering example | Visible side ratio 1, 2/3, or 0 on a 3 m bed |
| Terrain scaling | Original fixed/sparse examples vs moving-patch example | Full-bed side 3, 6, 10 m; patch stays 3 m |
| Moving-patch overhead | Original sparse vs moving-patch example | Identical 2 m allocation, stationary and moving box |

Each run executes in a fresh subprocess, sequentially on one GPU. Successful
cases have three repetitions. A failed case is recorded and not retried with
lower resolution or changed physics. Workers have a 600-second timeout.

The moving-patch scaling line reuses its measured constant-size workload across
conceptual operating-area sizes. There is no large background terrain map in this
prototype. This is a computation/allocation scaling comparison, not a measured
traversal of the entire operating area or a terrain-persistence test.

## Normalization

The runner subclasses the requested example classes and temporarily wraps solver
constructors **inside each worker process**. Interactive example files are not
changed. All cases use:

- A single box with half-extents `(0.25, 0.35, 0.25)` m, initial center at
  `(0, 0, 2)` m, and the same builder mass/shape settings. Finalized mass is saved.
- Sand depth 0.5 m, density 2500 kg/m3, friction 0.75, and voxel size 0.05 m.
- Three particle samples per voxel axis, identical horizontal cell-centered
  sampling and both vertical endpoints. Jitter equals particle diameter.
- Coupled timestep 0.01 s, four MuJoCo substeps, one proxy iteration, proxy mass
  scale 1, and the same MuJoCo settings (`use_mujoco_contacts=False`, `njmax=100`).
- MPM P0 strain basis, 50 maximum iterations, critical fraction 0, no warmstart,
  and forward collider velocity mode.
- The shared `ObjectController`: 1 s settling, 1 s acceleration, then a 0.2 m/s
  +X reference with dynamic horizontal PD forces and orientation torques. Vertical
  motion is free. Stationary cases set controller speed to zero before capture.
- CUDA-graph replay enabled in every case, including recycling for moving patch.

Capacity is the next power of two above 1.5 times the nominal bed voxel count:
`ceil_pow2(1.5 * (width / 0.05)**2 * (0.5 / 0.05))`.
This gives the same capacity for equal-sized workloads; capacity is not a separate
sweep. Sparse grid padding is zero. Fixed grids use padding 50 (2.5 m per side),
matching the chosen fixed-grid configuration and allowing particles to spread.
That extra fixed spatial allowance is part of the fixed-grid cost, not an
identical-topology comparison.

## Timing

Physics: warm up for 200 steps (2 s), then measure 200 steps (another 2 s).
Reported wall times include controller, contacts, coupling, and recycling where
applicable, ending only after GPU completion. CUDA-event intervals are retained
separately in the raw JSON and CSV. Per-step synchronization prevents CPU/GPU
queuing from disguising the actual duration; it does not measure fully pipelined
interactive throughput. Initialization and graph capture are excluded from step
timing and recorded separately.

Rendering: simulate the same 200-step warmup, freeze that snapshot, warm up 20
draws, and time 200 render calls at 960 x 720 with VSync off. Each timed draw ends
with OpenGL and CUDA completion. This includes compaction, visible-count readback,
viewer updates, and drawing rigid bodies/ground. Ratio zero skips particle
compaction and drawing. Ratio one still clips to the full original XY footprint;
particles outside the bed bounds are not shown. This measures a static snapshot,
not animated rendering throughput.

The mean and p95 pool all measured calls from the successful repetitions; p95 is
the 95th percentile, not a maximum or confidence interval. Per-run means and their
range are saved. No GPU clocks are locked; run-to-run variation is retained.

## VRAM and workload diagnostics

NVML total-device VRAM is sampled every 50 ms without CUDA synchronization.
The reported values subtract each worker's pre-initialization device baseline:

- **Steady-state:** average sampled VRAM during the measured phase and subsequent
  untimed diagnostics.
- **Peak:** maximum sampled VRAM across initialization, warmup, measurement, and
  validation.

These are baseline-subtracted **device-wide** observations, not exact per-process
allocations. Desktop/other-process activity can affect them, and sub-50 ms peaks
may be missed. Raw samples and baselines are saved. Warp allocator current/high
watermark byte counts are also saved as complementary diagnostics; they do not
include all CUDA, MuJoCo, or OpenGL allocations.

Particle count and reserved cell capacity are recorded per run. Active-cell counts
are read from the actual FEM partition map at measurement start, midpoint, and
end; padded capacity entries are not counted. These are workload observations,
not extra swept parameters.

## Interpretation limits

This benchmark measures performance of the current implementation, not validated
physical equivalence. Actual dynamic trajectories can differ despite common
reference motion and material settings. Final poses, window-shift counts, and
example validation failures are reported in `RESULTS.md`; failures are not silently
removed from timing results. Example-specific final body-height checks differ, so
read the final positions as well as pass/fail labels.

Moving patch discards distant terrain history and simulates fewer particles in
the scaling comparison. A speedup does not establish equivalent force continuity,
track persistence, or terrain deformation. The equal-allocation overhead comparison measures net step cost, not isolated
recycling-kernel latency: recycling can also change active cells and subsequent
solver work. Physics runs are headless, so differences
between the interactive examples' rendering implementations do not contaminate
the grid, scaling, or moving-patch-overhead comparisons.
