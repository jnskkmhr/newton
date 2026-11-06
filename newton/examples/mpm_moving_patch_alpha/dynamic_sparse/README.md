# Benchmark rerun: dynamic sparse allocation

This rerun uses `example_mujoco_mpm_coupled_solver_sparse.py` for the sparse
method. It excludes rendering overhead and uses line plots with mean points and
+/- one standard deviation. Previous bounded-sparse results remain in the parent
directory.

- [Fixed versus dynamic sparse](01_grid_comparison.png)
- [Terrain scaling, including moving patch](02_terrain_scaling.png)
- [Equal-size stationary/moving comparison](03_equal_size_comparison.png)
- [Results and validation](RESULTS.md)
- [CSV summary](summary.csv)

## Methods and fairness

| Method | Grid | Active-cell capacity | Outer CUDA graph |
|---|---|---|---|
| Fixed | Fixed grid, padding 50 | Positive, size-based | Yes |
| Sparse | Sparse, allocated each step, padding 0 | -1 | No |
| Moving-patch | Rebuildable sparse, padding 0 | Positive, size-based | Yes |

The original example class is used for fixed-grid runs, the dedicated sparse
example for dynamic sparse, and the moving-patch example for moving patch.
The MPM solver may still use internal rheology graphs; "no outer graph" refers
to capture of the entire coupled step.

Process-local subclasses normalize particle generation and solver settings;
interactive example files are not edited.

All use the same 0.05 m voxel size, three samples per voxel axis, 0.5 m bed depth,
2500 kg/m3 density, 0.75 friction, jitter, initial box geometry/mass, dynamic
force controller, 0.01 s timestep, four MuJoCo substeps, and one proxy iteration.
MPM uses P0 strain, 50 maximum iterations, critical fraction 0, no warmstart,
and forward collider velocities. MuJoCo uses the same contact settings and
`njmax=100`. These match the normalization in the previous benchmark.

Positive capacity uses the next power of two above 1.5 times nominal bed voxel
count. Dynamic sparse intentionally has no positive reserved capacity. Therefore
this compares the requested execution methods; it does not isolate allocation
cost from CUDA-graph effects. The equal-size comparison is not a pure measurement
of recycling-kernel overhead. Recycling can also change active cells and physics.

## What stationary means

**Stationary means zero horizontal reference speed**, not paused simulation or a
kinematic/frozen box. The controller holds initial XY position and orientation;
gravity, vertical settling, sand deformation, and corrective forces remain active.

**Moving means the shared straight +X reference:** 1 s before translation, 1 s
smooth acceleration, then 0.2 m/s. The actual box stays dynamic and can lag the
reference. Testing both shows whether relative method cost changes when the box
moves and the moving patch shifts. It does not split boundary-checking and
recycling into separately timed kernels.

## Cases and timing

- Fixed/dynamic-sparse full beds: 3, 6, and 10 m square.
- Moving-patch scaling reference: one constant 3 m square patch. Its line across
  operating-area sizes repeats this same measurement, not separate traversals.
- Equal allocation: 2 m square beds for all three methods, stationary and moving.
- Three fresh-process repetitions per successful case, sequential GPU execution.
- 200 untimed warmup steps, then 200 timed steps per run, with GPU completion.
- No viewer rendering, compilation, or graph creation inside timed step calls.
- Failed/OOM/timeout cases are recorded, not silently retuned or omitted.

Timing error bars are the sample standard deviation of all measured step times
pooled across repetitions. They describe step-time variation, including variation
between runs. **They are not confidence intervals or the standard error.**

VRAM error bars use the standard deviation across the three per-run steady means
or sampled peaks. Peak figures show the mean sampled peak with SD; the maximum
observed peak is also saved in CSV. NVML device VRAM is sampled every 50 ms,
subtracting each process's pre-initialization baseline. Steady memory is averaged
during measurement and subsequent untimed diagnostics; peak covers the whole run.
These are device-wide baseline-subtracted observations, not exact process-only
allocations. Short peaks can be missed. GPU clocks are not locked.

Particle count, actual FEM-domain/partition cell counts at start/midpoint/end,
and configured capacity are saved. Dynamic sparse uses the whole allocated sparse
domain; bounded methods use an active partition. Final poses, window shifts,
example validation errors, allocator memory statistics, and full samples are
preserved in `raw/`. Physical equivalence is not established by timing results.

## Reproduce

From the repository root:

```bash
uv run --extra examples newton/examples/mpm_moving_patch_alpha/dynamic_sparse/run_benchmarks.py
uv run --extra examples newton/examples/mpm_moving_patch_alpha/dynamic_sparse/plot_results.py
```

The runner resumes existing results. Move `raw/` aside for a new complete run.
Do not mix results with different settings. `environment.json` and
`source_hashes.json` record the environment and source fingerprints for this run.
