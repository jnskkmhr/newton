# GPU-only beta2: matched timing check

This check compares beta's fixed background height arrays with beta2's GPU tile cache. Beta2 stores saved heights and reference volumes directly on the GPU; terrain updates do not download or upload these arrays. Earlier CPU-cache results are archived in [cpu_cache_initial](cpu_cache_initial/).

## Setup

Both runs use a dynamic box with two-way MuJoCo/implicit-MPM coupling, a force controller targeting +X motion at 0.4 m/s, 972,000 particles, a 3 × 3 × 0.5 m particle bed, 0.05 m voxels, three particles per cell per axis, and a 0.005 s physics step. Material settings match: density 2,500 kg/m³, Young's modulus 40 MPa, Poisson's ratio 0.3, and friction tan(50°). The benchmark explicitly uses four physics steps per 50 Hz frame; the interactive examples default to two.

Both benchmark cases use the same static ground plane to isolate terrain maintenance. These results predate the replacement of beta2's following slab with a static background heightfield; they do not measure the current interactive ground mesh. Rendering is disabled. Each mode runs in a separate process, with 150 warmup steps followed by 300 measured steps. Timing includes Python work and GPU completion. These are one pair of runs, not repeated trials or confidence intervals.

## Results

| Method | Physics step, mean ± standard deviation [ms] | Mean steady-state VRAM [GiB] | Sampled peak VRAM [GiB] | Background arrays reserved [MiB] |
| --- | ---: | ---: | ---: | ---: |
| beta | 39.27 ± 3.19 | 4.09 | 4.60 | 7.34 |
| beta2 | 38.11 ± 1.99 | 4.11 | 5.04 | 14.07 |

Beta2 is about 3% faster in this short run. The difference is small relative to step variation. Both versions still perform MPM on the same number of local particles; GPU height storage does not reduce the main material solve.

VRAM is sampled device-wide at 20 Hz and reported above each process's pre-initialization baseline. Peak includes initialization; other graphics activity can affect these readings, and the baseline differed between runs. These measurements do not establish a peak-memory improvement. The exact background-array reservation is a more direct comparison: beta allocates a 40 × 40 m background, while beta2 reserves 1,024 tiles, each 2 × 2 m. Only four beta2 tiles were used in this short route. Beta2's default cache reserves more memory than beta's 40 m background in this case; its benefit is saving visited tiles at arbitrary positions without allocating the rectangle between them. Capacity remains finite and exhaustion raises an error without eviction or CPU fallback.

## Particle-property refresh

A separate microbenchmark measures 30 completed notifications after five warmups, outside the physics timing. On the beta2 model, full property notification took 6.066 ± 0.911 ms, while targeted mass/radius/flag refresh took 0.122 ± 0.002 ms. This isolated difference does not translate directly into whole-step speedup: the MPM solve remains the dominant work, and notifications occur only when particle properties change.

## Reproduce

```bash
uv run --no-sync python newton/examples/mpm_moving_patch_beta2/benchmark/compare.py --mode beta2 --output newton/examples/mpm_moving_patch_beta2/benchmark/beta2.json
uv run --no-sync python newton/examples/mpm_moving_patch_beta2/benchmark/compare.py --mode beta --output newton/examples/mpm_moving_patch_beta2/benchmark/beta.json
```

Run sequentially on an otherwise idle GPU. Raw per-step timings and memory samples are in the JSON files. [environment.json](environment.json) records the GPU, dependency versions, and source hashes.

The separate test suite passed 22 tests, including GPU tile collisions/concurrent writes, negative coordinates, saved empty terrain, revisiting deformation, capacity exhaustion, and a guard against terrain-array transfers. The coupled integration test travels to 1,024 m and back. This short timing check does not establish long-duration force continuity or terrain reconstruction accuracy.
