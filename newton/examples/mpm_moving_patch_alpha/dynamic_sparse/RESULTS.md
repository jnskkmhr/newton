# Dynamic sparse benchmark rerun

No rendering benchmark is included. All cases are headless.

Sparse uses per-step allocation (`max_active_cell_count=-1`) without an outer CUDA graph. Fixed and moving-patch use bounded capacity and graph replay.

Timing SD pools measured step durations across successful runs. VRAM SD is across per-run means/peaks. These are standard deviations, not confidence intervals.

## Main observations

- 3 m bed: fixed 16.81 +/- 1.91 ms; dynamic sparse 22.66 +/- 1.68 ms. Steady VRAM: 3.42 vs 1.97 GiB.
- 6 m bed: fixed 66.93 +/- 3.62 ms; dynamic sparse 37.36 +/- 2.19 ms. Steady VRAM: 10.13 vs 5.99 GiB.
- 10 m bed: dynamic sparse completed at 81.34 +/- 5.25 ms with 15.58 GiB steady VRAM; fixed failed under the tested configuration.
- Relative method performance changes with terrain size and capacity. Do not generalize the smallest-bed timing to larger beds.
- Equal-size results combine allocation, graph execution, recycling, and their effects on subsequent solver work; they do not isolate recycling overhead.

| Case | Runs | Step mean +/- SD (ms) | Steady mean +/- SD (GiB) | Peak mean +/- SD (GiB) | Particles | Active cells | Capacity |
|---|---:|---:|---:|---:|---:|---:|---:|
| equal_fixed_moving | 3 | 11.55 +/- 1.44 | 1.99 +/- 0.05 | 2.04 +/- 0.06 | 446,400 | 16,202-17,453 | 32,768 |
| equal_fixed_stationary | 3 | 11.19 +/- 1.38 | 2.05 +/- 0.13 | 2.10 +/- 0.13 | 446,400 | 16,382-17,463 | 32,768 |
| equal_moving-patch_moving | 3 | 14.14 +/- 1.94 | 2.18 +/- 0.02 | 2.23 +/- 0.03 | 446,400 | 12,747-13,975 | 32,768 |
| equal_moving-patch_stationary | 3 | 12.26 +/- 1.38 | 2.12 +/- 0.03 | 2.18 +/- 0.01 | 446,400 | 12,719-13,972 | 32,768 |
| equal_sparse_moving | 3 | 19.45 +/- 1.33 | 1.24 +/- 0.02 | 1.34 +/- 0.04 | 446,400 | 16,920-17,674 | Dynamic (-1) |
| equal_sparse_stationary | 3 | 19.18 +/- 1.37 | 1.26 +/- 0.02 | 1.34 +/- 0.01 | 446,400 | 17,084-17,669 | Dynamic (-1) |
| fixed_10m | 0 | Failed | - | - | - | - | - |
| fixed_3m | 3 | 16.81 +/- 1.91 | 3.42 +/- 0.07 | 3.47 +/- 0.11 | 1,004,400 | 34,014-37,055 | 65,536 |
| fixed_6m | 3 | 66.93 +/- 3.62 | 10.13 +/- 0.32 | 10.29 +/- 0.37 | 4,017,600 | 126,104-139,105 | 262,144 |
| patch_3m | 3 | 18.52 +/- 2.37 | 3.48 +/- 0.01 | 3.53 +/- 0.02 | 1,004,400 | 28,912-31,917 | 65,536 |
| sparse_10m | 3 | 81.34 +/- 5.25 | 15.58 +/- 0.19 | 16.47 +/- 0.23 | 11,160,000 | 375,348-409,220 | Dynamic (-1) |
| sparse_3m | 3 | 22.66 +/- 1.68 | 1.97 +/- 0.00 | 2.10 +/- 0.01 | 1,004,400 | 36,561-38,589 | Dynamic (-1) |
| sparse_6m | 3 | 37.36 +/- 2.19 | 5.99 +/- 0.02 | 6.37 +/- 0.02 | 4,017,600 | 138,535-149,423 | Dynamic (-1) |

## Interpretation

Stationary means the controller has zero horizontal reference velocity. Gravity, settling, deformation, and corrective forces remain active. Moving cases follow the same straight +X reference at 0.2 m/s after settling and acceleration.

The equal-size comparison measures net method performance, not pure recycling overhead. It now also changes sparse-grid allocation and graph execution. The earlier bounded-sparse experiment is preserved in the sibling `benchmark/` directory.

## Validation and failures

- equal_fixed_moving: final mean x=0.309 m, z=0.339 m; window shifts=[None, None, None]; checks=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- equal_fixed_stationary: final mean x=-0.000 m, z=0.375 m; window shifts=[None, None, None]; checks=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- equal_moving-patch_moving: final mean x=0.332 m, z=0.324 m; window shifts=[1, 1, 1]; checks=['passed', 'passed', 'passed'].
- equal_moving-patch_stationary: final mean x=-0.000 m, z=0.362 m; window shifts=[0, 0, 0]; checks=['passed', 'passed', 'passed'].
- equal_sparse_moving: final mean x=0.329 m, z=0.359 m; window shifts=[None, None, None]; checks=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- equal_sparse_stationary: final mean x=-0.000 m, z=0.401 m; window shifts=[None, None, None]; checks=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- fixed_10m: Graph creation error: Warp CUDA error 2: out of memory (in function wp_cuda_graph_create_exec, /builds/omniverse/warp/warp/native/warp.cu:3660)
- fixed_3m: final mean x=0.290 m, z=0.345 m; window shifts=[None, None, None]; checks=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- fixed_6m: final mean x=0.288 m, z=0.346 m; window shifts=[None, None, None]; checks=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- patch_3m: final mean x=0.314 m, z=0.332 m; window shifts=[1, 1, 1]; checks=['passed', 'passed', 'passed'].
- sparse_10m: final mean x=0.308 m, z=0.369 m; window shifts=[None, None, None]; checks=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- sparse_3m: final mean x=0.310 m, z=0.368 m; window shifts=[None, None, None]; checks=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- sparse_6m: final mean x=0.308 m, z=0.369 m; window shifts=[None, None, None]; checks=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
