# MPM benchmark results

Generated from the raw JSON files by `plot_results.py`.

## Main observations

- At 3 m, fixed averaged 17.86 ms and sparse 18.43 ms per completed step; neither has a large advantage here.
- Rendering the full footprint averaged 54.43 ms, partial visibility 20.04 ms, and no sand drawing 0.80 ms. Full/partial render-call ratio: 2.72x.
- Compared with 6 m fixed, the 3 m patch was 3.14x faster per step and used 2.89x less steady VRAM. This compares different simulated terrain sizes.
- Compared with 6 m sparse, the 3 m patch was 3.22x faster per step and used 2.97x less steady VRAM. This compares different simulated terrain sizes.
- At equal 2 m allocation (stationary), moving patch added 1.03 ms, or 8.4% net step time.
- At equal 2 m allocation (moving), moving patch added 1.99 ms, or 16.1% net step time.
- Both 10 m full-bed attempts exhausted GPU memory during graph creation with 1,048,576 reserved cells. This is a limit of the tested capacity/configuration, not proof that every 10 m configuration is infeasible.
- Validation and final-pose differences below limit claims of physical equivalence; these are performance observations for this prototype.

| Case | Runs | Mean ms | p95 ms | Steady GiB | Peak GiB | Particles | Active cells (sampled range) | Capacity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_10m | 0 | Failed | - | - | - | - | - | - |
| fixed_3m | 3 | 17.86 | 20.75 | 3.46 | 3.54 | 1,004,400 | 34,015-37,055 | 65,536 |
| fixed_6m | 3 | 62.42 | 68.43 | 10.09 | 10.18 | 4,017,600 | 126,104-139,105 | 262,144 |
| overhead_original_moving | 3 | 12.36 | 15.00 | 2.11 | 2.15 | 446,400 | 16,585-17,516 | 32,768 |
| overhead_original_stationary | 3 | 12.34 | 15.29 | 2.09 | 2.15 | 446,400 | 16,760-17,515 | 32,768 |
| overhead_patch_moving | 3 | 14.35 | 18.17 | 2.12 | 2.18 | 446,400 | 12,745-13,975 | 32,768 |
| overhead_patch_stationary | 3 | 13.38 | 16.09 | 2.13 | 2.19 | 446,400 | 12,719-13,971 | 32,768 |
| patch_3m | 3 | 19.88 | 23.48 | 3.49 | 3.54 | 1,004,400 | 28,918-31,919 | 65,536 |
| render_0.000 | 3 | 0.80 | 1.33 | 3.66 | 3.78 | 1,004,400 | 37,142-37,143 | 65,536 |
| render_0.667 | 3 | 20.04 | 23.89 | 3.69 | 3.83 | 1,004,400 | 37,142-37,143 | 65,536 |
| render_1.000 | 3 | 54.43 | 61.50 | 3.82 | 3.93 | 1,004,400 | 37,142-37,143 | 65,536 |
| sparse_10m | 0 | Failed | - | - | - | - | - | - |
| sparse_3m | 3 | 18.43 | 21.22 | 3.48 | 3.58 | 1,004,400 | 34,723-37,143 | 65,536 |
| sparse_6m | 3 | 64.04 | 72.03 | 10.35 | 10.48 | 4,017,600 | 127,607-139,421 | 262,144 |

## Validation and tracking

The force-controlled reference is shared; actual motion is not prescribed.

- **fixed_10m failed:** Graph creation error: Warp CUDA error 2: out of memory (in function wp_cuda_graph_create_exec, /builds/omniverse/warp/warp/native/warp.cu:3660)
- fixed_3m: final mean x=0.290 m, z=0.345 m; window shifts=[None, None, None]; validation=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- fixed_6m: final mean x=0.288 m, z=0.346 m; window shifts=[None, None, None]; validation=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- overhead_original_moving: final mean x=0.331 m, z=0.323 m; window shifts=[None, None, None]; validation=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- overhead_original_stationary: final mean x=-0.000 m, z=0.362 m; window shifts=[None, None, None]; validation=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- overhead_patch_moving: final mean x=0.332 m, z=0.324 m; window shifts=[1, 1, 1]; validation=['passed', 'passed', 'passed'].
- overhead_patch_stationary: final mean x=-0.000 m, z=0.362 m; window shifts=[0, 0, 0]; validation=['passed', 'passed', 'passed'].
- patch_3m: final mean x=0.314 m, z=0.332 m; window shifts=[1, 1, 1]; validation=['passed', 'passed', 'passed'].
- render_0.000: final mean x=0.033 m, z=0.450 m; window shifts=[None, None, None]; validation=['passed', 'passed', 'passed'].
- render_0.667: final mean x=0.033 m, z=0.450 m; window shifts=[None, None, None]; validation=['passed', 'passed', 'passed'].
- render_1.000: final mean x=0.033 m, z=0.450 m; window shifts=[None, None, None]; validation=['passed', 'passed', 'passed'].
- **sparse_10m failed:** Graph creation error: Warp CUDA error 2: out of memory (in function wp_cuda_graph_create_exec, /builds/omniverse/warp/warp/native/warp.cu:3660)
- sparse_3m: final mean x=0.314 m, z=0.332 m; window shifts=[None, None, None]; validation=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
- sparse_6m: final mean x=0.312 m, z=0.333 m; window shifts=[None, None, None]; validation=['Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]', 'Test "all bodies are above the sand" failed for the following bodies: [body_0]'].
