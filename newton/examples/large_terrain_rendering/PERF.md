# Prototype rendering measurements

Measured on an NVIDIA GeForce RTX 5090, driver 595.84, with Warp 1.17.0 and
OVRTX 0.3.0.312915. These are short prototype runs, not a sustained
90-second benchmark or GPU timestamp measurements.

The static scene uses a 1000 m terrain radius, 131,073 terrain
vertices, 261,632 terrain triangles, 4500 rock instances, and
the native Unitree G1 USD asset. Six shared rock prototypes supply the instanced
geometry. Terrain triangle counts exclude rocks and G1.

## Fixed camera, 2560 by 1440

180 rendered frames, with the first 32 excluded from frame-time statistics.
Synchronous RTX rendering; wall time measured around begin-frame, state logging,
and end-frame. Screenshot writing is outside this interval.

| Measurement | Result |
| --- | ---: |
| Scene preparation, excluding RTX initialization | 2.16 s |
| Measured frames | 148 |
| Median render wall time | 10.98 ms |
| p95 | 11.81 ms |
| p99 | 13.79 ms |
| Maximum | 16.27 ms |
| 1% low, from slowest 1% of measured render times | 65.8 FPS |

ViewerRTX requests frame generation by default. Generated frames were not
counted separately, so these figures must not be presented as native GPU FPS.
The headless run also excludes interactive window presentation and input cost.
Initial RTX scene loading and shader compilation are excluded from this table;
new runs additionally record `first_render_seconds` in their JSON report.

The [captured frame](prototype.jpg) is a reduced copy of the inspected 1440p
render. The full-resolution PNG and raw timing JSON are in `_generated/` locally.

## Moving camera, 2560 by 1440

300 rendered frames along the scripted orbit, with 32 warm-up frames excluded.
The terrain and all scene objects remained static.

| Measurement | Result |
| --- | ---: |
| Measured frames | 268 |
| First render, including RTX initialization | 1.96 s |
| Median render wall time | 10.87 ms |
| p95 | 12.26 ms |
| p99 | 14.42 ms |
| Maximum | 29.31 ms |
| 1% low, from measured render times | 49.3 FPS |

The maximum frame time and lower 1% result show that a stable median alone does
not establish a 60 FPS floor. Fine distant detail also needs further temporal
filtering; moving-camera captures can show thin reconstruction artifacts.

## Remaining work

Measure longer camera routes, separate GPU work from CPU synchronization, record
GPU memory, and test the interactive window before making sustained-performance
claims. Terrain is fixed in world space; this prototype does not measure terrain
streaming or dynamic mesh updates.

## G1 policy locomotion with local rigid contacts

The trained `config/policy_3d_rft.pt` policy was run with `config/g1.yaml` and its
improved-foot MJCF. No policy weights are bundled in this directory. The verified
run used CUDA graph capture, 600 Hz rigid physics, 50 Hz inference, and the default
0.6 m/s forward command after a 0.5 s settling period. Rocks remained visual-only.

600 headless frames at 1280 by 720 on the same RTX 5090 produced:

| Measurement | Result |
| --- | ---: |
| Simulated duration | 10.0 s |
| Actual policy evaluations | 500 |
| Rigid physics steps | 6000 |
| XY travel | 5.03 m |
| Final root height over terrain | 0.755 m |
| Final robot up direction dotted with world up | 0.9988 |
| Collision mesh vertices / triangles | 6561 / 12800 |
| Collision-patch recenters | 2 |
| Maximum observed contact count | 65 |
| Median step wall time with RTX active | 26.34 ms |
| Median render wall time | 11.83 ms |
| Median combined step/render wall time | 38.38 ms |
| p95 combined step/render wall time | 55.15 ms |
| Maximum combined step/render wall time | 66.68 ms |

[Walking frame](walking.jpg). Raw output is in
`_generated/walking_graph_rtx/timing.json` locally. This is not a sustained
real-time result: the simulation advances 1/60 s per frame regardless of wall
time. The measured combined frame time exceeds that budget. The renderer's frame
generation setting does not change the amount of simulated time advanced.

A separate null-viewer run with graph capture measured about 12.4 ms median step
time; the earlier uncaptured RTX run measured about 49.0 ms. These use different
rendering workloads and should not be interpreted as an isolated GPU speedup.
The graph-enabled RTX result above is the relevant combined measurement.

Validation passed for observation ordering/scales, history initialization,
previous-action observations, collision contacts after a 12 m patch move,
unchanged shared height samples, and render mesh connectivity. The ten-second
policy run additionally required finite state, actual contacts, an upright
robot, and measurable travel across patch updates.
# Alpha MPM + clipmap validation

The separate `desert_g1_mpm` demo was validated with 300 RTX frames (six simulated
seconds) on the RTX 5090 at 1280 × 720, using the default 128,000 particles,
4,500 rocks, 5 cm MPM voxels, and an 8 cm sand layer. The first 32 frames were
excluded from steady-state timing. CUDA graph capture was disabled. The render
clipmap had 7,137 vertices / 14,144 triangles across a 2,048 m square; the local
bedrock collider had 1,953 vertices / 3,776 triangles across a 16 m square.

| Measurement | Result |
| --- | --- |
| G1 travel | 4.965 m |
| Final upright dot product | 0.997 |
| Policy evaluations / MPM steps | 300 / 600 |
| Particle patch shifts / collider updates including initialization | 12 / 13 |
| Render clipmap updates including initialization | 16 |
| Median stepping wall time | 78.01 ms |
| Median rendering wall time | 48.19 ms |
| Median combined frame wall time | 125.95 ms |
| Combined p95 / maximum frame wall time | 136.76 / 142.85 ms |

These are CPU wall times around synchronous GPU work, not GPU timestamp results.
The 20 ms simulation frame took about 126 ms wall time: this initial integration
does not run in real time. Initialization and shader compilation are excluded.
The final five-second null-viewer validation also passed, with 3.80 m travel and
nine patch shifts. Its median stepping time was 65.50 ms and combined frame time
was 66.35 ms, including the checks for bedrock penetration and measurable travel.

An earlier 25 cm sand layer with 10 cm voxels failed the upright check, including
with alpha's existing material settings. The validated shallow layer is not
evidence of stable G1 locomotion in deeper sand. Alpha discards departed terrain
state, and the renderer currently shows particles rather than a reconstructed
MPM surface. See [MPM_DEMO.md](MPM_DEMO.md) for the implemented architecture.
