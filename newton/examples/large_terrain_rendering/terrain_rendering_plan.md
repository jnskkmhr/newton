# Sand dune rendering with Unitree G1 in Newton

Status: the prototype now supports G1 policy locomotion with rigid-body contact.
See [README.md](README.md) for launch commands and [PERF.md](PERF.md) for
measurements. Rendering uses procedural dunes, a shallow basin, instanced rocks,
generated materials, sky, and sunlight. A separate controller drives a rigid G1
against a bounded collision mesh that follows it. Terrain streaming, atmosphere,
vegetation, and rock collision selection remain future work.

We will build a dedicated sand dune rendering showcase with a Unitree G1 humanoid in
`newton/examples/large_terrain_rendering`, using Python, Warp, and `ViewerRTX`.
The objective is a convincing landscape from close-up sand detail to distant
dune silhouettes, with measured rendering performance during camera movement.

This project owns its terrain generation, rendering data, camera, and visual
settings. It has no dependency on the MPM beta examples, their sampling settings,
or their physics caches. MPM integration is outside the current scope. Terrain
resolution will follow visual quality and rendering cost.

The [original Snowflow brief](snow_mountain_game.md) remains a technical reference.
The MAWI video below defines the current visual direction. The
current scope is dunes, rock formations, sand and rock materials, lighting,
atmosphere, a Unitree G1 figure, and RTX performance. G1 replaces the reference's stylized human.
Rigid G1 locomotion is implemented; spells, surfing, cloth, and deformable sand
interaction are outside the current milestones. The reference's frame rate is a comparison point, not an
established Newton target.

## Visual direction and G1

The primary visual reference is MAWI's
[Unreal Engine 5.3 — Dune Desert Landscape — Stone Pit](https://www.youtube.com/watch?v=tXw5O9HtP84).
We inspected its thumbnail and frames at 00:15, 01:05, and 02:00 from a
480p reference copy. These support landscape and lighting observations; this
resolution does not establish the source material's finest texture detail.
The earlier dune photograph is superseded by this video.

| Reference frame | Observed appearance | Planned rendering target |
| --- | --- | --- |
| [00:15](https://www.youtube.com/watch?v=tXw5O9HtP84&t=15s) | Warm dunes behind a rocky basin; rippled sand, boulders, gravel patches, and sparse dry scrub in the foreground | Combine broad dune silhouettes with a detailed rocky basin and localized surface variation |
| [01:05](https://www.youtube.com/watch?v=tXw5O9HtP84&t=65s) | Close view of a large rock beside rough ground and broken stones | Give near-camera rock faces and the transition from sand to gravel sufficient geometric and material detail |
| [02:00](https://www.youtube.com/watch?v=tXw5O9HtP84&t=120s) | A strong foreground rock shadow, bright warm sand, scattered boulders, and low dunes beneath a blue sky | Preserve readable shadow shapes, surface detail in shade, and a clear separation of nearby and distant terrain |

We will build a rocky desert basin surrounded by sand dunes. Long dune crests
will have broad windward slopes and steeper sheltered slopes, with wind ripples
on suitable surfaces. Exposed rock ledges, scattered boulders, and gravel patches
will give the foreground its character. Sand will collect around and partly
cover the rocks rather than end at uniform material boundaries. Sparse dry scrub
will follow once the sand and rocks are convincing.

We will use warm ochre and tan sand, subdued brown-gray rock, a blue sky, and
directional sunlight. We will tune the sun direction and elevation against the
reference's shadow shapes rather than assume a sunset scene. Distant dunes will
have less contrast and fine detail. Sand grain will come mainly from material
detail; large rock shapes and dune silhouettes will use geometry.

The video is an appearance reference. Its engine, procedural biome implementation,
and advertised landscape extent do not establish Newton's implementation or
performance target. We will build the scene in Python/Warp with `ViewerRTX` and
measure it locally. Reference media are for inspection; no MAWI assets have been
added to the project.

We will use the existing Unitree G1 visual asset loaded by the standalone
[`robot_g1` example](../robot/example_robot_g1.py):
`newton.utils.download_asset("unitree_g1")` and
`usd_structured/g1_29dof_with_hand_rev_1_0.usda`. We will reuse its visual geometry
and articulated transforms through Newton's public model and viewer interfaces.
We will preserve the asset's attribution and licensing requirements.

The original posed G1 remains available through `--g1-mode posed`. The default
mode now imports the policy-configured MJCF/URDF and advances rigid contacts. We will verify mesh
materials in RTX and keep collision proxies hidden. A camera view from behind
G1 will complement the terrain-only close, middle-distance, and wide views.
Locomotion uses a separate `controller.py` and `simulation.py`; `collision.py`
selects an 8 m region of the dune surface for contact. No MPM code is imported.

## What Snowflow actually implements

We inspected the [reference repository](https://github.com/Noniv/snowflow_demo)
at revision `545039733b74eec742862f161990142c7ca7c7ec`. The author's
[Reddit post](https://www.reddit.com/r/ClaudeAI/comments/1v94nal/people_liked_my_desert_so_heres_a_waterbending/)
contains the initial prompt; the source describes the implemented behavior.

| Component | Implementation found in the source | Consequence |
| --- | --- | --- |
| Nearby terrain | One nested-ring mesh, with eight levels and 0.085 m inner vertex spacing; vertex shaders evaluate the surface | Detail decreases with distance without uploading a new terrain mesh every frame |
| Snow interaction | Footfall and movement events stamp depression, berm height, compression, and ice into a moving field | Snow contact is an authored surface response rather than an MPM constitutive/contact solve |
| Deformation storage | Two ping-pong RGBA16F targets, default 2048 by 2048 over 80 m | Approximately 0.039 m texels; memory and update area stay bounded |
| Character | Height sampling, a movement controller, procedural gait, foot planting, and cloth | Convincing movement does not require solving bulk snow around the character |
| Snow-surf wake | A fixed mesh evaluated along a moving spine, plus spray particles | The coherent curling mass is a mesh; individual particles add airborne detail |
| Distant mountains | Heightfield ray marching in the sky shader | The mountain backdrop does not contain a mountain-sized particle simulation |
| Appearance | Custom snow shading, sky illumination, shadows, and temporal post-processing | Much of the apparent detail comes from shading and lighting |

Source entry points at the inspected revision:

- [Terrain mesh](https://github.com/Noniv/snowflow_demo/blob/545039733b74eec742862f161990142c7ca7c7ec/src/terrain/clipmapMesh.js).
- [Deformation field](https://github.com/Noniv/snowflow_demo/blob/545039733b74eec742862f161990142c7ca7c7ec/src/terrain/deformation.js)
  and [update shader](https://github.com/Noniv/snowflow_demo/blob/545039733b74eec742862f161990142c7ca7c7ec/src/shaders/deformSim.fragment.wgsl).
- [Character controller](https://github.com/Noniv/snowflow_demo/blob/545039733b74eec742862f161990142c7ca7c7ec/src/character/controller.js)
  and [snow contact](https://github.com/Noniv/snowflow_demo/blob/545039733b74eec742862f161990142c7ca7c7ec/src/character/snowContact.js).
- [Surf wake](https://github.com/Noniv/snowflow_demo/blob/545039733b74eec742862f161990142c7ca7c7ec/src/vfx/surfWake.js)
  and [sky shader](https://github.com/Noniv/snowflow_demo/blob/545039733b74eec742862f161990142c7ca7c7ec/src/shaders/sky.fragment.wgsl).

The field retains trails while they remain in its moving window and gradually
relaxes them. Newly exposed texels are cleared; it is not a permanent world cache.
Depression and berm brushes have separate amplitudes, and the update includes
decay and clamping. This is not a guarantee of physical mass conservation.
Also, deformation texel spacing does not imply equally fine displaced geometry:
the terrain mesh spacing grows across successive rings.

The reference README reports a 3.22 ms GPU frame at 2560 by 1440 on an RTX 5070 Ti.
We have not reproduced that measurement. Its subsystem figures should not be
treated as an additive budget, and GPU time alone does not establish end-to-end
FPS or 1% lows. The Newton RTX path has different geometry transfer and scene update costs. Similar visual scale is feasible to investigate; equivalent performance
remains unverified.

## Newton renderer capabilities to validate

- [`ViewerRTX.log_mesh()`](../../_src/viewer/viewer_rtx.py) currently calls
  `.numpy()` for updated Warp vertices and normals. Fixed topology avoids index
  updates, but does not eliminate position and normal transfer. We will keep
  unchanged meshes static and measure the cost of updating terrain coverage.
- [`ViewerUSD` material authoring](../../_src/viewer/viewer_usd.py) exposes base
  color/texture, roughness, metallic, and opacity through `UsdPreviewSurface`.
  We will validate fine normal detail, roughness, and stable grain highlights
  for sand. Snow-specific scattering is not a requirement. We will demonstrate
  the required material path in the installed OVRTX version before depending on it.
- Python will orchestrate the scene, and Warp will generate terrain geometry and
  procedural data. OVRTX will render it. Custom shading cannot be assumed to be
  expressible as a Warp kernel through the current viewer API.

## Terrain representation

We will define a deterministic world-space height function, using XY for the
terrain plane and Z for height. Broad dune crests and interdune valleys will establish the
silhouette; wind-directed ripples will add surface structure.
The same world coordinates and seed will produce matching terrain across tiles
and levels of detail. Each layer will have an explicit physical scale in metres.

| Representation | Responsibility | Update policy |
| --- | --- | --- |
| Detailed nearby mesh | Dune forms that affect the silhouette, shadows, and close views | Generate initially; update when terrain coverage changes |
| Coarser terrain meshes | Middle-distance slopes and distant dune silhouettes | Reuse while covered; generate newly required tiles |
| Rock meshes and instances | Basin ledges, scattered boulders, and larger stones | Build static geometry initially; update visible coverage as needed |
| Surface material detail | Sand grain, gravel, rock variation, normals, and roughness | Evaluate through supported RTX materials, with distance filtering |
| Lighting and atmosphere | Warm sunlight, cool shadows, and distance-dependent contrast | Update when camera or lighting settings require it |

We will begin with a fixed terrain extent to establish appearance and measure a
static scene. We will then add camera-following levels of detail: increasingly
coarse terrain farther from the camera. We will compare nested rings and tiled
meshes against RTX upload and acceleration structure costs before choosing the
final layout. Snowflow's shader-displaced clipmap cannot be assumed to have the
same update cost through `ViewerRTX`.

Mesh density will be chosen by projected visual detail and measured cost. Fine
sand grain belongs in material detail when it does not affect silhouettes.
Normals will agree with the rendered surface, and detail will fade with distance
to avoid shimmering. Shared boundaries will use consistent samples and an
explicit transition strategy to prevent cracks and distracting changes in shape.

We will generate distant dune silhouettes as geometry first. A sky-shader
backdrop is an optional optimization only if the renderer supports it and
profiling justifies it. We will build the rocky basin and ledges as separate meshes, since a dune
heightfield alone cannot represent vertical faces and overhangs. Repeated stones
will use instancing where supported. Small gravel will transition to material
detail at distance; we will avoid uploading individual stone transforms each
frame when their positions are unchanged.

## Planned milestones and acceptance gates

1. **Create the standalone scene and baseline.** We will add an `Example` in this
   directory with deterministic dunes, a rocky basin, a posed Unitree G1, a useful
   initial camera, and RTX lighting. We will capture close, middle-distance, and wide views at 2560 by
   1440. Gate: a runnable example with recorded settings and measured preparation,
   upload, rendering, and presentation costs.

2. **Establish dune and rocky basin appearance.** We will tune dune crests,
   asymmetric slopes, wind ripples, ledges, boulders, and sand-to-gravel transitions.
   We will establish directional sunlight, warm sand, subdued rock colors,
   readable shadows, roughness, and controlled exposure. We will validate normal detail and restrained grain
   highlights in a small material scene before applying them to the landscape.
   Gate: inspected images show the reference's contrast between smooth dunes
   and rough rocky ground, readable shadows, preserved highlight detail, and a
   grounded G1 with credible scale. We will record any missing renderer capability.

3. **Extend visible scale with terrain detail levels.** We will add a repeatable
   camera route and bounded mesh coverage with progressively coarser distant
   geometry. We will preallocate reusable buffers where practical and update
   changed terrain only. Gate: camera movement reveals no cracks or distracting
   terrain transitions, and terrain generation and uploads do not cause large
   frame-time spikes.

4. **Unify the atmosphere and distant landscape.** We will validate the available
   sky and fog controls and tune aerial perspective: distant terrain loses
   contrast and shifts toward the atmosphere's color. Gate: the horizon, distant
   dunes, nearby sand, rock formations, and G1 share consistent lighting and depth
   cues. Sparse dry scrub will be added only after this composition is established.

5. **Profile and polish motion.** We will tune supported antialiasing, exposure,
   and detail filtering using the camera route. We will evaluate decorative
   wind-blown sand only after terrain appearance and frame times pass their
   gates. Gate: stable fine detail in motion, reproducible rendering measurements,
   and a documented quality/performance tradeoff.

The example and rendering helpers will stay in
`newton/examples/large_terrain_rendering`. We will use existing public Newton
geometry and viewer interfaces where suitable. Any required viewer extension
will address a demonstrated rendering need and follow Newton's API guidelines.

## Measurement and validation

The local machine currently reports an RTX 5090 with 32607 MiB and driver 595.84.
We will record actual software versions and renderer settings with each run.
Initial static and moving-camera measurements are recorded in [PERF.md](PERF.md);
longer profiling runs remain planned.

We will measure a stationary camera, a route within existing terrain coverage,
and a route that requires new terrain coverage. Timed runs will last at least
90 seconds after warm-up, with fixed resolution, seed, camera route, and quality
settings. We will report initial generation and warm-up costs separately.

We will record terrain generation, normal generation, mesh transfers, RTX scene
updates, rendering, presentation, vertex/triangle counts, and GPU memory use.
We will report median, p95, p99, maximum frame time, and 1% low FPS, defined as
1000 divided by the mean of the slowest 1% of frame times in milliseconds.
GPU work, CPU waiting, and any frame generation will be reported separately.
We will compare the same scene with G1 shown and hidden, and with rock and gravel
detail shown and hidden, to separate these costs from dune rendering.

A 90 FPS target permits about 11.1 ms per real frame. We will allocate subsystem
budgets after the baseline identifies dominant costs. We will compare visual
quality at several mesh densities and visible extents rather than infer
performance from triangle count alone.

We will validate finite geometry and normals, deterministic terrain samples,
matching tile boundaries, and bounded geometry counts where applicable. We will
use focused `unittest` coverage for meaningful invariants and regressions, plus
screenshot inspection for appearance. Before committing a regression fix, we
will verify that its regression test fails without the fix. We will run commands
with `uv` and run `uvx pre-commit run -a` before committing. A new public example
will include `test_final()` or `test_post_step()`, README registration, a 320 by
320 JPEG, and a Towncrier fragment.

## Decisions and remaining validation

- The scope is a dedicated sand dune rendering showcase with Unitree G1,
  independent of the MPM beta projects. The posed mode keeps rendering measurements independent of physics; policy mode
  adds rigid-body simulation against the local dune surface. The stylized human and snow environment are replaced by G1 and sand.
- We will preserve the original reference brief; this plan defines the current
  implementation scope.
- The main unresolved choices are terrain detail layout, RTX material features,
  atmosphere controls, and geometry update cost. We will validate these before
  promising visual or performance parity with the reference.
- No external source code or assets have been copied into Newton. The reference
  is MIT licensed; any later substantial source adaptation will retain its
  required notice and follow Newton's licensing conventions.
