<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Chrono CRM and Newton moving terrain

Newton's moving terrain examples aim to simulate a small volume of deformable
terrain around a moving body while limiting compute and GPU memory. This note
compares their current behavior with Project Chrono's continuum representation
model (CRM), which uses smoothed particle hydrodynamics (SPH), and identifies
work for a possible MPM beta2 implementation.

The main finding is that Newton already recycles a bounded particle bed.
Chrono's additional useful idea is to compact solver work around individual
interacting solids within the stored bed. Beta already separates dynamic
terrain from a fixed boundary band, but both retain active MPM particles.
Terrain persistence is a separate feature with additional costs. Switching from MPM to
SPH is not required to explore that separation.

This is a source inspection, not a runtime benchmark. The proposed work below
is not implemented, and no speedup or physical equivalence has been measured.

> This comparison records the source inspection before beta2 was implemented.
> See [README.md](README.md) for current implemented behavior and validation.

## Versions inspected

The original request lists `mpm_moving_patch_alpha` twice. The working interpretation
is that `mpm_coupling` is the early prototype and `mpm_moving_patch_alpha` is alpha:

| Version | Relevant implementation | Current behavior |
| --- | --- | --- |
| Early prototype | [`mpm_coupling/...moving_patch.py`](../mpm_coupling/example_mujoco_mpm_coupled_solver_moving_patch.py), `_recycle_terrain` | A tile-aligned XY window follows body 0. Escaping particles wrap to the opposite side and reset to their reference Z and fresh material state. |
| Alpha | [`moving_patch_mpm.py`](../mpm_moving_patch_alpha/impl/moving_patch_mpm.py), `MovingPatchMPMWrapper` | The wrapper supports sampled ground heights and a selected tracked body. Particles remaining inside the window retain state; recycled particles reset. All particles in the bed remain physically simulated. |
| Beta, default | [`moving_terrain_mpm.py`](../mpm_moving_patch_beta/impl/moving_terrain_mpm.py), `MovingTerrainMPMWrapper` | Dynamic simulated terrain is surrounded by fixed boundary particles. A finite background stores measured surface height and reference material volume per area for revisits. |
| Beta, optional full-state cache | [`particle_state_cache.py`](../mpm_moving_patch_beta/impl/particle_state_cache.py), `ParticleStateCache` | GPU records preserve individual particle state outside the loaded terrain. A separate snapshot retains boundary state; Python tracks tiles and free simulation slots. |
| Beta2 | This directory | Contains design documents; no beta2 simulation implementation was present at inspection. |

Newton was inspected in the local working tree based on commit
`66d9d50dde76f2d88184682028e6f9a886cb20f5`. The example directories are untracked
in that tree, so this commit alone does not reproduce them. Chrono source links
below are pinned to `aa3922df4ad18cc8571cb71b7fb274fd7e1612fa`, retrieved
2026-09-23. The online API documentation can change independently.

## What Chrono actually separates

### Active domains: reduce work within an existing particle bed

An active domain is a box associated with an interacting solid. Chrono tests
particles against the registered domains and also maintains extended domains
that provide neighboring particles. Inactive particles have their velocities
zeroed; extended-domain particles provide support without necessarily being
dynamically advanced. This is a local approximation: distant terrain does not
continue flowing as it would in a fully simulated bed.
See [`UpdateActivityD` and the integration kernels](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/chrono_fsi/sph/physics/SphFluidDynamics.cu).

The original particle state still exists. Chrono builds a compact list for the
extended domains and hashes/sorts that subset for neighbor search; some work
still scans all markers. Therefore active domains can reduce solver work and
working storage without eliminating the memory occupied by the full terrain's
particle records.
See [`SphCollisionSystem::ArrangeData`](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/chrono_fsi/sph/physics/SphCollisionSystem.cu).

The solver can reuse neighbor lists over several steps. Its default search
interval is one step, and a forced rebuild overrides that interval. The current
API also supports per-body domains and an initial free-flow duration during
which activity restrictions are disabled.
See [`ChFsiFluidSystemSPH`](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/chrono_fsi/sph/ChFsiFluidSystemSPH.cpp).

### Moving patch: limit the stored particle bed during forward travel

`ConstructMovingPatch` constructs a rectangular bed. A tracked sentinel body
approaching the front triggers relocation of a rear slab into a new front slab.
The implementation assumes world +X travel. It shifts the boundary condition
enforcement (BCE) markers and computational domain and forces another proximity
search. BCE markers are samples used to impose solid boundary conditions on
SPH particles.
See [`CRMTerrain`](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/chrono_vehicle/terrain/CRMTerrain.h)
and [`Synchronize`](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/chrono_vehicle/terrain/CRMTerrain.cpp).

The relocator partitions the existing marker array and assigns selected markers
to regular grid positions ahead. It resets velocity, density/pressure/viscosity,
and the stress components shown in the relocation kernel. This path does not
archive the vacated soil for later restoration. Partitioning can inspect the
whole marker range; relocation should not be assumed to cost only the number
of particles moved.
See [`SphParticleRelocator::MoveAABB2AABB`](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/chrono_fsi/sph/physics/SphParticleRelocator.cu).

These are independent facilities. In particular, the inspected
[moving-patch demo](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/demos/vehicle/terrain/demo_VEH_CRMTerrain_MovingPatch.cpp)
registers no bodies for fluid-solid interaction (FSI) and does not demonstrate active-domain
acceleration around vehicle wheels. It is not evidence for the performance of
the two facilities combined in a coupled vehicle simulation.

## Differences that matter for beta2

| Concern | Newton alpha / beta | Chrono CRM implications |
| --- | --- | --- |
| Numerical method | Material point method (MPM): particles carry state and transfer information through a background grid. These examples use `SolverImplicitMPM`. | SPH computes continuum interactions using particle neighbors. Search reuse is not directly equivalent to reusing an MPM grid solve. |
| Particle storage | Alpha and default beta allocate a finite local bed. Beta full-state caching additionally stores departed particles. | Moving-patch recycling also bounds the bed; active domains alone leave the original terrain particles allocated. |
| Where dynamics runs | Alpha simulates the whole bed. Beta uses one dynamic XY rectangle around a tracked body and a fixed boundary band. | Active domains can follow interacting solids separately, potentially avoiding dynamics between distant contact regions. |
| Boundary treatment | Beta boundary particles have zero mass and velocity but retain `ParticleFlags.ACTIVE`. | SPH distinguishes active and extended support particles and uses BCE markers. These roles cannot be reproduced just by copying an activity flag into MPM. |
| Movement | Newton follows XY motion, including reversal, within its terrain constraints. Beta snaps to background samples with a one-sample deadband. | The inspected moving patch relocates slabs along +X when the sentinel enters a front buffer. |
| Revisiting terrain | Alpha resets recycled terrain. Default beta reconstructs columns; optional beta caching restores individual records. | The inspected moving-patch path does not restore soil left behind. |
| Scheduling | Alpha recycling supports CUDA graph capture. Beta explicitly runs layer updates and its simulation loop without an outer captured graph. | Chrono's SPH update schedule is separate; GPU relocation does not establish that Newton can capture its current property refresh. |

The Chrono entries above follow the linked source paths. Newton details follow
the local implementations listed earlier and
[`MovingTerrainExample.setup_solver` / `simulate`](../mpm_moving_patch_beta/impl/base_example.py).
These examples couple MPM and MuJoCo through `SolverCoupledProxy`; Chrono's
SPH/solid coupling is a different numerical system. Particle counts or frame
rates alone would not make an accuracy-matched comparison.

## Concrete costs in the current beta

1. **The fixed boundary band still costs solver work.** Its particles stay
   active in the transfer and material masks. With the wrapper's default 2 m
   by 2 m dynamic region and a 0.5 m boundary band, the outer area is 9 m²
   versus 4 m² dynamically moving. At uniform depth and sampling, the band is
   about 56% of the initial particles. This is a particle-count calculation,
   not a prediction that 56% of frame time can be removed.

2. **Layer updates synchronize with the host.** Default beta reads status and
   change counters and scans all simulation slots for transitions each update.
   Height deposition runs when the patch center changes. Full-state mode also
   reads outgoing particle IDs and tile keys, groups them using NumPy, and
   creates GPU batches for departed and entering terrain.

3. **Property refresh does more than the wrapper's scalar reads suggest.**
   `_refresh_particle_properties()` calls
   `notify_model_changed(ModelFlags.MODEL_PROPERTIES)` and clears numerical
   warm starts with `reset(flags=0)`. The notification reaches
   `ImplicitMPMModel._refresh_particle_flags_and_extrema`, which reads particle
   flags, volumes, and several material arrays back to the CPU, then constructs
   replacement GPU arrays. This happens when transitions change properties,
   not necessarily on every step. The wrapper's comment about scalar readback
   does not describe this complete call path.

4. **Persistence has two different memory costs.** Height mode allocates dense
   ground, surface-height, and reference-depth arrays for the finite background.
   Full-state mode additionally reserves approximately 50% extra simulation
   slots, allocates a full boundary snapshot, and keeps departed GPU records.
   Its simulation-slot capacity remains fixed and can be exceeded by material
   accumulation even when there is still GPU memory available.

5. **Rendering savings do not imply simulation savings.** Beta's particle
   rendering compaction and level of detail reduce submitted graphics data.
   They do not shrink the MPM particle allocation or active simulation region.

The solver refresh behavior is verified in the implementation files
`newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py` and
`newton/_src/solvers/implicit_mpm/implicit_mpm_model.py`. Those paths explain the
current implementation; example code should continue using public solver APIs.

For memory planning, let `N` be the initial local particle count, `H` the number
of background samples, and `V` the number of currently cached departed
particles. Default beta has local particle storage proportional to `N` plus
background storage proportional to `H`. Full-state beta adds storage
proportional to `V` and increases the local allocation to about `1.5 N`.

One `ParticleData` record contains two three-component vectors, four 3×3
matrices, four scalar floats, and one integer: **188 bytes** at the current
32-bit field widths. One million cached records therefore contribute about
188 MB before allocation overhead and metadata. This excludes live particle
arrays, solver/proxy states, the boundary snapshot, rendering arrays, and the
MPM grid. Retaining arbitrarily much exact particle history on the GPU is
incompatible with a fixed GPU memory budget.

Default height reconstruction also has a fidelity cost: it preserves estimated
column height and reference volume, but resets material history on restoration.
Full-state restoration preserves more information, but the departed particles
are frozen in time. Neither mode represents continued remote avalanches or
settling while the terrain is outside the simulation.

## Proposed beta2 work

We will keep MPM and first establish which costs dominate. Chrono suggests
useful scheduling and storage choices, but its timings and support distances
cannot be transferred directly to an implicit MPM solver.

1. **Measure the existing modes with matched inputs.** We will compare alpha,
   beta height caching, and beta full-state caching using the same material,
   particle spacing, bed dimensions, rigid-body motion, simulated duration,
   and coupled timestep. We will measure steady motion and patch-shift frames
   separately after warmup. Metrics will include wall time per simulated
   second, layer/cache updates, property refresh, MPM solve, rendering, peak
   GPU memory, and cached particle count. Rendering will be disabled for the
   physics baseline. Existing benchmark results need their settings checked
   before being reused for this comparison.

2. **Reduce unnecessary transition work while preserving beta behavior.**
   We will evaluate larger sample-aligned shift distances and a front/side
   margin before moving the outer bounds. We will retain a per-step check for
   particles flowing across boundaries even when the body is stationary.
   We will profile a narrowly scoped public solver update path for changing
   particle properties without repeatedly copying unchanged material arrays
   through the CPU. Any solver API change requires separate design and tests.
   CUDA graph capture will be validated only after array lifetimes and update
   scheduling are stable; simply removing the host branch is insufficient.

3. **Prototype separate storage and dynamic regions.** We will first use one
   dynamic region plus enough supporting terrain inside a larger stored bed.
   We will validate that arrangement before considering separate regions
   around wheels or feet. MPM interpolation and the implicit stress/contact
   solve determine the required support; Chrono's SPH neighbor radius is not
   a valid MPM boundary-width rule. Clearing `ACTIVE` on today's boundary
   particles without replacing their support could change the solution.

4. **Choose persistence explicitly.** We will retain height reconstruction as
   the baseline for routes that revisit terrain. A forward-only experiment
   can reset departed soil, as alpha already does, to isolate recycling cost.
   Exact revisit history will require a declared cache budget and a policy
   when it is exceeded. CPU storage or height reconstruction are possible
   policies with different transfer and fidelity costs; neither is selected
   here. A reset mode must clearly state that revisits lose deformation.

Before claiming a performance improvement, we will validate the proposed
implementation against a larger fully simulated MPM bed. Cases will include
forward travel, reversal over a rut, diagonal shifts, particles crossing a
stationary boundary, and separated contacts. We will compare sinkage, rigid
body forces, material volume, and discontinuities at patch shifts, and sweep
support width. Cache restoration tests will check the chosen persistence
contract. A boundary that freezes terrain must remain outside the region
whose flow matters to the task.

The first deliverable should be a profile of beta's transition and property
refresh path. That will establish whether reducing host transfers and shift
frequency is sufficient before adding multiple dynamic regions or changing
terrain persistence.
