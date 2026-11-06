<!-- SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Chrono CRM and Newton moving terrain: comparison and beta2 direction

## Implementation update

The initial implementation is now described in [README.md](README.md). It adds
GPU-only height-based background tiles with fixed capacity, a following ground
collider, and a targeted MPM particle-property update. It supports one tracked
body and flat or sloped ground. Multiple wheel/foot regions, a cheaper boundary
representation, and CUDA graph capture remain proposed work. The comparison and
implementation handoff below record the original design; they are not a claim
that every listed optimization has been implemented or measured.

## 1. Motivation and main finding

The goal is to simulate terrain around a moving robot without simulating every particle in a large environment. The local terrain must still support the robot, exchange forces with it, and recover previously disturbed ground when the robot returns. Compute cost, GPU memory, and terrain memory are related requirements, but they need separate solutions.

**Newton beta already has the bounded particle allocation provided by Chrono's moving patch. The most useful additional idea from Chrono is to distinguish particles that are stored, particles that supply nearby support, and particles whose dynamics are advanced.** This is a work-selection strategy; it does not require replacing MPM with SPH.

There is also an important difference in scope: **beta's terrain memory goes beyond the inspected Chrono moving-patch implementation.** Chrono's relocation path creates fresh material ahead of the tracked body. Beta can reconstruct previously visited terrain from a height cache, or optionally restore cached particle state. Removing that work would simplify beta, but it would also remove a feature that the earlier Newton prototype lacked.

This document compares source code, not measured Chrono/Newton performance. It does not establish that Chrono is faster or that either method gives equivalent soil forces. The work below was proposed during the comparison; see the implementation update above for current status.

### Implementation handoff: what the first beta2 will improve

**Original handoff scope:** this section records the proposal. Current implemented behavior and measurements are linked in the implementation update above. This section defines the first implementation scope for a later main-thread task.

The first beta2 will improve the execution of beta's existing moving terrain. It will retain:

- The same simulated terrain dimensions, particle sampling, material parameters, and supporting boundary treatment for the initial comparison.
- Height-based background reconstruction, including cached column height and reference material volume.
- Two-way rigid-body coupling and dynamic robot/object motion.
- Handling of particles that escape because material flows, including when the tracked body is stationary.

It will **not initially restrict MPM to particles colliding with the robot**. Simulated terrain and active boundary particles will continue to participate in MPM. Collision constraints and selection of material for simulation remain separate concepts.

| Current beta cost | First beta2 implementation target | Expected benefit, requiring measurement |
| --- | --- | --- |
| Particle-property changes trigger broad CPU readbacks and replacement GPU arrays. | Implement a targeted property-update path, keeping updates on the GPU where feasible. Check derived volumes, masks, and solver state for consistency. | Lower maintenance time; potentially fewer temporary allocations. |
| Patch movement can repeatedly trigger maintenance. | Reduce unnecessary shifts and update affected terrain data, while saving information before material becomes fixed or is recycled. | Lower maintenance time. Little direct reduction in persistent storage. |
| The current update path prevents outer CUDA graph capture. | Make update scheduling and array lifetimes compatible with capture, then validate capture and replay. | Lower CPU launch overhead. No guaranteed VRAM reduction. |
| Grid and NanoVDB capacities may reserve more storage than needed. | Measure occupancy under motion and deformation, then tune capacities with sufficient headroom and overflow checks. | Lower unused reserved VRAM. Runtime improvement is not guaranteed. |

We will profile before selecting which changes to implement first. If MPM solving dominates step time, cheaper maintenance alone will produce a limited improvement. With the same particle count and grid requirements, we will not assume a large reduction in persistent solver memory.

The goal is to preserve beta's intended physical behavior, not to promise bitwise-identical trajectories. Changes in patch-shift scheduling or numerical warm starts will require checks of forces, sinkage, boundary transitions, and restored terrain shape.

**Deferred work:** thinner boundaries, a cheaper solver-supported representation of fixed support, and separate simulated regions around wheels or feet. These may reduce actual particle/grid work, but can change physical behavior and require additional validation. Clearing `ParticleFlags.ACTIVE` on the existing boundary is not an accepted shortcut. Full-state background caching is also not a prerequisite for this first beta2 implementation.

**Chrono distinction:** active domains retain the original terrain particle records while reducing local dynamics. Chrono's inspected moving-patch relocation instead resets departed particles as fresh material. The first beta2 will retain Newton's bounded local particle allocation and height-based background reconstruction; it will not switch to storing every background particle.

**Completion criteria:** retain beta as the comparison baseline; compare matched scenes and physics settings with rendering disabled; report mean and standard deviation of completed physics-step time, measured GPU memory, and sampled peak GPU memory. Verify terrain support, stationary material outflow, patch shifts, reversals, and returning to disturbed terrain. Record any optimization that does not improve the measured result rather than claiming a benefit from its implementation alone.

Suggested request when returning to the main thread:

> Implement the first beta2 scope in `newton/examples/mpm_moving_patch_beta2/chrono_comparison_main.md`, under “Implementation handoff.” Start by profiling beta's maintenance and solver costs. Preserve height-based terrain reconstruction and boundary support. Keep the deferred local-domain and boundary-physics changes separate, and measure improvements against beta.

## 2. Which Newton versions are being compared?

The request lists `mpm_moving_patch_alpha` twice. This comparison treats the early example in `mpm_coupling` as the first prototype and the wrapper in `mpm_moving_patch_alpha` as alpha.

| Version | What is stored and simulated? | What happens when terrain leaves the patch? |
| --- | --- | --- |
| Early prototype: [`mpm_coupling`](../mpm_coupling/example_mujoco_mpm_coupled_solver_moving_patch.py) | A finite particle bed follows the box. | Particles wrap to the other side and return to fresh material state. |
| Alpha: [`MovingPatchMPMWrapper`](../mpm_moving_patch_alpha/impl/moving_patch_mpm.py) | A finite particle bed follows a selected body; terrain heights can come from a sampled ground profile. The entire particle bed participates in MPM, including particles hidden by rendering. | Escaping particles are recycled. The original terrain generator supplies their ground profile, not a cache of departed deformation. |
| Beta, default height cache: [`MovingTerrainMPMWrapper`](../mpm_moving_patch_beta/impl/moving_terrain_mpm.py) | Dynamic simulated terrain, a fixed boundary band, and a GPU background height cache. Both local particle regions retain `ParticleFlags.ACTIVE`. | Cached column height and reference material volume determine reconstructed particles. Full material history is not restored. |
| Beta, optional full-state cache: [`ParticleStateCache`](../mpm_moving_patch_beta/impl/particle_state_cache.py) | Local particles plus GPU records for departed terrain, extra simulation slots, and a full-size boundary snapshot. | Cached particle records can be restored. Departed terrain is frozen while it is outside the simulation. |
| Beta2 | The comparison directory currently contains design documents. | Behavior remains to be implemented and validated. |

Two distinctions from the earlier development matter throughout this comparison:

- **Rendering visibility is separate from simulation participation.** Hiding particles, changing their colors, or submitting fewer particles to the viewer does not by itself reduce MPM work.
- **Sparse grid occupancy is separate from robot contact.** Newton builds the MPM grid around participating material particles. Soil without robot contact still has gravity, internal stress, and material motion to solve.

## 3. Chrono has two separate mechanisms

Chrono CRM means **continuum representation model**. It uses **smoothed particle hydrodynamics (SPH)**: nearby particles interact through a smoothing kernel. Newton's **material point method (MPM)** instead transfers particle information to a computational grid, solves on that grid, and transfers results back to particles. Chrono's [CRM guide](https://api.projectchrono.org/vehicle_terrain_crm_api_.html) exposes both active domains and moving patches; they address different costs.

### 3.1 Active domains: reduce work within stored terrain

An active domain defines a local interaction region associated with a solid. Multiple registered solids can produce a union of local regions. Chrono also includes an extended region around them to supply neighboring particles. See the [active-domain guide](https://api.projectchrono.org/vehicle_terrain_crm_performance.html).

The inspected integration code distinguishes three roles:

| Chrono particle role | Included in the local neighbor data? | Position and velocity advanced as ordinary SPH material? |
| --- | --- | --- |
| Inside an active domain | Yes | Yes |
| Outside the active domain but inside its extended support region | Yes | No in the inspected Euler/midpoint integration kernels |
| Outside both regions | No | No |

The last two roles must not be conflated. A supporting neighbor can remain available to nearby material without undergoing ordinary dynamics. `UpdateActivityD` zeros the velocity of particles outside the active domains; the integration kernels skip particles whose activity identifier is not positive. Being present in the compact neighbor list therefore does **not** mean being dynamically advanced. This source-level distinction is more precise than the guide's broad description of active and extended particles being processed together. [Chrono activity and integration source](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L228-L288).

Chrono builds a compact particle-index list for the active and extended regions. Hashing and sorting then operate on that selected set. However, classification/list construction still scans the original marker population, and the original particle records remain allocated. **An active domain is not, by itself, a constant-memory representation of an arbitrarily large terrain.** The activity-count path also reads results back to the CPU when sizing working arrays. The useful distinction is the smaller downstream workload, not a claim that Chrono has no synchronization. [Chrono neighbor-search source](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/chrono_fsi/sph/physics/SphCollisionSystem.cu#L335-L387).

### 3.2 Moving patch: reuse a bounded particle bed

`ConstructMovingPatch` addresses particle storage. In the inspected implementation, a tracked body approaches the front of the patch, and a strip of particles behind it is relocated ahead. The implementation advances along **positive X**; it is not a general replacement for beta's XY tracking, reversals, and revisiting terrain. Its boundary markers and computational domain move too, and relocation forces a new proximity search. [Chrono moving-patch source](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/chrono_vehicle/terrain/CRMTerrain.cpp#L75-L173).

**Boundary condition enforcing (BCE) markers** supply SPH boundary conditions. They are not Newton's ordinary zero-mass MPM particles, so their implementation cannot be transferred by changing one Newton particle flag.

The relocation code resets velocity, density/pressure/viscosity, and stress components. It does not retrieve material previously left at the destination from a terrain-history cache. It also partitions the original marker range to find particles to move; its cost is not necessarily proportional only to the number relocated. [Chrono particle relocation source](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/chrono_fsi/sph/physics/SphParticleRelocator.cu#L120-L202).

Conceptually, the mechanisms are independent:

```text
Active domains:
    retain the stored terrain
        -> select local particles and supporting neighbors
        -> perform local SPH work

Moving patch:
    retain a bounded particle allocation
        -> relocate departed material ahead
        -> reset it as fresh material

Newton beta:
    retain a bounded particle allocation
        -> save departed terrain information
        -> reuse particles at entering terrain
        -> reconstruct material from the cache
```

The inspected [moving-patch demo](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/demos/vehicle/terrain/demo_VEH_CRMTerrain_MovingPatch.cpp) does not register interacting vehicle bodies with the fluid-solid system. It demonstrates relocation, not the performance of a complete vehicle using both mechanisms together.

### 3.3 Neighbor-search reuse is a separate optimization

Chrono can rebuild neighbor data every several steps instead of every step. A forced rebuild overrides that schedule. [Chrono update scheduling](https://github.com/projectchrono/chrono/blob/aa3922df4ad18cc8571cb71b7fb274fd7e1612fa/src/chrono_fsi/sph/ChFsiFluidSystemSPH.cpp#L2036-L2063).

This does not mean Newton can skip its material solve or reuse old particle-to-grid weights. Particle positions, transfer weights, stresses, and contact constraints change. Reusing MPM grid storage or topology is a different question from reusing numerical values. Newton already has a rebuildable sparse-grid path; any further reuse must preserve correct occupancy and transfer data.

## 4. What this means for beta's three terrain layers

Use the existing terms consistently:

| Newton terrain layer | Current beta behavior | Closest Chrono concept |
| --- | --- | --- |
| **Simulated terrain** | Dynamic MPM particles around the tracked body. | Material inside active domains. |
| **Boundary terrain** | Zero-mass MPM particles with `ACTIVE` retained; velocity and velocity gradient are held at zero. | Supporting material around an active domain, conceptually. The numerical treatment differs. |
| **Background terrain** | Terrain information stored outside the local particle simulation. Height mode stores arrays, not all the original sand particles. | No direct equivalent in the inspected fresh-material relocation path. Inactive full-terrain SPH records are a different storage choice. |

The boundary exists for a physical reason. Loads and material motion extend beyond the portion of sand touching a wheel or foot. Removing everything outside the contact footprint would remove supporting material and truncate stress transmission.

**Clearing `ParticleFlags.ACTIVE` on the current boundary is therefore not equivalent to Chrono's extended support region.** It removes those particles from MPM participation. Chrono retains supporting particles in its local neighbor data. A cheaper MPM boundary must still supply an appropriate boundary condition or support contribution.

MPM transfer uses a finite grid stencil, but the implicit material solve couples neighboring grid unknowns. Sufficient support cannot be inferred simply by copying Chrono's SPH neighbor distance. Boundary width must be validated against forces, sinkage, material flow, and sensitivity to making the simulated region larger.

The supporting ground profile and cached sand height are also different data. Beta's ground mesh represents the underlying terrain surface. The sand-height cache describes the material above it. A cheap static ground mesh does not demonstrate that retaining full particle state over the same area would also be cheap.

## 5. What beta currently pays for

### 5.1 The fixed boundary is still represented in MPM

Zero mass does not make the boundary disappear from particle transfer or sparse-grid construction. The current [`ImplicitMPMModel` masks](../../_src/solvers/implicit_mpm/implicit_mpm_model.py) do not exclude a particle merely because its mass is zero. The [`grid construction`](../../_src/solvers/implicit_mpm/solver_implicit_mpm.py) uses the resulting participating particles.

For the wrapper defaults, a 2 × 2 m simulated region plus a 0.5 m boundary on each side produces a 3 × 3 m particle bed. At uniform depth and sampling, the boundary contains about **56% of the particles**. This is a particle-count calculation, **not** a prediction that removing it would save 56% of runtime. It identifies a substantial cost worth measuring, while also explaining why boundary removal changes the physics.

### 5.2 Property changes can trigger broad CPU/GPU work

[`update_terrain_layers()`](../mpm_moving_patch_beta/impl/moving_terrain_mpm.py) reads a status value to the CPU, runs transitions over the allocated particle array, and reads a changed flag. When properties change, `_refresh_particle_properties()` copies mass/radius/flag arrays to the MPM model, calls `notify_model_changed(MODEL_PROPERTIES)`, and invalidates numerical warm starts.

The notification reaches `_refresh_particle_flags_and_extrema()`, which reads flags, particle volumes, and material arrays to the CPU and creates replacement GPU arrays. Thus, **the complete update is not just a few scalar transfers**. This is a concrete optimization candidate, although its fraction of total step time has not been measured here. It occurs when properties change, not necessarily on every physics step.

The beta loop runs outside an outer CUDA graph capture. Alpha's simpler reset kernel is designed for capture. Graph capture could reduce launch overhead, but only after data lifetimes, allocations, property updates, and synchronization are made compatible; merely deleting a host branch is insufficient.

### 5.3 Height caching preserves a reduced description of terrain

The current height cache stores measured column height and reference material volume. Reconstructed particles are distributed through that column using their reference vertical fractions. It cannot encode arbitrary voids, overhangs, particle rearrangement, or material loading history.

There is a further detail in `_transition`: in height-only mode, a particle becoming dynamic from the boundary receives default elastic strain, plastic determinant, stress, and rendering transform, even if it did not wrap across the outer patch. A particle that stays inside simulated terrain retains its state. Therefore material-history loss is not confined to newly wrapped particles.

This helps explain the earlier observation that revisited terrain may settle again. Restoring a column height does not restore the state that supported that height. It is a plausible mechanism, not a measured attribution of every observed height mismatch.

### 5.4 Full-state caching purchases more fidelity with more storage and bookkeeping

The optional cache retains positions, velocities, deformation/stress fields, and particle properties. Its `ParticleData` fields total **188 bytes per cached particle** at the current field widths, excluding allocation overhead and other solver arrays. One million departed particle records alone therefore require about 188 MB.

Current full-state mode also adds roughly 50% spare simulation slots, a full-size frozen snapshot, and CPU processing of outgoing particle IDs and tile groups. These are additional costs beyond copying a few heights. This matches the reason for keeping the default height-based version as the practical starting point rather than silently reintroducing full-state caching in beta2.

Full-state caching still freezes departed material. It improves restoration, but does not simulate remote settling or avalanches while the robot is elsewhere.

### 5.5 Local MPM cost and background memory have different scaling

Let `N` be the number of local particle slots, `H` the number of background height samples, and `V` the number of departed cached particle records:

| Mode | Main storage scaling, excluding detailed grid/robot allocations |
| --- | --- |
| Alpha reset-only patch | Local particles and grid; terrain-generator/ground storage is separate. |
| Beta height cache | Local particles and grid, plus background arrays proportional to `H`. |
| Beta full-state cache | The above, larger local particle storage, and cached records proportional to `V`. |
| Chrono active domains without recycling | Original terrain particle storage remains; some working arrays depend on the selected local particles. |
| Chrono moving patch | Bounded particle storage, with departed deformation discarded by the inspected relocation path. |

A local moving patch can bound MPM work without making an unlimited background free. Beta's ground, sand-height, and reference-depth arrays alone use approximately `12 * H` bytes as float32 arrays, before the ground mesh and other buffers. A tiled or externally stored background becomes necessary if the background itself exceeds the desired memory budget.

## 6. Beta2 implementation direction

We will preserve the existing behavior first: one tracked body, two-way rigid coupling, simulated terrain, boundary terrain, and the default height cache. We will keep material parameters outside the wrapper and reuse existing parameter names and units. Separate wheel/foot regions will be an optional later experiment, not an initial requirement.

### Stage 1: measure and reduce maintenance costs

We will distinguish ordinary physics steps from steps that move the patch or change particle layers. We will measure layer updates, height updates, property refresh, the coupled MPM solve, and GPU memory with rendering disabled. These component timings are diagnostic measurements; the main performance result will remain completed physics-step time and GPU memory.

We will then evaluate:

1. **A larger movement margin before shifting the patch.** The alpha `shift_step` distance provides an existing convention if explicit shift control is needed. The shift must occur before the interaction reaches the boundary, and fast mouse dragging must remain supported.
2. **A narrower property-update path.** We will determine whether changed masses, volumes, and masks can be refreshed on the GPU without reading unrelated constant material parameters back to the CPU. Any required solver API will be identified explicitly; this is not assumed to be an example-only change.
3. **Height updates limited to affected columns, if profiling justifies them.** The current accumulation arrays are already patch-sized. We will save terrain before it becomes fixed boundary material or is recycled, not only when it reaches the outer edge. Particle flow can cross a layer boundary while the tracked body is stationary, so robot movement alone is not a sufficient correctness condition.
4. **CUDA graph capture after the update path is compatible.** We will retain per-step handling of particles that escape through material flow. Less frequent patch shifts must not leave escaped particles untreated.

These changes target beta's extra maintenance work while preserving its terrain-memory feature.

### Stage 2: reduce boundary cost without removing support

We will first test how much the existing boundary can be narrowed while preserving an adequately large simulated region. This is a parameter study, not a claim that one or two voxel layers are universally sufficient.

If boundary particles remain a dominant cost, we will investigate a solver-supported distinction between dynamic material and material that supplies fixed support. A supporting particle may still need to contribute mass/velocity constraints and occupied grid cells even if some constitutive updates can be omitted. Which operations can be omitted safely requires an MPM derivation and validation.

We will not implement this by simply clearing `ACTIVE`, or by assuming a rigid wall is equivalent to surrounding deformable sand.

### Stage 3: optional regions around separate wheels or feet

Chrono's local domains are most attractive when a large stored bed contains widely separated interaction regions. For a compact robot whose disturbed terrain overlaps, several separate regions may save little and introduce extra boundaries.

A later MPM implementation could form the union of regions around selected bodies, with sufficient supporting terrain. Overlapping regions must not duplicate particles or forces. Material connecting two contacts must remain simulated when it carries relevant loads or flow. A compact index list would only save substantial work if particle transfers, grid construction, and solver assembly actually use it; changing a flag while continuing to scan and allocate every particle array does not remove every cost.

This stage will require more solver integration than moving the existing rectangular patch. It will be considered only after the simpler maintenance and boundary changes have been measured.

## 7. Validation and decision criteria

Performance comparisons will use the same material, sampling, voxel size, rigid-body motion, and simulated duration. SPH and MPM runtimes will not be compared as though equal particle counts implied equal accuracy or stable timestep.

Before accepting a beta2 optimization, we will check:

- A box resting on sand: sinkage, vertical force, and vibration compared with a larger simulated region.
- A shovel pushing material while the tracked body remains nearly stationary: escaping particles and saved heights.
- Forward motion, reversal, and a turn: layer transitions and retained terrain history.
- A return to disturbed terrain: reconstructed surface before and after further settling.
- A rover or G1 scene: force continuity across patch shifts and continued two-way coupling.

The first beta2 target is **the existing height-cache behavior with less maintenance overhead**. Chrono-inspired selection of smaller dynamic regions is a later option. Exact particle history remains a separate, more expensive feature; it will not be treated as a prerequisite for improving the current default.

## Source scope

Local behavior was checked in the working-tree files linked above, including the implicit MPM model and solver. Chrono implementation details refer to revision `aa3922df4ad18cc8571cb71b7fb274fd7e1612fa`. The corresponding source copies available in `/tmp/newton-chrono-comparison` were inspected directly; links are pinned to that revision. The live API pages were also consulted and can differ from a pinned implementation.

This document does not use the ongoing Newton benchmark as a completed result, run a new GPU workload, or change either implementation. It is separate from [`chrono_comparison.md`](chrono_comparison.md).
