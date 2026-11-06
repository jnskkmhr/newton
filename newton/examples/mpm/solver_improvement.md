# Implicit MPM solver performance investigation

## Motivation and scope

We will investigate ways to reduce the cost of one implicit MPM physics step without changing the simulated terrain, material, grid resolution, or timestep. The main targets are repeated grid and matrix construction, particle–grid transfers, temporary storage, CPU/GPU synchronization, collision queries, and iterative solver work.

This document lists source-based improvement candidates. The first measurements are
now available in the [component profiling report](solver_profile/README.md), with PNG
figures, raw samples, memory accounting, and reproduction commands. No numerical
solver optimization has been implemented. Local solver changes mean these findings
may differ from upstream Newton.

The first profiles prioritize **work and storage sized by reserved grid capacity**,
followed by **material/contact solve preparation and iterations**. With the same
120,000 particles, changing reserved cells from 131,072 to 98,304 reduced mean step
time from 23.32 to 19.86 ms and CUDA-pool used memory from 1.689 to 1.385 GiB.
These controlled core-solver cases do not include moving-patch recycling or a coupled
robot. Invalid residual norms appeared in some reference diagnostics; convergence
needs investigation before changing stopping conditions.

Instrumentation and subsequent experimental solver changes belong in
`newton/_src/solvers/implicit_mpm_improve`. The profiling adapter calls the unchanged
reference solver, and baseline runs instantiate the reference class directly. The
existing `implicit_mpm` implementation remains the numerical reference.

The investigation covers the core solver. Particle rendering, video capture, policy
inference, and the moving-patch wrapper must be measured separately. A faster renderer
does not imply faster MPM physics.

## What one physics step does

`SolverImplicitMPM.step()` calls `_particles_to_cells()`, `_rebuild_scratchpad()`, and `_step_impl()`:

```text
Particle positions and flags
    → build/update grid and identify occupied cells
    → group particles by cell; construct numerical spaces
    → query colliders at grid/sample locations
    → transfer particle mass and momentum to the grid
    → assemble material and contact operators
    → solve material deformation and contact together
    → update particle velocity, position, and material state
    → retain numerical guesses for the next step
```

The material solve covers the simulated material, including material away from contact. Collision rasterization adds contact information; it does not restrict the entire material solve to cells touching the robot.

Definitions used below:

- **Grid topology:** which cells/nodes exist and how they connect.
- **Matrix structure:** row offsets and column indices identifying the nonzero blocks. **Matrix values** are the numerical coefficients in those blocks.
- **BSR:** block sparse row storage, which groups matrix entries into small dense blocks.
- **Warm start:** a previous numerical solution used as the initial guess for the next solve. It is different from the particle's physical material state.
- **Reserved capacity:** allocated storage, which can exceed the number of occupied cells or useful matrix entries.
- **PIC integration:** one particle sample is assigned to a cell. **GIMP integration:** a particle can contribute samples to several cells. This distinction is separate from choosing PIC versus APIC velocity transfer.

## Prioritized candidates

Priority is an investigation order, not a prediction of speedup. “High” effort means a structural solver change or substantial numerical validation.

| # | Candidate | Where it matters most | Effort |
| --- | --- | --- | --- |
| 1 | Remove full particle-property refresh from boundary transitions | Moving patches with changing particle mass | Medium |
| 2 | Reuse the iterative solve's CUDA graph | Uncaptured outer simulation loop | Medium–high |
| 3 | Specialize ordinary PIC particle updates | Particle-heavy scenes | Medium |
| 4 | Fuse particle-to-grid transfer passes | Particle-heavy scenes using APIC | Medium–high |
| 5 | Reuse explicit temporary arrays and matrix buffers | Uncaptured execution; higher-order strain bases | Low–medium |
| 6 | Avoid topology and coloring work when connectivity is unchanged | Slowly changing terrain | High |
| 7 | Reuse sparse matrix structure and transpose mappings | Assembly-heavy scenes | High |
| 8 | Reduce work proportional to reserved capacity | Generously provisioned sparse grids | Medium–high |
| 9 | Improve warm starts before increasing solver complexity | Iteration-heavy scenes; rebuilt sparse grids | Medium–high |
| 10 | Prune collider queries and cache static geometry data | Many robot links or large support meshes | Medium–high |
| 11 | Simplify the finite-mass collider coupling operator | Scenes using finite-mass MPM colliders | Medium–high |
| 12 | Tune sparse arithmetic layout and kernel scheduling | Iterative solve dominates runtime | Medium–high |
| 13 | Evaluate matrix-free material operators | Matrix assembly/storage dominates runtime | High |
| 14 | Stop converged environments independently | Many parallel robot environments | High |

### 1. Update only changed particle properties during boundary transitions

**Observed:** [`ImplicitMPMModel.notify_particle_properties_changed()`](../../_src/solvers/implicit_mpm/implicit_mpm_model.py) copies particle flags, launches `_refresh_particle_properties` over all particles, and reads `_particle_property_extrema.numpy()`. The four summary values select material-dependent branches on the CPU. The public solver method also states that this refresh cannot run inside CUDA graph capture and that the caller must invalidate numerical warm starts when appropriate.

**We will investigate:** add a narrowly defined update path for particles whose mass, inverse mass, or flags changed. For homogeneous terrain, unchanged material parameters may permit retaining conservative material summaries instead of recomputing and reading them after every transition. Keep the general refresh for arbitrary edits.

**Why it could help:** the benefit could exceed the small readback size: a CPU read can wait for preceding GPU work, and the current refresh processes the full particle array. A device-only path would also remove one obstacle to capturing the surrounding simulation step.

**Correctness checks:** update derived particle volume, density, transfer flags, and material flags consistently. A particle changing between simulated and zero-mass boundary terrain must preserve the intended mechanical behavior. Do not infer that unchanged material arrays alone make summaries valid: changing the material-active particle set can change their extrema. Preserve a correct numerical warm-start invalidation policy.

### 2. Avoid rebuilding the iterative solve graph every uncaptured step

**Observed:** [`_run_solver_loop()`](../../_src/solvers/implicit_mpm/solve_rheology.py) uses a conditional CUDA graph when supported. If the caller is already capturing, it records into that capture. Otherwise it creates a `wp.ScopedCapture`, captures the loop, and launches the resulting graph. `solve_rheology()` returns this graph, but `_solve_rheology()` currently does not retain that return value.

**We will investigate:** retain the solve objects, their device buffers, and a reusable graph while dimensions, topology-dependent operations, and settings remain compatible. Measure capture/instantiation time separately from GPU execution.

**Important constraint:** storing only the graph handle is insufficient. The current implementation releases temporary solver storage after the call. Reuse must guarantee that every captured pointer remains valid and refers to the intended data. Topology changes, solver changes, and buffer growth require invalidation or recapture.

**Do not assume:** an uncaptured outer loop means the iterative solve also runs entirely uncaptured. The internal conditional-graph path already exists. CPU residual reads on every iteration are also **not** universal: the non-graph loop reads residuals once per iteration group, while graph execution normally keeps its stopping condition on the GPU. Verbose diagnostics add readbacks.

### 3. Use direct particle writes for ordinary PIC integration

**Observed:** [`_update_particles()`](../../_src/solvers/implicit_mpm/solver_implicit_mpm.py) clones the previous elastic strain and plastic volume ratio when a strain update is needed and the input/output state is the same object. It then clears output strain/stress/plastic arrays. Advection clears velocity and velocity-gradient arrays and assigns the input particle positions to the output positions.

[`update_particle_strains()` and `advect_particles()`](../../_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py) use atomic additions. These support accumulation when a particle has multiple quadrature contributions, as in GIMP.

**We will investigate:** a PIC-only kernel where one thread owns one particle, reads its old state, computes the update, and writes each output once. This may avoid atomics, separate zeroing passes, and the in-place history copies. First verify that the chosen PIC path really has one contribution per output particle.

For scale, copying a float32 3×3 elastic strain and one float32 plastic ratio involves 40 bytes of payload per particle per update, before counting the source reads and other state traffic. This is an accounting estimate, not measured bandwidth or runtime.

**Correctness checks:** preserve inactive-particle behavior, APIC velocity gradients, interpolation weights, and in-place/out-of-place equivalence. Keep the accumulating implementation for GIMP until a particle-owned implementation explicitly gathers all contributions. Simply removing the existing clears or atomics would be incorrect.

### 4. Fuse particle-to-grid mass and momentum work

**Observed:** [`_compute_unconstrained_velocity()`](../../_src/solvers/implicit_mpm/solver_implicit_mpm.py) separately integrates particle momentum, the APIC contribution when enabled, and mass. A further kernel computes unconstrained grid velocity and inverse mass.

**We will investigate:** accumulate mass and momentum together and include the APIC term in the same particle traversal. Reuse particle position, density, weights, and cell/node lookup results within that traversal.

**Tradeoff:** fewer passes do not automatically mean faster execution. A fused kernel can use more registers or increase atomic contention. Compare a fused implementation against the current implementation for both compact sand beds and scattered particles, at identical transfer settings.

### 5. Reuse explicit temporaries rather than adding another general allocator

**Observed:** much of the solver already uses `fem.borrow_temporary(..., temporary_store=...)` and releases those buffers back to a store. Persistent grid and scratchpad objects also already exist. These calls must not all be reported as fresh GPU allocations.

Concrete explicit allocation sites worth measuring include:

- `_build_strain_eigenbasis()`: a BSR copy, eigenvector storage, and rotated volume storage for applicable strain bases.
- `_apply_strain_eigenbasis()`: temporary strain and compliance matrix values when rotation is required.
- `_update_particles()`: in-place state clones described above.
- `solve_rheology()`: a velocity clone for a configured Jacobi warm-start smoother.
- `compute_diagonal_factorization()`: arrays for Jacobi mass splitting, plus empty placeholder arrays on the nonsplit path.
- `build_rigidity_operator()`: triplet arrays and two BSR matrices, described in item 11.

**We will investigate:** retained buffers or the existing temporary store, sized by capacity and reused after checking shape/dtype/device. Measure allocator calls, copies, and zeroing separately. Even when allocation is pooled or captured, the memory traffic from initializing an array remains relevant.

**Scope check:** the strain eigenbasis work is conditional. It is not evidence that every default `P0` sand step performs a large eigendecomposition and matrix rotation.

### 6. Reuse topology and coloring when occupied connectivity is unchanged

**Observed:** `_particles_to_cells()` rebuilds a capacity-bounded sparse grid in place, reconstructs an active partition, and creates particle quadrature. `ImplicitMPMScratchpad.rebuild_function_spaces()` refreshes retained topologies and builds the required partitions/restrictions. `_compute_coloring()` performs color assignment, radix sorting, and run-length encoding when coloring is enabled.

Coloring groups strain updates that can run together without conflicting writes during Gauss–Seidel iteration.

**We will investigate:** distinguish changes in particle positions/weights from changes in occupied cells and connectivity. Retain topology, restrictions, and color ordering when their dependencies are unchanged. Start with fixed-grid connectivity, then evaluate a sparse topology-change marker.

**Correctness checks:** particles can change cells without changing the union of occupied cells. Their cell membership, quadrature, and matrix values still need updating. Particle-based spaces can also change when occupancy counts change. Do not skip the entire setup merely because the robot or patch center has not moved. A topology-change check must cost less than the work it saves and should avoid introducing a CPU wait each step.

### 7. Separate sparse matrix structure construction from coefficient updates

**Observed:** `_build_plasticity_system()` assembles the strain matrix; `_build_elasticity_system()` conditionally assembles compliance. Subgrid contact also builds a collider matrix. The iterative solver requests strain and collider transposes when needed. The transpose-ready flags are held in solver objects constructed for the solve.

**We will investigate:** retain row offsets, column indices, and an original-to-transpose value mapping while structure is unchanged. Update values directly instead of repeating triplet processing, sorting, or transpose structure generation. Inspect the installed Warp FEM/BSR implementation first: a call to `fem.integrate()` does not by itself prove all symbolic work is repeated.

**Correctness checks:** fixed topology alone may not imply an unchanged matrix pattern if assembly drops numerical zeros or material activity changes. Define an explicit stable pattern, invalidate it correctly, and measure the memory cost of any retained zero blocks.

### 8. Reduce scans and clears over unused reserved capacity

**Observed:** the rebuildable sparse path uses capacity-sized grid buffers and an explicit partition to mask unused cells. `_create_geometry_partition()` borrows an active-cell mask with `grid.cell_count()` entries and clears it before identifying occupied cells. `_update_grid_point_mask()` processes the particle flags each step.

**We will investigate:** trace the distinction between valid counts and reserved lengths through partition building, topology rebuilding, coloring, matrix assembly, and reductions. Use device-side valid counts or compact lists where they avoid meaningful work. Generation counters or touched-entry lists may avoid clearing a large mask, but introduce their own storage and bookkeeping.

**Measurement:** hold the physical scene constant and sweep reserved capacity only. Separately record occupied cells, partition sizes, allocated bytes, and kernel launch dimensions. This reveals capacity overhead without confusing it with more simulated sand.

Reducing capacity is not itself a scalable algorithm. The solver must still detect overflow and remain correct as the material moves or spreads.

### 9. Improve useful warm starts and reduce iterations

**Observed:** rebuilt sparse topology invalidates ordinary previous-grid lookup. `_warmstart_fields()` explicitly avoids those grid-to-grid transfers; some paths clear impulses or stresses instead. The code already supports particle-backed stress warm starts, and `auto` selects that mode for rebuildable sparse grids with `P1d` or `Q1d` strain bases.

**We will investigate:** measure setup cost plus iterations for the supported warm-start modes. If repeated cold starts dominate, preserve a valid mapping through particle state or world-cell coordinates. Contact impulse reuse must also track collider identity and changing contact normals.

**Correctness checks:** prevent stress/impulse leakage between worlds, recycled particles, or unrelated cells. Respect resets and material/property changes. A warm start that costs less per iteration but causes worse convergence is not an improvement.

### 10. Reduce unnecessary collider distance queries

**Observed:** [`collision_sdf()`](../../_src/solvers/implicit_mpm/rasterized_collisions.py) iterates over the relevant collider list for each query. There is already world-specific filtering. `_query_collider_sdf()` performs the geometric query for each candidate. `_rasterize_colliders()` also computes collider-node quantities and may interpolate normals or assemble subgrid mappings.

**We will investigate:** add spatial candidate lists or conservative bounds before expensive queries. Cache geometry-dependent data for a static support mesh while its sample positions and query semantics remain unchanged. Measure candidate count and BVH-query time separately.

**Correctness checks:** reject a collider only when its bound proves it cannot affect the required result. A nearest signed-distance query is not interchangeable with a simple contact-radius test. Account for margins, motion during a timestep, deformable meshes, world ownership, and moving sample locations. A separate no-collider fast path could skip work, but must initialize all downstream fields consistently.

### 11. Avoid generic sparse assembly for simple rigid contact mappings

**Observed:** [`build_rigidity_operator()`](../../_src/solvers/implicit_mpm/rasterized_collisions.py) allocates triplets with up to three blocks per collider node and constructs `J` and `IJtm` using `bsr_from_triplets()`. `apply_rigidity_operator()` uses two BSR matrix-vector products to apply the coupling.

**We will investigate:** retain triplet/matrix storage first. Then compare direct indexed node-to-body accumulation and body-to-node evaluation with the generic sparse representation. The direct representation may suit the small fixed number of contributions per node.

**Scope check:** `_build_collider_rigidity_operator()` skips this work unless finite-mass collider coupling is needed. Verify that the benchmark actually enters this path; an external proxy coupling configuration may have a different cost distribution. Deformable colliders and shared body reductions require separate validation.

### 12. Tune sparse layout, color scheduling, and local factorization

**Observed:** the solver already offers several GS/Jacobi implementations, including SoA and batched variants. `_DelassusOperator` computes a local diagonal factorization used by the iterative solve. This is not a codebase with only one naive sparse matrix-vector kernel.

**We will investigate:** profile memory throughput, register pressure, occupancy, row-length distribution, and kernel launch count for the existing variants. Compare cost per iteration **and** iterations to the same residual. Investigate repeated matrix reads, tiny color batches, and redundant local factorization only after identifying the dominant kernels.

**Correctness checks:** numerical coefficients change with particle weights, material state, and timestep even if matrix structure is constant. Reusing a factorization across steps requires stronger conditions than unchanged topology. Do not introduce approximate reciprocals or reduced precision without force/convergence validation.

### 13. Evaluate a matrix-free material operator

**Idea requiring a prototype:** compute strain/velocity operator products from local grid stencils or particle interpolation data instead of assembling and reading a global strain matrix.

**Potential benefit:** less sparse matrix storage and assembly. **Potential cost:** recomputing weights/connectivity for every iteration, extra gathers/scatters, and harder integration with colored GS and its local preconditioner. This is an algorithmic project, not a simple replacement of `bsr_mv()`.

Start with one supported basis combination and compare full-step cost, including setup. Retain the assembled implementation as a numerical reference. Linear Krylov methods are not drop-in replacements for contact-constrained sand: the existing code explicitly rejects its linear-solver path when colliders are active.

### 14. Avoid solving already-converged environments repeatedly

**Observed:** the implementation has environment offsets, per-environment tolerance scaling, and a batched stopping-condition path. A global solve still has to continue until the required environments satisfy the stopping condition.

**We will investigate:** first inspect which existing iteration kernels already mask completed work. If converged environments still execute the expensive updates, propagate an active-environment mask through material and contact iterations. Compare this with grouping environments by expected difficulty.

**Correctness checks:** material and contact iterations are coupled, so an environment cannot be marked complete from an obsolete material residual alone. Shared colliders and accidental cross-world matrix entries invalidate independent stopping. Measure throughput and worst-environment convergence, not just average residual.

## Things we should not call unnecessary without evidence

- **Zero-mass boundary particles:** they can provide support through transfer/contact behavior. Removing their active flag changes the boundary condition; it is not an equivalent performance optimization.
- **Non-contacting material:** it can deform, carry stress, and influence contact indirectly. Skipping it requires a deliberate model of the omitted material.
- **Array clearing before atomic accumulation:** currently required in the accumulating particle-update path.
- **All temporary allocations:** many already borrow pooled storage. Retained capacity still costs memory, but allocation syntax alone does not identify runtime allocation overhead.
- **Initialization readbacks:** `_rebuild_capacity()` inspects initial voxels on the CPU. That is different from a readback in every rebuilt-grid step. Dense-grid bounds readbacks and dynamic sparse allocation also belong to specific branches, not every grid mode.
- **Rendering-only work:** particle surfaces and grain rendering have separate entry points. Do not attribute their cost to the normal material solve unless the application calls them inside the measured interval.
- **Disabling GIMP, APIC, elasticity, contact detail, or reducing iterations:** these may change accuracy or physical behavior. Report them as configuration/quality tradeoffs, separately from equivalent implementation improvements.

## Measurement and implementation order

1. **Establish a stage profile.** Use the existing timer sections to locate work, then GPU profiling to separate launches, memory operations, graph construction, and synchronization. Keep verbose residual logging off in the timing run. Compare instrumented and uninstrumented runs because timing itself can affect execution.
2. **Use controlled workloads.** Include a settled bed, a moving box, a robot with many colliders, and a moving-patch boundary transition. Add several independent environments for batching candidates. Hold material, density, voxel size, particles per cell, timestep, solver tolerance, and reserved capacities constant within each comparison.
3. **Measure complete steady-state steps.** Record mean ± standard deviation over repeated runs, GPU stage time, CPU wall time, VRAM, iteration count, and final residual. Separate initialization/JIT/capture from steady-state execution, but count graph recapture if it happens every step. Report ordinary frames and boundary-transition frames separately as well as their combined average.
4. **Start with items 1–5.** They target concrete repeated work without requiring a new MPM formulation. Implement one change at a time and retain a reference path for comparison.
5. **Proceed to items 6–9 if setup or iteration count dominates.** Validate invalidation logic before relying on cached topology, matrix structure, or warm starts.
6. **Use items 10–14 where profiles justify them.** Collider-heavy scenes, sparse algebra, and thousands of environments can have different bottlenecks.

For every implementation, compare settled height, penetration, contact impulse/force, particle state, and convergence against the reference. Include material yielding, changing particle mass/flags, particles crossing cells, patch shifts, sparse overflow, and independent-world resets. Floating-point reduction order may change; use justified numerical tolerances rather than demanding bitwise equality.

The first decision should come from the profile: **does time go into grid/assembly setup, iterative solving, particle updates, or CPU waiting?** That determines which candidate can improve the complete simulation rather than only a small kernel.
