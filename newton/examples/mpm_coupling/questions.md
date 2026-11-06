* This part produces (1000, 3) arrys, where is that coming from? Is it collision between box and ground and max collision pairs is specified somewhere? 
```python
print(self.contacts.rigid_contact_point0.numpy().shape)
print(self.contacts.rigid_contact_point1.numpy().shape)
```

* how do you get MPM's state and contact properties like
    * MPM material point position in world 
    * Contact Point
    * Contact Force 
---

## Answers

### 1. Why `(1000, 3)`?

That is the *capacity*, not the number of contacts. `CollisionPipeline` allocates every
rigid-contact buffer with `rigid_contact_max` slots, and `rigid_contact_point0/1` are
`wp.vec3`, so `.numpy()` gives `(rigid_contact_max, 3)`.

You did not pass `rigid_contact_max`, so it is auto-estimated by
`_estimate_rigid_contact_max()` (`newton/_src/sim/collide.py:707`). This scene has one box
shape plus the ground plane, i.e. one contact pair x 5 contacts per primitive pair = 5
contacts. That falls below `_RIGID_CONTACT_MIN_CAPACITY = 1000`
(`newton/_src/sim/collide.py:559`), so the floor wins. Verified with the same
box + ground plane: `rigid_contact_max == 1000`.

So it is not "max collision pairs"; it is the minimum allocation floor.

How to use it correctly:

```python
n = int(self.contacts.rigid_contact_count.numpy()[0])   # contacts actually found this frame
pts0 = self.contacts.rigid_contact_point0.numpy()[:n]   # only first n slots are valid
shape0 = self.contacts.rigid_contact_shape0.numpy()[:n]
shape1 = self.contacts.rigid_contact_shape1.numpy()[:n]
normal = self.contacts.rigid_contact_normal.numpy()[:n] # points shape0 -> shape1
```

Two caveats:

* `rigid_contact_point0/1` are in **body-local** frames
  (`newton/_src/sim/contacts.py:266-269`). For world-space points use
  `newton.eval_rigid_contact_kinematics(model, state, contacts,
  out_point0_world=..., out_point1_world=...)`.
* To shrink the allocation, pass it explicitly:
  `newton.CollisionPipeline(self.model, soft_contact_max=0, rigid_contact_max=64)`.

Note also that in this example these contacts only feed the `mujoco` entry
(`use_mujoco_contacts=False`); the MPM proxy has `collision_pipeline=lambda _model: None`,
so no `Contacts` are generated on the MPM side at all.

### 2. Getting MPM state and "contact" quantities

**Material point positions in world.** The MPM entry is `in_place=True`, so the particles
live directly in the shared state - no mapping needed:

```python
q = self.state_0.particle_q.numpy()    # (particle_count, 3), world space [m]
v = self.state_0.particle_qd.numpy()   # world-space velocity [m/s]
```

`render()` already does this via `newton.examples.get_coupled_view_state(self).particle_q`,
which is the right accessor if you switch the coupled render view away from "combined".

Extra MPM per-particle fields hang off the state's `mpm` extended-attribute namespace
(allocated because of `SolverImplicitMPM.register_custom_attributes(builder)`):

```python
self.state_0.mpm.particle_stress           # Cauchy stress mat33 [Pa]
self.state_0.mpm.particle_qd_grad          # velocity gradient mat33 (APIC)
self.state_0.mpm.particle_elastic_strain   # elastic deformation gradient mat33
self.state_0.mpm.particle_transform        # per-particle frame (rendering/projection)
self.state_0.mpm.particle_Jp               # plastic volume ratio
```

**Contact point and contact force.** MPM never produces a `Contacts` object - there are no
per-pair contact points. Coupling happens through grid impulses on the rasterized
colliders:

```python
impulses, positions, collider_ids = self.mpm_solver.collect_collider_impulses(state)
```

(`newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py:2176`)

* `positions` - world-space application points [m]; these are the closest thing to
  "contact points".
* `impulses` - world-units impulses [N.s] applied to the collider by the material.
* `collider_ids` - collider index; map to a model body with
  `self.mpm_solver.collider_body_index` (value `-1` when the collider is not a body).

Force is simply `impulse / dt` - that is exactly what the harvest kernel does
(`solver_implicit_mpm.py:4003`). Net wrench on a body:

```python
imp, pos, cid = self.mpm_solver.collect_collider_impulses(self.state_0)
imp, pos, cid = imp.numpy(), pos.numpy(), cid.numpy()
body_of = self.mpm_solver.collider_body_index.numpy()

body = 0
sel = cid >= 0
sel[sel] = body_of[cid[sel]] == body
f = imp[sel].sum(axis=0) / self.frame_dt                      # net force [N]
com = self.state_0.body_q.numpy()[body][:3] + self.model.body_com.numpy()[body]
tau = np.cross(pos[sel] - com, imp[sel] / self.frame_dt).sum(axis=0)   # net torque [N.m]
```

Two caveats: call it *after* a step (before the first step it falls back to the
preallocated scratchpad and returns zeros), and it cannot be called from inside the
captured graph - do it in `render()`/`step()` like the existing `show_impulses` path.
`newton/examples/mpm/example_mpm_twoway_coupling.py:218` is the reference usage.

**Rigid-side contact forces** (the MuJoCo half) are a separate mechanism: request the
extended attribute before allocating contacts, then ask the solver to fill it.

```python
self.model.request_contact_attributes("force")   # before pipeline.contacts()
...
self.solver.solver("mujoco").update_contacts(self.contacts)
forces = self.contacts.force.numpy()[:n]         # spatial forces
```

I have not run this path in this coupled configuration (`use_mujoco_contacts=False`), so
check it produces non-zero values before relying on it.

### 3. Why do the impulse positions look like a grid around `particle_q`?

Because they are a grid. `collect_collider_impulses()` returns
`state.collider_position_field.dof_values` verbatim
(`solver_implicit_mpm.py:2192`), and that field is filled in
`rasterize_collider()` (`rasterized_collisions.py:715-724`) with:

```python
collider_position_field.dof_values.fill_(wp.vec3(fem.OUTSIDE))
fem.interpolate(world_position, dest=collider_position_field,
                at=collider_space_restriction, ...)
```

i.e. each collision-space DOF is assigned its own **background-grid node position**,
not a point on the collider surface. Implicit MPM resolves contact by rasterizing the
collider SDF onto grid nodes, so there is no per-pair contact point.

Three practical consequences:

* The array length is `collision_space_partition.node_count()` - every collision node,
  including ones nowhere near a collider. Those get `collider_ids == -1` and zero
  impulse (`rasterized_collisions.py:488-496`).
* Nodes outside the domain keep the sentinel `fem.OUTSIDE == -1e8` from the `fill_`.
  Plotting raw positions therefore includes points at (-1e8, -1e8, -1e8).
* Node spacing is `voxel_size / 2` = 0.025 m here, not 0.05, because the default
  `collider_basis` is `"S2"` (degree-2, nodes at cell corners *and* edge midpoints,
  `solver_implicit_mpm.py:913`).

Filter to the nodes that actually carry contact - they lie within ~0.5 voxel of the
collider surface (`_COLLIDER_ACTIVATION_DISTANCE = 0.5`):

```python
imp, pos, cid = self.mpm_solver.collect_collider_impulses(render_state)
imp, pos, cid = imp.numpy(), pos.numpy(), cid.numpy()
active = (cid >= 0) & (np.linalg.norm(imp, axis=1) > 0.0)
pos, imp = pos[active], imp[active]   # hugs the box surface instead of filling the grid
```

The existing `show_impulses` path does not filter; inactive nodes happen to draw
zero-length lines, so it mostly hides the issue except for the `OUTSIDE` sentinels.

### 4. What does "rasterize" mean exactly?

Sampling the collider's signed-distance field at every background-grid node and storing
per-node contact data - the same sense as graphics rasterization (continuous geometry ->
values on a fixed lattice). There are no predefined collision points on the collider.

**1. Every collider becomes a triangle mesh.** `_get_shape_mesh()`
(`implicit_mpm_model.py:89`) tessellates each shape at model build time:
`BOX -> Mesh.create_box`, `SPHERE -> create_sphere`, plus capsule/cylinder/cone, and
`PLANE -> a 1000x1000 quad`. All shapes on a body are merged into one mesh
(`_create_body_collider_mesh`, line 218) and uploaded as a `wp.Mesh` BVH in
`Collider.collider_mesh`. There is no analytic primitive SDF path - the box is queried as
a mesh.

**2. Per-node SDF query.** `_query_collider_sdf()` (`rasterized_collisions.py:175`), for
each collision node `x`:

* transform `x` into the body frame,
* `wp.mesh_query_point_no_sign(mesh, x_local, query_max_dist + thickness)` -> closest
  surface point `cp`,
* sign the distance from the *average* face normal around `cp`
  (`_SDF_SIGN_FROM_AVERAGE_NORMAL`; more robust than raycast signs),
* `sdf = sign * |x - cp| - material_thickness`, `sdf_grad` = normalized offset = contact
  normal,
* evaluate collider velocity at `cp` (mesh velocity + rigid-body velocity, backward
  differenced from `body_q_prev` or forward from `body_qd`).

`collision_sdf()` loops over the colliders in the node's world and keeps the **minimum**
sdf, so each node is assigned at most one collider - that is the `collider_ids` array.

**3. Activation.** A node participates only if `sdf < activation_distance * voxel_size`,
with `activation_distance = 0.5 / degree` (`rasterized_collisions.py:727`) - 0.25 voxels
for the default `S2` basis. Inactive nodes get `friction = -1`, `collider_id = -1`, and
are skipped downstream.

**4. Nodal contact solve.** Each active node is one contact constraint on the grid
momentum system (`solve_nodal_friction`, `contact_solver_kernels.py:126`):

```python
u0 = velocities[i] - collider_velocities[i]     # relative velocity at the node
w  = inv_mass[i] + collider_inv_mass[i]         # node + collider effective inverse mass
u  = solve_coulomb_isotropic(friction, n, u0 - (impulse[i] + adhesion * n) * w)
delta_lambda = (u - u0) / w
impulse[i] += delta_lambda
velocities[i] += inv_mass[i] * delta_lambda
```

a Coulomb-cone projection iterated Gauss-Seidel-style (graph coloured) inside
`solve_rheology`, jointly with the plastic/rheology strain solve. `collider_inv_mass` is
nonzero only for compliant/dynamic colliders via `build_rigidity_operator` - that is what
makes coupling two-way. When `collider_basis != velocity_basis`, the sparse
`collider_matrix` (from `collision_weight_field`) maps collision nodes onto velocity DOFs
and `apply_subgrid_impulse` spreads the impulses.

**5. Output.** The accumulated per-node `impulse_field` is exactly what
`collect_collider_impulses()` returns and what is harvested into body wrenches / proxy
particle forces.

**6. Particle-level safety net (opt-in).** `project_outside_collider`
(`rasterized_collisions.py:344`) runs a per-particle SDF query and pushes particles that
would end up inside a collider back out with a Coulomb velocity response. It is **not**
part of `_step_impl`: the caller invokes `solver.project_outside(state, state, dt)`
explicitly after `step()`, as the `newton/examples/mpm/*` examples do.
`example_mujoco_mpm_coupled_solver.py` does not call it.

Implication: contact is resolved only to grid resolution. Collider features thinner than a
voxel, or gaps narrower than one, are invisible to the grid solve; the particle projection
pass is what prevents leakage - when you call it. That is the trade-off of rasterized contact versus the
pairwise narrow phase used on the rigid side.

### 5. Are collision nodes the same as the MPM grid?

Same grid *geometry*, different *function space*. All three bases are built on one
`fem.Geometry` (`self.grid`, `solver_implicit_mpm.py:328-344`):

| space    | default basis            | nodes per hex cell             |
|----------|--------------------------|--------------------------------|
| velocity | `Q1` (trilinear Lagrange)| 8, at voxel corners            |
| collider | `S2` (quadratic serendipity) | 20, at corners + edge midpoints |
| strain   | `P0`                     | 1, cell interior               |

Collision nodes therefore sit on the same voxel lattice and, with the defaults, are a
*superset* of the velocity nodes: all corners plus the 12 edge midpoints per cell.

Special cases:

* `collider_basis="Q1"` - `_create_collider_function_space` short-circuits with
  `self._collision_space = self._velocity_space` (line 392-396); collision nodes are then
  literally the MPM velocity nodes.
* `collider_basis="pic"` / `"pic8"` / `"pic27"` - the basis is a `PointBasisSpace` over the
  particle quadrature, so nodes are at particles, not on the grid.

This is why `collider_matrix` exists: when the collider basis differs from the velocity
basis, nodal impulses must be mapped onto velocity DOFs (`collision_weight_field` ->
`apply_subgrid_impulse`). With `Q1` that mapping is skipped.

Correction to section 3: serendipity omits face-center and cell-interior nodes, so the
collision nodes are not a uniform `voxel_size/2` lattice. Nearest-neighbour spacing along
cell edges is 0.025 m here, but there are no nodes at face or cell centres.

### 6. Terminology: "collider" = the object, never the grid

In this solver **"collider" always means the object** - a rigid body's merged collision
mesh (your box, the ground plane). The grid sample points are **collision nodes**. The
confusion is real, because many arrays named `collider_*` are indexed by *node*, not by
object:

| Indexed by collider **object** (length = collider count) | Indexed by collision **node** (length = node count) |
|---|---|
| `collider_mesh`, `collider_max_thickness` | `collider_ids` |
| `collider_body_index`, `collider_world` | `collider_position_field`, `collider_distance_field` |
| `collider_particle_offsets`, `collider_face_offset` | `collider_normal_field`, `collider_velocity` |
| `collider_total_volumes` | `collider_friction`, `collider_adhesion`, `collider_node_volume`, `impulse_field` |

So `collider_ids[node]` is a **node -> object** map: "which object is nearest to this
node", or -1 for none. `collision_sdf()` loops over *objects* for one node and keeps the
minimum signed distance; the kernel launch around it loops over *nodes*.

Figures: `rasterize_pipeline.png` (the four stages) and `rasterize_vs_pairwise.png`
(pairwise narrow phase vs rasterized contact, and what voxel size buys).

### 7. `inv_mass` vs `collider_inv_mass` in `solve_nodal_friction`

Inverted from the obvious reading. At the call site (`solve_rheology.py:1649-1663`):

* `inv_mass` <- `momentum.inv_volume` = `scratch.inv_mass_matrix`: one value per **MPM
  velocity node**, the inverse of that node's material mass.
* `collider_inv_mass` <- the **object's** effective inverse mass felt at that node, from
  `compute_collider_inv_mass` (`contact_solver_kernels.py:11`). It is `zero_()`'d when
  there is no rigidity operator, i.e. all colliders are effectively infinite mass
  (`min_collider_mass == INFINITY`) - the one-way-coupling case.

`w = inv_mass + collider_inv_mass` is the Delassus diagonal of the contact;
`delta_lambda = delta_u / w` is the standard contact-solver step.

**Does the grid have mass?** Yes - ordinary MPM lumped P2G mass:

1. `model.particle_mass` from the builder (here `cell_volume * 2500`).
2. `particle_volume = 8 r**3`, `particle_density = particle_mass / particle_volume`
   (`implicit_mpm_model.py:185-197`).
3. Node mass assembled by `fem.integrate(integrate_mass, quadrature=pic, ...)`, which
   returns `phi(s) * density * inv_cell_volume` - the FEM spelling of
   `m_i = sum_p w_ip * m_p`.
4. `free_velocity` (`implicit_mpm_solver_kernels.py:190`) divides and stores the
   reciprocal:

```python
inv_particle_mass = 1.0 / (pmass + drag)     # drag = air_drag * dt
velocity_avg[i] = velocity_int[i] * inv_particle_mass
inv_mass_matrix[i] = inv_particle_mass
```

* `drag` regularizes empty nodes (`pmass == 0` would otherwise be a division by zero), so
  air nodes move nearly freely.
* Kinematic particles (`density == 0`) map to `INFINITY` mass in `integrate_mass`
  (line 134-136), so `inv_mass ~ 0` and the grid velocity there is prescribed.

**Units.** It is not kg but mass *per cell volume* - `integrate_mass` carries
`inv_cell_volume`, and the field is named `inv_volume` in `MomentumData` for that reason.
That is exactly why `collect_collider_impulses()` multiplies by `voxel_size**3` to return
world units.

**Gotcha.** Density is derived from `8 r**3`, not from the emit spacing. That equals the
emit cell volume only for cubic cells. In this example all three `cell_size` components
are 0.016667 so the density is exactly 2500; anisotropic `cell_x/cell_y/cell_z` with
`radius_mean = max(cell_size)/2` would silently give a different effective density than
the `density` value you passed.

### 8. One MPM step, end to end

**Once, at `SolverImplicitMPM(...)` construction:** tessellate every collider object to
triangles, merge per body, upload as `wp.Mesh` + BVH, pack into `Collider`. Not repeated
per step (only deformable collider meshes get their points refreshed and refit).

**Per `step(state_in, state_out, control, contacts, dt)`:**

*Phase 0 - grid and binning* (`step` -> `_particles_to_cells`, line 2680), driven by
particles, not colliders:

* `fixed` grid reused as-is; rebuildable `sparse` calls `grid.rebuild(positions)`; plain
  `sparse` reallocates around the current particle cloud.
* Build the active-cell partition, bin particles into cells -> `PicQuadrature`.
* `_rebuild_scratchpad(pic)` rebuilds the velocity / collider / strain function spaces and
  allocates node arrays.

*Then `_step_impl` (line 2875), in order:*

1. `_rasterize_colliders` - per collision node, closest-point query against each object
   mesh, nearest wins; writes sdf, normal, object velocity, friction, adhesion,
   `collider_id`, node volumes.
2. `_compute_unconstrained_velocity` - particle-to-grid. Node momentum (+ `dt * gravity`)
   and node mass are integrated, then divided -> unconstrained velocity and
   `inv_mass_matrix`. This is where the grid acquires mass.
3. `_build_collider_rigidity_operator` - only when a collider has finite mass; enables
   two-way push-back.
4. `_build_elasticity_system` + `_build_plasticity_system` - strain matrix, compliance,
   yield-surface parameters.
5. `_load_warmstart` - previous step's impulses and stresses as initial guess.
6. `_solve_rheology` - implicit solve alternating the stress/plasticity projection with
   the nodal Coulomb contact solve (`solve_nodal_friction`). Outputs final node velocities
   and `impulse_field`.
7. `_save_for_next_warmstart`.
8. `_update_particles` - grid-to-particle: `update_particle_strains`, then
   `advect_particles` (velocity + APIC gradient interpolation, `x += v*dt`, clamped by
   `particle_max_velocity`).
9. `_save_data` - attach `impulse_field`, `collider_ids`, collider position/distance/normal
   fields and `velocity_field` to `state_out`; copy `body_q` to `body_q_prev`.

So: the grid is built around the *particles* first, and the collider is rasterized onto
that existing grid - it has to be that way, since a sparse grid's extent is defined by
where the material is.

**Post-step, caller-driven** (not inside `step()`):

* `collect_collider_impulses()` / the body-wrench harvest - done by `SolverCoupledProxy`
  in this example.
* `project_outside(state, state, dt)` - the anti-leak pass. Corrects section 4: it is a
  separate public call, and this coupled example never makes it.

### 9. What `project_outside_collider` does

Geometric cleanup applied *outside* the implicit solve. Wrapper:
`solver.project_outside(state_in, state_out, dt, gap=None)`
(`solver_implicit_mpm.py:2205`) - optionally overrides `collider.query_max_dist` with
`gap`, then launches the kernel over `particle_count`. Writes only `particle_q`,
`particle_qd`, `mpm.particle_qd_grad`.

Per particle (`rasterized_collisions.py:344`):

1. **Skip** inactive (`~ParticleFlags.ACTIVE`) and kinematic (`particle_mass == 0`)
   particles.
2. **Query** `collision_sdf(pos_adv, ...)` - same closest-point machinery as the grid
   rasterization - for `sdf`, outward normal, collider surface velocity, `material_id`.
3. **Predict end-of-step penetration:**

```python
sdf_end = sdf - wp.dot(sdf_vel, sdf_gradient) * dt + material_projection_threshold[material_id]
if sdf_end < 0:
```

   The middle term accounts for the collider advancing over `dt`. The threshold is allowed
   penetration, default `0.01 * voxel_size` (`_DEFAULT_PROJECTION_THRESHOLD`), which keeps
   particles resting exactly on the surface from jittering.

4. **Coulomb velocity response:**

```python
delta_vel = solve_coulomb_isotropic(friction, sdf_gradient, p_vel - sdf_vel) + sdf_vel - p_vel
p_vel += delta_vel
pos_adv += delta_vel * dt
```

   `solve_coulomb_isotropic` takes relative velocity `u = v_particle - v_collider`; if the
   normal component is approaching, it removes it, then applies the cone - stick
   (`u = 0`) when `|u_t| <= mu * |u_n|`, otherwise slide with tangential speed reduced by
   `mu * |u_n|`.

5. **Push out the remainder:**
   `pos_adv -= min(0, sdf_end + dt * dot(delta_vel, n)) * n` - lands the particle at the
   threshold offset, not exactly on the surface.
6. **Rigidify the velocity gradient:** `0.5 * (vel_grad - vel_grad^T)` keeps only the
   rotational part, so the affine APIC mode does not drive material back into the collider
   on the next transfer.

Two caveats:

* **No reaction on the body.** Only particle arrays are written; no impulse is harvested.
  It is a momentum leak traded for no visible penetration - a safety net, not physics.
* **Deep penetration is not fixed.** The query uses `max_dist = query_max_dist + thickness`,
  default `voxel_size * sqrt(3)` (`implicit_mpm_model.py:430`). A particle deeper than that
  finds no closest point, gets `sdf = INFINITY`, and is skipped. That is what `gap` is for
  (`test_implicit_mpm.py` passes `gap=1.0`).

### 10. `grid_type`: `fixed` vs `sparse` vs `dense`

Three config values plus a derived fourth mode. The geometry class differs, but the
operational difference is **when the grid is allocated**.

| | `"sparse"` (plain) | `"sparse"` rebuildable | `"fixed"` | `"dense"` |
|---|---|---|---|---|
| Geometry | `fem.Nanogrid` | `fem.Nanogrid`, persistent buffers | `fem.Grid3D` | `fem.Grid3D` |
| Extent | voxels containing particles | same | particle bbox at **first** step + `grid_padding`, frozen | particle bbox, recomputed each step |
| Allocated | every step | once, then `grid.rebuild()` in place | once | every step |
| Host sync | no | no | no | **yes** (bounds readback, `solver_implicit_mpm.py:2563`) |
| CUDA graph capture | no | yes | yes | no |
| Memory | proportional to occupied voxels | capacity-bounded | proportional to bbox volume | proportional to bbox volume |

"Rebuildable sparse" is derived, not a config value (line 1469). It needs **all** of:
`grid_type == "sparse"`, `max_active_cell_count > 0`, `grid_padding == 0`,
`velocity_basis == "Q1"`, and a rebuild-safe strain and collider basis. It is the only
mode that is both sparse and graph-capturable - the target if you want to move off
`fixed`. Call `check_sparse_grid_rebuild_status()` after replay to catch overflow.

**The `fixed` trap:** the lattice never grows. A particle that leaves it gets
`NULL_ELEMENT_INDEX` from `fem.lookup` in `mark_active_cells` (line 1283-1286), marks no
cell, never enters the `PicQuadrature`, contributes no mass, and is never advected - it
freezes in place. `grid_padding = 50` at `voxel_size = 0.05` buys 2.5 m of headroom around
the initial bed, which is why this example works.

Note `fixed` still rebuilds the **active partition** every step from particle positions
(`_create_geometry_partition`, line 2602) - only the lattice is frozen. That is why
`collect_collider_impulses()` positions track the material (section 3): the returned nodes
are the nodes of cells that currently contain particles, and node indices are not stable
across steps.

Figure: `mpm_step_flow.png` - the whole step, stage by stage.

### 11. Switching this example to a sparse grid

Setting `grid_type = "sparse"` alone is not enough. With `grid_padding = 50` still set,
`_sparse_rebuildable` (line 1469) evaluates False and the solver silently falls back to
**plain** sparse, which allocates a fresh NanoVDB volume every step. That cannot run inside
the CUDA graph this example captures in `__init__`:

```
Warp CUDA error 900: operation not permitted when stream is capturing (wp_cuda_stream_synchronize)
RuntimeError: Failed to create volume
  _particles_to_cells -> _allocate_grid:2518 -> allocate_by_voxels
```

Nothing warns that the rebuildable path was declined - that is the trap.

**Fix applied:** `mpm_config.grid_padding = 0`. Verified 30 frames headless, exit 0, and:

```
grid_type=sparse  grid_padding=0  max_active_cell_count=32768
velocity_basis=Q1  strain_basis=P0  collider_basis=S2
rebuildable=True  rebuild_status=OK
```

Padding is not a loss here: it exists to give a *static* grid headroom so material cannot
escape it (the `fixed` trap in section 10). A rebuildable grid is rebuilt around the
particles every step, so the padding would be wasted voxels - which is why the mode
requires zero.

Remaining requirements, all already satisfied by this config:

* `max_active_cell_count > 0` and large enough. 32768 covers the 2x2x0.5 m bed at 0.05 m
  (~16000 cells) plus the disturbed region. Overflow raises a message naming the knob.
* `velocity_basis == "Q1"` (default); `strain_basis` in `pic*`/`P0`/`P1d`/`Q1d`/`Q1`;
  `collider_basis` in `pic*`/`Q1`/`S2`/`S3`.
* `warmstart_mode` must not be `"grid"` or `"smoothed"` - rejected at construction for
  rebuildable grids since topology changes in place. `"auto"` resolves to `"particles"`.
* Call `check_sparse_grid_rebuild_status()` **outside** graph capture to detect overflow.
  It synchronizes the device, so prefer `test_final()` or every N frames.

If you ever need plain sparse (e.g. you really want padding), drop graph capture:
`self.graph = _capture_frame_graph(self.model, self.simulate, enabled=False)`.

### 12. `solve_rheology`

The actual physics solve; everything else in the step is assembly around it.

**Problem.** After P2G there is a free velocity `v*`. The end-of-step velocity must satisfy
two nonlinear conditions simultaneously: stress inside the yield surface at every strain
node (with elastic compliance), and the Coulomb cone / non-penetration at every collision
node. Both are second-order-cone complementarity problems coupled through the shared grid
velocities.

**Unknowns.** Stress `sigma` per strain node (`vec6`, Voigt) and collider impulse `r` per
collision node. Velocity is dependent: `v = v* + M^-1 (B^T sigma + J^T r)`, where
`B = strain_mat` and `M^-1 = inv_volume`.

**Delassus operator.** `W = B M^-1 B^T + compliance` maps a stress change to the resulting
strain-rate change - the stress-space analogue of `w = inv_mass + collider_inv_mass`
(section 7). Only its block diagonal is formed, then eigendecomposed per node into
`delassus_rotation` (mat55) and `delassus_diagonal` (vec6) so each local 6-DOF problem is
solved in a decoupled basis (`compute_delassus_diagonal`, line 502).

**Loop** (`_run_solver_loop`, line 1785):

```python
while condition:                 # evaluated on device
    for _k in range(5):          # solve_granularity
        contact_solver.solve()   # one Coulomb sweep over collision nodes
        rheology_solver.solve()  # one sweep over colors: local stress projection
    residual = rheology_solver.eval_residual()
    update_condition(...)        # clears `condition` at res < tol or max_iterations
```

Alternating block Gauss-Seidel between the two physics, each half itself a GS sweep over
nodes. `wp.capture_while` makes it a **conditional CUDA graph**: loop, residual reduction
and termination test all run on device with no host sync - which is what lets the whole
frame be captured. Convergence is checked every 5 iterations, so `max_iterations = 50`
allows up to 10 checks.

**One GS sweep** (`_GaussSeidelSolver.solve`): one launch per color. Coloring guarantees
nodes in a color share no velocity DOF, so the sequential-in-spirit update runs in parallel
within a color. Persistent-kernel launch shape (`dim = sm_count*2*64`,
`max_blocks = sm_count*2`) sized to the GPU, not the data. `make_gs_solve_kernel`
specializes on `has_viscosity` / `has_dilatancy` / `has_compliance_mat`, so unused physics
compiles away. Per node: local strain rate -> stress increment projected onto the yield
surface in the rotated eigenbasis -> immediately scattered to neighbouring velocity DOFs
(outputs `[velocity, stress, delta_stress]`). That write-back is the Gauss-Seidel part.

**Contact half.** `_NodalContactSolver` (one constraint per node, `solve_nodal_friction`)
or `_SubgridContactSolver` when `collider_mat.nnz > 0`. After each sweep,
`apply_rigidity_operator` pushes the impulse delta onto body velocities - two-way coupling
closes inside the solve, not after it.

**Variants.** `Config.solver`, default `"auto"` -> `"gs"` for a `Q1` velocity basis,
`"gs-batched"` for B2/B3.

* `gs` / `gs-soa` / `gs-batched` - Gauss-Seidel flavours; batched is Jacobi-like across
  batches with mass splitting.
* `jacobi` - fully parallel, built with `split_mass=True` (each node's inverse mass divided
  by the number of strain nodes touching it) to stay stable.
* `cg` / `cr` / `gmres` - **linear** solvers; they cannot handle contact and raise if
  colliders are present. Useful contact-free or as a warm-start prefix.

Sequence grammar: `[linear, ][jacobi, ]<final>` - e.g. `("cr", "gs")`, or `("jacobi", "gs")`
for 5 Jacobi smoother iterations first.

**Before / after.** `preprocess_stress_and_strain` projects the warm-started stress onto
the yield surface before iterating; `postprocess_stress_and_strain` rotates stress back to
world space and emits `elastic_strain_delta` / `plastic_strain_delta`, consumed by
`_update_particles`.

**Knobs.** `max_iterations` (50 here) and `tolerance` (default 1e-4). Contact and rheology
share the iteration budget, so an under-converged solve shows up as soft material *and*
soft contact at the same time. `verbose=True` prints the achieved residual and iteration
count.
