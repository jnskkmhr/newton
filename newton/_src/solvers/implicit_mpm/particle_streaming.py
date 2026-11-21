# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed pool of implicit-MPM particles that follows an anchor body in each world."""

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import warp as wp

from .particle_cache_kernels import refresh_cached_properties, update_patch_center


@wp.func
def _surface_height(mesh: wp.uint64, x: float, y: float, ray_z: float, ray_length: float):
    """Cast downward at world XY [m] and return whether the mesh was hit and its height [m]."""
    query = wp.mesh_query_ray(mesh, wp.vec3(x, y, ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
    height = ray_z - query.t
    return query.result, height


@wp.kernel(enable_backward=False)
def _mark_resets(selected: wp.array[wp.bool], pending: wp.array[int]):
    """Mark selected worlds for particle reinitialization."""
    i = wp.tid()
    if selected[i]:
        pending[i] = 1


@wp.kernel(enable_backward=False)
def _update_window_center(
    body_q: wp.array[wp.transform],
    body_indices: wp.array[int],
    world_origins: wp.array[wp.vec3],
    surface_center: wp.vec2,
    surface_size: wp.vec2,
    pool_size: wp.vec2,
    tile_size: float,
    window_centers: wp.array[wp.vec2],
):
    """Snap each anchor's XY position [m] to the tile grid and keep the whole pool on the surface."""
    i = wp.tid()
    p = wp.transform_get_translation(body_q[body_indices[i]])
    o = world_origins[i]
    allowed_half = (surface_size - pool_size) * 0.5
    x = wp.floor((p[0] - o[0]) / tile_size + 0.5) * tile_size
    y = wp.floor((p[1] - o[1]) / tile_size + 0.5) * tile_size
    window_centers[i] = wp.vec2(
        wp.clamp(o[0] + x, surface_center[0] - allowed_half[0], surface_center[0] + allowed_half[0]),
        wp.clamp(o[1] + y, surface_center[1] - allowed_half[1], surface_center[1] + allowed_half[1]),
    )


@wp.kernel(enable_backward=False)
def _recycle_particles(
    particle_world: wp.array[int],
    window_centers: wp.array[wp.vec2],
    template_positions: wp.array[wp.vec3],
    reset_requested: wp.array[int],
    dynamic_mass: wp.array[float],
    initial_plastic_volume_ratio: wp.array[float],
    window_size: wp.vec2,
    pool_size: wp.vec2,
    surface_mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
    reset_positions: wp.array[wp.vec3],
    is_dynamic: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_qd_grad: wp.array[wp.mat33],
    particle_elastic_strain: wp.array[wp.mat33],
    particle_plastic_volume_ratio: wp.array[float],
    particle_stress: wp.array[wp.mat33],
    particle_transform: wp.array[wp.mat33],
    recycled: wp.array[int],
    mass_change_count: wp.array[int],
    surface_miss_count: wp.array[int],
    recycle_count: wp.array[int],
):
    """Wrap particles into the pool, restore the default material state, and zero the mass of the ring.

    ``template_positions`` holds fixed world XY coordinates [m] and particle-layer Z offsets [m].
    ``reset_positions`` holds surface-adjusted world XYZ positions [m]. ``reset_requested`` flags
    request reinitialization of a world. Each counter is a one-element array shared by the launch.
    """
    i = wp.tid()
    world = particle_world[i]
    center = window_centers[world]
    template = template_positions[i]
    # Toroidal wrapping retains overlap and restores fresh material at the entering edge.
    x = template[0] + wp.floor((center[0] + 0.5 * pool_size[0] - template[0]) / pool_size[0]) * pool_size[0]
    y = template[1] + wp.floor((center[1] + 0.5 * pool_size[1] - template[1]) / pool_size[1]) * pool_size[1]
    # Float32 wrapping can round past the pool edge and miss the surface at its boundary.
    x = wp.clamp(x, center[0] - 0.5 * pool_size[0], center[0] + 0.5 * pool_size[0])
    y = wp.clamp(y, center[1] - 0.5 * pool_size[1], center[1] + 0.5 * pool_size[1])
    reset_position = reset_positions[i]
    if wp.abs(reset_position[0] - x) > 1.0e-5 or wp.abs(reset_position[1] - y) > 1.0e-5 or reset_requested[world] != 0:
        hit, height = _surface_height(surface_mesh, x, y, ray_z, ray_length)
        if not hit:
            wp.atomic_add(surface_miss_count, 0, 1)
        reset_position = wp.vec3(x, y, height + template[2])
    dynamic = int(wp.abs(x - center[0]) < 0.5 * window_size[0] and wp.abs(y - center[1]) < 0.5 * window_size[1])
    reset_position_changed = wp.length(reset_position - reset_positions[i]) > 1.0e-5
    escaped = (
        wp.abs(particle_q[i][0] - center[0]) >= 0.5 * pool_size[0]
        or wp.abs(particle_q[i][1] - center[1]) >= 0.5 * pool_size[1]
    )
    # Ring particles are rewritten every step; only a new site or state counts as recycled.
    moved = reset_position_changed or escaped or dynamic != is_dynamic[i] or reset_requested[world] != 0
    recycled[i] = wp.where(moved, 1, 0)
    if moved:
        wp.atomic_add(recycle_count, 0, 1)
    if moved or dynamic == 0:
        particle_q[i] = reset_position
        particle_qd[i] = wp.vec3(0.0)
        particle_qd_grad[i] = wp.mat33(0.0)
        particle_elastic_strain[i] = wp.identity(n=3, dtype=float)
        particle_plastic_volume_ratio[i] = initial_plastic_volume_ratio[i]
        particle_stress[i] = wp.mat33(0.0)
        particle_transform[i] = wp.identity(n=3, dtype=float)
    target_mass = float(dynamic) * dynamic_mass[i]
    if particle_mass[i] != target_mass:
        wp.atomic_add(mass_change_count, 0, 1)
    particle_mass[i] = target_mass
    particle_inv_mass[i] = 0.0
    if dynamic != 0:
        particle_inv_mass[i] = 1.0 / dynamic_mass[i]
    reset_positions[i] = reset_position
    is_dynamic[i] = dynamic


@wp.kernel(enable_backward=False)
def _pin_ring_particles(
    is_dynamic: wp.array[int],
    reset_positions: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    """Return ring particles to their reset positions [m] with zero velocity [m/s]."""
    i = wp.tid()
    if is_dynamic[i] == 0:
        particle_q[i] = reset_positions[i]
        particle_qd[i] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def _mark_arrived_material(
    layers: wp.array[int],
    next_layers: wp.array[int],
    previous_material_flags: wp.array[int],
    previous_density: wp.array[float],
    arrived: wp.array[int],
):
    """Select the slots whose sand became simulated material in the cache update that just ran."""
    i = wp.tid()
    # Layer 1 is the dynamic window. A slot loaded by this update was planned as empty (layer 0).
    loaded = next_layers[i] == 0
    was_material = (previous_material_flags[i] & 1) != 0 and previous_density[i] > 0.0
    arrived[i] = wp.where(layers[i] == 1 and (loaded or not was_material), 1, 0)


@wp.kernel(enable_backward=False)
def _refresh_density(mass: wp.array[float], volume: wp.array[float], density: wp.array[float]):
    """Update particle densities [kg/m^3] from masses [kg] and fixed reference volumes [m^3]."""
    i = wp.tid()
    density[i] = mass[i] / volume[i]


@wp.kernel(enable_backward=False)
def _set_world_dynamic_mass(
    particle_world: wp.array[int],
    selected: wp.array[wp.bool],
    density: wp.array[float],
    radius: wp.array[float],
    dynamic_mass: wp.array[float],
):
    """Store the mass [kg] restored to dynamic particles of the selected worlds."""
    i = wp.tid()
    world = particle_world[i]
    if selected[world]:
        r = radius[i]
        dynamic_mass[i] = density[world] * (8.0 * r * r * r)


class ParticleStreaming:
    """Fixed pool of particles that follows one anchor body per world.

    The pool is a rectangle of the dynamic window plus a ring of width ``margin`` on each side.
    Every particle keeps a lattice site that wraps around the pool, so when the window center moves
    by one ``tile_size`` step, the strip left behind reappears at the leading edge in the default
    material state, placed on ``surface_mesh``. Ring particles are kinematic: they carry zero mass and
    are returned to their sites by :meth:`post_step`. No particle is created or destroyed, and no
    history is kept for sand that leaves the pool.

    Create with :meth:`SolverImplicitMPM.set_particle_streaming`. Call :meth:`update` before each step
    with the state holding the anchor poses and particle kinematics, and :meth:`post_step` after it.
    Both only launch kernels, so they can be recorded in a CUDA graph.

    With particle sleeping enabled, ring particles stay kinematic and recycled particles are woken
    with a fresh quiet timer.

    .. experimental::

        Runtime-only pool management. The interface may change.
    """

    def __init__(
        self,
        solver,
        state,
        *,
        model,
        body_indices: Sequence[int],
        size: Sequence[float],
        margin: float,
        tile_size: float,
        surface_mesh: wp.Mesh,
        world_origins,
        spawn_heights,
    ):
        self.solver = solver
        self.model = model
        self.material_state = state
        if solver.model.particle_count != model.particle_count:
            raise ValueError("Particle streaming requires the MPM solver to own all particles in original order.")
        device = model.device
        world_count = model.world_count
        self.window_size = tuple(float(v) for v in size)
        self.margin = float(margin)
        self.tile_size = float(tile_size)
        if len(self.window_size) != 2 or min(self.window_size) <= 0.0 or self.margin <= 0.0 or self.tile_size <= 0.0:
            raise ValueError("Streaming window size, margin and tile size must be positive.")
        if self.tile_size > self.margin:
            raise ValueError("Streaming tile size must not exceed the margin.")
        self.pool_size = tuple(v + 2.0 * self.margin for v in self.window_size)

        indices = np.asarray(body_indices, dtype=np.int32)
        if indices.shape != (world_count,):
            raise ValueError("Particle streaming needs exactly one anchor body index per world.")
        self.body_indices = wp.array(indices, dtype=int, device=device)

        particle_world = model.particle_world.numpy()
        if np.any(particle_world < 0) or np.any(particle_world >= world_count):
            raise ValueError("Streamed particles must belong to valid worlds.")
        origins = np.asarray(world_origins, dtype=np.float32)
        heights = np.asarray(spawn_heights, dtype=np.float32)
        if origins.shape != (world_count, 3) or heights.shape != (world_count,):
            raise ValueError("World origins and spawn heights must provide one entry per world.")
        self.world_origins = wp.array(origins, dtype=wp.vec3, device=device)
        self.window_centers = wp.array(origins[:, :2], dtype=wp.vec2, device=device)

        self.surface_mesh = surface_mesh
        points = surface_mesh.points.numpy()
        self.surface_bounds = np.stack((points.min(axis=0), points.max(axis=0)))
        self._ray_z = float(self.surface_bounds[1, 2] + 1.0)
        self._ray_length = float(self.surface_bounds[1, 2] - self.surface_bounds[0, 2] + 2.0)
        if np.any(origins[:, :2] - 0.5 * np.asarray(self.pool_size) < self.surface_bounds[0, :2]) or np.any(
            origins[:, :2] + 0.5 * np.asarray(self.pool_size) > self.surface_bounds[1, :2]
        ):
            raise ValueError("The surface mesh must contain every initial particle pool.")

        template = model.particle_q.numpy()
        template[:, 2] -= heights[particle_world]
        self.template_positions = wp.array(template, dtype=wp.vec3, device=device)
        self.reset_positions = wp.clone(model.particle_q)
        self.particle_is_dynamic = wp.ones(model.particle_count, dtype=int, device=device)
        self.dynamic_mass = wp.clone(model.particle_mass)
        if np.any(self.dynamic_mass.numpy() <= 0.0):
            raise ValueError("Initial particle masses must be positive; streaming assigns zero mass to the ring.")
        self.initial_plastic_volume_ratio = wp.clone(state.mpm.particle_Jp)
        self.reset_requested = wp.ones(world_count, dtype=int, device=device)
        self.recycled = wp.zeros(model.particle_count, dtype=int, device=device)
        self._counts = wp.zeros(3, dtype=int, device=device)
        # Mass changes and recycles control GPU branches; surface misses accumulate until check_status().
        self.mass_change_count = self._counts[:1]
        self.surface_miss_count = self._counts[1:2]
        self.recycle_count = self._counts[2:]

    def update(self, state) -> None:
        """Move the pools to their anchors and recycle particles before one solver step.

        Args:
            state: State holding the anchor ``body_q`` and the particle kinematics to update.
        """
        model = self.model
        bounds = self.surface_bounds
        wp.launch(
            _update_window_center,
            dim=model.world_count,
            inputs=[
                state.body_q,
                self.body_indices,
                self.world_origins,
                wp.vec2(*((bounds[0, :2] + bounds[1, :2]) / 2)),
                wp.vec2(*(bounds[1, :2] - bounds[0, :2])),
                wp.vec2(*self.pool_size),
                self.tile_size,
            ],
            outputs=[self.window_centers],
            device=model.device,
        )
        self.mass_change_count.zero_()
        self.recycle_count.zero_()
        material = self.material_state.mpm
        wp.launch(
            _recycle_particles,
            dim=model.particle_count,
            inputs=[
                model.particle_world,
                self.window_centers,
                self.template_positions,
                self.reset_requested,
                self.dynamic_mass,
                self.initial_plastic_volume_ratio,
                wp.vec2(*self.window_size),
                wp.vec2(*self.pool_size),
                self.surface_mesh.id,
                self._ray_z,
                self._ray_length,
            ],
            outputs=[
                self.reset_positions,
                self.particle_is_dynamic,
                state.particle_q,
                state.particle_qd,
                model.particle_mass,
                model.particle_inv_mass,
                material.particle_qd_grad,
                material.particle_elastic_strain,
                material.particle_Jp,
                material.particle_stress,
                material.particle_transform,
                self.recycled,
                self.mass_change_count,
                self.surface_miss_count,
                self.recycle_count,
            ],
            device=model.device,
        )
        self.reset_requested.zero_()
        wp.capture_if(self.mass_change_count, self._refresh_particle_mass)
        if self.solver._particle_activity is not None:
            wp.capture_if(self.recycle_count, self._reset_recycled_history)

    def post_step(self, state) -> None:
        """Remove numerical drift of the kinematic ring after the solver advected it."""
        wp.launch(
            _pin_ring_particles,
            dim=self.model.particle_count,
            inputs=[self.particle_is_dynamic, self.reset_positions, state.particle_q, state.particle_qd],
            device=self.model.device,
        )

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Request reinitialization of the selected worlds' pools at the next :meth:`update`.

        Args:
            world_mask: Boolean mask with at least ``world_count`` entries; None selects every world.
        """
        if world_mask is None:
            self.reset_requested.fill_(1)
        else:
            wp.launch(
                _mark_resets,
                dim=self.model.world_count,
                inputs=[world_mask, self.reset_requested],
                device=self.model.device,
            )

    def set_dynamic_particle_density(self, density: wp.array, world_mask: wp.array) -> None:
        """Update the mass restored to dynamic particles after a material density change.

        This does not change which particles are dynamic and does not write solver material parameters.

        Args:
            density: Nominal material density [kg/m^3] per world.
            world_mask: Boolean mask selecting the worlds whose stored masses are updated.
        """
        wp.launch(
            _set_world_dynamic_mass,
            dim=self.model.particle_count,
            inputs=[self.model.particle_world, world_mask, density, self.model.particle_radius],
            outputs=[self.dynamic_mass],
            device=self.model.device,
        )

    def check_status(self) -> None:
        """Raise if a recycled particle missed the surface mesh since the last check. Reads from the device."""
        if self.surface_miss_count.numpy()[0]:
            self.surface_miss_count.zero_()
            raise ValueError("Recycled particles left the streaming surface mesh.")

    def _reset_recycled_history(self) -> None:
        """Wake recycled particles and restart their quiet timers inside the GPU branch."""
        self.solver._particle_activity.reset_particles(self.material_state, self.recycled)

    def _refresh_particle_mass(self) -> None:
        """Synchronize changed masses and invalidate warm starts inside the GPU branch."""
        solver_model = self.solver.model
        for name in ("particle_mass", "particle_inv_mass"):
            if getattr(solver_model, name) is not getattr(self.model, name):
                wp.copy(getattr(solver_model, name), getattr(self.model, name))
        # Radius, volume, collider membership and constitutive parameters are unchanged.
        derived = self.solver._mpm_model
        wp.launch(
            _refresh_density,
            dim=self.model.particle_count,
            inputs=[solver_model.particle_mass, derived.particle_volume],
            outputs=[derived.particle_density],
            device=self.model.device,
        )
        activity = self.solver._particle_activity
        if activity is not None:
            wp.launch(
                _refresh_density,
                dim=self.model.particle_count,
                inputs=[solver_model.particle_mass, derived.particle_volume],
                outputs=[activity.base_density],
                device=self.model.device,
            )
        # A full solver reset also clears sparse-grid error status and is forbidden during capture.
        # Recycling only invalidates warm starts and collider history.
        self.solver._clear_reset_warmstarts(None, (None, None))
        self.solver._last_step_data.save_collider_current_position(
            self.material_state.body_q,
            body_world=solver_model.body_world,
            world_count=solver_model.world_count,
        )


class CachedParticleStreaming:
    """Pool of particles that follows anchor bodies and remembers the sand it leaves behind.

    The ground is divided into square tiles of ``tile_size``. Each anchor loads the tiles of its pool
    footprint: the dynamic window plus a kinematic ring of width ``margin``. When a pool moves, the
    tiles it leaves are saved to a per-world record store and their simulation slots are freed. Tiles
    it enters are restored from that store, or generated from a tile template settled once at
    construction if they were never visited. Spare simulation slots hold no sand until a tile needs them.
    A world that holds no particles has no pool.
    With particle sleeping enabled, sand entering the dynamic window arrives at rest and may sleep at once;
    sand that stays keeps its sleep history.

    Create with :meth:`SolverImplicitMPM.set_particle_streaming`. Call :meth:`update` before each step
    with the state holding the anchor poses and particle kinematics, and :meth:`post_step` after it.
    Both only launch kernels, so they can be recorded in a CUDA graph.

    .. experimental::

        Runtime-only pool management. The interface may change.
    """

    def __init__(
        self,
        solver,
        state,
        *,
        model,
        body_indices,
        size: Sequence[float],
        margin: float,
        tile_size: float,
        depth: float,
        surface_mesh: wp.Mesh,
        world_origins,
        tile_template,
        tile_radius: float,
        max_cached_particles: int,
        max_visited_tiles: int,
        settle_time: float,
        settle_speed: float,
        settle_dt: float,
    ):
        from .particle_cache import GpuParticleStateCache  # noqa: PLC0415

        self.solver = solver
        self.model = model
        self.material_state = state
        if solver.model.particle_count != model.particle_count:
            raise ValueError("Particle streaming requires the MPM solver to own all particles in original order.")
        device = model.device
        world_count = model.world_count
        self.window_size = tuple(float(v) for v in size)
        self.margin = float(margin)
        self.tile_size = float(tile_size)
        if len(self.window_size) != 2 or min(self.window_size) <= 0.0 or self.margin <= 0.0 or self.tile_size <= 0.0:
            raise ValueError("Streaming window size, margin and tile size must be positive.")
        if self.tile_size > self.margin:
            raise ValueError("Streaming tile size must not exceed the margin.")
        self.pool_size = tuple(v + 2.0 * self.margin for v in self.window_size)

        indices = np.asarray(body_indices, dtype=np.int32)
        if indices.ndim == 1:
            indices = indices[:, None]
        if indices.ndim != 2 or indices.shape[0] != world_count or np.any(indices[:, 0] < 0):
            raise ValueError("Particle streaming needs at least one anchor body index per world.")
        self.body_counts = (indices >= 0).sum(axis=1).astype(np.int32)
        self.bodies = wp.array(indices, dtype=int, device=device)

        particle_world = model.particle_world.numpy()
        if np.any(particle_world < 0) or np.any(particle_world >= world_count):
            raise ValueError("Streamed particles must belong to valid worlds.")
        # A world without particles has no pool: its anchors load no tiles.
        self.body_counts[np.bincount(particle_world, minlength=world_count) == 0] = 0
        origins = np.asarray(world_origins, dtype=np.float32)
        if origins.shape != (world_count, 3):
            raise ValueError("World origins must provide one entry per world.")
        # The center-update kernel uses these origins to quantize pool movement.
        self.origins = wp.array(origins, dtype=wp.vec3, device=device)
        self.patch_centers = wp.array(
            np.repeat(origins[:, None, :2], indices.shape[1], axis=1), dtype=wp.vec2, device=device
        )
        self.particle_reset_positions = wp.clone(model.particle_q)
        self.particle_is_dynamic = wp.ones(model.particle_count, dtype=int, device=device)
        self.particle_dynamic_mass = wp.clone(model.particle_mass)
        if np.any(self.particle_dynamic_mass.numpy() <= 0.0):
            raise ValueError("Initial particle masses must be positive; streaming assigns zero mass to the ring.")
        self.initial_particle_plastic_volume_ratio = wp.clone(state.mpm.particle_Jp)
        self.env_reset_requested = wp.ones(world_count, dtype=int, device=device)
        self.terrain_query_miss_count = wp.zeros(1, dtype=int, device=device)
        self._arrived = wp.zeros(model.particle_count, dtype=int, device=device)

        points = surface_mesh.points.numpy()
        bounds = np.stack((points.min(axis=0), points.max(axis=0)))
        self.background_mesh = SimpleNamespace(
            mesh=surface_mesh,
            bounds=bounds,
            height_query_start_z=float(bounds[1, 2] + 1.0),
            height_query_max_distance=float(bounds[1, 2] - bounds[0, 2] + 2.0),
        )
        self.terrain = SimpleNamespace(
            total_patch_size=self.pool_size,
            simulated_terrain_size=self.window_size,
            particle_depth=float(depth),
            voxel_size=float(solver.voxel_size),
            particle_cache=SimpleNamespace(
                tile_size=self.tile_size,
                max_cached_particles_per_env=int(max_cached_particles),
                max_visited_tiles_per_env=int(max_visited_tiles),
                settle_time_s=float(settle_time),
                settle_speed=float(settle_speed),
                settle_dt=float(settle_dt),
            ),
        )
        self.tile_template = np.asarray(tile_template, dtype=np.float32)
        self.tile_radius = float(tile_radius)
        self.cache = GpuParticleStateCache(self)

    @property
    def window_centers(self) -> wp.array:
        """Snapped pool centers [m], shape ``(world_count, anchors)``."""
        return self.patch_centers

    def update(self, state) -> None:
        """Move the pools to their anchors and page tiles before one solver step.

        Args:
            state: State holding the anchor ``body_q`` and the particle kinematics to update.
        """
        bounds = self.background_mesh.bounds
        wp.launch(
            update_patch_center,
            dim=self.bodies.shape,
            inputs=[
                state.body_q,
                self.bodies,
                self.origins,
                wp.vec2(*((bounds[0, :2] + bounds[1, :2]) / 2)),
                wp.vec2(*(bounds[1, :2] - bounds[0, :2])),
                wp.vec2(*self.pool_size),
                self.tile_size,
            ],
            outputs=[self.patch_centers],
            device=self.model.device,
        )
        # Everything stays on the device, so this is valid inside a CUDA graph.
        self.cache.update(state)
        wp.capture_if(self.cache.changed, self._refresh_particle_mass)
        self.env_reset_requested.zero_()

    def post_step(self, state) -> None:
        """Remove numerical drift of the kinematic ring after the solver advected it."""
        wp.launch(
            _pin_ring_particles,
            dim=self.model.particle_count,
            inputs=[self.particle_is_dynamic, self.particle_reset_positions, state.particle_q, state.particle_qd],
            device=self.model.device,
        )

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Request reinitialization of the selected worlds' pools and history at the next :meth:`update`.

        Args:
            world_mask: Boolean mask with at least ``world_count`` entries; None selects every world.
        """
        if world_mask is None:
            self.env_reset_requested.fill_(1)
        else:
            wp.launch(
                _mark_resets,
                dim=self.model.world_count,
                inputs=[world_mask, self.env_reset_requested],
                device=self.model.device,
            )

    def set_dynamic_particle_density(self, density: wp.array, world_mask: wp.array) -> None:
        """Update the mass given to newly generated dynamic particles after a material density change.

        Args:
            density: Nominal material density [kg/m^3] per world.
            world_mask: Boolean mask selecting the worlds whose stored masses are updated.
        """
        wp.launch(
            _set_world_dynamic_mass,
            dim=self.model.particle_count,
            inputs=[self.model.particle_world, world_mask, density, self.model.particle_radius],
            outputs=[self.particle_dynamic_mass],
            device=self.model.device,
        )

    def check_status(self) -> None:
        """Raise if a generated particle missed the surface or a capacity was exceeded. Reads from the device."""
        if self.terrain_query_miss_count.numpy()[0]:
            self.terrain_query_miss_count.zero_()
            raise ValueError("Generated particles left the streaming surface mesh.")
        self.cache.check_errors()

    def _refresh_particle_mass(self) -> None:
        """Synchronize paged particle properties and invalidate solver warm starts inside the GPU branch."""
        solver_model = self.solver.model
        for name in ("particle_mass", "particle_inv_mass", "particle_radius", "particle_flags"):
            if getattr(solver_model, name) is not getattr(self.model, name):
                wp.copy(getattr(solver_model, name), getattr(self.model, name))
        # The pool holds only ordinary material particles, with no particle-backed colliders.
        derived = self.solver._mpm_model
        wp.launch(
            refresh_cached_properties,
            dim=self.model.particle_count,
            inputs=[self.model.particle_mass, self.model.particle_radius, self.model.particle_flags],
            outputs=[
                derived.particle_volume,
                derived.particle_density,
                derived.material_particle_volume,
                derived.particle_flags,
                derived.material_particle_flags,
            ],
            device=self.model.device,
        )
        activity = self.solver._particle_activity
        if activity is not None:
            # Sand entering the dynamic window is restored or template-settled, so it arrives at rest and may
            # sleep at once. Sand that stayed keeps its sleep history.
            wp.launch(
                _mark_arrived_material,
                dim=self.model.particle_count,
                inputs=[
                    self.cache.layers,
                    self.cache.next_layers,
                    activity.base_material_flags,
                    activity.base_density,
                ],
                outputs=[self._arrived],
                device=self.model.device,
            )
            activity.refresh_material(derived)
            activity.settle_particles(self.material_state, self._arrived)
        # A full solver reset also clears sparse-grid error status and is forbidden during capture.
        # Paging only invalidates warm starts and collider history.
        self.solver._clear_reset_warmstarts(None, (None, None))
        self.solver._last_step_data.save_collider_current_position(
            self.material_state.body_q,
            body_world=solver_model.body_world,
            world_count=solver_model.world_count,
        )
