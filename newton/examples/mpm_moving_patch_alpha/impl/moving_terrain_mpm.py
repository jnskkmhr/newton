# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Moving terrain with fresh particle recycling and no terrain-history cache."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from .moving_patch_mpm import HeightFieldTerrain, MovingPatchMPMWrapper, _sample_height, _TerrainHeightField
from .terrain_generator import TerrainGenerator
from .terrain_scene import TerrainScene

__all__ = ["MovingTerrainMPMConfig", "MovingTerrainMPMWrapper"]


@dataclass(frozen=True)
class MovingTerrainMPMConfig:
    """Configure alpha terrain dimensions using the beta/beta2 names."""

    moving_terrain_size: tuple[float, float] = (2.0, 2.0)
    """Full X/Y size of the normally visible simulated terrain [m]."""
    boundary_terrain_size: float = 0.5
    """Width of the hidden, zero-mass active boundary on each side [m]."""
    background_terrain_size: tuple[float, float] = (40.0, 16.0)
    """Full X/Y size of the visual background terrain [m]."""
    background_terrain_origin: tuple[float, float] = (0.0, 0.0)
    """World X/Y center of the visual background terrain [m]."""
    background_height_spacing: float = 0.5
    """Spacing of the generated supporting-ground samples [m]; not a history cache."""
    shift_step: float = 0.4
    """Distance between discrete patch centers [m], retaining alpha's shift control."""

    def __post_init__(self):
        for name in ("moving_terrain_size", "background_terrain_size", "background_terrain_origin"):
            values = np.asarray(getattr(self, name))
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain two finite numbers")
        for value in (
            *self.moving_terrain_size,
            *self.background_terrain_size,
            self.background_height_spacing,
            self.shift_step,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Terrain lengths and spacing must be positive and finite")
        if not math.isfinite(self.boundary_terrain_size) or self.boundary_terrain_size < 0:
            raise ValueError("boundary_terrain_size must be nonnegative and finite")
        if np.any(
            np.asarray(self.background_terrain_size)
            < np.asarray(self.moving_terrain_size) + 2 * self.boundary_terrain_size
        ):
            raise ValueError("The background ground mesh must fit the full particle bed")


@wp.kernel
def _update_particle_layers(
    center: wp.array[wp.vec2i],
    step: float,
    size: wp.vec2,
    reference: wp.array[wp.vec3],
    terrain: _TerrainHeightField,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    dynamic_mass: wp.array[float],
    mass: wp.array[float],
    inv_mass: wp.array[float],
    layers: wp.array[int],
    changed: wp.array[int],
    gradients: wp.array[wp.mat33],
    elastic: wp.array[wp.mat33],
    plastic: wp.array[float],
    initial_plastic: wp.array[float],
    stress: wp.array[wp.mat33],
    transforms: wp.array[wp.mat33],
):
    i = wp.tid()
    p = positions[i]
    c = wp.vec2(float(center[0][0]) * step, float(center[0][1]) * step)
    layer = int(2)
    if wp.abs(p[0] - c[0]) < size[0] * 0.5 and wp.abs(p[1] - c[1]) < size[1] * 0.5:
        layer = 1
    if layers[i] != layer:
        wp.atomic_add(changed, 0, 1)
        if layer == 1:
            r = reference[i]
            z = r[2] - _sample_height(terrain, r[0], r[1]) + _sample_height(terrain, p[0], p[1])
            positions[i] = wp.vec3(p[0], p[1], z)
            velocities[i] = wp.vec3(0.0)
            gradients[i] = wp.mat33(0.0)
            elastic[i] = wp.identity(n=3, dtype=float)
            plastic[i] = initial_plastic[i]
            stress[i] = wp.mat33(0.0)
            transforms[i] = wp.identity(n=3, dtype=float)
    layers[i] = layer
    mass[i] = dynamic_mass[i]
    inv_mass[i] = 1.0 / dynamic_mass[i]
    if layer == 2:
        mass[i] = 0.0
        inv_mass[i] = 0.0
        velocities[i] = wp.vec3(0.0)
        gradients[i] = wp.mat33(0.0)


@wp.kernel
def _render_radii(
    center: wp.array[wp.vec2i],
    step: float,
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    size: wp.vec2,
    ratio: float,
    show_buffer: bool,
    output: wp.array[float],
):
    i = wp.tid()
    c = wp.vec2(float(center[0][0]) * step, float(center[0][1]) * step)
    p = positions[i]
    visible = wp.abs(p[0] - c[0]) < size[0] * ratio * 0.5 and wp.abs(p[1] - c[1]) < size[1] * ratio * 0.5
    output[i] = float(0.0)
    if show_buffer or visible:
        output[i] = radii[i]


class MovingTerrainMPMWrapper:
    """Share beta's setup/bind/update API while discarding departed terrain history.

    Boundary particles remain ACTIVE with zero mass. Particles staying inside
    simulated terrain retain their physical state; incoming particles start fresh.
    No saved terrain height, stress, or particle-state cache is allocated.
    """

    def __init__(
        self,
        *,
        config: MovingTerrainMPMConfig,
        tracked_body: str,
        material_parameters: Mapping[str, float],
        density: float,
        jitter: float,
        voxel_size: float = 0.05,
        particles_per_cell: int = 3,
        sand_depth: float = 0.5,
        ground_height: float = 0.0,
        terrain_gen: TerrainGenerator | None = None,
    ):
        if not tracked_body:
            raise ValueError("tracked_body must be a nonempty body name")
        if isinstance(particles_per_cell, bool) or not isinstance(particles_per_cell, int) or particles_per_cell < 1:
            raise ValueError("particles_per_cell must be a positive integer")
        if not all(math.isfinite(v) and v > 0 for v in (voxel_size, density, sand_depth)):
            raise ValueError("voxel_size, density, and sand_depth must be positive and finite")
        if not math.isfinite(jitter) or jitter < 0 or not math.isfinite(ground_height):
            raise ValueError("jitter must be nonnegative and ground_height finite")
        self.config, self.tracked_body = config, tracked_body
        self.material_parameters = dict(material_parameters)
        self.voxel_size, self.particles_per_cell = voxel_size, particles_per_cell
        self.sand_depth, self.density, self.jitter = sand_depth, density, jitter
        self.moving_size = np.asarray(config.moving_terrain_size)
        self.outer_size = self.moving_size + 2 * config.boundary_terrain_size
        self.bed_lo = np.array([-self.outer_size[0] / 2, -self.outer_size[1] / 2, 0.0])
        self.bed_hi = np.array([self.outer_size[0] / 2, self.outer_size[1] / 2, sand_depth])
        self.terrain_gen = terrain_gen or TerrainGenerator(type="flat", params={"height": ground_height})
        size = np.asarray(config.background_terrain_size)
        origin = np.asarray(config.background_terrain_origin)
        res = np.maximum(1, np.ceil(size / config.background_height_spacing).astype(int))
        x = np.linspace(origin[0] - size[0] / 2, origin[0] + size[0] / 2, res[0] + 1)
        y = np.linspace(origin[1] - size[1] / 2, origin[1] + size[1] / 2, res[1] + 1)
        heights = self.terrain_gen.sample_height(x[:, None], y[None, :])
        # Construction is on CPU; bind uploads the immutable ground samples once.
        self.terrain = HeightFieldTerrain(
            heights, origin=(x[0], y[0], 0.0), spacing=(x[1] - x[0], y[1] - y[0]), device="cpu"
        )
        self.scene = TerrainScene(self)
        self.allocated = False
        self.bound = False

    def _body_index(self, labels):
        matches = [i for i, label in enumerate(labels) if label == self.tracked_body]
        if len(matches) != 1:
            raise ValueError(f"Expected one body named {self.tracked_body!r}; found {len(matches)}")
        return matches[0]

    def sample_ground_height(self, x):
        """Sample the generated mesh along X at Y=0 [m], matching the robot hooks."""
        x = np.asarray(x)
        xy = np.stack((x, np.zeros_like(x)), axis=-1)
        return self.terrain.sample_height(xy)

    def add_ground(self, builder: newton.ModelBuilder, *, texture: np.ndarray | None = None):
        """Add a hidden local collider; background rendering is separate and collision-free."""
        return self.scene.add_ground(builder, texture)

    def allocate_particle_storage(self, builder: newton.ModelBuilder):
        """Emit one jittered bed using alpha's original vertical-endpoint sampling."""
        if self.allocated or builder.particle_count:
            raise ValueError("Allocate once in a builder with no existing particles")
        body = self._body_index(builder.body_label)
        center = np.asarray(builder.body_q[body])[:2]
        center = np.floor(center / self.config.shift_step + 0.5) * self.config.shift_step
        if np.any(center - self.outer_size / 2 < self.terrain.origin[:2]) or np.any(
            center + self.outer_size / 2 > self.terrain.upper_xy
        ):
            raise ValueError("Initial particle bed must fit inside the configured background extent")
        extent = self.bed_hi - self.bed_lo
        res = np.ceil(np.nextafter(self.particles_per_cell * extent / self.voxel_size, -np.inf)).astype(int)
        cell = extent / res
        builder.add_particle_grid(
            pos=wp.vec3(center[0] + self.bed_lo[0] + cell[0] / 2, center[1] + self.bed_lo[1] + cell[1] / 2, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(res[0]),
            dim_y=int(res[1]),
            dim_z=int(res[2]) + 1,
            cell_x=float(cell[0]),
            cell_y=float(cell[1]),
            cell_z=float(cell[2]),
            mass=float(self.density * np.prod(cell)),
            radius_mean=float(np.max(cell) / 2),
            jitter=self.jitter,
            custom_attributes=self.material_parameters,
        )
        q = np.asarray(builder.particle_q).copy()
        q[:, 2] += self.terrain.sample_height(q[:, :2])
        builder.particle_q = [wp.vec3(*p) for p in q]
        self.dynamic_masses = np.asarray(builder.particle_mass, dtype=np.float32).copy()
        dynamic = np.all(np.abs(q[:, :2] - center) < self.moving_size / 2, axis=1)
        self.initial_layers = np.where(dynamic, 1, 2).astype(np.int32)
        builder.particle_mass = np.where(dynamic, self.dynamic_masses, 0.0).tolist()
        self.initial_particle_count = builder.particle_count
        self.allocated = True

    def bind(self, *, model: newton.Model, state: newton.State, coupled_solver=None, mpm_entry="mpm"):
        """Bind existing solver arrays after model construction; allocate no history cache."""
        if not self.allocated or self.bound:
            raise ValueError("Allocate particles first and bind only once")
        if model.particle_count != self.initial_particle_count:
            raise ValueError("The model particle count must match the allocated terrain")
        self.model, self.state, self.device = model, state, model.device
        self.body_index = self._body_index(model.body_label)
        self.mpm_solver = None if coupled_solver is None else coupled_solver.solver(mpm_entry)
        self.mpm_state = state if coupled_solver is None else coupled_solver.entry_state(mpm_entry, phase="input")
        if self.terrain.data.heights.device != model.device:
            self.terrain = HeightFieldTerrain(
                self.terrain.samples, origin=self.terrain.origin, spacing=self.terrain.spacing, device=model.device
            )
        self._patch = MovingPatchMPMWrapper(
            model,
            state,
            self.mpm_state,
            bed_lo=self.bed_lo,
            bed_hi=self.bed_hi,
            shift_step=self.config.shift_step,
            terrain=self.terrain,
            body_index=self.body_index,
        )
        self.reference_positions = self._patch.reference_positions
        self.render_radii = self._patch.render_radii
        self.render_colors = wp.full(model.particle_count, wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=model.device)
        self.window_center = self._patch.window_center
        self.window_shifts = self._patch.window_shifts
        self.window_starts = self._patch.window_starts
        self.window_ends = self._patch.window_ends
        self.layers = wp.array(self.initial_layers, dtype=int, device=self.device)
        self.dynamic_mass = wp.array(self.dynamic_masses, dtype=float, device=self.device)
        self.changed = wp.zeros(1, dtype=int, device=self.device)
        self.bound = True
        self._patch.reset_particle_background()
        if self.scene.shape is not None:
            self.scene.bind()

    def update_terrain_layers(self):
        """Reset escaped particles to the initial material state and generated ground."""
        self._patch.reset_particle_background()
        self.changed.zero_()
        material = self.mpm_state.mpm
        wp.launch(
            _update_particle_layers,
            dim=self.model.particle_count,
            inputs=[
                self.window_center,
                self.config.shift_step,
                wp.vec2(*self.moving_size),
                self.reference_positions,
                self.terrain.data,
                self.state.particle_q,
                self.state.particle_qd,
                self.dynamic_mass,
                self.model.particle_mass,
                self.model.particle_inv_mass,
                self.layers,
                self.changed,
                material.particle_qd_grad,
                material.particle_elastic_strain,
                material.particle_Jp,
                self._patch.initial_plastic,
                material.particle_stress,
                material.particle_transform,
            ],
            device=self.device,
        )
        if int(self.changed.numpy()[0]) > 0 and self.mpm_solver is not None:
            view = self.mpm_solver.model
            for name in ("particle_mass", "particle_inv_mass"):
                wp.copy(getattr(view, name), getattr(self.model, name))
            self.mpm_solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
            self.mpm_solver.reset(self.mpm_state, flags=0)
        if self.scene.shape is not None:
            self.scene.update_collider()

    def render_background(self, viewer, *, show_buffer=False):
        """Draw terrain at the initial sand surface, leaving the particle region open."""
        if self.scene.shape is not None:
            self.scene.render(viewer, show_buffer=show_buffer)

    def render_active_particle(
        self, *, positions=None, visible_ratio=1.0, show_buffer=False, particle_color=(0.7, 0.6, 0.4)
    ):
        """Show the simulated terrain; optionally include the dynamic boundary buffer."""
        if not math.isfinite(visible_ratio) or not 0 <= visible_ratio <= 1:
            raise ValueError("visible_ratio must lie between zero and one")
        positions = self.state.particle_q if positions is None else positions
        wp.launch(
            _render_radii,
            dim=self.model.particle_count,
            inputs=[
                self.window_center,
                self.config.shift_step,
                positions,
                self.model.particle_radius,
                wp.vec2(*self.moving_size),
                visible_ratio,
                show_buffer,
                self.render_radii,
            ],
            device=self.device,
        )
        self.render_colors.fill_(wp.vec3(*particle_color))
        return self.render_radii

    def update_window_outline(self):
        """Update the optional outline of the full moving particle bed."""
        self._patch.update_window_outline()
