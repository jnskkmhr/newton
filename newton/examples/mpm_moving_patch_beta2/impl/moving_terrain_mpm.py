# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local MPM terrain with a tiled, height-based background."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from .terrain_generator import TerrainGenerator
from .terrain_height_cache import TerrainHeightCache, TerrainHeightData, ground_height, read_height, save_height

__all__ = ["MovingTerrainMPMConfig", "MovingTerrainMPMWrapper"]


@dataclass(frozen=True)
class MovingTerrainMPMConfig:
    """Configure local terrain and background tiles; all lengths are in meters."""

    moving_terrain_size: tuple[float, float] = (2.0, 2.0)
    """Full X/Y size of dynamic simulated terrain [m]."""

    boundary_terrain_size: float = 0.5
    """Thickness of fixed supporting terrain on each side [m]."""

    background_terrain_size: tuple[float, float] = (20.0, 20.0)
    """Full X/Y size of the static supporting ground mesh [m]."""
    background_terrain_origin: tuple[float, float] = (0.0, 0.0)
    """World X/Y center of the static supporting ground mesh [m]."""
    background_tile_size: float = 2.0
    """Side length of a saved background tile [m]. There is no world-size limit."""

    background_tile_capacity: int = 1024
    """Maximum number of saved GPU tiles; history is never silently discarded."""

    background_height_spacing: float = 0.05
    """Spacing of background height samples [m]."""

    def __post_init__(self):
        if (
            isinstance(self.background_tile_capacity, bool)
            or not isinstance(self.background_tile_capacity, int)
            or self.background_tile_capacity < 1
        ):
            raise ValueError("background_tile_capacity must be a positive integer")
        size = np.asarray(self.moving_terrain_size)
        if size.shape != (2,) or not np.isfinite(size).all():
            raise ValueError("moving_terrain_size must contain two finite lengths")
        for name in ("background_terrain_size", "background_terrain_origin"):
            value = np.asarray(getattr(self, name))
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain two finite numbers")
        h = self.background_height_spacing
        if not math.isfinite(h) or h <= 0:
            raise ValueError("background_height_spacing must be positive and finite")
        for length in (*size, *self.background_terrain_size, self.boundary_terrain_size, self.background_tile_size):
            if not math.isfinite(length) or length <= 0 or not np.isclose(length / h, round(length / h)):
                raise ValueError("Terrain sizes must be positive multiples of background_height_spacing")


@wp.func
def _inside(p: wp.vec3, center: wp.vec2, size: wp.vec2):
    return wp.abs(p[0] - center[0]) < size[0] * 0.5 and wp.abs(p[1] - center[1]) < size[1] * 0.5


@wp.kernel
def _choose_center(
    poses: wp.array[wp.transform],
    body: int,
    spacing: float,
    center: wp.array[wp.vec2],
    next_center: wp.array[wp.vec2],
    status: wp.array[int],
):
    p = wp.transform_get_translation(poses[body])
    origin = wp.vec2(0.5 * spacing)
    c = wp.vec2(
        origin[0] + wp.floor((p[0] - origin[0]) / spacing + 0.50001) * spacing,
        origin[1] + wp.floor((p[1] - origin[1]) / spacing + 0.50001) * spacing,
    )
    # A one-sample deadband avoids repeated shifts from tiny body oscillations.
    if wp.abs(p[0] - center[0][0]) <= spacing:
        c[0] = center[0][0]
    if wp.abs(p[1] - center[0][1]) <= spacing:
        c[1] = center[0][1]
    next_center[0] = c
    status[0] = 0
    if c[0] != center[0][0] or c[1] != center[0][1]:
        status[0] = 1


@wp.kernel
def _deposit_volume(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    flags: wp.array[int],
    spacing: float,
    center: wp.array[wp.vec2],
    outer: wp.vec2,
    volume: wp.array2d[float],
    ground: wp.vec2,
    height_sum: wp.array2d[float],
    fraction_sum: wp.array2d[float],
    fraction: wp.array[float],
    layers: wp.array[int],
    dynamic_columns: wp.array2d[int],
):
    i = wp.tid()
    if flags[i] & int(newton.ParticleFlags.ACTIVE):
        start_x = int(wp.floor((center[0][0] - 0.5 * outer[0]) / spacing))
        start_y = int(wp.floor((center[0][1] - 0.5 * outer[1]) / spacing))
        ix = int(wp.floor((positions[i][0]) / spacing + 0.5)) - start_x
        iy = int(wp.floor((positions[i][1]) / spacing + 0.5)) - start_y
        if ix >= 0 and iy >= 0 and ix < volume.shape[0] and iy < volume.shape[1]:
            r = radii[i]
            if layers[i] == 1:
                wp.atomic_max(dynamic_columns, ix, iy, 1)
            wp.atomic_add(volume, ix, iy, 8.0 * r * r * r)
            z = positions[i][2] - ground_height(positions[i][0], ground)
            wp.atomic_add(height_sum, ix, iy, wp.max(z, 0.0))
            wp.atomic_add(fraction_sum, ix, iy, fraction[i])


@wp.kernel
def _save_heights(
    h: float,
    center: wp.array[wp.vec2],
    size: wp.vec2,
    outer: wp.vec2,
    ground: wp.vec2,
    volume: wp.array2d[float],
    cache: TerrainHeightData,
    height_sum: wp.array2d[float],
    fraction_sum: wp.array2d[float],
    dynamic_columns: wp.array2d[int],
):
    i, j = wp.tid()
    ix = i + int(wp.floor((center[0][0] - 0.5 * outer[0]) / h))
    iy = j + int(wp.floor((center[0][1] - 0.5 * outer[1]) / h))
    p = wp.vec3(float(ix) * h, float(iy) * h, 0.0)
    # Save only fully covered samples, excluding the artificial outer edge.
    if _inside(p, center[0], size) or dynamic_columns[i, j] != 0:
        # Normalize mean height by the initial sampling fraction (approximately 0.5).
        # This removes finite-sample jitter bias and preserves affine compression.
        thickness = float(0.0)
        if fraction_sum[i, j] > 1.0e-6:
            thickness = height_sum[i, j] / fraction_sum[i, j]
        height = ground_height(float(ix) * h, ground) + thickness
        # Keep reference material volume independent of geometric compression.
        save_height(cache, ix, iy, height, volume[i, j] / (h * h))


@wp.kernel
def _detect_departures(
    positions: wp.array[wp.vec3],
    layers: wp.array[int],
    center: wp.array[wp.vec2],
    moving_size: wp.vec2,
    outer_size: wp.vec2,
    status: wp.array[int],
):
    i = wp.tid()
    if (layers[i] == 1 and not _inside(positions[i], center[0], moving_size)) or not _inside(
        positions[i], center[0], outer_size
    ):
        wp.atomic_max(status, 0, 1)


@wp.kernel
def _transition(
    center: wp.array[wp.vec2],
    moving_size: wp.vec2,
    outer_size: wp.vec2,
    h: float,
    ground: wp.vec2,
    depth: float,
    density: float,
    cache: TerrainHeightData,
    fraction: wp.array[float],
    base_volume: float,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    mass: wp.array[float],
    inv_mass: wp.array[float],
    radii: wp.array[float],
    flags: wp.array[int],
    layers: wp.array[int],
    changed: wp.array[int],
    recycled: wp.array[int],
    gradients: wp.array[wp.mat33],
    elastic: wp.array[wp.mat33],
    plastic: wp.array[float],
    initial_plastic: wp.array[float],
    stress: wp.array[wp.mat33],
    transforms: wp.array[wp.mat33],
):
    i = wp.tid()
    p = positions[i]
    recycle = not _inside(p, center[0], outer_size)
    if recycle:
        lo = center[0] - 0.5 * outer_size
        x = p[0] - wp.floor((p[0] - lo[0]) / outer_size[0]) * outer_size[0]
        y = p[1] - wp.floor((p[1] - lo[1]) / outer_size[1]) * outer_size[1]
        ix = int(wp.floor(x / h + 0.5))
        iy = int(wp.floor(y / h + 0.5))
        sample = read_height(cache, ix, iy)
        thickness = wp.max(sample[0] - ground_height(float(ix) * h, ground), 0.0)
        ground_z = ground_height(x, ground)
        p = wp.vec3(x, y, ground_z + fraction[i] * thickness)
        positions[i] = p
        volume = base_volume * sample[1] / depth
        radii[i] = 0.5 * wp.pow(wp.max(volume, 1.0e-18), 1.0 / 3.0)
        flags[i] = int(newton.ParticleFlags.ACTIVE)
        if thickness <= 1.0e-6:
            flags[i] = 0
        wp.atomic_add(recycled, 0, 1)
    layer = int(2)
    if _inside(p, center[0], moving_size):
        layer = 1
    old_layer = layers[i]
    if recycle or layer != old_layer:
        wp.atomic_add(changed, 0, 1)
        r = radii[i]
        m = density * 8.0 * r * r * r
        if layer == 2 or flags[i] == 0:
            m = 0.0
        mass[i] = m
        inv_mass[i] = 0.0
        if m > 0.0:
            inv_mass[i] = 1.0 / m
        velocities[i] = wp.vec3(0.0)
        gradients[i] = wp.mat33(0.0)
        if recycle or layer == 1:
            elastic[i] = wp.identity(n=3, dtype=float)
            plastic[i] = initial_plastic[i]
            stress[i] = wp.mat33(0.0)
            transforms[i] = wp.identity(n=3, dtype=float)
    if layer == 2:
        velocities[i] = wp.vec3(0.0)
        gradients[i] = wp.mat33(0.0)
    layers[i] = layer


@wp.kernel
def _render_radii(
    layers: wp.array[int],
    flags: wp.array[int],
    radii: wp.array[float],
    show_boundary: bool,
    positions: wp.array[wp.vec3],
    ground: wp.vec2,
    fraction: wp.array[float],
    depth: float,
    color_by_height: bool,
    height_color_range: float,
    render_radii: wp.array[float],
    colors: wp.array[wp.vec3],
    particle_color: wp.vec3,
    boundary_color: wp.vec3,
):
    i = wp.tid()
    render_radii[i] = 0.0
    if (layers[i] == 1 or show_boundary) and flags[i] & int(newton.ParticleFlags.ACTIVE):
        render_radii[i] = radii[i]
    colors[i] = particle_color
    if layers[i] == 2:
        colors[i] = boundary_color
    if color_by_height:
        p = positions[i]
        reference_z = ground_height(p[0], ground) + fraction[i] * depth
        lowering = wp.clamp((reference_z - p[2]) / height_color_range, 0.0, 1.0)
        colors[i] = wp.lerp(wp.vec3(0.10, 0.30, 0.90), wp.vec3(0.90, 0.15, 0.10), lowering)


class MovingTerrainMPMWrapper:
    """Experimental local MPM terrain with height history entirely on the GPU.

    Only simulated and boundary terrain allocate MPM particles. A GPU tile cache
    stores departed heights and reference volumes. Retained simulated particles
    keep physical state; reconstructed terrain starts with fresh material history.
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
        """Define terrain geometry and sampling before allocating particles.

        Args:
            config: Fixed terrain dimensions and background sample spacing [m].
            tracked_body: Exact body label; missing or duplicate labels are errors.
            material_parameters: MPM custom attributes chosen by the example, such
                as friction, Young's modulus [Pa], and Poisson's ratio. Forwarded
                to the builder without wrapper-specific material defaults.
            voxel_size: MPM voxel size [m].
            particles_per_cell: Number of samples per voxel along each axis.
            jitter: Position jitter [m], matching ModelBuilder.add_particle_grid:
                each coordinate receives (random - 0.5) * jitter, giving offsets
                in [-jitter / 2, jitter / 2). Zero disables jitter.
            sand_depth: Initial vertical sand thickness above ground [m].
            ground_height: Ground height [m] when terrain_gen is omitted.
            terrain_gen: Procedural ground profile; defaults to flat ground.
            density: Initial material density [kg/m^3].
        """
        if not isinstance(tracked_body, str) or not tracked_body:
            raise ValueError("tracked_body must be a nonempty body name")
        if not isinstance(particles_per_cell, int) or particles_per_cell < 1:
            raise ValueError("particles_per_cell must be a positive integer")
        if not all(math.isfinite(v) and v > 0 for v in (voxel_size, sand_depth, density)):
            raise ValueError("voxel_size, sand_depth and density must be finite and positive")
        if not math.isfinite(ground_height):
            raise ValueError("ground_height must be finite")
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError("jitter must be finite and nonnegative")
        self.jitter = jitter
        self.config, self.tracked_body = config, tracked_body
        self.material_parameters = dict(material_parameters)
        self.voxel_size, self.particles_per_cell = voxel_size, particles_per_cell
        self.sand_depth, self.ground_height, self.density = sand_depth, ground_height, density
        self.moving_size = np.asarray(config.moving_terrain_size)
        self.outer_size = self.moving_size + 2 * config.boundary_terrain_size
        self.terrain_gen = terrain_gen or TerrainGenerator(type="flat", params={"height": ground_height})
        if self.terrain_gen.type.value == "sinusoidal":
            raise ValueError(
                "Beta2 currently supports flat and slope ground; moving wave colliders are not implemented"
            )
        self.ground_parameters = wp.vec2(
            float(self.terrain_gen.sample_height(0.0)),
            float(self.terrain_gen.sample_height(1.0) - self.terrain_gen.sample_height(0.0)),
        )
        self.height_cache = TerrainHeightCache(
            spacing=config.background_height_spacing,
            tile_size=config.background_tile_size,
            capacity=config.background_tile_capacity,
            sand_depth=sand_depth,
            ground=self.ground_parameters,
        )
        if config.boundary_terrain_size < 2 * voxel_size:
            raise ValueError("Use a boundary at least two MPM voxels wide")
        self.bound = False
        self.allocated = False

    def _body_index(self, labels):
        matches = [i for i, label in enumerate(labels) if label == self.tracked_body]
        if len(matches) != 1:
            raise ValueError(f"Expected one body named {self.tracked_body!r}; found {len(matches)}")
        return matches[0]

    def sample_ground_height(self, x):
        """Sample the supporting ground in world coordinates [m]."""
        return self.terrain_gen.sample_height(x)

    def add_ground(self, builder: newton.ModelBuilder, *, texture: np.ndarray | None = None):
        """Add a static heightfield collider covering the configured background."""
        size = np.asarray(self.config.background_terrain_size)
        origin = np.asarray(self.config.background_terrain_origin)
        shape = np.rint(size / self.config.background_height_spacing).astype(int) + 1
        x = np.linspace(origin[0] - size[0] / 2, origin[0] + size[0] / 2, shape[0])
        y = np.linspace(origin[1] - size[1] / 2, origin[1] + size[1] / 2, shape[1])
        heights = self.terrain_gen.sample_height(x[:, None], y[None, :]).astype(np.float32)
        mesh = newton.Mesh.create_heightfield(
            heightfield=heights,
            extent_x=size[0],
            extent_y=size[1],
            center_x=origin[0],
            center_y=origin[1],
            ground_z=float(heights.min() - 1.0),
            compute_normals=True,
            compute_uvs=texture is not None,
        )
        if texture is not None:
            mesh.texture = texture
        color = (1.0, 1.0, 1.0) if texture is not None else (0.3, 0.25, 0.18)
        return builder.add_shape_mesh(-1, mesh=mesh, label="terrain_ground", color=color)

    def allocate_particle_storage(self, builder: newton.ModelBuilder):
        """Derive particle count and create jittered samples at the named body [m].

        This wrapper requires itself to own all model particles and the
        builder body pose to contain the initial world position of the box.
        """
        if self.allocated or builder.particle_count:
            raise ValueError("Allocate once into a builder without other particles")
        if builder.world_count > 1:
            raise ValueError("The beta2 wrapper supports one world only")
        body = self._body_index(builder.body_label)
        h = self.config.background_height_spacing
        pose = np.asarray(builder.body_q[body])[:2]
        origin = np.full(2, h / 2)
        self.initial_center = origin + np.floor((pose - origin) / h + 0.50001) * h
        cell = self.voxel_size / self.particles_per_cell
        # Equal sample counts per background column make a flat bed reconstruct flat.
        per_column = max(1, round(h / cell))
        self.cell_xy = h / per_column
        nz = max(1, math.ceil(self.sand_depth / cell))
        dz = self.sand_depth / nz
        nx, ny = np.rint(self.outer_size / self.cell_xy).astype(int)
        self.base_volume = self.cell_xy**2 * dz
        radius = 0.5 * self.base_volume ** (1 / 3)
        rng = np.random.default_rng(42)
        # Align background sample centers with particle columns; avoid boundary samples.
        lo = self.initial_center - self.outer_size / 2
        x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        positions = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(float)
        positions = positions * [self.cell_xy, self.cell_xy, dz] + [lo[0], lo[1], self.ground_height]
        positions += [self.cell_xy / 2, self.cell_xy / 2, dz / 2]
        # Match ModelBuilder.add_particle_grid's units and sampling convention.
        positions += (rng.random(positions.shape) - 0.5) * self.jitter
        self.fractions = ((positions[:, 2] - self.ground_height) / self.sand_depth).astype(np.float32)
        positions[:, 2] += self.sample_ground_height(positions[:, 0]) - self.ground_height
        dynamic = np.all(np.abs(positions[:, :2] - self.initial_center) < self.moving_size / 2, axis=1)
        masses = np.where(dynamic, self.density * self.base_volume, 0.0)
        builder.add_particles(
            pos=positions.tolist(),
            vel=np.zeros_like(positions).tolist(),
            mass=masses.tolist(),
            radius=[radius] * len(positions),
            custom_attributes=self.material_parameters,
        )
        self.initial_layers = np.where(dynamic, 1, 2).astype(np.int32)
        self.initial_particle_count = len(positions)
        self.allocated = True

    def bind(self, *, model: newton.Model, state: newton.State, coupled_solver=None, mpm_entry: str = "mpm"):
        """Bind stable particle arrays after construction; resolve the body by name.

        Passing no coupled solver permits CPU-only transition tests. Real MPM
        simulation requires the coupled solver so material caches are refreshed.
        """
        if not self.allocated or self.bound:
            raise ValueError("Allocate particles before binding, and bind only once")
        if model.particle_count != len(self.fractions):
            raise ValueError("The wrapper must own every model particle")
        self.model, self.state, self.device = model, state, model.device
        self.body_index = self._body_index(model.body_label)
        self.coupled_solver = coupled_solver
        self.mpm_solver = None if coupled_solver is None else coupled_solver.solver(mpm_entry)
        self.mpm_state = state if coupled_solver is None else coupled_solver.entry_state(mpm_entry, phase="input")
        if self.mpm_solver is not None:
            if self.mpm_solver.model.particle_count != model.particle_count:
                raise ValueError("MPM entry must own all terrain particles in model order")
        with wp.ScopedDevice(self.device):
            self.height_cache.bind(self.device)
            self.background_heights = self.height_cache.heights
            self.background_reference_depth = self.height_cache.reference_depth
            local_shape = tuple(np.rint(self.outer_size / self.config.background_height_spacing).astype(int) + 3)
            self.deposited_volume = wp.zeros(local_shape, dtype=float)
            self.deposited_height = wp.zeros(local_shape, dtype=float)
            self.deposited_fraction = wp.zeros(local_shape, dtype=float)
            self.dynamic_columns = wp.zeros(local_shape, dtype=int)
            self.center = wp.array([self.initial_center], dtype=wp.vec2)
            self.next_center = wp.clone(self.center)
            self.status = wp.zeros(1, dtype=int)
            self.changed = wp.zeros(1, dtype=int)
            self.recycled_count = wp.zeros(1, dtype=int)
            self.layers = wp.array(self.initial_layers, dtype=int)
            self.fraction = wp.array(self.fractions, dtype=float)
            self.initial_plastic = wp.clone(model.mpm.particle_Jp)
            self.render_radii = wp.empty_like(model.particle_radius)
            self.render_colors = wp.empty(model.particle_count, dtype=wp.vec3)
        self.bound = True

    def save_heights(self):
        """Save measured column heights and reference material volume on the GPU."""
        self.deposited_volume.zero_()
        self.deposited_height.zero_()
        self.deposited_fraction.zero_()
        self.dynamic_columns.zero_()
        h = self.config.background_height_spacing
        wp.launch(
            _deposit_volume,
            dim=self.model.particle_count,
            inputs=[
                self.state.particle_q,
                self.model.particle_radius,
                self.model.particle_flags,
                h,
                self.center,
                wp.vec2(*self.outer_size),
                self.deposited_volume,
                self.ground_parameters,
                self.deposited_height,
                self.deposited_fraction,
                self.fraction,
                self.layers,
                self.dynamic_columns,
            ],
            device=self.device,
        )
        wp.launch(
            _save_heights,
            dim=self.deposited_volume.shape,
            inputs=[
                h,
                self.center,
                wp.vec2(*self.moving_size),
                wp.vec2(*self.outer_size),
                self.ground_parameters,
                self.deposited_volume,
                self.height_cache.data,
                self.deposited_height,
                self.deposited_fraction,
                self.dynamic_columns,
            ],
            device=self.device,
        )
        self.height_cache.check_capacity()

    def update_terrain_layers(self):
        """Update local terrain and save heights before particles become fixed.

        This runs outside CUDA graph capture. Only small status counters return
        to Python; terrain heights and reference volumes remain on the GPU.
        """
        if not self.bound:
            raise RuntimeError("Bind the wrapper before updating terrain")
        wp.launch(
            _choose_center,
            dim=1,
            inputs=[
                self.state.body_q,
                self.body_index,
                self.config.background_height_spacing,
                self.center,
                self.next_center,
                self.status,
            ],
            device=self.device,
        )
        wp.launch(
            _detect_departures,
            dim=self.model.particle_count,
            inputs=[
                self.state.particle_q,
                self.layers,
                self.next_center,
                wp.vec2(*self.moving_size),
                wp.vec2(*self.outer_size),
                self.status,
            ],
            device=self.device,
        )
        if int(self.status.numpy()[0]) > 0:
            self.save_heights()
        wp.copy(self.center, self.next_center)
        self.changed.zero_()
        material = self.mpm_state.mpm
        wp.launch(
            _transition,
            dim=self.model.particle_count,
            inputs=[
                self.center,
                wp.vec2(*self.moving_size),
                wp.vec2(*self.outer_size),
                self.config.background_height_spacing,
                self.ground_parameters,
                self.sand_depth,
                self.density,
                self.height_cache.data,
                self.fraction,
                self.base_volume,
                self.state.particle_q,
                self.state.particle_qd,
                self.model.particle_mass,
                self.model.particle_inv_mass,
                self.model.particle_radius,
                self.model.particle_flags,
                self.layers,
                self.changed,
                self.recycled_count,
                material.particle_qd_grad,
                material.particle_elastic_strain,
                material.particle_Jp,
                self.initial_plastic,
                material.particle_stress,
                material.particle_transform,
            ],
            device=self.device,
        )
        if int(self.changed.numpy()[0]) > 0:
            self._refresh_particle_properties()

    def _refresh_particle_properties(self):
        if self.mpm_solver is None:
            return
        # MPM's compact view owns separate particle property arrays.
        view = self.mpm_solver.model
        for name in ("particle_mass", "particle_inv_mass", "particle_radius", "particle_flags"):
            wp.copy(getattr(view, name), getattr(self.model, name))
        self.mpm_solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        # flags=0 clears numerical warm starts without resetting material history.
        self.mpm_solver.reset(self.mpm_state, flags=0)

    def render_active_particle(
        self,
        *,
        positions=None,
        visible_ratio=1.0,
        show_buffer=False,
        color_by_height=False,
        height_color_range=0.1,
        particle_color=(0.72, 0.59, 0.34),
        boundary_color=(0.85, 0.38, 0.12),
    ):
        """Update display radii/colors without modifying simulation state.

        Height colors show vertical change from each slot's initial layer above
        the local ground: blue at or above the original height, red when lowered. The color range
        is the absolute change [m] at saturation. This is not elastic strain.
        """
        if not math.isfinite(height_color_range) or height_color_range <= 0.0:
            raise ValueError("height_color_range must be positive and finite")
        if positions is None:
            positions = self.state.particle_q
        wp.launch(
            _render_radii,
            dim=self.model.particle_count,
            inputs=[
                self.layers,
                self.model.particle_flags,
                self.model.particle_radius,
                show_buffer,
                positions,
                self.ground_parameters,
                self.fraction,
                self.sand_depth,
                color_by_height,
                height_color_range,
                self.render_radii,
                self.render_colors,
                wp.vec3(*particle_color),
                wp.vec3(*boundary_color),
            ],
            device=self.device,
        )
        return self.render_radii
