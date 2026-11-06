# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Visual background surface and a GPU-updated local ground collider."""

import numpy as np
import warp as wp

import newton

from .moving_patch_mpm import _sample_height, _TerrainHeightField


@wp.kernel
def _move_collider(
    center: wp.array[wp.vec2i],
    step: float,
    terrain: _TerrainHeightField,
    reference: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
):
    i = wp.tid()
    p = reference[i]
    c = wp.vec2(float(center[0][0]) * step, float(center[0][1]) * step)
    # Align vertices with the original heightfield, preserving its triangles.
    x0 = terrain.origin[0] + wp.floor((c[0] - terrain.origin[0]) / terrain.spacing[0] + 0.5) * terrain.spacing[0]
    y0 = terrain.origin[1] + wp.floor((c[1] - terrain.origin[1]) / terrain.spacing[1] + 0.5) * terrain.spacing[1]
    x, y = x0 + p[0], y0 + p[1]
    points[i] = wp.vec3(x, y, _sample_height(terrain, x, y) + p[2])


@wp.kernel
def _collider_bounds(
    center: wp.array[wp.vec2i],
    step: float,
    terrain: _TerrainHeightField,
    half: wp.vec2,
    zmin: float,
    zmax: float,
    shape: int,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
):
    c = wp.vec2(float(center[0][0]) * step, float(center[0][1]) * step)
    x = terrain.origin[0] + wp.floor((c[0] - terrain.origin[0]) / terrain.spacing[0] + 0.5) * terrain.spacing[0]
    y = terrain.origin[1] + wp.floor((c[1] - terrain.origin[1]) / terrain.spacing[1] + 0.5) * terrain.spacing[1]
    lower[shape] = wp.vec3(x - half[0], y - half[1], zmin)
    upper[shape] = wp.vec3(x + half[0], y + half[1], zmax)


@wp.kernel
def _visual_surface(
    center: wp.array[wp.vec2i],
    step: float,
    size: wp.vec2,
    terrain: _TerrainHeightField,
    depth: float,
    lo: wp.vec2,
    hi: wp.vec2,
    nx: int,
    ny: int,
    show_buffer: bool,
    outer: wp.vec2,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    uvs: wp.array[wp.vec2],
):
    i = wp.tid()
    strip = i // (nx * ny)
    j = i % (nx * ny)
    u, v = float(j // ny) / float(nx - 1), float(j % ny) / float(ny - 1)
    c = wp.vec2(float(center[0][0]) * step, float(center[0][1]) * step)
    width = size
    if show_buffer:
        width = outer
    a, b = wp.max(lo, c - width * 0.5), wp.min(hi, c + width * 0.5)
    xlo, xhi, ylo, yhi = lo[0], a[0], lo[1], hi[1]
    if strip == 1:
        xlo, xhi = b[0], hi[0]
    elif strip == 2:
        xlo, xhi, ylo, yhi = a[0], b[0], lo[1], a[1]
    elif strip == 3:
        xlo, xhi, ylo, yhi = a[0], b[0], b[1], hi[1]
    x, y = wp.lerp(xlo, xhi, u), wp.lerp(ylo, yhi, v)
    points[i] = wp.vec3(x, y, _sample_height(terrain, x, y) + depth)
    eps = float(0.01)
    dx = (_sample_height(terrain, x + eps, y) - _sample_height(terrain, x - eps, y)) / (2.0 * eps)
    dy = (_sample_height(terrain, x, y + eps) - _sample_height(terrain, x, y - eps)) / (2.0 * eps)
    normals[i] = wp.normalize(wp.vec3(-dx, -dy, 1.0))
    uvs[i] = wp.vec2((x - lo[0]) / (hi[0] - lo[0]), (y - lo[1]) / (hi[1] - lo[1]))


class TerrainScene:
    """Keep background drawing separate from the finite local collision mesh."""

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.shape = None
        self.texture = None

    def add_ground(self, builder, texture):
        w = self.wrapper
        self.texture = texture
        spacing = w.terrain.spacing
        margin = max(2 * w.voxel_size, w.config.shift_step, float(np.max(spacing)))
        cells = 2 * np.ceil((w.outer_size / 2 + margin) / spacing).astype(int)
        self.half = cells * spacing / 2
        self.source = newton.Mesh.create_heightfield(
            heightfield=np.zeros(tuple(cells + 1), dtype=np.float32),
            extent_x=float(2 * self.half[0]),
            extent_y=float(2 * self.half[1]),
            ground_z=-1.0,
            compute_inertia=False,
        )
        self.zmin = float(w.terrain.samples.min() - 1.0)
        self.zmax = float(w.terrain.samples.max())
        self.reference = self.source.vertices.copy()
        body = w._body_index(builder.body_label)
        center = np.asarray(builder.body_q[body])[:2]
        center = np.floor(center / w.config.shift_step + 0.5) * w.config.shift_step
        center = w.terrain.origin[:2] + np.floor((center - w.terrain.origin[:2]) / spacing + 0.5) * spacing
        vertices = self.reference.copy()
        vertices[:, :2] += center
        vertices[:, 2] += w.terrain.sample_height(vertices[:, :2])
        self.source.vertices = vertices
        self.shape = builder.add_shape_mesh(
            -1,
            mesh=self.source,
            label="terrain_collider",
            cfg=builder.ShapeConfig(is_visible=False, density=0.0),
        )
        return self.shape

    def bind(self):
        w = self.wrapper
        with wp.ScopedDevice(w.device):
            self.reference_gpu = wp.array(self.reference, dtype=wp.vec3)
            # This is the same Warp mesh that Newton's collision pipeline uses.
            self.source.finalize(w.device)
            self.mesh = self.source.mesh
            nx, ny = w.terrain.samples.shape
            self.nx, self.ny = nx, ny
            ids = np.arange(nx * ny).reshape(nx, ny)
            a = ids[:-1, :-1].ravel()
            triangles = np.stack((a, a + ny, a + 1, a + ny, a + ny + 1, a + 1), axis=1).reshape(-1, 3)
            indices = np.concatenate([triangles + k * nx * ny for k in range(4)]).astype(np.int32)
            self.visual_indices = wp.array(indices.ravel(), dtype=int)
            self.visual_points = wp.empty(4 * nx * ny, dtype=wp.vec3)
            self.visual_normals = wp.empty_like(self.visual_points)
            self.visual_uvs = wp.empty(4 * nx * ny, dtype=wp.vec2)
            self.visual_transform = wp.array([wp.transform_identity()], dtype=wp.transform)
            self.visual_scale = wp.array([wp.vec3(1.0)], dtype=wp.vec3)
            color = wp.vec3(1.0) if self.texture is not None else wp.vec3(0.7, 0.6, 0.4)
            self.visual_color = wp.array([color], dtype=wp.vec3)
            self.visual_material = wp.array([wp.vec4(0.5, 0.0, 0.0, float(self.texture is not None))], dtype=wp.vec4)
        if w.mpm_solver is not None:
            view = w.mpm_solver.model
            flags = view.shape_flags.numpy()
            bodies = view.shape_body.numpy()
            body_ids = sorted(
                {
                    int(b)
                    for b, f in zip(bodies, flags, strict=True)
                    if b >= 0 and f & int(newton.ShapeFlags.COLLIDE_PARTICLES)
                }
            )
            w.mpm_solver.setup_collider(
                collider_meshes=[self.mesh] + [None] * len(body_ids),
                collider_body_ids=[None, *body_ids],
            )
        self.update_collider()

    def update_collider(self):
        w = self.wrapper
        wp.launch(
            _move_collider,
            dim=self.mesh.points.size,
            inputs=[w.window_center, w.config.shift_step, w.terrain.data, self.reference_gpu, self.mesh.points],
            device=w.device,
        )
        self.mesh.refit()
        wp.launch(
            _collider_bounds,
            dim=1,
            inputs=[
                w.window_center,
                w.config.shift_step,
                w.terrain.data,
                wp.vec2(*self.half),
                self.zmin,
                self.zmax,
                self.shape,
                w.model.shape_collision_aabb_lower,
                w.model.shape_collision_aabb_upper,
            ],
            device=w.device,
        )

    def render(self, viewer, *, show_buffer=False):
        w = self.wrapper
        wp.launch(
            _visual_surface,
            dim=self.visual_points.size,
            inputs=[
                w.window_center,
                w.config.shift_step,
                wp.vec2(*w.moving_size),
                w.terrain.data,
                w.sand_depth,
                wp.vec2(*w.terrain.origin[:2]),
                wp.vec2(*w.terrain.upper_xy),
                self.nx,
                self.ny,
                show_buffer,
                wp.vec2(*w.outer_size),
                self.visual_points,
                self.visual_normals,
                self.visual_uvs,
            ],
            device=w.device,
        )
        viewer.log_mesh(
            "/terrain_background_mesh",
            self.visual_points,
            self.visual_indices,
            normals=self.visual_normals,
            uvs=self.visual_uvs,
            texture=self.texture,
            color=(1.0, 1.0, 1.0) if self.texture is not None else (0.7, 0.6, 0.4),
            dynamic=True,
            hidden=True,
        )
        viewer.log_instances(
            "/terrain_background",
            "/terrain_background_mesh",
            self.visual_transform,
            self.visual_scale,
            self.visual_color,
            self.visual_material,
        )
