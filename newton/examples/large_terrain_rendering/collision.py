# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bounded terrain collision geometry independent of the full render landscape."""

import math

import numpy as np
import warp as wp

import newton

from .terrain import dune_height


@wp.kernel
def _patch_vertices(side: int, spacing: float, center: wp.vec2, seed: int, vertices: wp.array[wp.vec3]):
    i = wp.tid()
    x = (float(i % side) - float(side - 1) / 2.0) * spacing
    y = (float(i // side) - float(side - 1) / 2.0) * spacing
    vertices[i] = wp.vec3(x, y, dune_height(x + center[0], y + center[1], seed))


class TerrainCollisionPatch:
    """Keep one fixed-topology contact mesh centered near the robot."""

    def __init__(self, *, size: float, spacing: float, seed: int, device, center=(0.0, -3.0)):
        if not math.isfinite(size) or not math.isfinite(spacing) or size < 4 or spacing <= 0 or spacing > size / 4:
            raise ValueError("Collision size must be at least 4 m, with spacing in (0, size/4]")
        cells = int(math.ceil(size / spacing))
        if cells > 512:
            raise ValueError("Collision patch is limited to 512 cells per side")
        self.side = cells + 1
        self.spacing = spacing
        self.size = cells * spacing
        self.seed = seed
        self.device = device
        self.center = np.round(np.asarray(center) / spacing) * spacing
        self.vertices = wp.empty(self.side * self.side, dtype=wp.vec3, device=device)
        self._sample()
        row, column = np.mgrid[:cells, :cells]
        a = (row * self.side + column).astype(np.int32)
        self.indices = np.stack([a, a + 1, a + self.side + 1, a, a + self.side + 1, a + self.side], axis=-1).ravel()
        self.mesh = newton.Mesh(self.vertices.numpy(), self.indices, compute_inertia=False)
        self.updates = 0

    def _sample(self):
        wp.launch(
            _patch_vertices,
            dim=self.vertices.shape[0],
            inputs=[self.side, self.spacing, wp.vec2(*self.center), self.seed, self.vertices],
            device=self.device,
        )

    def add_shape(self, builder):
        """Add only the local contact surface, hidden from rendering."""
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.75, gap=0.01, is_visible=False, has_particle_collision=False)
        self.shape = builder.add_shape_mesh(
            -1, xform=wp.transform(wp.vec3(*self.center, 0.0)), mesh=self.mesh, cfg=cfg, label="local_terrain_contact"
        )

    def update(self, position, model, solver):
        """Recenter on the sampling lattice before the robot approaches an edge."""
        if np.max(np.abs(np.asarray(position[:2]) - self.center)) < self.size / 4:
            return False
        self.center = np.round(np.asarray(position[:2]) / self.spacing) * self.spacing
        self._sample()
        wp.copy(self.mesh.mesh.points, self.vertices)
        self.mesh.mesh.refit()
        vertices = self.vertices.numpy()
        model.shape_transform[self.shape : self.shape + 1].assign(
            np.array([[*self.center, 0.0, 0.0, 0.0, 0.0, 1.0]], np.float32)
        )
        model.shape_collision_aabb_lower[self.shape : self.shape + 1].assign(vertices.min(axis=0)[None])
        model.shape_collision_aabb_upper[self.shape : self.shape + 1].assign(vertices.max(axis=0)[None])
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        self.updates += 1
        return True
