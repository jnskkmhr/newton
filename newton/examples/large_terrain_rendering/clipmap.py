# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Nested terrain grids with stitched edges, shared by rendering and local contact."""

import numpy as np
import warp as wp

from .terrain import dune_height, dune_visual_noise


def grid_topology(cells, levels, spacing):
    """Build a central grid and square rings; split coarse edges at fine vertices."""
    if cells < 8 or cells % 4 or not 1 <= levels <= 12 or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Use cells divisible by four (>=8), 1..12 levels, and positive finite spacing")
    vertices, triangles, lookup = [], [], {}

    def vertex(point):
        key = tuple(point)
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append(key)
        return lookup[key]

    half, inner = cells // 2, cells // 4
    for level in range(levels):
        scale = spacing * 2**level
        for x in range(-half, half):
            for y in range(-half, half):
                if level and -inner <= x < inner and -inner <= y < inner:
                    continue
                corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
                polygon = []
                for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
                    polygon.append(vertex(np.asarray(a) * scale))
                    midpoint = (np.asarray(a) + b) / 2
                    if level and max(abs(midpoint)) == inner:
                        polygon.append(vertex(midpoint * scale))
                if len(polygon) == 4:
                    triangles.extend([(polygon[0], polygon[1], polygon[2]), (polygon[0], polygon[2], polygon[3])])
                else:
                    center = vertex(np.array([x + 0.5, y + 0.5]) * scale)
                    triangles.extend((center, a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1], strict=True))
    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32).ravel()


@wp.func
def _surface(
    x: float,
    y: float,
    seed: int,
    offset: float,
    hole: wp.vec2,
    inner: float,
    outer: float,
    depth: float,
    visual_noise: bool,
):
    distance = wp.max(wp.abs(x - hole[0]), wp.abs(y - hole[1]))
    cut = depth * (1.0 - wp.smoothstep(inner, outer, distance))
    height = dune_height(x, y, seed) + offset - cut
    if visual_noise:
        # Keep the MPM opening aligned with particles; add roughness beyond it.
        blend = wp.smoothstep(outer, outer + 5.0, distance)
        height += blend * dune_visual_noise(x, y, seed)
    return height


@wp.kernel
def _sample(
    xy: wp.array[wp.vec2],
    center: wp.vec2,
    seed: int,
    offset: float,
    hole: wp.vec2,
    inner: float,
    outer: float,
    depth: float,
    visual_noise: bool,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    uvs: wp.array[wp.vec2],
):
    i = wp.tid()
    p = xy[i] + center
    z = _surface(p[0], p[1], seed, offset, hole, inner, outer, depth, visual_noise)
    dx = (
        _surface(p[0] + 0.05, p[1], seed, offset, hole, inner, outer, depth, visual_noise)
        - _surface(p[0] - 0.05, p[1], seed, offset, hole, inner, outer, depth, visual_noise)
    ) / 0.1
    dy = (
        _surface(p[0], p[1] + 0.05, seed, offset, hole, inner, outer, depth, visual_noise)
        - _surface(p[0], p[1] - 0.05, seed, offset, hole, inner, outer, depth, visual_noise)
    ) / 0.1
    points[i] = wp.vec3(p[0], p[1], z)
    normals[i] = wp.normalize(wp.vec3(-dx, -dy, 1.0))
    uvs[i] = p / 8.0


class TerrainClipmap:
    """Camera-centered nested grids with full-buffer updates on snapped movement.

    All levels share a center snapped to the finest spacing. This initial clipmap
    refreshes all levels together, rather than using incremental strip uploads.
    """

    def __init__(self, *, cells=32, levels=8, spacing=0.5, seed=7, visual_noise=False, device=None):
        xy, indices = grid_topology(cells, levels, spacing)
        self.xy = wp.array(xy, dtype=wp.vec2, device=device)
        self.indices = wp.array(indices, dtype=wp.int32, device=device)
        self.points = wp.empty(len(xy), dtype=wp.vec3, device=device)
        self.normals = wp.empty_like(self.points)
        self.uvs = wp.empty(len(xy), dtype=wp.vec2, device=device)
        self.spacing, self.seed = spacing, seed
        self.visual_noise = visual_noise
        self.extent = cells * spacing * 2 ** (levels - 1)
        self.key = None
        self.updates = 0

    def update(self, center, *, offset=0.0, hole=(0.0, 0.0), inner=0.0, outer=1.0, depth=0.0):
        """Resample when the camera or local MPM opening changes; return whether updated."""
        center = np.floor(np.asarray(center[:2]) / self.spacing + 0.5) * self.spacing
        key = (*center, offset, *hole, inner, outer, depth)
        if key == self.key:
            return False
        wp.launch(
            _sample,
            dim=len(self.xy),
            inputs=[
                self.xy,
                wp.vec2(*center),
                self.seed,
                offset,
                wp.vec2(*hole),
                inner,
                outer,
                depth,
                self.visual_noise,
                self.points,
                self.normals,
                self.uvs,
            ],
            device=self.points.device,
        )
        self.key = key
        self.updates += 1
        return True
