# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Procedural dunes and rock geometry for the desert rendering example."""

import numpy as np
import warp as wp


@wp.func
def dune_height(x: float, y: float, seed: int):
    """Evaluate terrain height [m] in world XY coordinates [m]."""
    u = 0.86 * x + 0.51 * y
    v = -0.51 * x + 0.86 * y
    phase = float(seed) * 0.173
    crest = 0.5 + 0.5 * wp.sin(u * 0.036 + 0.85 * wp.sin(v * 0.013 + phase))
    distant = 14.0 * wp.pow(crest, 3.5)
    distant += 6.0 * wp.pow(0.5 + 0.5 * wp.sin(u * 0.019 - v * 0.011 + 2.1), 4.0)
    r = wp.sqrt(x * x + y * y)
    blend = wp.smoothstep(45.0, 150.0, r)
    local = 0.35 * wp.sin(x * 0.12) * wp.cos(y * 0.09)
    local += 0.16 * wp.sin(u * 0.42 + 0.5 * wp.sin(v * 0.17))
    basin_r = wp.sqrt((x - 11.0) * (x - 11.0) + (y - 19.0) * (y - 19.0))
    basin = -1.8 * wp.exp(-wp.pow(basin_r / 6.0, 6.0))
    return local * (1.0 - blend) + distant * blend + basin


@wp.func
def dune_visual_noise(x: float, y: float, seed: int):
    """Add world-space bumps [m] to the visual mesh at three feature sizes."""
    xy = wp.vec2(x, y)
    return (
        0.6 * wp.noise(wp.rand_init(seed, 0), xy / 24.0)
        + 0.25 * wp.noise(wp.rand_init(seed, 1), xy / 8.0)
        + 0.1 * wp.noise(wp.rand_init(seed, 2), xy / 2.0)
    )


@wp.func
def visual_dune_height(x: float, y: float, seed: int):
    """Sample the radial visual mesh, leaving the central contact area smooth."""
    blend = wp.smoothstep(5.0, 15.0, wp.length(wp.vec2(x, y)))
    return dune_height(x, y, seed) + blend * dune_visual_noise(x, y, seed)


@wp.kernel
def _sample_heights(xy: wp.array[wp.vec2], seed: int, heights: wp.array[float]):
    i = wp.tid()
    heights[i] = dune_height(xy[i][0], xy[i][1], seed)


@wp.kernel
def _terrain_vertices(
    ring_count: int,
    sector_count: int,
    radius: float,
    seed: int,
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    uvs: wp.array[wp.vec2],
):
    i = wp.tid()
    x = float(0.0)
    y = float(0.0)
    if i > 0:
        ring = (i - 1) // sector_count + 1
        sector = (i - 1) % sector_count
        r = radius * wp.pow(float(ring) / float(ring_count), 2.0)
        angle = 2.0 * wp.pi * float(sector) / float(sector_count)
        x = r * wp.cos(angle)
        y = r * wp.sin(angle)
    points[i] = wp.vec3(x, y, visual_dune_height(x, y, seed))
    e = 0.05
    dx = (visual_dune_height(x + e, y, seed) - visual_dune_height(x - e, y, seed)) / (2.0 * e)
    dy = (visual_dune_height(x, y + e, seed) - visual_dune_height(x, y - e, seed)) / (2.0 * e)
    normals[i] = wp.normalize(wp.vec3(-dx, -dy, 1.0))
    uvs[i] = wp.vec2(x / 8.0, y / 8.0)


def sample_heights(xy: np.ndarray, *, seed: int, device) -> np.ndarray:
    """Sample terrain heights [m] at world XY positions [m]."""
    positions = wp.array(np.asarray(xy, dtype=np.float32), dtype=wp.vec2, device=device)
    heights = wp.empty(len(xy), dtype=float, device=device)
    wp.launch(_sample_heights, dim=len(xy), inputs=[positions, seed, heights], device=device)
    return heights.numpy()


def build_terrain(*, radius: float, ring_count: int, sector_count: int, seed: int, device):
    """Build a connected mesh with increasing vertex spacing toward the horizon."""
    count = 1 + ring_count * sector_count
    points = wp.empty(count, dtype=wp.vec3, device=device)
    normals = wp.empty(count, dtype=wp.vec3, device=device)
    uvs = wp.empty(count, dtype=wp.vec2, device=device)
    wp.launch(
        _terrain_vertices,
        dim=count,
        inputs=[ring_count, sector_count, radius, seed, points, normals, uvs],
        device=device,
    )
    sectors = np.arange(sector_count, dtype=np.int32)
    next_sectors = (sectors + 1) % sector_count
    fan = np.stack([np.zeros_like(sectors), 1 + sectors, 1 + next_sectors], axis=-1)
    rings = np.arange(ring_count - 1, dtype=np.int32)[:, None]
    a = 1 + rings * sector_count + sectors
    b = 1 + rings * sector_count + next_sectors
    c, d = a + sector_count, b + sector_count
    quads = np.stack([a, c, d, a, d, b], axis=-1).reshape(-1, 3)
    indices = np.concatenate([fan, quads]).astype(np.int32).ravel()
    return points.numpy(), normals.numpy(), uvs.numpy(), indices


def build_rock(*, seed: int):
    """Build an irregular rock with a shared pole at each end and UV coordinates."""
    rng = np.random.default_rng(seed)
    sectors, rings = 40, 24
    angles = rng.uniform(0.0, 2.0 * np.pi, 4)
    points = [[0.0, 0.0, 0.85]]
    uvs = [[0.5, 0.0]]
    for row in range(1, rings):
        phi = np.pi * row / rings
        for column in range(sectors + 1):
            theta = 2.0 * np.pi * column / sectors
            r = 1.0 + 0.13 * np.sin(3 * theta + angles[0]) * np.sin(phi)
            r += 0.10 * np.sin(5 * theta + 4 * phi + angles[1]) * np.sin(phi)
            r += 0.06 * np.cos(8 * theta - 3 * phi + angles[2]) * np.sin(phi)
            points.append(
                [
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    0.85 * np.cos(phi) + 0.08 * np.sin(4 * theta + angles[3]) * np.sin(phi),
                ]
            )
            uvs.append([column / sectors, row / rings])
    bottom = len(points)
    points.append([0.0, 0.0, -0.85])
    uvs.append([0.5, 1.0])
    triangles = []
    for column in range(sectors):
        triangles.append([0, 1 + column, 2 + column])
    for row in range(rings - 2):
        for column in range(sectors):
            a = 1 + row * (sectors + 1) + column
            b = a + sectors + 1
            triangles.extend([[a, b, b + 1], [a, b + 1, a + 1]])
    last = 1 + (rings - 2) * (sectors + 1)
    for column in range(sectors):
        triangles.append([last + column, bottom, last + column + 1])
    points = np.asarray(points, np.float32)
    triangles = np.asarray(triangles, np.int32)
    # Weld duplicated UV seam positions for continuous smooth shading.
    _, welded = np.unique(np.round(points, decimals=5), axis=0, return_inverse=True)
    smooth = np.zeros((welded.max() + 1, 3), np.float32)
    face_normals = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]], points[triangles[:, 2]] - points[triangles[:, 0]]
    )
    for corner in range(3):
        np.add.at(smooth, welded[triangles[:, corner]], face_normals)
    normals = smooth[welded]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
    return points, np.asarray(uvs, np.float32), triangles.ravel(), normals
