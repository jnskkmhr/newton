# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""A hand-trowel-shaped shovel with a curved blade and convex collision pieces."""

import math

import numpy as np
import warp as wp

import newton


def _mesh(vertices, faces, *, compute_inertia=True):
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    normals = np.zeros_like(vertices)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    return newton.Mesh(vertices, faces.ravel(), normals=normals, compute_inertia=compute_inertia)


def _blade_surface(nx, ny):
    """Sample the pointed outline and shallow crosswise curvature [m]."""
    x = np.linspace(-0.24, 0.36, nx)
    width = np.interp(
        x,
        [-0.24, -0.16, -0.06, 0.06, 0.17, 0.26, 0.32, 0.36],
        [0.13, 0.155, 0.16, 0.15, 0.125, 0.087, 0.048, 0.004],
    )
    u = np.linspace(-1.0, 1.0, ny)
    points = np.empty((nx, ny, 3))
    points[:, :, 0] = x[:, None]
    points[:, :, 1] = width[:, None] * u[None, :]
    points[:, :, 2] = 0.035 * (width[:, None] / 0.16) * u[None, :] ** 2 - 0.015 * (x[:, None] + 0.24) / 0.6
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b = i * ny + j, (i + 1) * ny + j
            faces.extend([(a, b, b + 1), (a, b + 1, a + 1)])
    return points.reshape(-1, 3), faces


def add_shovel(builder: newton.ModelBuilder) -> str:
    """Add one passive rigid trowel, opening upward with its tip toward +X.

    The reference hand tool is enlarged for the example's MPM resolution: the
    blade is 0.60 m long and up to 0.32 m wide. A smooth display mesh covers
    separate triangular-prism colliders so the curved interior remains empty.
    """
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.9), wp.quat_identity()), label="shovel")
    thickness = 0.014
    metal = (0.20, 0.21, 0.22)
    # A closed visual blade: top, underside, and a thin rim around the outline.
    top, faces = _blade_surface(25, 13)
    n = len(top)
    bottom = top - np.array([0.0, 0.0, thickness])
    closed_faces = list(faces) + [(a + n, c + n, b + n) for a, b, c in faces]
    perimeter = list(range(0, n, 13)) + list(range(n - 12, n)) + list(range(n - 14, 11, -13)) + list(range(11, 0, -1))
    for a, b in zip(perimeter, perimeter[1:] + perimeter[:1], strict=True):
        closed_faces.extend([(a, a + n, b + n), (a, b + n, b)])
    builder.add_shape_mesh(
        body,
        mesh=_mesh(np.vstack([top, bottom]), closed_faces, compute_inertia=False),
        cfg=builder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False),
        color=metal,
        label="shovel_blade_visual",
    )
    # Small convex pieces approximate the concave blade without bridging its bowl.
    # Register as MESH: implicit MPM supports MESH, but not CONVEX_MESH.
    # MuJoCo uses the convex hull of each separate mesh collider.
    points, triangles = _blade_surface(9, 5)
    prism_faces = [(0, 1, 2), (3, 5, 4), (0, 3, 4), (0, 4, 1), (1, 4, 5), (1, 5, 2), (2, 5, 3), (2, 3, 0)]
    for i, triangle in enumerate(triangles):
        upper = points[list(triangle)]
        lower = upper - np.array([0.0, 0.0, thickness])
        builder.add_shape_mesh(
            body,
            mesh=_mesh(np.vstack([upper, lower]), prism_faces),
            cfg=builder.ShapeConfig(density=7800.0, mu=0.5, is_visible=False),
            label=f"shovel_blade_collision_{i}",
        )
    # Short steel neck, angled slightly upward toward the wooden handle.
    start, end = wp.vec3(-0.21, 0.0, 0.018), wp.vec3(-0.48, 0.0, 0.085)
    axis = end - start
    builder.add_shape_cylinder(
        body,
        xform=wp.transform(
            0.5 * (start + end), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.atan2(axis[0], axis[2]))
        ),
        radius=0.022,
        half_height=0.5 * wp.length(axis),
        cfg=builder.ShapeConfig(density=7800.0, mu=0.5),
        color=metal,
        label="shovel_neck",
    )
    # Rounded wooden grip: a lathed profile with a bulge and a narrower neck.
    stations = [-0.80, -0.795, -0.78, -0.75, -0.68, -0.58, -0.52, -0.47]
    radii = [0.002, 0.018, 0.032, 0.042, 0.044, 0.036, 0.027, 0.024]
    segments = 24
    vertices = []
    for x, radius in zip(stations, radii, strict=True):
        z = 0.085 + (-x - 0.48) * 0.25
        for j in range(segments):
            angle = 2.0 * math.pi * j / segments
            vertices.append((x, radius * math.cos(angle), z + radius * math.sin(angle)))
    faces = []
    for i in range(len(stations) - 1):
        for j in range(segments):
            a, b = i * segments + j, i * segments + (j + 1) % segments
            faces.extend([(a, b, b + segments), (a, b + segments, a + segments)])
    for j in range(1, segments - 1):
        faces.append((0, j + 1, j))
        offset = (len(stations) - 1) * segments
        faces.append((offset, offset + j, offset + j + 1))
    builder.add_shape_mesh(
        body,
        mesh=_mesh(vertices, faces),
        cfg=builder.ShapeConfig(density=650.0, mu=0.5),
        color=(0.57, 0.34, 0.14),
        label="shovel_wooden_handle",
    )
    return "shovel"
