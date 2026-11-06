# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Alpha particle recycling adapted locally from mpm_moving_patch_alpha.

Retained dynamic particles keep their state. Boundary particles have zero mass
and remain fixed in world space. Departed particles are wrapped and reset;
there is no terrain history cache. Ground sampling uses the desert height function.
"""

import numpy as np
import warp as wp

import newton

from .terrain import dune_height, sample_heights


@wp.kernel
def _recycle(
    center: wp.vec2,
    width: float,
    seed: int,
    depth: float,
    reference: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    gradients: wp.array[wp.mat33],
    elastic: wp.array[wp.mat33],
    plastic: wp.array[float],
    stress: wp.array[wp.mat33],
    transforms: wp.array[wp.mat33],
    initial_plastic: wp.array[float],
    recycled: wp.array[int],
):
    i = wp.tid()
    p = positions[i]
    lower = center - wp.vec2(width / 2.0)
    if p[0] < lower[0] or p[0] >= lower[0] + width or p[1] < lower[1] or p[1] >= lower[1] + width:
        x = p[0] - wp.floor((p[0] - lower[0]) / width) * width
        y = p[1] - wp.floor((p[1] - lower[1]) / width) * width
        layer = reference[i][2] - (dune_height(reference[i][0], reference[i][1], seed) - depth)
        positions[i] = wp.vec3(x, y, dune_height(x, y, seed) - depth + layer)
        velocities[i] = wp.vec3(0.0)
        gradients[i] = wp.mat33(0.0)
        elastic[i] = wp.identity(n=3, dtype=float)
        plastic[i] = initial_plastic[i]
        stress[i] = wp.mat33(0.0)
        transforms[i] = wp.identity(n=3, dtype=float)
        wp.atomic_add(recycled, 0, 1)


@wp.kernel
def _classify_particles(
    center: wp.vec2,
    size: float,
    dynamic_mass: float,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    mass: wp.array[float],
    inv_mass: wp.array[float],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    changed: wp.array[int],
    gradients: wp.array[wp.mat33],
    elastic: wp.array[wp.mat33],
    plastic: wp.array[float],
    stress: wp.array[wp.mat33],
    transforms: wp.array[wp.mat33],
    initial_plastic: wp.array[float],
):
    i = wp.tid()
    p = positions[i]
    lower = center - wp.vec2(size / 2.0)
    active = int(p[0] >= lower[0] and p[0] < lower[0] + size and p[1] >= lower[1] and p[1] < lower[1] + size)
    if active != dynamic[i]:
        wp.atomic_add(changed, 0, 1)
        velocities[i] = wp.vec3(0.0)
        gradients[i] = wp.mat33(0.0)
    dynamic[i] = active
    mass[i] = wp.where(active == 1, dynamic_mass, 0.0)
    inv_mass[i] = wp.where(active == 1, 1.0 / dynamic_mass, 0.0)
    if active == 0:
        anchors[i] = p
        velocities[i] = wp.vec3(0.0)
        gradients[i] = wp.mat33(0.0)
        elastic[i] = wp.identity(n=3, dtype=float)
        plastic[i] = initial_plastic[i]
        stress[i] = wp.mat33(0.0)
        transforms[i] = wp.identity(n=3, dtype=float)


@wp.kernel
def _pin_boundary(
    dynamic: wp.array[int], anchors: wp.array[wp.vec3], positions: wp.array[wp.vec3], velocities: wp.array[wp.vec3]
):
    i = wp.tid()
    if dynamic[i] == 0:
        positions[i] = anchors[i]
        velocities[i] = wp.vec3(0.0)


class AlphaPatch:
    """A dynamic central square surrounded by a zero-mass boundary and fresh recycling."""

    def __init__(self, args, builder, center, device):
        self.seed, self.depth = args.seed, args.sand_depth
        self.size, self.buffer = args.mpm_size, args.mpm_buffer
        self.width, self.shift = self.size + 2 * self.buffer, args.patch_shift
        self.center = np.floor(np.asarray(center[:2]) / self.shift + 0.5) * self.shift
        self.updates = 0
        res = np.ceil(
            np.nextafter(
                np.array([self.width, self.width, self.depth]) * args.particles_per_cell / args.voxel_size, -np.inf
            )
        ).astype(int)
        cell = np.array([self.width, self.width, self.depth]) / res
        self.dynamic_mass = float(3000.0 * np.prod(cell))
        builder.add_particle_grid(
            pos=wp.vec3(*(self.center - self.width / 2 + cell[:2] / 2), 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(res[0]),
            dim_y=int(res[1]),
            dim_z=int(res[2]) + 1,
            cell_x=float(cell[0]),
            cell_y=float(cell[1]),
            cell_z=float(cell[2]),
            mass=self.dynamic_mass,
            radius_mean=float(max(cell) / 2),
            jitter=float(min(cell) * 0.5),
            custom_attributes={
                "mpm:friction": float(np.tan(np.deg2rad(50.0))),
                "mpm:young_modulus": 40.0e6,
                "mpm:poisson_ratio": 0.3,
            },
        )
        positions = np.asarray(builder.particle_q).copy()
        positions[:, 2] += sample_heights(positions[:, :2], seed=self.seed, device=device) - self.depth
        builder.particle_q = [wp.vec3(*p) for p in positions]
        lower = self.center - self.size / 2
        self.initial_dynamic = np.all((positions[:, :2] >= lower) & (positions[:, :2] < lower + self.size), axis=1)
        builder.particle_mass = np.where(self.initial_dynamic, self.dynamic_mass, 0.0).tolist()

    def bind(self, model, state, mpm_state, mpm_solver):
        self.model, self.state, self.material = model, state, mpm_state.mpm
        self.mpm_solver, self.mpm_state = mpm_solver, mpm_state
        self.reference = wp.clone(model.particle_q)
        self.initial_plastic = wp.clone(self.material.particle_Jp)
        self.recycled = wp.zeros(1, dtype=int, device=model.device)
        self.dynamic = wp.array(self.initial_dynamic.astype(np.int32), dtype=int, device=model.device)
        self.anchors = wp.clone(state.particle_q)
        self.changed = wp.zeros(1, dtype=int, device=model.device)
        self.render_radius = float(model.particle_radius.numpy()[0])

    def update(self, position):
        center = np.floor(np.asarray(position[:2]) / self.shift + 0.5) * self.shift
        if not np.array_equal(center, self.center):
            self.center = center
            self.updates += 1
        wp.launch(
            _recycle,
            dim=self.model.particle_count,
            inputs=[
                wp.vec2(*self.center),
                self.width,
                self.seed,
                self.depth,
                self.reference,
                self.state.particle_q,
                self.state.particle_qd,
                self.material.particle_qd_grad,
                self.material.particle_elastic_strain,
                self.material.particle_Jp,
                self.material.particle_stress,
                self.material.particle_transform,
                self.initial_plastic,
                self.recycled,
            ],
            device=self.model.device,
        )
        self.changed.zero_()
        wp.launch(
            _classify_particles,
            dim=self.model.particle_count,
            inputs=[
                wp.vec2(*self.center),
                self.size,
                self.dynamic_mass,
                self.state.particle_q,
                self.state.particle_qd,
                self.model.particle_mass,
                self.model.particle_inv_mass,
                self.dynamic,
                self.anchors,
                self.changed,
                self.material.particle_qd_grad,
                self.material.particle_elastic_strain,
                self.material.particle_Jp,
                self.material.particle_stress,
                self.material.particle_transform,
                self.initial_plastic,
            ],
            device=self.model.device,
        )
        if int(self.changed.numpy()[0]):
            # The coupled MPM entry owns gathered property arrays and derived density masks.
            for name in ("particle_mass", "particle_inv_mass"):
                wp.copy(getattr(self.mpm_solver.model, name), getattr(self.model, name))
            self.mpm_solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
            self.mpm_solver.reset(self.mpm_state, flags=0)

    def pin_boundary(self):
        """Keep kinematic particles exactly fixed despite numerical grid interpolation."""
        wp.launch(
            _pin_boundary,
            dim=self.model.particle_count,
            inputs=[self.dynamic, self.anchors, self.state.particle_q, self.state.particle_qd],
            device=self.model.device,
        )

    def render_points(self):
        """Partition particles for two independently visible RTX batches without editing physics."""
        # RTX's point interface uploads CPU arrays. Compact here so hidden particles
        # do not become zero-scale instances that the renderer cannot reliably restore.
        positions = self.state.particle_q.numpy()
        dynamic = self.dynamic.numpy().astype(bool)
        return (
            wp.array(positions[dynamic], dtype=wp.vec3, device="cpu"),
            wp.array(positions[~dynamic], dtype=wp.vec3, device="cpu"),
        )
