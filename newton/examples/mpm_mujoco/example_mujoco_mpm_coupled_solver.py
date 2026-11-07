# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the G1 policy testbed with SolverCoupledProxy two-way coupling.

Terrain, material presets, policy observations, control rates, and rendering
are shared with example_mpm_g1_policy. Only the coupling implementation differs.

uv run python newton/examples/mpm_mujoco/example_mujoco_mpm_coupled_solver.py --viewer rtx
"""

from __future__ import annotations

import copy

import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
from newton.examples.mpm_mujoco.example_mpm_g1_policy import (
    RIGID_SUBSTEPS,
    NewtonEnv,
    main,
)
from newton.solvers import SolverImplicitMPM


class Example(NewtonEnv):
    """Use the shared G1 setup with solver-owned proxy impulse exchange."""

    def setup_physics(self, builder, sand_builder):
        # Merge particles into the rigid model, retaining their world assignment.
        # Both builders contain world 0; add_builder otherwise assigns global world -1.
        builder.add_builder(sand_builder)
        builder.particle_world[:] = sand_builder.particle_world
        self.model = builder.finalize()
        self.sand_model = self.model
        mpm_options = self.create_mpm_options(self.model.particle_count)

        rigid_shapes = []
        mpm_shapes = []
        proxy_bodies = set()
        for shape, (body, flags) in enumerate(zip(builder.shape_body, builder.shape_flags, strict=True)):
            if body >= 0 or flags & newton.ShapeFlags.COLLIDE_SHAPES:
                rigid_shapes.append(shape)
            elif flags & newton.ShapeFlags.COLLIDE_PARTICLES:
                mpm_shapes.append(shape)
            if body >= 0 and flags & newton.ShapeFlags.COLLIDE_PARTICLES:
                proxy_bodies.add(body)

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mujoco",
                    solver=self.create_rigid_solver,
                    bodies=list(range(self.model.body_count)),
                    joints=list(range(self.model.joint_count)),
                    shapes=rigid_shapes,
                    substeps=RIGID_SUBSTEPS,
                ),
                SolverCoupledProxy.Entry(
                    name="mpm",
                    solver=lambda view: SolverImplicitMPM(model=view, config=mpm_options),
                    particles=list(range(self.model.particle_count)),
                    shapes=mpm_shapes,
                    substeps=1,
                    in_place=True,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mujoco",
                        destination="mpm",
                        bodies=sorted(proxy_bodies),
                        mass_scale=self.config.get("coupling_relaxation", 26.6),
                        mode="lagged",
                        collision_pipeline=lambda _model: None,
                    )
                ],
                iterations=1,
            ),
        )
        self.rb_solver = self.solver.solver("mujoco")
        self.mpm_solver = self.solver.solver("mpm")
        self.state_0 = self.model.state()
        self.state_1 = self.state_0
        self.sand_state_0 = self.state_0
        self.control = self.model.control()
        self.rigid_collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.rigid_collision_pipeline.contacts()

    def setup_coupling_buffers(self):
        """Let SolverCoupledProxy allocate and manage all feedback buffers."""

    def render_model(self):
        """Draw particles once through the shared material-colored /sand point set."""
        model = copy.copy(self.model)
        model.particle_count = 0
        return model

    def _apply_action(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.rigid_collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)

    def reset_physics(self):
        """Restore the scene and clear MPM history and proxy feedback together."""
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        wp.copy(self.state_0.particle_q, self._initial_sand_q)
        wp.copy(self.state_0.particle_qd, self._initial_sand_qd)
        self.state_0.clear_forces()
        self.solver.reset(self.state_0)


if __name__ == "__main__":
    main(Example)
