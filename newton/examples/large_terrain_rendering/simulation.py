# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Rigid-body G1 simulation against a small moving terrain collision mesh."""

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo

from .collision import TerrainCollisionPatch
from .controller import G1Controller
from .terrain import sample_heights


class G1Simulation:
    """Step rigid contacts at 600 Hz and the G1 policy at 50 Hz."""

    def __init__(self, args, *, device):
        self.device = device
        self.seed = args.seed
        self.controller = G1Controller(
            args.config, policy_path=args.policy, asset_path=args.robot_asset, command=args.command, device=device
        )
        floor = float(sample_heights(np.array([[0, -3]], np.float32), seed=args.seed, device=device)[0])
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.75
        self.controller.add_robot(builder, floor_height=floor)
        self.patch = TerrainCollisionPatch(
            size=args.collision_size, spacing=args.collision_spacing, seed=args.seed, device=device
        )
        self.patch.add_shape(builder)
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.next_state = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=512, max_triangle_pairs=65536)
        self.contacts = self.pipeline.contacts()
        self.solver = SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            njmax=1024,
            nconmax=512,
            iterations=100,
            ls_iterations=100,
            impratio=100,
        )
        self.physics_steps = 0
        self.sim_dt = 1.0 / 600
        self.max_contacts = 0
        self.position = self.controller.initial_position.copy()
        self.use_graph = args.use_cuda_graph and device.is_cuda
        self.graph = None

    def _physics_pair(self, viewer):
        """Advance two substeps so capture retains the same input/output buffers."""
        for _ in range(2):
            self.state.clear_forces()
            viewer.apply_forces(self.state)
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.next_state, self.control, self.contacts, self.sim_dt)
            self.state, self.next_state = self.next_state, self.state

    def step(self, viewer):
        """Advance one 60 Hz example frame, refreshing collision coverage first."""
        self.position = self.state.joint_q[:3].numpy()
        self.patch.update(self.position, self.model, self.solver)
        for _ in range(5):
            if self.physics_steps % 12 == 0:
                self.controller.update(self.state, self.control, viewer, time=self.physics_steps * self.sim_dt)
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._physics_pair(viewer)
                if self.use_graph:
                    # The first real pair warms lazy kernels before capture.
                    with wp.ScopedCapture(device=self.device) as capture:
                        self._physics_pair(viewer)
                    self.graph = capture.graph
            self.physics_steps += 2
        self.max_contacts = max(self.max_contacts, int(self.contacts.rigid_contact_count.numpy()[0]))
        self.position = self.state.joint_q[:3].numpy()

    def metrics(self):
        """Report actual policy execution, contact coverage, and robot motion."""
        q = self.state.joint_q.numpy()
        ground = float(sample_heights(q[:2][None], seed=self.seed, device=self.device)[0])
        quat = q[3:7]
        up = 1 - 2 * (quat[0] ** 2 + quat[1] ** 2)
        return {
            "policy_calls": self.controller.policy.calls,
            "policy_file": self.controller.policy_path.name,
            "policy_sha256": self.controller.policy_sha256,
            "physics_cuda_graph": self.graph is not None,
            "physics_steps": self.physics_steps,
            "simulated_seconds": self.physics_steps * self.sim_dt,
            "collision_vertices": self.patch.side**2,
            "collision_triangles": len(self.patch.indices) // 3,
            "collision_patch_size_m": self.patch.size,
            "collision_patch_updates": self.patch.updates,
            "max_contacts": self.max_contacts,
            "root_position_m": q[:3].tolist(),
            "root_height_above_terrain_m": float(q[2] - ground),
            "root_up_dot": float(up),
            "displacement_xy_m": float(np.linalg.norm(q[:2] - self.controller.initial_position[:2])),
        }

    def test_final(self):
        """Require real inference, finite dynamics, ground contacts, and an upright robot."""
        result = self.metrics()
        assert np.isfinite(self.state.joint_q.numpy()).all()
        assert np.isfinite(self.state.joint_qd.numpy()).all()
        assert result["policy_calls"] > 0
        if result["simulated_seconds"] >= 1.0:
            assert result["max_contacts"] > 0, "Robot never contacted the terrain"
            assert result["root_height_above_terrain_m"] > 0.45, f"Robot fell: {result}"
            assert result["root_up_dot"] > 0.5, f"Robot is not upright: {result}"
        if result["simulated_seconds"] >= 3.0 and np.linalg.norm(self.controller.command_velocity[:2]) > 0.1:
            assert result["displacement_xy_m"] > 0.15, "Policy ran but the robot did not move"
