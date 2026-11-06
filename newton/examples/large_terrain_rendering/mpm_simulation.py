# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""G1 policy, alpha moving sand, and a bounded collider with nested resolutions."""

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

from .alpha_patch import AlphaPatch
from .clipmap import TerrainClipmap
from .controller import G1Controller
from .terrain import sample_heights


class AdaptiveCollider:
    """Fine bedrock under the full particle bed and a coarser surrounding ring.

    Both solvers share the same Warp mesh. Explicit MPM collider registration is
    necessary: default discovery would build its own copy of the rigid shape.
    """

    def __init__(self, args, builder, center, device):
        self.depth = args.sand_depth
        self.grid = TerrainClipmap(cells=32, levels=2, spacing=args.collider_spacing, seed=args.seed, device=device)
        if 32 * args.collider_spacing < args.mpm_size + 2 * args.mpm_buffer + 2 * args.voxel_size:
            raise ValueError("The collider's fine grid must cover the particle bed plus two MPM voxels")
        self.grid.update(center, offset=-self.depth)
        self.mesh = newton.Mesh(self.grid.points.numpy(), self.grid.indices.numpy(), compute_inertia=False)
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.75, is_visible=False, has_particle_collision=False)
        self.shape = builder.add_shape_mesh(-1, mesh=self.mesh, cfg=cfg, label="adaptive_bedrock")

    def update(self, center, model, solver):
        if not self.grid.update(center, offset=-self.depth):
            return False
        wp.copy(self.mesh.mesh.points, self.grid.points)
        self.mesh.mesh.refit()
        points = self.grid.points.numpy()
        model.shape_collision_aabb_lower[self.shape : self.shape + 1].assign(points.min(axis=0)[None])
        model.shape_collision_aabb_upper[self.shape : self.shape + 1].assign(points.max(axis=0)[None])
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        return True


class MPMSimulation:
    """Advance coupled MPM at 100 Hz, rigid dynamics at 600 Hz, and policy at 50 Hz."""

    def __init__(self, args, *, device):
        if not device.is_cuda:
            raise ValueError("The alpha desert demo requires a CUDA device")
        self.device, self.seed = device, args.seed
        self.controller = G1Controller(
            args.config, policy_path=args.policy, asset_path=args.robot_asset, command=args.command, device=device
        )
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.75
        floor = float(sample_heights(np.array([[0, -3]], np.float32), seed=args.seed, device=device)[0])
        self.controller.add_robot(builder, floor_height=floor)
        self.position = self.controller.initial_position.copy()
        self.patch = AlphaPatch(args, builder, self.position, device)
        self.collider = AdaptiveCollider(args, builder, self.patch.center, device)
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        config = SolverImplicitMPM.Config()
        config.voxel_size = args.voxel_size
        config.grid_type = "sparse"
        config.grid_padding = 0
        config.max_active_cell_count = 1 << 16
        config.strain_basis = "P0"
        config.max_iterations = args.mpm_iterations
        config.critical_fraction = 0.0
        config.warmstart_mode = "none"
        config.collider_velocity_mode = "forward"
        bodies = list(range(self.model.body_count))
        self.solver = SolverCoupledProxy(
            self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mujoco",
                    solver=lambda v: SolverMuJoCo(
                        v,
                        use_mujoco_contacts=False,
                        njmax=1024,
                        nconmax=512,
                        iterations=100,
                        ls_iterations=100,
                        impratio=100,
                    ),
                    bodies=bodies,
                    joints=list(range(self.model.joint_count)),
                    substeps=6,
                ),
                SolverCoupledProxy.Entry(
                    name="mpm",
                    solver=lambda v: SolverImplicitMPM(v, config=config),
                    particles=list(range(self.model.particle_count)),
                    in_place=True,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mujoco",
                        destination="mpm",
                        bodies=bodies,
                        mass_scale=1.0,
                        mode="lagged",
                        collision_pipeline=lambda _model: None,
                    ),
                ],
                iterations=1,
            ),
        )
        mpm = self.solver.solver("mpm")
        view = self.solver.view("mpm")
        shape_bodies, flags = view.shape_body.numpy(), view.shape_flags.numpy()
        collider_bodies = sorted(
            {
                int(b)
                for b, f in zip(shape_bodies, flags, strict=True)
                if b >= 0 and f & int(newton.ShapeFlags.COLLIDE_PARTICLES)
            }
        )
        mpm.setup_collider(
            collider_body_ids=[*collider_bodies, None],
            collider_meshes=[*[None] * len(collider_bodies), self.collider.mesh.mesh],
            collider_friction=[0.75] * (len(collider_bodies) + 1),
        )
        self.patch.bind(self.model, self.state, self.solver.entry_state("mpm", phase="input"), mpm)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=512, soft_contact_max=0)
        self.contacts = self.pipeline.contacts()
        self.physics_steps = 0
        self.sim_dt = 0.01
        self.voxel_size = args.voxel_size
        self.next_state = self.state

    def step(self, viewer):
        for _ in range(2):
            self.position = self.state.joint_q[:3].numpy()
            self.patch.update(self.position)
            self.collider.update(self.patch.center, self.model, self.solver)
            if self.physics_steps % 2 == 0:
                self.controller.update(self.state, self.control, viewer, time=self.physics_steps * self.sim_dt)
            self.state.clear_forces()
            viewer.apply_forces(self.state)
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.state, self.control, self.contacts, self.sim_dt)
            self.patch.pin_boundary()
            self.physics_steps += 1
        self.position = self.state.joint_q[:3].numpy()

    def metrics(self):
        q = self.state.joint_q.numpy()
        ground = float(sample_heights(q[:2][None], seed=self.seed, device=self.device)[0])
        return {
            "policy_calls": self.controller.policy.calls,
            "policy_file": self.controller.policy_path.name,
            "simulated_seconds": self.physics_steps * self.sim_dt,
            "mpm_steps": self.physics_steps,
            "particle_count": self.model.particle_count,
            "simulated_terrain_size_m": [self.patch.size, self.patch.size],
            "boundary_terrain_width_m": self.patch.buffer,
            "total_terrain_size_m": [self.patch.width, self.patch.width],
            "dynamic_particles": int(np.count_nonzero(self.model.particle_mass.numpy())),
            "boundary_particles": int(np.count_nonzero(self.model.particle_mass.numpy() == 0)),
            "patch_updates": self.patch.updates,
            "recycled_particles": int(self.patch.recycled.numpy()[0]),
            "collider_updates": self.collider.grid.updates,
            "collider_vertices": len(self.collider.grid.points),
            "collider_triangles": len(self.collider.grid.indices) // 3,
            "root_position_m": q[:3].tolist(),
            "root_up_dot": float(1 - 2 * (q[3] ** 2 + q[4] ** 2)),
            "root_height_above_sand_m": float(q[2] - ground),
            "displacement_xy_m": float(np.linalg.norm(q[:2] - self.controller.initial_position[:2])),
        }

    def test_final(self):
        result = self.metrics()
        assert self.model.particle_count == len(self.patch.reference)
        for array in (self.state.joint_q, self.state.joint_qd, self.state.particle_q, self.state.particle_qd):
            assert np.isfinite(array.numpy()).all()
        assert result["policy_calls"] > 0 and result["mpm_steps"] > 0
        boundary = self.patch.dynamic.numpy() == 0
        assert boundary.any() and (~boundary).any()
        np.testing.assert_array_equal(self.model.particle_mass.numpy()[boundary], 0)
        np.testing.assert_array_equal(self.model.particle_inv_mass.numpy()[boundary], 0)
        np.testing.assert_array_equal(self.state.particle_q.numpy()[boundary], self.patch.anchors.numpy()[boundary])
        np.testing.assert_array_equal(self.state.particle_qd.numpy()[boundary], 0)
        if result["simulated_seconds"] >= 1.0:
            assert result["root_up_dot"] > 0.5 and result["root_height_above_sand_m"] > 0.35, result
        particles = self.state.particle_q.numpy()
        ground = sample_heights(particles[:, :2], seed=self.seed, device=self.device) - self.patch.depth
        assert float(np.min(particles[:, 2] - ground)) > -2 * self.voxel_size, "Sand passed through bedrock"
        if result["simulated_seconds"] >= 5.0 and np.linalg.norm(self.controller.command_velocity[:2]) > 0.1:
            assert result["displacement_xy_m"] > 0.5 and result["patch_updates"] > 0, result
