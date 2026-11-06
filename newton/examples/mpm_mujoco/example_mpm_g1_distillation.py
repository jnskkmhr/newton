# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM G1 Policy — Box-Throwing MPM Setup
#
# Unitree G1 locomotion with a distillation policy coupled with implicit
# MPM sand.  MPM material and grid parameters match example_mpm_box_throwing.py
# (basalt, voxel_size=0.025, fixed grid).
#
# Ported from example_mpm_g1_distillation_running_decode.py (old newton)
# to the new newton_dev API.
#
# Command:
#   python newton/examples/mpm/example_mpm_g1_box_throwing_policy.py \
#       --policy models/policy.pt --decoder models/decoder.pt \
#       --config misc/g1_config.yaml --max-trial 3 --video
###########################################################################

from __future__ import annotations

import math
import os

import numpy as np
import torch
import warp as wp
import yaml
from common.kernel import compute_body_forces, subtract_body_force, subtract_body_force_no_inv_mass
from common.mpm_material_cfg import MPMMaterialCfg
from utils import (
    CircularBuffer,
    DictBenchmarkLogger,
    UniformVelocityGenerator,
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    quat_to_euler,
    save_video_cv2,
    yaw_quat,
)

import newton
import newton.examples
from newton import JointTargetMode
from newton.solvers import SolverImplicitMPM

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


class NewtonEnv:
    def __init__(
        self,
        viewer,
        config: dict,
        mjc_to_physx: list[int],
        physx_to_mjc: list[int],
        video: bool = False,
        device: str = "cuda:0",
        episode_length_s: float = 10.0,
        sand_type: str = "sand",
    ):
        self.config = config
        self.num_envs = 1
        self.device = device

        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8  # 400 Hz physics
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.max_episode_length = int(episode_length_s * self.fps)

        self.viewer = viewer
        self.history_length = 30

        """
        define material parameters
        """
        if sand_type == "poppy":
            # poppy seed
            packing_density = 0.6
            self.mpm_material_cfg = MPMMaterialCfg(
                bulk_density=1100.0 * packing_density,  # 1000-1100 for poppy seed
                friction_angle=30.0,  # spawn poppy seed and stops flowing, measure angle.
                youngs_modulus=15.0e6,  # 15-20 MPa is reasonable range
                poissons_ratio=0.3,
                critical_fraction=0.5,
                particles_per_cell=3.0,
                voxel_size=0.025,
                voxel_size_vis=0.04,
                transfer_scheme="apic",
                collider_basis="S2",
                collider_velocity_mode="forward",
                # transfer_scheme="pic",
                # collider_basis="pic27",
                # collider_velocity_mode="finite_difference",
                particle_color=(0.32, 0.24, 0.21),
            )
        elif sand_type == "sand":
            # dry sand
            packing_density = 0.6
            self.mpm_material_cfg = MPMMaterialCfg(
                bulk_density=1500.0 * packing_density,  # newton's default parameters
                friction_angle=30.0,  # spawn poppy seed and stops flowing, measure angle.
                youngs_modulus=15.0e6,  # 15-20 MPa is reasonable range
                poissons_ratio=0.3,
                critical_fraction=0.5,  # 0.5 would be suitable? If particle density is lower than this -> particle stops interacting
                particles_per_cell=3.0,
                voxel_size=0.025,
                voxel_size_vis=0.04,
                transfer_scheme="apic",
                collider_basis="S2",
                collider_velocity_mode="forward",
                # transfer_scheme="pic",
                # collider_basis="pic27",
                # collider_velocity_mode="finite_difference",
                particle_color=(0.45, 0.4, 0.28),
            )
        elif sand_type == "basalt":
            # basalt
            packing_density = 1.0
            self.mpm_material_cfg = MPMMaterialCfg(
                bulk_density=2700.0 * packing_density,  # 2700-3000 for basalt
                friction_angle=40.0,  # spawn poppy seed and stops flowing, measure angle.
                youngs_modulus=40.0e9,  # 15-20 GPa is reasonable range
                poissons_ratio=0.3,
                critical_fraction=0.5,  # 0.5 would be suitable? If particle density is lower than this -> particle stops interacting
                particles_per_cell=3.0,
                voxel_size=0.025,
                voxel_size_vis=0.04,
                transfer_scheme="apic",
                collider_basis="S2",
                collider_velocity_mode="forward",
                # transfer_scheme="pic",
                # collider_basis="pic27",
                # collider_velocity_mode="finite_difference",
                particle_color=(0.28, 0.24, 0.21),
            )
        else:
            raise ValueError(f"Unknown sand type: {sand_type}")

        """
        setup rigid body builder
        """
        builder = newton.ModelBuilder()
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        self.add_robot(builder)
        self.add_scene(builder)

        """
        setup sand builder
        """
        sand_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand_builder)
        self.add_sand(sand_builder, voxel_size=self.mpm_material_cfg.voxel_size_vis)

        """
        finalize rigid body and sand models
        """
        self.model = builder.finalize()
        self.sand_model = sand_builder.finalize()

        """
        set MPM parameters
        """
        MAX_ACTIVE_CELLS = 2**17
        MAX_NODE = 2**20

        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = self.mpm_material_cfg.voxel_size
        mpm_options.tolerance = 1.0e-6
        mpm_options.grid_type = "fixed"
        mpm_options.grid_padding = 50
        mpm_options.max_active_cell_count = MAX_ACTIVE_CELLS
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.transfer_scheme = self.mpm_material_cfg.transfer_scheme
        mpm_options.collider_basis = self.mpm_material_cfg.collider_basis
        mpm_options.collider_velocity_mode = self.mpm_material_cfg.collider_velocity_mode
        mpm_options.critical_fraction = self.mpm_material_cfg.critical_fraction

        """
        setup MPM solver
        """
        self.mpm_solver = SolverImplicitMPM(self.sand_model, mpm_options)

        # Coupling relaxation: foot mass << total body mass, so scale apparent
        # foot mass up to prevent under-coupling (same approach as G1 reference).
        # G1: total=32.34 kg, foot=0.608 kg, 2 feet support → 32.34/(2*0.608) ≈ 26.6
        coupling_relaxation = config.get("coupling_relaxation", 26.6)
        self.mpm_solver.setup_collider(
            model=self.model,
            body_mass=self.model.body_mass * coupling_relaxation,
            body_inv_inertia=self.model.body_inv_inertia / coupling_relaxation,
        )

        """
        setup rigid body solver
        """
        self.rb_solver = newton.solvers.SolverMuJoCo(self.model, ls_parallel=True, njmax=1000)

        """
        prepare simulation states
        """
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.sand_state_0 = self.sand_model.state()
        self.sand_state_0.body_q = wp.empty_like(self.state_0.body_q)
        self.sand_state_0.body_qd = wp.empty_like(self.state_0.body_qd)
        self.sand_state_0.body_f = wp.empty_like(self.state_0.body_f)

        self.control = self.model.control()
        self.contacts = self.model.contacts()

        """
        setup viewer
        """
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True
        self.show_impulses = True
        self.follow_cam = True

        # ------------------------------------------------------------------ #
        # Initial state
        # ------------------------------------------------------------------ #
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)
        self._initial_sand_q = wp.clone(self.sand_state_0.particle_q)
        self._initial_sand_qd = wp.clone(self.sand_state_0.particle_qd)

        # ------------------------------------------------------------------ #
        # Two-way coupling buffers
        # ------------------------------------------------------------------ #
        self.collider_impulses = wp.zeros(MAX_NODE, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(MAX_NODE, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(MAX_NODE, value=-1, dtype=int, device=self.model.device)
        self.collect_collider_impulses()

        self.collider_body_id = self.mpm_solver.collider_body_index
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)

        self.particle_render_colors = wp.full(
            self.sand_model.particle_count,
            value=wp.vec3(*self.mpm_material_cfg.particle_color),
            dtype=wp.vec3,
            device=self.sand_model.device,
        )

        self.capture()

        """
        policy setups
        """
        self.physx_to_mjc_indices = torch.tensor(physx_to_mjc, device=self.device, dtype=torch.long)
        self.mjc_to_physx_indices = torch.tensor(mjc_to_physx, device=self.device, dtype=torch.long)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).unsqueeze(0)

        self.create_obs_buffer()
        self.create_time_buffer()
        self.load_policy_and_setup_tensors()

        # # Seed control targets to standing pose before CUDA graph capture so the
        # # recorded step doesn't drive all joints to zero and corrupt initial state.
        # a_init = torch.cat([torch.zeros(6, device=self.device, dtype=torch.float32), self.joint_pos_initial.squeeze(0)])
        # wp.copy(self.control.joint_target_pos, wp.from_torch(a_init, dtype=wp.float32, requires_grad=False))

        seed = self.config.get("seed", 10)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        root_quat_w = torch.tensor(joint_q[3:7], device=self.device, dtype=torch.float32).unsqueeze(0)[
            :, [3, 0, 1, 2]
        ]  # xyzw → wxyz
        self.root_quat_w_init = root_quat_w.clone()

        self.velocity_generator = UniformVelocityGenerator(
            num_envs=self.num_envs,
            device=self.device,
            step_dt=self.frame_dt,
            # vx_range=self.config.get("vx_range", (1.0, 1.0)),
            vx_range=self.config.get("vx_range", (1.0, 2.5)),
            vy_range=self.config.get("vy_range", (0.0, 0.0)),
            wz_range=self.config.get("wz_range", (0.0, 0.0)),
        )
        self.velocity_generator.sample(root_quat_w)

        # ------------------------------------------------------------------ #
        # Video
        # ------------------------------------------------------------------ #
        self.video = video
        self.frames = [] if video else None

    """
    initializations
    """

    def add_robot(self, builder: newton.ModelBuilder):
        asset_path = self.config["asset_path"]
        if asset_path.endswith(".usd"):
            builder.add_usd(
                source=asset_path,
                xform=wp.transform(wp.vec3(*self.config["mjw_init_pos"])),
                collapse_fixed_joints=False,
                enable_self_collisions=False,
                joint_ordering="dfs",
                hide_collision_shapes=True,
            )
        elif asset_path.endswith(".urdf"):
            builder.add_urdf(
                source=asset_path,
                xform=wp.transform(wp.vec3(*self.config["mjw_init_pos"])),
                floating=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                ignore_inertial_definitions=False,
            )
        elif asset_path.endswith(".xml"):
            builder.add_mjcf(
                source=asset_path,
                xform=wp.transform(wp.vec3(*self.config["mjw_init_pos"])),
                floating=True,
                enable_self_collisions=False,
                ignore_inertial_definitions=False,
            )

        builder.joint_q[:3] = self.config.get("mjw_init_pos", [0.0, -0.5, 0.95])
        builder.joint_q[3:7] = self.config.get("mjw_init_quat", [0.0, 0.0, 0.0, 1.0])
        builder.joint_q[7:] = self.config["mjw_joint_pos"]

        for i in range(len(self.config["mjw_joint_stiffness"])):
            builder.joint_target_ke[i + 6] = self.config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[i + 6] = self.config["mjw_joint_damping"][i]
            builder.joint_armature[i + 6] = self.config["mjw_joint_armature"][i]
            builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

        # Only foot/ankle bodies collide with sand particles
        for body in range(builder.body_count):
            if "ankle" not in builder.body_label[body] and "knee" not in builder.body_label[body]:
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES

        self.foot_indices = [
            body_id
            for body_id in range(builder.body_count)
            # if "ankle_roll" in builder.body_label[body_id]
            if "foot_contact_point" in builder.body_label[body_id]
        ]

    def add_scene(self, builder: newton.ModelBuilder):
        # Flat platform before the sand bed and a landing box after it
        color = (0.3, 0.3, 0.3)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.0, -0.5, 0.05), q=wp.quat_identity()),
            color=wp.vec3(*color),
            hx=1.2,
            hy=0.5,
            hz=0.1,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(0.0, 5.0, 0.05), q=wp.quat_identity()),
            color=wp.vec3(*color),
            hx=1.2,
            hy=1.0,
            hz=0.1,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(1.2 + 0.1, 2.5, 0.05), q=wp.quatf(*(0.0, 0.0, 0.7071068, 0.7071068))),
            color=wp.vec3(*color),
            hx=3.5,
            hy=0.1,
            hz=0.1,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(-1.2 - 0.1, 2.5, 0.05), q=wp.quatf(*(0.0, 0.0, 0.7071068, 0.7071068))),
            color=wp.vec3(*color),
            hx=3.5,
            hy=0.1,
            hz=0.1,
        )
        builder.add_ground_plane()

    def add_sand(self, sand_builder: newton.ModelBuilder, voxel_size: float = 0.04):
        """Sand bed matching example_mpm_box_throwing.py dimensions."""
        bulk_density = self.mpm_material_cfg.bulk_density
        youngs_modulus = self.mpm_material_cfg.youngs_modulus
        poissons_ratio = self.mpm_material_cfg.poissons_ratio
        friction = math.tan(math.radians(self.mpm_material_cfg.friction_angle))
        particles_per_cell = self.mpm_material_cfg.particles_per_cell

        # bed dimensions from example_mpm_box_throwing.py
        bed_lo = np.array([-1.2, 0.0, 0.0])
        bed_hi = np.array([1.2, 4.0, 0.15])
        bed_res = np.array(np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size), dtype=int)

        cell_size = (bed_hi - bed_lo) / bed_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_volume) * bulk_density)

        sand_builder.add_particle_grid(
            pos=wp.vec3(bed_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=bed_res[0] + 1,
            dim_y=bed_res[1] + 1,
            dim_z=bed_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            custom_attributes={
                "mpm:young_modulus": youngs_modulus,
                "mpm:poisson_ratio": poissons_ratio,
                "mpm:friction": friction,
            },
        )

    def create_obs_buffer(self):
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.velocity_command = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float32)
        self.joint_pos = torch.zeros((self.num_envs, self.config["num_dofs"]), device=self.device, dtype=torch.float32)
        self.joint_vel = torch.zeros((self.num_envs, self.config["num_dofs"]), device=self.device, dtype=torch.float32)
        self.last_actions = torch.zeros(
            (self.num_envs, self.config["num_dofs"]), device=self.device, dtype=torch.float32
        )
        self.group_obs_term_history_buffer = {
            "base_ang_vel": CircularBuffer(self.history_length, self.num_envs, self.device),
            "projected_gravity": CircularBuffer(self.history_length, self.num_envs, self.device),
            "joint_pos": CircularBuffer(self.history_length, self.num_envs, self.device),
            "joint_vel": CircularBuffer(self.history_length, self.num_envs, self.device),
        }
        obs_dim = (
            +self.base_ang_vel.shape[1]
            + self.projected_gravity.shape[1]
            + self.velocity_command.shape[1]
            + self.joint_pos.shape[1]
            + self.joint_vel.shape[1]
            + self.last_actions.shape[1]
        )

        encoder_obs_dim = +self.base_ang_vel.shape[1] + self.projected_gravity.shape[1] + 8 + 8
        self.policy_obs = torch.zeros(1, obs_dim, device=self.device, dtype=torch.float32)
        self.encoder_obs = torch.zeros(1, self.history_length, encoder_obs_dim, device=self.device, dtype=torch.float32)

    def create_time_buffer(self):
        self.sim_time = 0.0
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def load_policy_and_setup_tensors(self):
        print("[INFO] Loading policy from:", self.config["policy_path"])
        self.policy = torch.jit.load(self.config["policy_path"], map_location=self.device)
        print("[INFO] Loading decoder from:", self.config["decoder_path"])
        self.decoder = torch.jit.load(self.config["decoder_path"], map_location=self.device)

        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        self.joint_pos_initial = torch.tensor(joint_q[7:], device=self.device, dtype=torch.float32).unsqueeze(0)
        self.act = torch.zeros(1, self.config["num_dofs"], device=self.device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(1, self.config["num_dofs"], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            _dummy_act, dummy_encoder_output = self.policy(self.policy_obs, self.encoder_obs)
            dummy_decoder_output = self.decoder(dummy_encoder_output)
        self.encoder_output = torch.zeros_like(dummy_encoder_output)
        self.decoder_output = torch.zeros_like(dummy_decoder_output)

    """
    graph capture
    """

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self._apply_action()
            self.graph = capture.graph
        else:
            self.graph = None

    """
    physics step
    """

    def step(self):
        self.update_command()
        self._pre_physics_step()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._apply_action()

        self.sim_time += self.frame_dt
        self.common_step_counter += 1
        self.episode_length_buf += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset()

        self._get_observations()
        return self.policy_obs, self.encoder_obs, self.reset_terminated, self.reset_time_outs

    """
    Implementation-specific functions.
    """

    def _get_observations(self):
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        joint_qd = self.state_0.joint_qd if self.state_0.joint_qd is not None else []

        root_quat_xyzw = torch.tensor(joint_q[3:7], device=self.device, dtype=torch.float32).unsqueeze(0)
        root_lin_vel_w = torch.tensor(joint_qd[:3], device=self.device, dtype=torch.float32).unsqueeze(0)
        root_ang_vel_w = torch.tensor(joint_qd[3:6], device=self.device, dtype=torch.float32).unsqueeze(0)
        joint_pos_current = torch.tensor(joint_q[7:], device=self.device, dtype=torch.float32).unsqueeze(0)
        joint_vel_current = torch.tensor(joint_qd[6:], device=self.device, dtype=torch.float32).unsqueeze(0)

        self.base_ang_vel = quat_rotate_inverse(root_quat_xyzw, root_ang_vel_w)
        self.base_lin_vel = quat_rotate_inverse(root_quat_xyzw, root_lin_vel_w)
        self.projected_gravity = quat_rotate_inverse(root_quat_xyzw, self.gravity_vec)
        self.joint_pos = joint_pos_current - self.joint_pos_initial
        self.joint_vel = joint_vel_current
        self.last_actions = self.act

        scale = self.config
        self.group_obs_term_history_buffer["base_ang_vel"].append(self.base_ang_vel * scale.get("ang_vel_scale", 1.0))
        self.group_obs_term_history_buffer["projected_gravity"].append(
            self.projected_gravity * scale.get("projected_gravity_scale", 1.0)
        )
        self.group_obs_term_history_buffer["joint_pos"].append(
            self.joint_pos[:, [0, 1, 2, 3, 6, 7, 8, 9]] * scale.get("joint_pos_scale", 1.0)
        )
        self.group_obs_term_history_buffer["joint_vel"].append(
            self.joint_vel[:, [0, 1, 2, 3, 6, 7, 8, 9]] * scale.get("joint_vel_scale", 1.0)
        )

        self.policy_obs[:, :] = torch.cat(
            [
                self.base_ang_vel * scale.get("ang_vel_scale", 1.0),
                self.projected_gravity * scale.get("projected_gravity_scale", 1.0),
                self.velocity_command * scale.get("command_scale", 1.0),
                self.joint_pos * scale.get("joint_pos_scale", 1.0),
                self.joint_vel * scale.get("joint_vel_scale", 1.0),
                self.last_actions * scale.get("last_action_scale", 1.0),
            ],
            dim=1,
        )
        self.encoder_obs[:, :] = torch.cat([buf.buffer for buf in self.group_obs_term_history_buffer.values()], dim=2)
        return self.policy_obs, self.encoder_obs

    def _pre_physics_step(self):
        with torch.no_grad():
            self.act, self.encoder_output = self.policy(self.policy_obs, self.encoder_obs)
            self.decoder_output = self.decoder(self.encoder_output)
            self.rearranged_act = torch.index_select(self.act, 1, self.mjc_to_physx_indices)
            a = self.joint_pos_initial + self.config["action_scale"] * self.rearranged_act
            a_with_zeros = torch.cat([torch.zeros(6, device=self.device, dtype=torch.float32), a.squeeze(0)])
            a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            wp.copy(self.control.joint_target_pos, a_wp)

    def _apply_action(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            wp.launch(
                compute_body_forces,
                dim=self.collider_impulse_ids.shape[0],
                inputs=[
                    self.sim_dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.state_0.body_f,
                ],
            )
            self.body_sand_forces.assign(self.state_0.body_f)

            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.rb_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            self.simulate_sand()

    def simulate_sand(self):
        if self.sand_state_0.body_q is not None:
            # wp.launch(
            #     subtract_body_force,
            #     dim=self.sand_state_0.body_q.shape,
            #     inputs=[
            #         # self.frame_dt,
            #         self.sim_dt,
            #         self.state_0.body_q,
            #         self.state_0.body_qd,
            #         self.body_sand_forces,
            #         self.model.body_inv_inertia,
            #         self.model.body_inv_mass,
            #         self.sand_state_0.body_q,
            #         self.sand_state_0.body_qd,
            #     ],
            # )
            wp.launch(
                subtract_body_force_no_inv_mass,
                # dim=self.sand_state_0.body_q.shape,
                dim=self.mpm_solver._mpm_model.collider_body_mass.shape,
                inputs=[
                    self.sim_dt,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.body_sand_forces,
                    self.mpm_solver._mpm_model.collider_body_inv_inertia,
                    self.mpm_solver._mpm_model.collider_body_mass,
                    self.sand_state_0.body_q,
                    self.sand_state_0.body_qd,
                ],
            )
        # self.mpm_solver.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=self.frame_dt)
        self.mpm_solver.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=self.sim_dt)
        self.collect_collider_impulses()

    def collect_collider_impulses(self):
        impulses, pos, ids = self.mpm_solver.collect_collider_impulses(self.sand_state_0)
        self.collider_impulse_ids.fill_(-1)
        n = min(impulses.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n].assign(impulses[:n])
        self.collider_impulse_pos[:n].assign(pos[:n])
        self.collider_impulse_ids[:n].assign(ids[:n])

    def update_command(self):
        self.velocity_generator.update_command()
        self.velocity_command[:, :3] = self.velocity_generator.vel_command
        self.velocity_command[:, 3:7] = self.velocity_generator.heading_command

    """
    episode termination
    """

    def _get_dones(self):
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        root_pos = torch.tensor(joint_q[:3], device=self.device, dtype=torch.float32).unsqueeze(0)
        root_quat_xyzw = torch.tensor(joint_q[3:7], device=self.device, dtype=torch.float32).unsqueeze(0)
        root_euler = quat_to_euler(root_quat_xyzw)

        time_out = self.episode_length_buf >= self.max_episode_length
        out_of_bounds = (torch.abs(root_pos[:, 0]) > 1.2) | (root_pos[:, 1] > 5.5)
        roll_violation = torch.abs(root_euler[:, 0]) > torch.pi / 6
        pitch_violation = torch.abs(root_euler[:, 1]) > torch.pi / 6
        return out_of_bounds | roll_violation | pitch_violation, time_out

    def reset(self):
        # Rigid body reset
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

        # Sand reset
        wp.copy(self.sand_state_0.particle_q, self._initial_sand_q)
        wp.copy(self.sand_state_0.particle_qd, self._initial_sand_qd)
        wp.copy(self.sand_state_0.body_q, self.state_0.body_q)
        wp.copy(self.body_sand_forces, wp.zeros_like(self.body_sand_forces))
        self.sand_state_0.clear_forces()

        lsd = self.mpm_solver._last_step_data
        if lsd.ws_impulse_field is not None:
            lsd.ws_impulse_field.dof_values.zero_()
        lsd.save_collider_current_position(self.state_0.body_q)

        self.collider_impulses.zero_()
        self.collider_impulse_pos.zero_()
        self.collider_impulse_ids.fill_(-1)

        # Control reset
        a_with_zeros = torch.cat(
            [torch.zeros(6, device=self.device, dtype=torch.float32), self.joint_pos_initial.squeeze(0)]
        )
        wp.copy(self.control.joint_target_pos, wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False))
        wp.copy(self.control.joint_target_vel, wp.zeros_like(self.control.joint_target_vel))

        # Obs reset
        for buf in self.group_obs_term_history_buffer.values():
            buf.reset()
        self.last_actions.zero_()
        self.act.zero_()

        # Sample new command
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        root_quat_wxyz = torch.tensor(joint_q[3:7], device=self.device, dtype=torch.float32).unsqueeze(0)[
            :, [3, 0, 1, 2]
        ]
        self.velocity_generator.sample(root_quat_wxyz)

        # Episode tracking reset
        self.episode_length_buf[:] = 0
        self.sim_time = 0.0
        self.common_step_counter = 0

        v = self.velocity_generator.vel_command.cpu().numpy().squeeze().tolist()
        print(f"[INFO] Resetting environment. New command: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}] m/s")

        return self._get_observations()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.1,
        )
        voxel_size = self.mpm_solver.voxel_size
        newton.examples.test_particle_state(
            self.sand_state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

    """
    log
    """

    def _get_log(self):
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []

        root_pos_w = torch.tensor(joint_q[:3], device=self.device, dtype=torch.float32).unsqueeze(0)
        root_quat_wxyz = torch.tensor(joint_q[3:7], device=self.device, dtype=torch.float32).unsqueeze(0)[
            :, [3, 0, 1, 2]
        ]
        root_quat_wxyz = quat_mul(root_quat_wxyz, quat_conjugate(self.root_quat_w_init))

        command = self.velocity_command.clone()
        command[:, 3:] = quat_mul(command[:, 3:], quat_conjugate(self.root_quat_w_init))
        yaw_q = yaw_quat(self.velocity_command[:, 3:])
        velocity_cmd_world = quat_rotate(yaw_q, command[:, :3])

        foot_pos = torch.tensor(self.state_0.body_q.numpy(), device=self.device, dtype=torch.float32)[
            self.foot_indices, :3
        ]
        foot_pos[:, 2] -= 0.0398
        foot_height = foot_pos[:, 2].reshape(1, -1)

        contact_force = wp.to_torch(self.state_0.body_f)[self.foot_indices, :3].reshape(1, -1)

        return torch.cat(
            [
                root_pos_w,
                root_quat_wxyz,
                self.base_lin_vel,
                self.base_ang_vel,
                command,
                contact_force,
                velocity_cmd_world,
                foot_height,
            ],
            dim=1,
        )

    """
    rendering
    """

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        if self.follow_cam:
            root_pos = self.state_0.joint_q.numpy()[:3]
            self.viewer.set_camera(pos=wp.vec3(*root_pos) + wp.vec3(2.0, 0.0, 0.0), pitch=-5.0, yaw=-180.0)

        self.viewer.log_points(
            "/sand",
            points=self.sand_state_0.particle_q,
            radii=self.sand_model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )

        if self.show_impulses:
            impulses, pos, _cid = self.mpm_solver.collect_collider_impulses(self.sand_state_0)
            self.viewer.log_lines(
                "/impulses",
                starts=pos,
                ends=pos + 5.0 * impulses,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/impulses", None, None, None)

        self.viewer.end_frame()

        if self.video:
            self.frames.append(self.viewer.get_frame().numpy())

    def render_ui(self, imgui):
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)
        _changed, self.follow_cam = imgui.checkbox("Follow Camera", self.follow_cam)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--video", action="store_true", help="Save video of the simulation.")
    parser.add_argument("--video-speed", type=float, default=1.0, help="Speed multiplier for the saved video.")
    parser.add_argument("--log", action="store_true", help="Log state data to disk.")
    parser.add_argument("--max-trial", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--tag", type=str, default="sand", help="Tag for the log directory.")
    parser.add_argument("--episode-length-s", type=float, default=6.0, help="Episode length in seconds.")
    parser.add_argument(
        "--sand_type", type=str, default="medium", choices=["basalt", "sand", "poppy"], help="Type of sand to simulate."
    )

    viewer, args = newton.examples.init(parser)

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    # asset_dir = str(newton.utils.download_asset("unitree_g1"))
    asset_dir = "/home/jkamohara3/.cache/newton/newton-assets_unitree_g1_540d4676/unitree_g1"

    yaml_file_path = "./misc/g1_beyondmimic.yaml"
    try:
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Robot config file not found: {yaml_file_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file: {e}")
        exit(1)

    config["asset_path"] = os.path.join(asset_dir, config["asset_path"])
    config["policy_path"] = "models/distillation_ppo/v9/policy.pt"
    config["decoder_path"] = "models/distillation_ppo/v9/teacher_decoder.pt"

    # Joint-order mapping between MuJoCo-warp and PhysX conventions
    mjc_to_physx = list(range(config["num_dofs"]))
    physx_to_mjc = list(range(config["num_dofs"]))

    num_envs = 1
    MAX_TRIAL = args.max_trial
    log_dir = "./logs/g1_ours"

    env = NewtonEnv(
        viewer=viewer,
        config=config,
        mjc_to_physx=mjc_to_physx,
        physx_to_mjc=physx_to_mjc,
        video=args.video,
        episode_length_s=args.episode_length_s,
        sand_type=args.sand_type,
    )

    if args.log:
        logger = DictBenchmarkLogger(
            log_dir=log_dir,
            tag=args.tag,
            num_envs=num_envs,
            max_trials=MAX_TRIAL,
            max_episode_length=env.max_episode_length,
            log_item=["obs", "encoder_output", "decoder_output"],
        )

    policy_obs, encoder_obs = env._get_observations()
    log = env._get_log()

    episode_length_log = [0] * num_envs
    episode_counter = [0] * num_envs

    while env.viewer.is_running():
        # while True:
        if args.log:
            logger.log(
                {
                    "obs": log.cpu().numpy(),
                    "encoder_output": env.encoder_output.cpu().numpy(),
                    "decoder_output": env.decoder_output.cpu().numpy(),
                }
            )

        if not env.viewer.is_paused():
            with wp.ScopedTimer("step", active=False):
                policy_obs, encoder_obs, dones, time_out = env.step()
                dones = dones | time_out
                log = env._get_log()

        env.render()

        for i in range(num_envs):
            episode_length_log[i] += 1

        for i, done_flag in enumerate(dones.cpu().numpy()):
            if episode_counter[i] < MAX_TRIAL:
                if done_flag:
                    print(
                        f"[INFO] Env {i}: Episode {episode_counter[i]} ended at {episode_length_log[i] / env.fps:.2f} s"
                    )
                    if args.log:
                        logger.save_to_buffer(trial_id=episode_counter[i], env_idx=i)
                        logger.save_episode_length_to_buffer(
                            trial_id=episode_counter[i],
                            env_idx=i,
                            episode_length=episode_length_log[i],
                        )
                    episode_length_log[i] = 0
                    episode_counter[i] += 1

        if all(tc >= MAX_TRIAL for tc in episode_counter):
            print("[INFO] Max trials reached.")
            break

    if args.log:
        logger.save()
        print(f"[INFO] Saved logs in {logger.log_dir}")

    if args.video:
        video_filename = os.path.join(log_dir, "logs", args.tag, "g1_box_mpm.mp4")
        save_video_cv2(env.frames, video_filename, fps=int(env.fps * args.video_speed))
        print(f"[INFO] Video saved to {video_filename}")

    env.viewer.close()
