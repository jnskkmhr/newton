# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM G1 Policy — Granular Terrain Setup
#
# Unitree G1 locomotion with a distillation policy coupled with implicit
# MPM terrain. The default material is closely packed poppy seeds with
# 0.02 m voxels and 1.25 particles per cell along each axis (sparse grid).
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

import newton
import newton.examples
from newton import JointTargetMode
from newton.examples.mpm_mujoco.common.kernel import (
    compute_body_forces,
    subtract_body_force,
)
from newton.examples.mpm_mujoco.common.mpm_material_cfg import MPMMaterialCfg
from newton.examples.mpm_mujoco.utils import (
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
from newton.solvers import SolverImplicitMPM

# Particle/material settings match the moving-patch configuration.
MPM_VOXEL_SIZE = 0.02
MPM_COLLIDER_MARGIN = 0.0
FOOT_CONTACT_MARGIN = 0.04 * 0.5
PLATFORM_CONTACT_MARGIN = 0.004
PLATFORM_CONTACT_GAP = 0.002
MPM_PARTICLES_PER_CELL = 1.25
MPM_VISUAL_COLOR = (0.32, 0.24, 0.21)

# Bulk bed density [kg/m³] and friction coefficient. These are tuning presets,
# not calibrated material models. The elastic/plastic settings are shared.
# Supplied density references: FHWA (dense sand), Manick et al. (basalt),
# and Yang et al. (poppy seeds).
_COMMON_MATERIAL = {
    "young_modulus": 1.0e15,
    "poisson_ratio": 0.3,
    "yield_pressure": 1.0e15,
    "tensile_yield_ratio": 0.0,
    "yield_stress": 0.0,
    "hardening": 0.0,
    "dilatancy": 0.0,
    "viscosity": 0.0,
}
MATERIAL_PRESETS = {
    "sand": {**_COMMON_MATERIAL, "density": 2700.0, "friction": 0.58},
    "beach_sand": {**_COMMON_MATERIAL, "density": 1800.0, "friction": 0.80},
    "basalt": {**_COMMON_MATERIAL, "density": 1660.0, "friction": 0.88},
    "poppy_loose": {**_COMMON_MATERIAL, "density": 640.0, "friction": 0.73},
    "poppy": {**_COMMON_MATERIAL, "density": 682.0, "friction": 1.07},
}

# The 2 cm bed needs more cells than the previous 131,072-cell allocation.
# Keep a bounded allocation so the sparse grid can rebuild during graph replay.
MPM_MAX_ACTIVE_CELL_COUNT = 1 << 19
RIGID_SUBSTEPS = 2

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
        sand_type: str = "poppy",
    ):
        self.config = config
        self.num_envs = 1
        self.device = device

        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4  # 200 Hz MPM, matching IsaacLab; rigid integration runs at 400 Hz.
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.max_episode_length = int(episode_length_s * self.fps)

        self.viewer = viewer
        self.history_length = self.config.get("history_length", 10)

        """
        define material parameters
        """
        if sand_type not in MATERIAL_PRESETS:
            raise ValueError(f"Unknown sand type: {sand_type}; choose from {tuple(MATERIAL_PRESETS)}")
        material = MATERIAL_PRESETS[sand_type]
        self.mpm_material_cfg = MPMMaterialCfg(
            bulk_density=material["density"],
            friction_angle=math.degrees(math.atan(material["friction"])),
            youngs_modulus=material["young_modulus"],
            poissons_ratio=material["poisson_ratio"],
            critical_fraction=0.0,
            particles_per_cell=MPM_PARTICLES_PER_CELL,
            voxel_size=MPM_VOXEL_SIZE,
            voxel_size_vis=MPM_VOXEL_SIZE,
            transfer_scheme="apic",
            collider_basis="pic27",
            collider_velocity_mode="forward",
            particle_color=MPM_VISUAL_COLOR,
        )
        self.particle_material_attributes = {
            f"mpm:{name}": value for name, value in material.items() if name != "density"
        }

        """
        setup rigid body builder
        """
        builder = newton.ModelBuilder()
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 160000.0
        builder.default_shape_cfg.kd = 1100.0
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        builder.default_shape_cfg.margin = 0.0
        builder.begin_world()
        self.add_robot(builder)
        builder.end_world()
        self.add_scene(builder)

        """
        setup sand builder
        """
        sand_builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(sand_builder)
        sand_builder.begin_world()
        self.add_sand(sand_builder, voxel_size=self.mpm_material_cfg.voxel_size)
        sand_builder.end_world()

        """
        finalize rigid body and sand models
        """
        self.model = builder.finalize()
        self.sand_model = sand_builder.finalize()

        """
        set MPM parameters
        """
        MAX_NODE = 2**20

        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = MPM_VOXEL_SIZE
        mpm_options.grid_type = "sparse"
        mpm_options.grid_padding = 0
        mpm_options.max_active_cell_count = MPM_MAX_ACTIVE_CELL_COUNT
        mpm_options.strain_basis = "P0"
        mpm_options.transfer_scheme = "apic"
        mpm_options.max_iterations = 25
        mpm_options.tolerance = 1.0e-4
        mpm_options.warmstart_mode = "auto"
        mpm_options.velocity_basis = "Q1"
        mpm_options.collider_basis = "pic27"
        mpm_options.collider_velocity_mode = "forward"
        mpm_options.solver = "auto"
        mpm_options.critical_fraction = 0.0
        mpm_options.separate_worlds = True
        mpm_options.project_outside_colliders = False

        """
        setup MPM solver
        """
        self.mpm_solver = SolverImplicitMPM(self.sand_model, mpm_options)

        # Coupling relaxation: foot mass << total body mass, so scale apparent
        # foot mass up to prevent under-coupling (same approach as G1 reference).
        # G1: total=32.34 kg, foot=0.608 kg, 2 feet support → 32.34/(2*0.608) ≈ 26.6
        coupling_relaxation = config.get("coupling_relaxation", 26.6)
        # One MPM collider per body (including the static body -1).
        # Override MPM margins without changing rigid foot/platform contacts.
        collider_body_ids = sorted(
            {
                body
                for body, flags in zip(builder.shape_body, builder.shape_flags, strict=True)
                if flags & newton.ShapeFlags.COLLIDE_PARTICLES
            }
        )
        self.mpm_solver.setup_collider(
            model=self.model,
            collider_body_ids=collider_body_ids,
            collider_margins=[MPM_COLLIDER_MARGIN] * len(collider_body_ids),
            body_mass=self.model.body_mass * coupling_relaxation,
            body_inv_inertia=self.model.body_inv_inertia / coupling_relaxation,
        )

        """
        setup rigid body solver
        """
        self.rb_solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            iterations=100,
            use_mujoco_contacts=False,
        )

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
        self.rigid_collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.rigid_collision_pipeline.contacts()

        """
        setup viewer
        """
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True
        # pic27 provides one impulse slot per particle, including zero impulses.
        # Drawing every slot as an RTX line is expensive; keep this debug view opt-in.
        self.show_impulses = False
        self.show_velocity = True
        self.follow_cam = True
        self._rtx_particle_style_initialized = False

        # ------------------------------------------------------------------ #
        # Initial state
        # ------------------------------------------------------------------ #
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)
        self._initial_sand_q = wp.clone(self.sand_state_0.particle_q)
        self._initial_sand_qd = wp.clone(self.sand_state_0.particle_qd)
        # wp.launch(kernel=sample_sinusoidal_terrain, dim=self.sand_model.particle_count, inputs=[self._initial_sand_q, 0.1, 0.1])
        # wp.copy(self.sand_state_0.particle_q, self._initial_sand_q)

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
            vx_range=self.config.get("vx_range", (1.0, 1.0)),
            # vx_range=self.config.get("vx_range", (1.0, 2.5)),
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

        # Joint targets use the coordinate layout, so the free base takes the
        # first 7 slots. Seed the actuated tail with the standing pose, else the
        # PD controllers drive every joint to zero before the first policy step.
        builder.joint_target_q[7:] = self.config["mjw_joint_pos"]

        for i in range(len(self.config["mjw_joint_stiffness"])):
            builder.joint_target_ke[i + 6] = self.config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[i + 6] = self.config["mjw_joint_damping"][i]
            builder.joint_armature[i + 6] = self.config["mjw_joint_armature"][i]
            builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

        for body, label in enumerate(builder.body_label):
            if "ankle" in label or "foot" in label:
                for shape in builder.body_shapes[body]:
                    builder.shape_margin[shape] = FOOT_CONTACT_MARGIN

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
        platform_cfg = builder.default_shape_cfg.copy()
        platform_cfg.margin = PLATFORM_CONTACT_MARGIN
        platform_cfg.gap = PLATFORM_CONTACT_GAP
        builder.add_shape_box(
            body=-1,
            cfg=platform_cfg,
            xform=wp.transform(p=wp.vec3(0.0, -0.5, 0.05), q=wp.quat_identity()),
            color=wp.vec3(*color),
            hx=1.2,
            hy=0.5,
            hz=0.1,
        )
        builder.add_shape_box(
            body=-1,
            cfg=platform_cfg,
            xform=wp.transform(p=wp.vec3(0.0, 4.0, 0.05), q=wp.quat_identity()),
            color=wp.vec3(*color),
            hx=1.2,
            hy=1.0,
            hz=0.1,
        )
        builder.add_shape_box(
            body=-1,
            cfg=platform_cfg,
            xform=wp.transform(p=wp.vec3(1.2 + 0.1, 2.5, 0.05), q=wp.quatf(*(0.0, 0.0, 0.7071068, 0.7071068))),
            color=wp.vec3(*color),
            hx=3.5,
            hy=0.1,
            hz=0.1,
        )
        builder.add_shape_box(
            body=-1,
            cfg=platform_cfg,
            xform=wp.transform(p=wp.vec3(-1.2 - 0.1, 2.5, 0.05), q=wp.quatf(*(0.0, 0.0, 0.7071068, 0.7071068))),
            color=wp.vec3(*color),
            hx=3.5,
            hy=0.1,
            hz=0.1,
        )
        builder.add_ground_plane(cfg=platform_cfg)

    def add_sand(self, sand_builder: newton.ModelBuilder, voxel_size: float = 0.04):
        """Sand bed matching example_mpm_box_throwing.py dimensions."""
        bulk_density = self.mpm_material_cfg.bulk_density
        particles_per_cell = self.mpm_material_cfg.particles_per_cell

        # bed dimensions from example_mpm_box_throwing.py
        bed_lo = np.array([-1.2, 0.0, 0.0])
        bed_hi = np.array([1.2, 3.0, 0.2])
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
            custom_attributes=self.particle_material_attributes,
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
            "velocity_command": CircularBuffer(self.history_length, self.num_envs, self.device),
            "joint_pos": CircularBuffer(self.history_length, self.num_envs, self.device),
            "joint_vel": CircularBuffer(self.history_length, self.num_envs, self.device),
            "last_actions": CircularBuffer(self.history_length, self.num_envs, self.device),
        }
        obs_dim = self.history_length * (
            +self.base_ang_vel.shape[1]
            + self.projected_gravity.shape[1]
            + self.velocity_command.shape[1]
            + self.joint_pos.shape[1]
            + self.joint_vel.shape[1]
            + self.last_actions.shape[1]
        )
        self.obs = torch.zeros(1, obs_dim, device=self.device, dtype=torch.float32)

    def create_time_buffer(self):
        self.sim_time = 0.0
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def load_policy_and_setup_tensors(self):
        """Load policy and setup initial tensors for robot control."""
        print("[INFO] Loading policy from:", self.config["policy_path"])
        self.policy = torch.jit.load(self.config["policy_path"], map_location=self.device)

        # Handle potential None state
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        self.joint_pos_initial = torch.tensor(joint_q[7:], device=self.device, dtype=torch.float32).unsqueeze(0)
        self.act = torch.zeros(1, self.config["num_dofs"], device=self.device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(1, self.config["num_dofs"], device=self.device, dtype=torch.float32)

        # `control.joint_target_q` follows the coordinate layout, so the floating
        # base occupies 7 slots (3 position + 4 quaternion) ahead of the actuated
        # DOFs -- not 6. Keep a persistent torch buffer mirroring that layout and
        # seed the base slots from the model so only the tail is rewritten.
        num_targets = self.control.joint_target_q.shape[0]
        self.joint_target_offset = num_targets - self.config["num_dofs"]
        self.joint_target_q_torch = torch.zeros(num_targets, device=self.device, dtype=torch.float32)
        self.joint_target_q_torch[: self.joint_target_offset] = torch.as_tensor(
            self.model.joint_target_q.numpy()[: self.joint_target_offset], device=self.device
        )
        self.joint_target_q_wp = wp.from_torch(self.joint_target_q_torch, dtype=wp.float32, requires_grad=False)

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
            self.mpm_solver.check_sparse_grid_rebuild_status()
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

        return self.obs, self.reset_terminated, self.reset_time_outs

    """
    Implementation-specific functions.
    """

    def _get_observations(self):
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        joint_qd = self.state_0.joint_qd if self.state_0.joint_qd is not None else []

        # root_pos_w = torch.tensor(joint_q[:3], device=self.device, dtype=torch.float32).unsqueeze(0)
        root_quat_w = torch.tensor(joint_q[3:7], device=self.device, dtype=torch.float32).unsqueeze(0)
        root_lin_vel_w = torch.tensor(joint_qd[:3], device=self.device, dtype=torch.float32).unsqueeze(0)  # maybe used
        root_ang_vel_w = torch.tensor(joint_qd[3:6], device=self.device, dtype=torch.float32).unsqueeze(0)

        joint_pos_current = torch.tensor(joint_q[7:], device=self.device, dtype=torch.float32).unsqueeze(0)
        joint_vel_current = torch.tensor(joint_qd[6:], device=self.device, dtype=torch.float32).unsqueeze(0)

        self.base_ang_vel = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
        self.base_lin_vel = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
        self.projected_gravity = quat_rotate_inverse(root_quat_w, self.gravity_vec)
        self.joint_pos = joint_pos_current - self.joint_pos_initial
        self.joint_vel = joint_vel_current
        self.last_actions = self.act

        # add to history buffer
        self.group_obs_term_history_buffer["base_ang_vel"].append(
            self.base_ang_vel * self.config.get("ang_vel_scale", 1.0)
        )
        self.group_obs_term_history_buffer["projected_gravity"].append(
            self.projected_gravity * self.config.get("projected_gravity_scale", 1.0)
        )
        self.group_obs_term_history_buffer["velocity_command"].append(
            self.velocity_command * self.config.get("command_scale", 1.0)
        )
        self.group_obs_term_history_buffer["joint_pos"].append(self.joint_pos * self.config.get("joint_pos_scale", 1.0))
        self.group_obs_term_history_buffer["joint_vel"].append(self.joint_vel * self.config.get("joint_vel_scale", 1.0))
        self.group_obs_term_history_buffer["last_actions"].append(
            self.last_actions * self.config.get("last_action_scale", 1.0)
        )
        self.obs[:, :] = torch.cat(
            [group_obs.buffer.reshape(1, -1) for group_obs in self.group_obs_term_history_buffer.values()], dim=1
        )

        return self.obs

    def _pre_physics_step(self):
        with torch.no_grad():
            self.act = self.policy(self.obs)
            self.rearranged_act = torch.index_select(self.act, 1, self.mjc_to_physx_indices)
            a = self.joint_pos_initial + self.config["action_scale"] * self.rearranged_act
            self.joint_target_q_torch[self.joint_target_offset :] = a.squeeze(0)
            wp.copy(self.control.joint_target_q, self.joint_target_q_wp)

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

            # Hold the lagged MPM force over both rigid integration steps.
            for _ in range(RIGID_SUBSTEPS):
                self.state_0.clear_forces()
                self.state_0.body_f.assign(self.body_sand_forces)
                self.viewer.apply_forces(self.state_0)
                self.rigid_collision_pipeline.collide(self.state_0, self.contacts)
                self.rb_solver.step(
                    self.state_0, self.state_1, self.control, self.contacts, self.sim_dt / RIGID_SUBSTEPS
                )
                self.state_0, self.state_1 = self.state_1, self.state_0

            self.simulate_sand()

    def simulate_sand(self):
        if self.sand_state_0.body_q is not None:
            wp.launch(
                subtract_body_force,
                dim=self.sand_state_0.body_q.shape,
                inputs=[
                    # self.frame_dt,
                    self.sim_dt,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.body_sand_forces,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.sand_state_0.body_q,
                    self.sand_state_0.body_qd,
                ],
            )
            # wp.launch(
            #     subtract_body_force_no_inv_mass,
            #     # dim=self.sand_state_0.body_q.shape,
            #     dim=self.mpm_solver._mpm_model.collider_body_mass.shape,
            #     inputs=[
            #         self.sim_dt,
            #         self.state_0.body_q,
            #         self.state_0.body_qd,
            #         self.body_sand_forces,
            #         self.mpm_solver._mpm_model.collider_body_inv_inertia,
            #         self.mpm_solver._mpm_model.collider_body_mass,
            #         self.sand_state_0.body_q,
            #         self.sand_state_0.body_qd,
            #     ],
            # )
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
        root_pos_w = torch.tensor(joint_q[:3], device=self.device, dtype=torch.float32).unsqueeze(0)
        root_quat_w = torch.tensor(joint_q[3:7], device=self.device, dtype=torch.float32).unsqueeze(0)
        root_euler_w = quat_to_euler(root_quat_w)

        time_out = self.episode_length_buf >= self.max_episode_length
        bound_x = 3.0
        bound_y = 3.0
        out_of_bounds_x = torch.abs(root_pos_w[:, 0]) > bound_x
        out_of_bounds_y = root_pos_w[:, 1] > bound_y
        roll_violation = torch.abs(root_euler_w[:, 0]) > torch.pi / 4
        pitch_violation = torch.abs(root_euler_w[:, 1]) > torch.pi / 4
        # reset = out_of_bounds_x | out_of_bounds_y | roll_violation | pitch_violation
        reset = out_of_bounds_x | out_of_bounds_y

        # Print termination reason when any condition fires
        if reset.any():
            reasons = []
            if out_of_bounds_x.any():
                reasons.append(f"out_of_bounds_x (|x|={root_pos_w[:, 0].abs().item():.2f} > {bound_x})")
            if out_of_bounds_y.any():
                reasons.append(f"out_of_bounds_y (y={root_pos_w[:, 1].item():.2f} > {bound_y})")
            if roll_violation.any():
                reasons.append(
                    f"roll_violation (|roll|={torch.abs(root_euler_w[:, 0]).item():.2f} > {torch.pi / 4:.2f})"
                )
            if pitch_violation.any():
                reasons.append(
                    f"pitch_violation (|pitch|={torch.abs(root_euler_w[:, 1]).item():.2f} > {torch.pi / 4:.2f})"
                )
            print(f"[INFO] Termination type: {' | '.join(reasons)}")
        if time_out.any():
            print(f"[INFO] Termination type: time_out ({self.episode_length_buf.item()}/{self.max_episode_length})")

        return reset, time_out

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
        self.joint_target_q_torch[self.joint_target_offset :] = self.joint_pos_initial.squeeze(0)
        wp.copy(self.control.joint_target_q, self.joint_target_q_wp)
        self.control.joint_target_qd.zero_()

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
        root_quat_w = torch.tensor(joint_q[3:7], device=self.device, dtype=torch.float32).unsqueeze(0)[
            :, [3, 0, 1, 2]
        ]  # xyzw -> wxyz
        root_quat_w = quat_mul(root_quat_w, quat_conjugate(self.root_quat_w_init))

        command = self.velocity_command.clone()
        command[:, 3:] = quat_mul(command[:, 3:], quat_conjugate(self.root_quat_w_init))

        yaw_quat_w = yaw_quat(self.velocity_command[:, 3:])
        velocity_cmd_world = quat_rotate(yaw_quat_w, command[:, :3])

        foot_pos = torch.tensor(self.state_0.body_q, device=self.device, dtype=torch.float32)[self.foot_indices, :3]
        foot_pos[:, 2] -= 0.0398
        foot_height = foot_pos[:, 2].reshape(1, -1)

        body_force = wp.to_torch(self.state_0.body_f)
        contact_force = body_force[self.foot_indices, :3].reshape(1, -1)

        log_obs = torch.cat(
            [
                root_pos_w,
                root_quat_w,
                self.base_lin_vel,
                self.base_ang_vel,
                command,
                contact_force,
                velocity_cmd_world,
                foot_height,
            ],
            dim=1,
        )

        return log_obs

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

        # RTX retains particle style when only positions are supplied on later frames.
        # Other viewers still receive radii and colors on every call.
        is_rtx = isinstance(self.viewer, newton.viewer.ViewerRTX)
        update_style = not is_rtx or not self._rtx_particle_style_initialized
        self.viewer.log_points(
            "/sand",
            points=self.sand_state_0.particle_q,
            radii=self.sand_model.particle_radius if update_style else None,
            colors=self.particle_render_colors if update_style else None,
            hidden=not self.viewer.show_particles,
        )
        if is_rtx:
            self._rtx_particle_style_initialized = True

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

        if self.show_velocity:
            self._log_velocity_arrows()
        else:
            self.viewer.log_arrows("/velocity_command", None, None, None)
            self.viewer.log_arrows("/velocity_actual", None, None, None)

        self.viewer.end_frame()

        if self.video:
            self.frames.append(self.viewer.get_frame().numpy())

    def _log_velocity_arrows(self):
        """Draw commanded (green) and actual (blue) base linear velocity arrows above the base.

        Mirrors IsaacLab's velocity command visualizer: both XY velocities are
        expressed in the world frame and drawn as arrows floating above the robot
        base, scaled by velocity magnitude.
        """
        arrow_z_offset = 0.5  # place arrows above the base [m]
        arrow_scale = 0.4  # world-space length per (m/s)

        joint_q = self.state_0.joint_q.numpy() if self.state_0.joint_q is not None else np.zeros(7)
        joint_qd = self.state_0.joint_qd.numpy() if self.state_0.joint_qd is not None else np.zeros(6)
        base_pos = np.asarray(joint_q[:3], dtype=np.float32)

        # Robot yaw (wxyz) used to lift the base-frame command into the world frame.
        root_quat_wxyz = torch.tensor(joint_q[3:7], device=self.device, dtype=torch.float32).unsqueeze(0)[
            :, [3, 0, 1, 2]
        ]
        yaw_q = yaw_quat(root_quat_wxyz)

        # Commanded velocity is expressed in the base heading frame (vx, vy).
        cmd_xy = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        cmd_xy[:, :2] = self.velocity_command[:, :2]
        cmd_world = quat_rotate(yaw_q, cmd_xy).squeeze(0).cpu().numpy()

        # Actual base linear velocity is already in the world frame (root twist).
        act_world = np.array([joint_qd[0], joint_qd[1], 0.0], dtype=np.float32)

        start = base_pos + np.array([0.0, 0.0, arrow_z_offset], dtype=np.float32)
        starts = wp.array([wp.vec3(*start), wp.vec3(*start)], dtype=wp.vec3, device=self.model.device)
        ends = wp.array(
            [wp.vec3(*(start + arrow_scale * cmd_world)), wp.vec3(*(start + arrow_scale * act_world))],
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.viewer.log_arrows("/velocity_command", starts[:1], ends[:1], (0.0, 0.8, 0.0))
        self.viewer.log_arrows("/velocity_actual", starts[1:], ends[1:], (0.0, 0.2, 1.0))

    def render_ui(self, imgui):
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)
        _changed, self.show_velocity = imgui.checkbox("Show Velocity", self.show_velocity)
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
        "--sand-type",
        "--sand_type",
        dest="sand_type",
        type=str,
        default="poppy",
        choices=tuple(MATERIAL_PRESETS),
        help="Granular material preset (poppy: closely packed seeds).",
    )

    viewer, args = newton.examples.init(parser)

    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    from newton._src.utils.download_assets import download_git_folder

    asset_dir = str(download_git_folder("https://github.com/jnskkmhr/newton-assets.git", "unitree_g1", ref="main"))

    yaml_file_path = "config/g1.yaml"
    try:
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Robot config file not found: {yaml_file_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file: {e}")
        exit(1)

    asset_path = newton.utils.download_asset("unitree_g1")
    config["asset_path"] = os.path.join(asset_dir, config["asset_path"])

    # Joint-order mapping between MuJoCo-warp and PhysX conventions
    mjc_to_physx = list(range(config["num_dofs"]))
    physx_to_mjc = list(range(config["num_dofs"]))

    num_envs = 1
    MAX_TRIAL = args.max_trial
    log_dir = f"./logs/{args.tag}"

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
            log_item=["obs"],
        )

    obs = env._get_observations()
    log = env._get_log() if args.log else None

    episode_length_log = [0] * num_envs
    episode_counter = [0] * num_envs

    # while env.viewer.is_running():
    while True:
        if args.log:
            logger.log(
                {
                    "obs": log.cpu().numpy(),
                }
            )

        if not env.viewer.is_paused():
            with wp.ScopedTimer("step", active=False):
                obs, dones, time_out = env.step()
                dones = dones | time_out
                if args.log:
                    log = env._get_log()

        with wp.ScopedTimer("render", active=False):
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
        os.makedirs(os.path.join(log_dir, "logs", args.tag), exist_ok=True)
        video_filename = os.path.join(log_dir, "logs", args.tag, "rl_play.mp4")
        save_video_cv2(env.frames, video_filename, fps=int(env.fps * args.video_speed))
        print(f"[INFO] Video saved to {video_filename}")

    env.viewer.close()
