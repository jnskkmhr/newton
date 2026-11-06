# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""G1 TorchScript locomotion on beta moving terrain with two-way coupling.

Run ``uv run -m newton.examples moving_terrain_mpm_g1 --config config/g1.yaml``.
I/K, J/L, U/O increment forward, sideways, and yaw velocity; Backspace stops.
The policy runs at 50 Hz; coupled physics runs at fps * sub_step, without graph capture.
"""

from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml

import newton
import newton.examples
import newton.utils
from newton.examples.mpm_moving_patch_beta.impl.base_example import MovingTerrainExample
from newton.examples.mpm_moving_patch_beta.impl.controller.g1_policy import G1Policy
from newton.examples.mpm_moving_patch_beta.impl.controller.keyboard_velocity import update_keyboard_velocity
from newton.examples.mpm_moving_patch_beta.impl.scene_visuals import VelocityArrows, make_earth_texture
from newton.solvers import SolverMuJoCo


class Example(MovingTerrainExample):
    """Run the 29-joint G1 policy on the shared beta terrain scene."""

    def __init__(self, viewer, args):
        config_path = Path(args.config).expanduser().resolve()
        with config_path.open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        names = self.config["mjw_joint_names"]
        if self.config["num_dofs"] != 29 or len(names) != 29 or len(set(names)) != 29:
            raise ValueError("G1 config must describe 29 uniquely named joints")
        for key in ("mjw_joint_pos", "mjw_joint_stiffness", "mjw_joint_damping", "mjw_joint_armature"):
            if len(self.config[key]) != 29 or not np.isfinite(self.config[key]).all():
                raise ValueError(f"{key} must contain 29 finite values")
        if self.config.get("history_length", 10) < 1:
            raise ValueError("history_length must be positive")
        policy_path = Path(args.policy or self.config["policy_path"]).expanduser()
        if not policy_path.is_absolute() and not policy_path.is_file():
            # Existing g1.yaml paths are relative to the project, e.g. config/policy.pt.
            candidates = (config_path.parent / policy_path, config_path.parent.parent / policy_path)
            policy_path = next((p for p in candidates if p.is_file()), candidates[0])
        if not policy_path.is_file():
            raise FileNotFoundError(f"TorchScript policy not found: {policy_path}; supply --policy")
        if args.asset:
            self.asset_path = Path(args.asset).expanduser().resolve()
        else:
            local_assets = Path(__file__).resolve().parents[4] / "newton-assets" / "unitree_g1"
            asset_root = local_assets if local_assets.is_dir() else newton.utils.download_asset("unitree_g1")
            self.asset_path = Path(asset_root) / self.config["asset_path"]
        if not self.asset_path.is_file():
            raise FileNotFoundError(f"G1 model not found: {self.asset_path}; supply --asset")
        self.auto_forward = args.auto_forward
        self.command_velocity = np.array(args.command, dtype=np.float32)
        if not np.isfinite(self.command_velocity).all():
            raise ValueError("Velocity commands must be finite")
        self.current_velocity = np.zeros(3, dtype=np.float32)
        self.policy_step = 0
        # Keep policy inference outside the uncaptured coupled physics step.
        super().__init__(viewer, args)
        self.particle_color = (0.7, 0.6, 0.4)
        self.boundary_color = self.particle_color
        self.velocity_arrows = VelocityArrows(self.model.device)
        policy_steps = self.fps * self.sub_step / 50.0
        if policy_steps < 1 or not np.isclose(policy_steps, round(policy_steps)):
            raise ValueError("G1 physics rate must be an integer multiple of the 50 Hz policy rate")
        self.policy_decimation = round(policy_steps)
        self.device = self.model.device
        self.torch_device = str(self.device)
        self.policy = G1Policy(self.config, policy_path, self.torch_device)
        self.q_indices = torch.tensor(self.policy_q_indices, device=self.torch_device)
        self.qd_indices = torch.tensor(self.policy_qd_indices, device=self.torch_device)
        self.targets = wp.to_torch(self.control.joint_target_q)
        self.command = torch.zeros((1, 7), device=self.torch_device)
        x, y, z, w = self.initial_quaternion
        self.heading = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _emit_ground(self, builder):
        """Match the reference G1 earth texture without changing ground geometry."""
        texture = make_earth_texture(self.mpm_wrapper.config.background_terrain_size)
        self.mpm_wrapper.add_ground(builder, texture=texture)

    def _render_overlays(self):
        """Show command velocity in green and measured XY velocity in blue."""
        self.velocity_arrows.render(self.viewer, self.state_0, self.current_velocity)

    def _emit_rigid_bodies(self, builder):
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 0.75
        if self.asset_path.suffix == ".xml":
            builder.add_mjcf(str(self.asset_path), enable_self_collisions=False)
        elif self.asset_path.suffix == ".urdf":
            builder.add_urdf(
                str(self.asset_path),
                floating=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                ignore_inertial_definitions=False,
            )
        else:
            raise ValueError("--asset must be an MJCF (.xml) or URDF (.urdf) model")
        if builder.joint_coord_count != 36 or builder.joint_dof_count != 35:
            raise ValueError("Expected a floating-base G1 with 29 actuated joints")
        position = list(self.config.get("mjw_init_pos", [0.0, 0.0, 0.92]))
        position[2] += 0.5  # The inherited sand surface is 0.5 m above the ground.
        self.initial_quaternion = np.asarray(self.config.get("mjw_init_quat", [0, 0, 0, 1]), dtype=float)
        norm = np.linalg.norm(self.initial_quaternion)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("mjw_init_quat must be a finite, nonzero XYZW quaternion")
        self.initial_quaternion /= norm
        builder.joint_q[:3] = position
        builder.joint_q[3:7] = self.initial_quaternion.tolist()
        joint_ids = {label.rsplit("/", 1)[-1]: i for i, label in enumerate(builder.joint_label)}
        self.policy_q_indices = []
        self.policy_qd_indices = []
        for i, name in enumerate(self.config["mjw_joint_names"]):
            if name not in joint_ids:
                raise ValueError(f"Configured joint {name!r} is missing from {self.asset_path}")
            joint = joint_ids[name]
            q = builder.joint_q_start[joint]
            dof = builder.joint_qd_start[joint]
            self.policy_q_indices.append(q)
            self.policy_qd_indices.append(dof)
            builder.joint_q[q] = self.config["mjw_joint_pos"][i]
            builder.joint_target_q[q] = self.config["mjw_joint_pos"][i]
            builder.joint_target_ke[dof] = self.config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[dof] = self.config["mjw_joint_damping"][i]
            builder.joint_armature[dof] = self.config["mjw_joint_armature"][i]
            builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        # Match the source G1 example's ankle/knee particle contacts.
        for body, label in enumerate(builder.body_label):
            if "ankle" not in label and "knee" not in label:
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] &= ~newton.ShapeFlags.COLLIDE_PARTICLES
        root_joint = next(j for j, kind in enumerate(builder.joint_type) if kind == newton.JointType.FREE)
        self.root_body = builder.joint_child[root_joint]
        builder.body_q[self.root_body] = wp.transform(wp.vec3(*position), wp.quat(*self.initial_quaternion))
        return builder.body_label[self.root_body]

    def apply_control(self):
        dt = self.sim_dt * self.policy_decimation
        velocity = self.command_velocity if self.auto_forward else self.current_velocity
        if self.auto_forward and self.sim_time < 0.1:
            velocity = np.zeros(3, dtype=np.float32)
        velocity, edited = update_keyboard_velocity(self.viewer, velocity, dt)
        if edited:
            self.auto_forward = False
        self.current_velocity = velocity
        self.heading += float(velocity[2]) * dt
        # Commands carry XYZ velocity followed by a WXYZ heading quaternion.
        self.command[0, :3] = torch.as_tensor(velocity, device=self.torch_device)
        self.command[0, 3:] = torch.tensor(
            [np.cos(self.heading / 2), 0, 0, np.sin(self.heading / 2)], device=self.torch_device
        )
        if self.command[0, 3] < 0:
            self.command[0, 3:] *= -1
        # Explicit stream handoff: Warp and Torch need not use the same CUDA stream.
        wp.synchronize_device(self.device)
        q = wp.to_torch(self.state_0.joint_q)
        qd = wp.to_torch(self.state_0.joint_qd)
        target = self.policy.targets(
            q[3:7].unsqueeze(0),
            qd[3:6].unsqueeze(0),
            q[self.q_indices].unsqueeze(0),
            qd[self.qd_indices].unsqueeze(0),
            self.command,
        )
        self.targets[self.q_indices] = target[0]
        if self.device.is_cuda:
            torch.cuda.synchronize(self.torch_device)

    def simulate(self):
        """Update policy at 50 Hz, then advance one coupled physics substep."""
        if self.policy_step % self.policy_decimation == 0:
            self.apply_control()
        super().simulate()
        self.policy_step += 1

    def render_ui(self, imgui):
        """Show policy commands and the shared terrain rendering controls."""
        imgui.text("XY velocity: green = command, blue = measured")
        imgui.text("Hold I/K, J/L, U/O to change forward, sideways, and yaw velocity.")
        imgui.text("Release keeps the command; Backspace stops.")
        vx, vy, wz = self.current_velocity
        imgui.text(f"Command: {vx:.2f}, {vy:.2f} m/s; yaw {wz:.2f} rad/s")
        _, self.auto_forward = imgui.checkbox("Walk automatically", self.auto_forward)
        super().render_ui(imgui)

    def test_post_step(self):
        """Check that coupled G1 coordinates remain finite."""
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("Non-finite G1 joint state")

    def test_final(self):
        """Check terrain state, policy outputs, and actual policy execution."""
        super().test_final()
        assert torch.isfinite(self.policy.obs).all()
        assert torch.isfinite(self.policy.action).all()
        assert self.policy_step > 0

    @staticmethod
    def create_parser():
        parser = MovingTerrainExample.create_parser()
        # Humanoid ground contacts can exceed the box example's constraint budget.
        parser.set_defaults(rigid_njmax=1024, rigid_nconmax=512)
        parser.add_argument("--config", default=str(Path(__file__).resolve().parents[3] / "config/g1.yaml"))
        parser.add_argument(
            "--auto-forward", action="store_true", help="Start with automatic motion instead of keyboard control"
        )
        parser.add_argument("--policy", help="TorchScript policy; defaults to policy_path in the config")
        parser.add_argument("--asset", help="Override the configured G1 MJCF or URDF file")
        parser.add_argument(
            "--command",
            type=float,
            nargs=3,
            default=(1.0, 0.0, 0.0),
            metavar=("VX", "VY", "WZ"),
            help="Velocity for --auto-forward or the automatic-motion checkbox [m/s, m/s, rad/s]",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
