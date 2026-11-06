# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""G1 TorchScript observations, history, velocity commands, and joint targets."""

import hashlib
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml

import newton
import newton.utils


def _rotate_inverse(q, vector):
    xyz, w = q[:, :3], q[:, 3:4]
    return (
        vector * (2 * w * w - 1)
        - 2 * w * torch.cross(xyz, vector, dim=-1)
        + 2 * xyz * (xyz * vector).sum(-1, keepdim=True)
    )


class G1Policy:
    """Evaluate the reference policy's term-major, oldest-first observation history."""

    def __init__(self, config, path, device):
        self.config = config
        self.length = int(config.get("history_length", 10))
        self.dofs = int(config["num_dofs"])
        if self.length < 1 or self.dofs != 29:
            raise ValueError("G1 requires 29 joints and a positive history length")
        self.network = torch.jit.load(str(path), map_location=device).eval()
        self.default_q = torch.tensor(config["mjw_joint_pos"], dtype=torch.float32, device=device).unsqueeze(0)
        self.action = torch.zeros((1, self.dofs), device=device)
        self.gravity = torch.tensor([[0.0, 0.0, -1.0]], device=device)
        self.history = {
            name: torch.zeros((1, self.length, width), device=device)
            for name, width in [
                ("ang_vel", 3),
                ("projected_gravity", 3),
                ("command", 7),
                ("joint_pos", self.dofs),
                ("joint_vel", self.dofs),
                ("last_action", self.dofs),
            ]
        }
        self.obs = torch.zeros((1, self.length * (13 + 3 * self.dofs)), device=device)
        self.calls = 0
        with torch.inference_mode():
            self._infer()

    def _infer(self):
        action = self.network(self.obs)
        if not isinstance(action, torch.Tensor) or action.shape != self.action.shape:
            raise ValueError("G1 policy must return a tensor of shape (1, 29)")
        if not torch.isfinite(action).all():
            raise ValueError("G1 policy returned non-finite actions")
        return action

    @torch.inference_mode()
    def targets(self, root_quat, angular_velocity, joint_q, joint_qd, command):
        """Return position targets [rad] with the reference policy's observation scales."""
        terms = (
            _rotate_inverse(root_quat, angular_velocity),
            _rotate_inverse(root_quat, self.gravity),
            command,
            joint_q - self.default_q,
            joint_qd,
            self.action,
        )
        offset = 0
        for (name, history), value in zip(self.history.items(), terms, strict=True):
            scaled = value * self.config.get(name + "_scale", 1.0)
            if self.calls == 0:
                history[:] = scaled.unsqueeze(1)
            else:
                history[:, :-1] = history[:, 1:].clone()
                history[:, -1] = scaled
            width = history.shape[1] * history.shape[2]
            self.obs[:, offset : offset + width] = history.reshape(1, -1)
            offset += width
        self.action.copy_(self._infer())
        self.calls += 1
        return self.default_q + self.config["action_scale"] * self.action


class G1Controller:
    """Load the policy model and translate 50 Hz velocity commands into joint targets."""

    policy_dt = 0.02

    def __init__(self, config_path, *, policy_path=None, asset_path=None, command=(0.6, 0.0, 0.0), device):
        config_path = Path(config_path).expanduser().resolve()
        self.config = yaml.safe_load(config_path.read_text())
        config = self.config
        names = config["mjw_joint_names"]
        if config["num_dofs"] != 29 or len(names) != 29 or len(set(names)) != 29:
            raise ValueError("G1 config must describe 29 uniquely named joints")
        for key in ("mjw_joint_pos", "mjw_joint_stiffness", "mjw_joint_damping", "mjw_joint_armature"):
            if len(config[key]) != 29 or not np.isfinite(config[key]).all():
                raise ValueError(f"{key} must contain 29 finite values")
        policy = Path(policy_path or config["policy_path"]).expanduser()
        if not policy.is_absolute() and not policy.is_file():
            candidates = (config_path.parent / policy, config_path.parent.parent / policy)
            policy = next((p for p in candidates if p.is_file()), candidates[0])
        if not policy.is_file():
            raise FileNotFoundError(f"TorchScript policy not found: {policy}; use --policy")
        if asset_path:
            asset = Path(asset_path).expanduser().resolve()
        else:
            local_root = Path(__file__).resolve().parents[4] / "newton-assets/unitree_g1"
            root = (
                local_root
                if (local_root / config["asset_path"]).is_file()
                else newton.utils.download_asset("unitree_g1")
            )
            asset = root / config["asset_path"]
        if not asset.is_file():
            raise FileNotFoundError(f"Configured G1 model not found: {asset}; use --robot-asset")
        self.asset = asset
        self.policy_path = policy.resolve()
        self.policy_sha256 = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        self.device = device
        self.torch_device = str(device)
        self.command_velocity = np.asarray(command, dtype=np.float32)
        if not np.isfinite(self.command_velocity).all():
            raise ValueError("Velocity command must be finite")
        self.command = torch.zeros((1, 7), device=self.torch_device)
        self.policy = G1Policy(config, policy, self.torch_device)
        self.heading = 0.0

    def add_robot(self, builder, *, floor_height: float):
        """Add the configured floating-base robot and its policy gains to a builder."""
        if self.asset.suffix == ".xml":
            builder.add_mjcf(str(self.asset), enable_self_collisions=False)
        elif self.asset.suffix == ".urdf":
            builder.add_urdf(
                str(self.asset),
                floating=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                ignore_inertial_definitions=False,
            )
        else:
            raise ValueError("--robot-asset must be MJCF (.xml) or URDF (.urdf)")
        if builder.joint_coord_count != 36 or builder.joint_dof_count != 35:
            raise ValueError("Policy requires a floating base and 29 actuated joints")
        quat = np.asarray(self.config.get("mjw_init_quat", [0, 0, 0, 1]), dtype=float)
        if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1.0e-8:
            raise ValueError("Initial XYZW quaternion must be finite and nonzero")
        quat /= np.linalg.norm(quat)
        self.initial_position = np.array([0.0, -3.0, floor_height + self.config.get("mjw_init_pos", [0, 0, 0.92])[2]])
        builder.joint_q[:3] = self.initial_position.tolist()
        builder.joint_q[3:7] = quat.tolist()
        x, y, z, w = quat
        self.heading = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        joints = {label.rsplit("/", 1)[-1]: i for i, label in enumerate(builder.joint_label)}
        self.q_indices, self.qd_indices = [], []
        for i, name in enumerate(self.config["mjw_joint_names"]):
            if name not in joints:
                raise ValueError(f"Configured joint {name!r} is missing from {self.asset}")
            joint = joints[name]
            q, dof = builder.joint_q_start[joint], builder.joint_qd_start[joint]
            self.q_indices.append(q)
            self.qd_indices.append(dof)
            builder.joint_q[q] = self.config["mjw_joint_pos"][i]
            builder.joint_target_q[q] = self.config["mjw_joint_pos"][i]
            builder.joint_target_ke[dof] = self.config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[dof] = self.config["mjw_joint_damping"][i]
            builder.joint_armature[dof] = self.config["mjw_joint_armature"][i]
            builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        root_joint = next(i for i, kind in enumerate(builder.joint_type) if kind == newton.JointType.FREE)
        self.root_body = builder.joint_child[root_joint]
        builder.body_q[self.root_body] = wp.transform(wp.vec3(*self.initial_position), wp.quat(*quat))
        self.q_indices = torch.tensor(self.q_indices, device=self.torch_device)
        self.qd_indices = torch.tensor(self.qd_indices, device=self.torch_device)

    def update(self, state, control, viewer, *, time: float):
        """Update the held velocity command and evaluate one policy sample."""
        if viewer.is_key_down("backspace"):
            self.command_velocity[:] = 0
        for positive, negative, axis, rate in [("i", "k", 0, 1.0), ("j", "l", 1, 0.5), ("u", "o", 2, 1.0)]:
            self.command_velocity[axis] += (
                (float(viewer.is_key_down(positive)) - float(viewer.is_key_down(negative))) * rate * self.policy_dt
            )
        velocity = self.command_velocity if time >= 0.5 else np.zeros(3, np.float32)
        self.heading += float(velocity[2]) * self.policy_dt
        self.command[0, :3] = torch.as_tensor(velocity, device=self.torch_device)
        heading = np.array([np.cos(self.heading / 2), 0.0, 0.0, np.sin(self.heading / 2)], np.float32)
        if heading[0] < 0:
            heading *= -1
        self.command[0, 3:] = torch.as_tensor(heading, device=self.torch_device)
        # Explicit synchronization preserves the Warp/Torch handoff used by the reference.
        wp.synchronize_device(self.device)
        q, qd = wp.to_torch(state.joint_q), wp.to_torch(state.joint_qd)
        targets = self.policy.targets(
            q[3:7][None], qd[3:6][None], q[self.q_indices][None], qd[self.qd_indices][None], self.command
        )
        wp.to_torch(control.joint_target_q)[self.q_indices] = targets[0]
        if self.device.is_cuda:
            torch.cuda.synchronize(self.torch_device)
