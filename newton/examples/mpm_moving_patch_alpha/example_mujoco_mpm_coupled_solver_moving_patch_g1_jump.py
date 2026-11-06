# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Motion-driven G1 jump on two-way coupled moving-patch MPM terrain.

Run ``uv run --no-sync -m newton.examples mujoco_mpm_coupled_solver_moving_patch_g1_jump``.
The supplied mimic policy and motion advance at 60 Hz; coupled physics advances
at 240 Hz. The final reference pose and phase are held after the clip ends.
Terrain options, camera following, and OpenGL video recording match the other
moving-patch robot examples. The policy is not retuned for moving sand or slopes.
"""

from copy import copy
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml

import newton
import newton.examples
import newton.utils
from newton.examples.mpm_moving_patch_alpha.impl.terrain_moving_patch import TerrainMovingPatchExample


class Example(TerrainMovingPatchExample):
    frame_rate = 60

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
        if self.config.get("history_length", 1) != 1:
            raise ValueError("The mimic policy requires history_length=1")
        motion_path = Path(args.motion).expanduser()
        if not motion_path.is_absolute() and not motion_path.is_file():
            motion_path = config_path.parent / motion_path
        self.motion = np.loadtxt(motion_path, delimiter=",", ndmin=2, dtype=np.float32)
        if self.motion.shape[0] == 0 or self.motion.shape[1] != 29 or not np.isfinite(self.motion).all():
            raise ValueError("Motion must contain 29 finite joint positions per frame")
        if args.num_frames is None:
            args.num_frames = len(self.motion) + 1
            viewer.num_frames = args.num_frames
        self.policy_step = 0
        setup_args = copy(args)
        setup_args.disable_cuda_graph = True
        super().__init__(viewer, setup_args)
        self.device = str(self.model.device)
        self.policy = torch.jit.load(str(policy_path), map_location=self.device).eval()
        self.q_indices = np.asarray(self.policy_q_indices)
        self.qd_indices = np.asarray(self.policy_qd_indices)
        self.default_pos = np.asarray(self.config["mjw_joint_pos"], dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.targets = self.control.joint_target_q.numpy()
        self.observation = np.zeros(123, dtype=np.float32)
        if not args.disable_cuda_graph:
            self.capture()

    def _emit_rigid_bodies(self, builder):
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
        return builder.body_label[0]

    def _get_observations(self):
        q = self.state_0.joint_q.numpy()
        qd = self.state_0.joint_qd.numpy()
        rotation = wp.quat(*q[3:7])
        angular_velocity = np.asarray(wp.quat_rotate_inv(rotation, wp.vec3(*qd[3:6])))
        gravity = np.asarray(wp.quat_rotate_inv(rotation, wp.vec3(0.0, 0.0, -1.0)))
        frame = min(self.policy_step, len(self.motion) - 1)
        self.observation[:] = np.concatenate(
            (
                self.motion[frame] * self.config.get("command_scale", 1.0),
                [min(self.policy_step / len(self.motion), 1.0)],
                angular_velocity * self.config.get("ang_vel_scale", 1.0),
                gravity * self.config.get("projected_gravity_scale", 1.0),
                (q[self.q_indices] - self.default_pos) * self.config.get("joint_pos_scale", 1.0),
                qd[self.qd_indices] * self.config.get("joint_vel_scale", 0.05),
                self.last_action * self.config.get("last_action_scale", 1.0),
            )
        )
        return torch.as_tensor(self.observation, device=self.device).unsqueeze(0)

    def step(self):
        with torch.inference_mode():
            action = self.policy(self._get_observations())
        if not isinstance(action, torch.Tensor) or action.shape != (1, 29):
            raise ValueError("Mimic policy must return a tensor with shape (1, 29)")
        action = action.cpu().numpy()
        if not np.isfinite(action).all():
            raise ValueError("Mimic policy returned non-finite actions")
        self.last_action[:] = action[0]
        self.targets[self.q_indices] = self.default_pos + self.config["action_scale"] * self.last_action
        self.control.joint_target_q.assign(self.targets)
        super().step()
        self.policy_step += 1

    def render_ui(self, imgui):
        frame = min(self.policy_step, len(self.motion) - 1)
        imgui.text(f"G1 jump reference: {frame + 1}/{len(self.motion)} (60 Hz)")
        imgui.text("Final reference is held; right-click to drag the robot.")
        _, self.show_buffer = imgui.checkbox("Show boundary particles", self.show_buffer)
        _, self.show_impulses = imgui.checkbox("Show MPM impulses", self.show_impulses)
        super().render_ui(imgui)

    def test_post_step(self):
        """Check that policy observations, actions, and coupled state remain finite."""
        for array in (self.state_0.joint_q, self.state_0.joint_qd, self.state_0.particle_q, self.state_0.particle_qd):
            assert np.isfinite(array.numpy()).all()
        assert np.isfinite(self.observation).all()
        assert np.isfinite(self.last_action).all()

    def test_final(self):
        """Check terrain-relative state and successful policy execution."""
        super().test_final()
        self.test_post_step()
        assert self.policy_step > 0

    @staticmethod
    def create_parser():
        parser = TerrainMovingPatchExample.create_parser()
        config_dir = Path(__file__).resolve().parents[3] / "config"
        parser.set_defaults(num_frames=None, rigid_substeps=1, sub_step=4)
        parser.add_argument("--config", default=str(config_dir / "g1_mimic.yaml"), help="Mimic policy configuration.")
        parser.add_argument(
            "--motion", default=str(config_dir / "motion.csv"), help="29-column reference joints at 60 Hz."
        )
        parser.add_argument("--policy", help="Override the TorchScript mimic policy.")
        parser.add_argument("--asset", help="Override the G1 MJCF or URDF model.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    except KeyboardInterrupt:
        print("\nInterrupted; saving captured video frames.", flush=True)
        viewer.close()
    finally:
        example.save_video()
