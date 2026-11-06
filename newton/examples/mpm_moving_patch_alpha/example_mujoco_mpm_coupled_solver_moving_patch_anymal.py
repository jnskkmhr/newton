# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Policy-driven ANYmal with two-way MuJoCo-MPM moving terrain.

Run with ``uv run --extra examples --extra onnx -m newton.examples
mujoco_mpm_coupled_solver_moving_patch_anymal``. Use I/K to walk forward/backward,
J/L to move sideways, U/O to turn, or right-click and drag the robot. Held keys ramp the command; release retains it.
Backspace resets the command to zero.

Select flat (default), slope, or sinusoidal ground with ``--terrain-type``.
The inherited 0.5 m sand bed uses genuine two-way proxy feedback. The pretrained
walking policy is not tuned for this terrain: the robot can sink and stall.

Use --video output.mp4 --video-duration 10 to record OpenGL frames and exit.
Add --headless for offscreen recording; Ctrl+C saves captured frames.
"""

from copy import copy

import numpy as np
import warp as wp
from warp_nn.runtime import OnnxRuntime

import newton
import newton.examples
import newton.utils
from newton.examples.mpm_moving_patch_alpha.impl.controller.keyboard_velocity import update_keyboard_velocity
from newton.examples.mpm_moving_patch_alpha.impl.terrain_moving_patch import TerrainMovingPatchExample
from newton.examples.robot.example_robot_anymal_c_walk import (
    _build_joint_target_q_kernel,
    _compute_obs_kernel,
    lab_to_mujoco,
    mujoco_to_lab,
)
from newton.examples.robot.onnx_policy_utils import validate_policy_io_shapes


class Example(TerrainMovingPatchExample):
    def __init__(self, viewer, args):
        # Delay graph capture until the policy's control buffer has been installed.
        setup_args = copy(args)
        setup_args.disable_cuda_graph = True
        super().__init__(viewer, setup_args)
        self.device = self.model.device
        self.policy_step = 0
        policy_steps = self.fps * self.sub_step / 50.0
        if policy_steps < 1 or not np.isclose(policy_steps, round(policy_steps)):
            raise ValueError("Coupled physics rate must be an integer multiple of the 50 Hz policy rate")
        self.policy_decimation = round(policy_steps)
        self.auto_forward = True
        self._command = wp.vec3(1.0, 0.0, 0.0)
        self.gravity_w = wp.vec3(0.0, 0.0, -1.0)

        policy_path = self.asset_path / "rl_policies" / "anymal_walking_policy_physx.onnx"
        self.policy = OnnxRuntime(str(policy_path), device=self.device)
        self.policy_input_name = self.policy.input_names[0]
        self.policy_output_name = self.policy.output_names[0]
        validate_policy_io_shapes(
            policy_path,
            self.policy_input_name,
            self.policy_output_name,
            obs_width=48,
            action_width=12,
            context="example_mujoco_mpm_coupled_solver_moving_patch_anymal",
        )
        self.initial_joint_positions = wp.clone(self.state_0.joint_q[7:])
        self.lab_to_mujoco_idx = wp.array(lab_to_mujoco, dtype=wp.int32, device=self.device)
        self.mujoco_to_lab_idx = wp.array(mujoco_to_lab, dtype=wp.int32, device=self.device)
        self.obs = wp.zeros((1, 48), dtype=wp.float32, device=self.device)
        self.prev_act = wp.zeros((1, 12), dtype=wp.float32, device=self.device)
        self.control.joint_target_q = wp.clone(self.state_0.joint_q)
        if not args.disable_cuda_graph:
            self.capture()

    def _emit_rigid_bodies(self, builder):
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.06, limit_ke=1.0e3, limit_kd=1.0e1)
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        self.asset_path = newton.utils.download_asset("anybotics_anymal_c")
        # Raise the original standing pose by the moving patch's 0.5 m bed depth.
        builder.add_urdf(
            str(self.asset_path / "urdf" / "anymal.urdf"),
            xform=wp.transform(
                wp.vec3(0.0, 0.0, 1.12),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            ),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        # Match the original ANYmal example's shank-only sand contacts.
        for body in range(builder.body_count):
            if "SHANK" not in builder.body_label[body]:
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] &= ~newton.ShapeFlags.COLLIDE_PARTICLES

        initial_q = {
            "RH_HAA": 0.0,
            "RH_HFE": -0.4,
            "RH_KFE": 0.8,
            "LH_HAA": 0.0,
            "LH_HFE": -0.4,
            "LH_KFE": 0.8,
            "RF_HAA": 0.0,
            "RF_HFE": 0.4,
            "RF_KFE": -0.8,
            "LF_HAA": 0.0,
            "LF_HFE": 0.4,
            "LF_KFE": -0.8,
        }
        for name, value in initial_q.items():
            joint = next(i for i, label in enumerate(builder.joint_label) if label.endswith(f"/{name}"))
            builder.joint_q[builder.joint_q_start[joint]] = value
        for dof in range(builder.joint_dof_count):
            builder.joint_target_ke[dof] = 150.0
            builder.joint_target_kd[dof] = 5.0
        return builder.body_label[0]

    def apply_control(self):
        wp.launch(
            _compute_obs_kernel,
            dim=1,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.initial_joint_positions,
                self.lab_to_mujoco_idx,
                self.gravity_w,
                self._command,
                self.prev_act,
                self.obs,
            ],
            device=self.device,
        )
        act = self.policy({self.policy_input_name: self.obs})[self.policy_output_name]
        wp.launch(
            _build_joint_target_q_kernel,
            dim=self.model.joint_coord_count,
            inputs=[act, self.initial_joint_positions, self.mujoco_to_lab_idx, 0.5, 7, self.control.joint_target_q],
            device=self.device,
        )
        wp.copy(self.prev_act, act)

    def step(self):
        velocity = (1.0, 0.0, 0.0) if self.auto_forward else self._command
        command, edited = update_keyboard_velocity(self.viewer, velocity, self.frame_dt)
        if edited:
            self.auto_forward = False
        self._command = wp.vec3(*command)
        super().step()

    def simulate(self):
        """Update the policy at 50 Hz and advance one coupled physics substep."""
        if self.policy_step % self.policy_decimation == 0:
            self.apply_control()
        super().simulate()
        self.policy_step += 1

    def _render_overlays(self):
        self._log_velocity_arrows(self._command)

    def render_ui(self, imgui):
        imgui.text("XY velocity: green = command, blue = measured")
        imgui.text("Hold I/K, J/L, U/O to ramp forward, sideways, yaw commands")
        imgui.text("Release keeps command; Backspace stops.")
        imgui.text("Right-click and drag the robot; blue outline: simulated terrain")
        _changed, self.auto_forward = imgui.checkbox("Walk forward automatically", self.auto_forward)
        _changed, self.show_buffer = imgui.checkbox("Show boundary particles", self.show_buffer)
        _changed, self.show_impulses = imgui.checkbox("Show coupling impulses", self.show_impulses)

        super().render_ui(imgui)

    def test_final(self):
        super().test_final()
        assert np.isfinite(self.obs.numpy()).all(), "Non-finite policy observations"
        assert np.isfinite(self.prev_act.numpy()).all(), "Non-finite policy actions"
        assert np.isfinite(self.control.joint_target_q.numpy()).all(), "Non-finite joint targets"


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
