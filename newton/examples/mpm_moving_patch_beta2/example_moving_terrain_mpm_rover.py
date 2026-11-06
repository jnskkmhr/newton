# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Four-wheel differential-drive rover on beta moving MPM terrain.

Run ``uv run -m newton.examples moving_terrain_mpm_rover``. I/K drives forward/back,
J/L turns, and O stops automatic driving. Use ``--wheel-type flat`` for smooth
wheels or ``--wheel-type grouser`` for convex-mesh lugs. All assets are local.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from newton.examples.mpm_moving_patch_beta2.impl.base_example import MovingTerrainExample
from newton.examples.mpm_moving_patch_beta2.impl.controller.differential_drive import DifferentialDriveController
from newton.solvers import SolverMuJoCo


class Example(MovingTerrainExample):
    """Reuse the beta terrain and coupling with a fully dynamic articulated rover."""

    def __init__(self, viewer, args):
        if not all(math.isfinite(value) for value in (args.drive_speed, args.turn_rate)):
            raise ValueError("Drive speed and turn rate must be finite")
        self.drive_speed = args.drive_speed
        self.turn_rate = args.turn_rate
        self.auto_drive = not args.manual
        self.wheel_type = args.wheel_type
        self.rover_asset = Path(__file__).with_name("assets") / f"rover_{self.wheel_type}_wheel.xml"
        root = ET.parse(self.rover_asset).getroot()
        metadata = {item.attrib["name"]: float(item.attrib["data"]) for item in root.findall("custom/numeric")}
        self.wheel_radius = metadata["wheel_radius"]
        self.track_width = metadata["track_width"]
        super().__init__(viewer, args)

    def _emit_rigid_bodies(self, builder):
        """Load local MJCF geometry, inertias and limited velocity actuators."""
        SolverMuJoCo.register_custom_attributes(builder)
        first_body = builder.body_count
        first_joint = builder.joint_count
        builder.add_mjcf(
            str(self.rover_asset),
            collapse_fixed_joints=False,
            parse_mujoco_options=False,
            enable_self_collisions=False,
            ignore_inertial_definitions=False,
            force_show_colliders=True,
        )
        joints = {builder.joint_label[j].rsplit("/", 1)[-1]: j for j in range(first_joint, builder.joint_count)}
        self._wheel_dofs = []
        self._wheel_sides = []
        for name, side in (("front_left", 1.0), ("front_right", -1.0), ("rear_left", 1.0), ("rear_right", -1.0)):
            joint = joints[f"{name}_drive"]
            self._wheel_dofs.append(builder.joint_qd_start[joint])
            self._wheel_sides.append(side)
        return builder.body_label[first_body]

    def _place_rigid_bodies(self, builder):
        """Start the rover parallel to the sampled ground beneath its chassis."""
        super()._place_rigid_bodies(builder)
        h = self.mpm_wrapper.config.background_height_spacing
        slope = (self.mpm_wrapper.sample_ground_height(h) - self.mpm_wrapper.sample_ground_height(-h)) / (2 * h)
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -math.atan(float(slope)))
        center = wp.transform_get_translation(builder.body_q[0])
        for body in range(builder.body_count):
            pose = builder.body_q[body]
            position = center + wp.quat_rotate(rotation, wp.transform_get_translation(pose) - center)
            builder.body_q[body] = wp.transform(position, rotation * wp.transform_get_rotation(pose))
        for joint in range(builder.joint_count):
            if builder.joint_type[joint] == newton.JointType.FREE:
                start = builder.joint_q_start[joint]
                builder.joint_q[start : start + 7] = list(builder.body_q[builder.joint_child[joint]])

    def _create_controller(self, args):
        """Use wheel actuators rather than the box's external force controller."""
        self.object_controller = None
        self.drive_controller = DifferentialDriveController(
            wheel_dofs=self._wheel_dofs,
            wheel_sides=self._wheel_sides,
            wheel_radius=self.wheel_radius,
            track_width=self.track_width,
            device=self.model.device,
        )

    def step(self):
        """Read the viewer's existing key state and apply wheel velocity targets."""
        forward = float(self.viewer.is_key_down("i")) - float(self.viewer.is_key_down("k"))
        turn = float(self.viewer.is_key_down("j")) - float(self.viewer.is_key_down("l"))
        stop = self.viewer.is_key_down("o")
        if forward or turn or stop:
            self.auto_drive = False
        if self.auto_drive:
            speed, yaw_rate = self.drive_speed, self.turn_rate
        else:
            speed, yaw_rate = forward * abs(self.drive_speed), turn * self.turn_rate
        if stop:
            speed, yaw_rate = 0.0, 0.0
        self.drive_controller.apply(self.control, speed=speed, yaw_rate=yaw_rate)
        super().step()

    def render_ui(self, imgui):
        """Show drive controls using the viewer's side-panel callback."""
        imgui.text("I/K forward/back; J/L turn; O stop. Right-click drag.")
        imgui.text(f"Wheels: {self.wheel_type}. Wheel slip and chassis motion are dynamic.")
        _, self.auto_drive = imgui.checkbox("Drive automatically", self.auto_drive)
        _, self.drive_speed = imgui.slider_float("Forward speed [m/s]", self.drive_speed, -5.0, 5.0)
        _, self.turn_rate = imgui.slider_float("Yaw rate [rad/s]", self.turn_rate, -5.0, 5.0)
        _, self.show_buffer = imgui.checkbox("Show fixed boundary particles", self.show_buffer)
        _, self.show_impulses = imgui.checkbox("Show coupling impulses", self.show_impulses)
        self.render_particle_ui(imgui)

    def test_final(self):
        """Check dynamic rover topology, wheel commands and terrain state."""
        super().test_final()
        assert self.model.body_count == 5
        assert self.object_controller is None
        assert np.isfinite(self.state_0.body_q.numpy()).all()
        assert np.isfinite(self.state_0.body_qd.numpy()).all()
        assert np.isfinite(self.control.joint_target_qd.numpy()).all()

    @staticmethod
    def create_parser():
        """Configure rover drive commands and locally stored wheel assets."""
        parser = MovingTerrainExample.create_parser()
        parser.add_argument("--manual", action="store_true", help="Start with automatic rover driving disabled.")
        parser.add_argument(
            "--wheel-type", choices=("flat", "grouser"), default="grouser", help="Local MJCF wheel asset"
        )
        parser.add_argument("--drive-speed", type=float, default=0.4, help="Forward drive command [m/s]")
        parser.add_argument("--turn-rate", type=float, default=0.0, help="Yaw command [rad/s]")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
