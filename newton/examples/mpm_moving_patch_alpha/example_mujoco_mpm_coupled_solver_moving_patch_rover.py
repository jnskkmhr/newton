# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Four-wheel differential-drive rover on two-way coupled moving MPM terrain.

Run with ``uv run --extra examples -m newton.examples
mujoco_mpm_coupled_solver_moving_patch_rover``. The rover drives forward
by default. I/K drive forward/backward, J/L turn, and O stops automatic
driving. Add ``--video rover.mp4`` to record the OpenGL viewer; ``--headless``
records offscreen. Recording stops automatically after ``--video-duration``
seconds of simulation (default: 10). Wheel slip and chassis motion remain fully dynamic.
"""

import xml.etree.ElementTree as ET
from copy import copy
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mpm_moving_patch_alpha.impl.terrain_moving_patch import TerrainMovingPatchExample


@wp.kernel
def _set_wheel_targets(
    wheel_dofs: wp.array[int],
    wheel_sides: wp.array[float],
    speed: float,
    yaw_rate: float,
    track_width: float,
    radius: float,
    targets: wp.array[float],
):
    i = wp.tid()
    # +Y is left; positive rotation about +Y rolls a wheel forward along +X.
    targets[wheel_dofs[i]] = (speed - wheel_sides[i] * yaw_rate * track_width * 0.5) / radius


class Example(TerrainMovingPatchExample):
    def __init__(self, viewer, args):
        if not np.isfinite(args.drive_speed) or not np.isfinite(args.turn_rate):
            raise ValueError("Drive speed and turn rate must be finite")
        self.drive_speed = args.drive_speed
        self.turn_rate = args.turn_rate
        self.auto_drive = True
        self.wheel_type = args.wheel_type
        self.rover_asset = Path(__file__).with_name("assets") / f"rover_{self.wheel_type}_wheel.xml"
        asset = ET.parse(self.rover_asset).getroot()
        dimensions = {item.attrib["name"]: float(item.attrib["data"]) for item in asset.findall("custom/numeric")}
        self.wheel_radius = dimensions["wheel_radius"]
        self.track_width = dimensions["track_width"]
        # The parent captures simulate(); initialize wheel control before capture.
        setup_args = copy(args)
        setup_args.disable_cuda_graph = True
        super().__init__(viewer, setup_args)
        self.wheel_dofs = wp.array(self._wheel_dofs, dtype=int, device=self.model.device)
        self.wheel_sides = wp.array(self._wheel_sides, dtype=float, device=self.model.device)
        self._apply_drive(self.drive_speed, self.turn_rate)
        if not args.disable_cuda_graph:
            self.capture()

    def _emit_particles(self, builder, voxel_size):
        super()._emit_particles(builder, voxel_size)
        terrain = self.terrain_heightfield
        # Fit the local sampled surface, rather than assuming a planar slope.
        dx = float(terrain.spacing[0])
        ground = terrain.sample_height(np.array([[-dx, 0.0], [0.0, 0.0], [dx, 0.0]]))
        pitch = float(np.arctan2(ground[2] - ground[0], 2.0 * dx))
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -pitch)
        height = ground[1] + self.bed_hi[2] + 0.4 / np.cos(pitch)
        pose = wp.transform(wp.vec3(0.0, 0.0, height), rotation)
        builder.body_q[0] = pose
        start = builder.joint_q_start[0]
        builder.joint_q[start : start + 7] = list(pose)

    def _emit_rigid_bodies(self, builder):
        # Import MJCF velocity actuators with their force limits and joint targets.
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        joint_start = builder.joint_count
        builder.add_mjcf(
            str(self.rover_asset),
            parse_mujoco_options=False,
            enable_self_collisions=False,
            ignore_inertial_definitions=False,
            force_show_colliders=True,
        )
        joints = {builder.joint_label[j].rsplit("/", 1)[-1]: j for j in range(joint_start, builder.joint_count)}
        self._wheel_dofs = []
        self._wheel_sides = []
        for name, side in (("front_left", 1.0), ("front_right", -1.0), ("rear_left", 1.0), ("rear_right", -1.0)):
            joint = joints[f"{name}_drive"]
            self._wheel_dofs.append(builder.joint_qd_start[joint])
            self._wheel_sides.append(side)
        return builder.body_label[0]

    def _apply_drive(self, speed, yaw_rate):
        wp.launch(
            _set_wheel_targets,
            dim=4,
            inputs=[
                self.wheel_dofs,
                self.wheel_sides,
                speed,
                yaw_rate,
                self.track_width,
                self.wheel_radius,
                self.control.joint_target_qd,
            ],
            device=self.model.device,
        )

    def step(self):
        forward = float(self.viewer.is_key_down("i")) - float(self.viewer.is_key_down("k"))
        turn = float(self.viewer.is_key_down("j")) - float(self.viewer.is_key_down("l"))
        stop = self.viewer.is_key_down("o")
        if forward or turn or stop:
            self.auto_drive = False
        if self.auto_drive:
            speed, yaw_rate = self.drive_speed, self.turn_rate
        else:
            speed = forward * abs(self.drive_speed)
            yaw_rate = turn * 0.6
        if stop:
            speed, yaw_rate = 0.0, 0.0
        # Update the existing target array; captured solver calls read it live.
        self._apply_drive(speed, yaw_rate)
        super().step()

    def render_ui(self, imgui):
        imgui.text("Rover: I/K forward/back, J/L turn, O stop")
        imgui.text(f"Wheels: {self.wheel_type}; right-click drag. Wheel slip remains dynamic.")
        _, self.auto_drive = imgui.checkbox("Drive automatically", self.auto_drive)
        _, self.drive_speed = imgui.slider_float("Forward speed [m/s]", self.drive_speed, -1.0, 1.0)
        _, self.turn_rate = imgui.slider_float("Yaw rate [rad/s]", self.turn_rate, -5.0, 5.0)
        _, self.show_buffer = imgui.checkbox("Show boundary particles", self.show_buffer)
        _, self.show_impulses = imgui.checkbox("Show MPM impulses", self.show_impulses)

        super().render_ui(imgui)

    def test_final(self):
        super().test_final()
        assert self.model.body_count == 5
        assert self.object_controller is None

        assert np.isfinite(self.control.joint_target_qd.numpy()).all()

    @staticmethod
    def create_parser():
        parser = TerrainMovingPatchExample.create_parser()
        parser.add_argument(
            "--wheel-type", choices=("flat", "grouser"), default="grouser", help="MJCF wheel asset to load."
        )
        parser.add_argument("--drive-speed", type=float, default=1.0, help="Forward command [m/s].")
        parser.add_argument("--turn-rate", type=float, default=1.0, help="Yaw command [rad/s].")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    except KeyboardInterrupt:
        print("\nInterrupted; saving captured video frames.", flush=True)
        viewer.close()
    finally:
        example.save_video()
