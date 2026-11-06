# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared procedural ground and moving sand patch for articulated robot examples."""

import numpy as np
import warp as wp

import newton
from newton.examples.mpm_moving_patch_alpha.impl.base_example import MovingTerrainExample
from newton.examples.mpm_moving_patch_alpha.impl.terrain_generator import TerrainGenerator, TerrainType


@wp.kernel
def _velocity_arrow_points(
    q: wp.array[float],
    qd: wp.array[float],
    command: wp.vec3,
    yaw_only: bool,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    rotation = wp.quat(q[3], q[4], q[5], q[6])
    if yaw_only:
        yaw = wp.atan2(2.0 * (q[6] * q[5] + q[3] * q[4]), 1.0 - 2.0 * (q[4] * q[4] + q[5] * q[5]))
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    velocity = wp.quat_rotate(rotation, wp.vec3(command[0], command[1], 0.0))
    start = wp.vec3(q[0], q[1], q[2] + 0.5)
    starts[0] = start
    starts[1] = start + wp.vec3(0.0, 0.0, 0.08)
    # Common scale: 0.4 m of arrow length per m/s; ignore vertical velocity.
    ends[0] = starts[0] + 0.4 * wp.vec3(velocity[0], velocity[1], 0.0)
    ends[1] = starts[1] + 0.4 * wp.vec3(qd[0], qd[1], 0.0)


class TerrainMovingPatchExample(MovingTerrainExample):
    """Share sampled collision, particle placement, and recycling terrain."""

    def __init__(self, viewer, args):
        self.frames = []
        self.video_filename = args.video
        self.video_speed = args.video_speed
        self.video_duration = args.video_duration
        if not np.isfinite(self.video_duration) or self.video_duration <= 0.0:
            raise ValueError("--video-duration must be finite and positive")
        if not np.isfinite(self.video_speed) or self.video_speed <= 0.0:
            raise ValueError("--video-speed must be finite and positive")
        if self.video_filename and not isinstance(viewer, newton.viewer.ViewerGL):
            raise ValueError("Video recording requires --viewer gl (optionally --headless)")
        terrain_type = TerrainType(args.terrain_type)
        params = {}
        if terrain_type == TerrainType.SLOPE:
            if not np.isfinite(args.slope_angle) or abs(args.slope_angle) >= 60.0:
                raise ValueError("--slope-angle must be finite and strictly between -60 and 60 degrees")
            params = {"angle_degrees": args.slope_angle}
        elif terrain_type == TerrainType.SINUSOIDAL:
            params = {
                "amplitude": args.wave_amplitude,
                "wavelength": args.wave_wavelength,
                "phase_degrees": args.wave_phase,
            }
        self.terrain_gen = TerrainGenerator(type=terrain_type, params=params)
        super().__init__(viewer, args)
        if self.video_filename and int(self.fps * self.video_speed) < 1:
            raise ValueError("--video-speed produces a video frame rate below 1 FPS")
        self._last_recorded_time = self.sim_time
        self.video_frame_limit = max(1, int(np.ceil(self.video_duration * self.fps)))
        if self.video_filename:
            # Duration controls recording in both windowed and headless mode.
            self.viewer.num_frames = None
        if self.video_filename:
            self._show_recording_progress()
        self.camera_offset = wp.vec3(2.5, -2.5, 1.0)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.camera.look_at(self.state_0.body_q[:1].numpy()[0, :3])

    def _emit_ground(self, builder):
        size = self.mpm_wrapper.config.background_terrain_size
        # Seeded earth tones at several spatial scales stay fixed to the ground.
        texels_per_meter = 32
        width = round(size[0] * texels_per_meter)
        height = round(size[1] * texels_per_meter)
        rng = np.random.default_rng(42)
        variation = np.zeros((height, width))
        for scale, weight in ((4.0, 0.5), (1.0, 0.3), (0.25, 0.2)):
            nx = max(2, round(size[0] / scale) + 1)
            ny = max(2, round(size[1] / scale) + 1)
            noise = rng.random((ny, nx))
            rows = np.array([np.interp(np.linspace(0, nx - 1, width), np.arange(nx), row) for row in noise])
            smooth = np.array([np.interp(np.linspace(0, ny - 1, height), np.arange(ny), column) for column in rows.T]).T
            variation += weight * smooth
        variation = np.clip(variation + rng.normal(0.0, 0.045, variation.shape), 0.0, 1.0)
        dark = np.array([65.0, 52.0, 36.0])
        light = np.array([170.0, 148.0, 108.0])
        texture = (dark + variation[..., None] * (light - dark)).astype(np.uint8)
        self.mpm_wrapper.add_ground(builder, texture=texture)
        return self.mpm_wrapper.terrain

    def _emit_particles(self, builder, voxel_size):
        super()._emit_particles(builder, voxel_size)
        terrain = self.terrain_heightfield
        root = next(i for i, kind in enumerate(builder.joint_type) if kind == newton.JointType.FREE)
        start = builder.joint_q_start[root]
        xy = np.asarray(builder.joint_q[start : start + 2])
        builder.joint_q[start + 2] += float(terrain.sample_height(xy))
        body = builder.joint_child[root]
        builder.body_q[body] = wp.transform(
            wp.vec3(*builder.joint_q[start : start + 3]), wp.quat(*builder.joint_q[start + 3 : start + 7])
        )

    def step(self):
        # This demo has a finite ground mesh. Stop before the patch reaches its edge.
        xy = self.state_0.body_q[:1].numpy()[0, :2]
        center = np.floor(xy / self.shift_step + 0.5) * self.shift_step
        terrain = self.terrain_heightfield
        if np.any(center + self.bed_lo[:2] < terrain.origin[:2]) or np.any(center + self.bed_hi[:2] > terrain.upper_xy):
            raise RuntimeError("Moving patch left the sampled ground; extend the terrain before continuing")
        super().step()

    def render(self):
        base_pos = self.state_0.body_q[:1].numpy()[0, :3]
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            # Follow translation while preserving the user's orbit and zoom.
            self.viewer.camera.follow(base_pos)
        else:
            self.viewer.set_camera(pos=wp.vec3(*base_pos) + self.camera_offset, pitch=-25.0, yaw=125.0)
        super().render()
        if self.video_filename and self.sim_time > self._last_recorded_time:
            self.frames.append(self.viewer.get_frame().numpy())
            self._last_recorded_time = self.sim_time
            if len(self.frames) % max(1, round(self.fps / 10)) == 0 or len(self.frames) == self.video_frame_limit:
                self._show_recording_progress()
            if len(self.frames) >= self.video_frame_limit:
                # Request loop termination; run() closes the viewer afterward.
                self.viewer.renderer.app.exit()

    def _log_velocity_arrows(self, command, *, yaw_only=False):
        if not hasattr(self, "_velocity_starts"):
            self._velocity_starts = wp.empty(2, dtype=wp.vec3, device=self.model.device)
            self._velocity_ends = wp.empty(2, dtype=wp.vec3, device=self.model.device)
        wp.launch(
            _velocity_arrow_points,
            dim=1,
            inputs=[self.state_0.joint_q, self.state_0.joint_qd, command, yaw_only],
            outputs=[self._velocity_starts, self._velocity_ends],
            device=self.model.device,
        )
        self.viewer.log_arrows("/velocity_command", self._velocity_starts[:1], self._velocity_ends[:1], (0.0, 0.8, 0.0))
        self.viewer.log_arrows("/velocity_actual", self._velocity_starts[1:], self._velocity_ends[1:], (0.0, 0.2, 1.0))

    def render_ui(self, imgui):
        imgui.text(f"Terrain: {self.terrain_gen.type.value}")
        if self.terrain_gen.type == TerrainType.SLOPE:
            imgui.text(f"Slope: {self.terrain_gen.params['angle_degrees']:.1f} degrees (+X uphill)")
        elif self.terrain_gen.type == TerrainType.SINUSOIDAL:
            imgui.text(
                f"Wave amplitude: {self.terrain_gen.params['amplitude']:.2f} m; "
                f"wavelength: {self.terrain_gen.params['wavelength']:.2f} m"
            )
        _, self.show_window = imgui.checkbox("Show simulation window", self.show_window)

    def _show_recording_progress(self):
        recorded = len(self.frames) / self.fps
        total = self.video_frame_limit / self.fps
        fraction = len(self.frames) / self.video_frame_limit
        filled = int(30 * fraction)
        bar = "=" * filled + "-" * (30 - filled)
        print(
            f"\rRecording [{bar}] {recorded:.2f} / {total:.2f} simulated seconds ({fraction:.0%})",
            end="\n" if len(self.frames) == self.video_frame_limit else "",
            flush=True,
        )

    def save_video(self):
        if self.video_filename and self.frames:
            # Keep OpenCV optional when recording is disabled.
            from newton.examples.mpm_moving_patch_alpha.impl.utils.video_util import save_video_cv2  # noqa: PLC0415

            if len(self.frames) < self.video_frame_limit:
                print()
            print(f"Saving video to {self.video_filename} ...", flush=True)
            save_video_cv2(self.frames, self.video_filename, fps=int(self.fps * self.video_speed))

    def test_final(self):
        """Check terrain-relative heights; world Z may be negative downhill."""
        assert self.model.particle_count == self.terrain_reference.shape[0]
        for array in (
            self.state_0.body_q,
            self.state_0.body_qd,
            self.state_0.particle_q,
            self.state_0.particle_qd,
            self.state_0.joint_q,
            self.state_0.joint_qd,
        ):
            assert np.isfinite(array.numpy()).all()
        terrain = self.terrain_heightfield
        particles = self.state_0.particle_q.numpy()
        height = terrain.sample_height(particles[:, :2])
        assert np.min(particles[:, 2] - height) > -2 * self.mpm_solver.voxel_size
        bodies = self.state_0.body_q.numpy()[:, :3]
        assert np.all(bodies[:, 2] > terrain.sample_height(bodies[:, :2]))

    @staticmethod
    def create_parser():
        parser = MovingTerrainExample.create_parser()
        parser.add_argument("--terrain-type", choices=[kind.value for kind in TerrainType], default="flat")
        parser.add_argument("--wave-amplitude", type=float, default=0.3, help="Sinusoidal height amplitude [m].")
        parser.add_argument("--wave-wavelength", type=float, default=4.0, help="Sinusoidal wavelength along X [m].")
        parser.add_argument("--wave-phase", type=float, default=0.0, help="Sinusoidal phase at X=0 [degrees].")
        parser.add_argument(
            "--slope-angle", type=float, default=30.0, help="Slope angle [degrees], positive uphill along +X."
        )
        parser.add_argument("--video", type=str, default=None, help="Save OpenGL frames to this video file.")
        parser.add_argument(
            "--video-duration",
            type=float,
            default=10.0,
            help="Recording length in simulation seconds; auto-exit afterward. Overrides --num-frames when recording.",
        )
        parser.add_argument("--video-speed", type=float, default=1.0, help="Video playback speed multiplier.")
        return parser
