# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render a procedural desert with rigid-body Unitree G1 policy locomotion.

Run: uv run --no-sync -m newton.examples.large_terrain_rendering.example_desert_g1
"""

import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
import newton.viewer

from .scene import write_scene
from .terrain import build_terrain
from .video import ViewerRTXVideo, configure_video


class Example:
    """Show the desert with a policy-driven or posed Unitree G1."""

    def __init__(self, viewer, args):
        if args.radius <= 0 or not math.isfinite(args.radius):
            raise ValueError("--radius must be finite and positive")
        if args.rings < 8 or args.sectors < 16 or args.rocks < 0:
            raise ValueError("Use at least 8 rings, 16 sectors, and a nonnegative rock count")
        if args.capture_frame < 1 or args.num_frames < 1:
            raise ValueError("Frame counts must be positive")
        if not math.isfinite(args.sun_elevation) or not 0 < args.sun_elevation <= 90:
            raise ValueError("--sun-elevation must be in (0, 90] degrees")
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.frame_count = 0
        self.frame_times = []
        self.step_times = []
        self.total_frame_times = []
        self.last_step_ms = 0.0
        self.simulation = None
        self.device = wp.get_device()
        started = time.perf_counter()
        self.directory = Path(args.output_dir).resolve()
        self.terrain = build_terrain(
            radius=args.radius, ring_count=args.rings, sector_count=args.sectors, seed=args.seed, device=self.device
        )
        asset = None
        if args.g1 and args.g1_mode == "posed":
            asset = (
                Path(args.g1_asset).resolve()
                if args.g1_asset
                else (newton.utils.download_asset("unitree_g1") / "usd_structured/g1_29dof_with_hand_rev_1_0.usda")
            )
            if not asset.is_file():
                raise FileNotFoundError(asset)
        self.scene_path, self.statistics = write_scene(
            self.directory,
            self.terrain,
            seed=args.seed,
            rock_count=args.rocks,
            device=self.device,
            sun_elevation=args.sun_elevation,
            g1_asset=asset,
        )
        if args.g1 and args.g1_mode == "policy":
            from .simulation import G1Simulation  # noqa: PLC0415 -- keep Torch optional for posed rendering

            self.simulation = G1Simulation(args, device=self.device)
            self.model = self.simulation.model
            self.state_0 = self.simulation.state
            self.state_1 = self.simulation.next_state
            self.control = self.simulation.control
            self.contacts = self.simulation.contacts
        else:
            self.model = newton.ModelBuilder().finalize()
            self.state_0 = self.model.state()
            self.state_1 = None
            self.control = None
            self.contacts = None
        viewer.set_model(self.model)
        if isinstance(viewer, newton.viewer.ViewerRTX):
            viewer.add_background_usd(str(self.scene_path))
        elif not isinstance(viewer, newton.viewer.ViewerNull):
            raise ValueError("This prototype supports the RTX and null viewers")
        self._set_camera()
        self.statistics["preparation_seconds"] = time.perf_counter() - started
        print(json.dumps(self.statistics, indent=2))

    def _set_camera(self):
        angle = 0.0
        if self.args.camera == "orbit":
            angle = 0.08 * self.sim_time
        x = 6.0 * math.cos(angle) + 12.0 * math.sin(angle)
        y = 6.0 * math.sin(angle) - 12.0 * math.cos(angle)
        z = 3.0
        target = (3.0, 17.0, 1.5)
        if self.args.camera == "follow" and self.simulation is not None:
            root = self.simulation.position
            x, y, z = root + np.array([-4.0, -6.0, 2.0])
            target = root + np.array([2.0, 1.0, 0.25])
        direction = (target[0] - x, target[1] - y, target[2] - z)
        yaw = math.degrees(math.atan2(direction[1], direction[0]))
        pitch = math.degrees(math.atan2(direction[2], math.hypot(direction[0], direction[1])))
        self.viewer.set_camera(wp.vec3(x, y, z), pitch=pitch, yaw=yaw)

    def gui(self, ui):
        """Toggle G1 tracking while preserving the view when it is disabled."""
        if self.simulation is None:
            return
        tracking = self.args.camera == "follow"
        label = "Deactivate camera tracking" if tracking else "Activate camera tracking"
        if ui.button(label):
            self.args.camera = "fixed" if tracking else "follow"
            if not tracking:
                self._set_camera()

    def step(self):
        """Advance the optional rigid simulation and the camera clock."""
        started = time.perf_counter()
        if self.simulation is not None:
            self.simulation.step(self.viewer)
            self.state_0 = self.simulation.state
            self.state_1 = self.simulation.next_state
        self.sim_time += self.frame_dt
        if self.args.camera in ("orbit", "follow"):
            self._set_camera()
        self.last_step_ms = (time.perf_counter() - started) * 1000
        if self.frame_count >= self.args.warmup_frames:
            self.step_times.append(self.last_step_ms)

    def render(self):
        """Render the static landscape and current robot pose."""
        started = time.perf_counter()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
        elapsed = time.perf_counter() - started
        if self.frame_count == 0:
            self.statistics["first_render_seconds"] = elapsed
        self.frame_count += 1
        if self.frame_count > self.args.warmup_frames:
            self.frame_times.append(elapsed * 1000)
            self.total_frame_times.append(elapsed * 1000 + self.last_step_ms)
        self.last_step_ms = 0.0
        if self.args.screenshot and self.frame_count == self.args.capture_frame:
            if not isinstance(self.viewer, newton.viewer.ViewerRTX):
                raise ValueError("--screenshot requires --viewer rtx")
            path = Path(self.args.screenshot).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.viewer.save_screenshot(str(path))
            print(f"Saved screenshot: {path}")

    def test_final(self):
        """Verify terrain geometry and the active G1 mode."""
        points, normals, uvs, indices = self.terrain
        assert np.isfinite(points).all() and np.isfinite(normals).all() and np.isfinite(uvs).all()
        assert indices.min() >= 0 and indices.max() < len(points)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-5)
        faces = points[indices.reshape(-1, 3)]
        cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        assert (cross[:, 2] > 0).all(), "Terrain triangles must face upward without degenerate faces"
        if self.simulation is not None:
            self.simulation.test_final()
        elif self.args.g1:
            assert 1.0 < self.statistics["g1_height_m"] < 2.0, "G1 must retain its real-world scale"
            assert abs(self.statistics["g1_ground_clearance_m"] - 0.015) < 1.0e-5

    def write_report(self):
        """Save separate physics, rendering, and combined wall-time measurements."""
        report = dict(self.statistics)
        report.update(
            seed=self.args.seed,
            radius_m=self.args.radius,
            device=str(self.device),
            resolution=[self.args.width, self.args.height],
            camera=self.args.camera,
            viewer=self.args.viewer,
            warmup_frames=self.args.warmup_frames,
            measured_frames=len(self.frame_times),
            timing="CPU wall time around render; not GPU timestamps",
        )
        if self.simulation is not None:
            report["simulation"] = self.simulation.metrics()
        if self.step_times:
            report["median_step_ms"] = float(np.median(self.step_times))
        if self.total_frame_times:
            total = np.asarray(self.total_frame_times)
            slowest = max(1, math.ceil(len(total) * 0.01))
            report["combined_step_and_render"] = {
                "median_ms": float(np.median(total)),
                "p95_ms": float(np.percentile(total, 95)),
                "max_ms": float(total.max()),
                "one_percent_low_fps": float(1000 / np.sort(total)[-slowest:].mean()),
            }
        report["warp_version"] = wp.__version__
        if self.args.viewer == "rtx":
            report["ovrtx_version"] = importlib.metadata.version("ovrtx")
            report["frame_generation"] = (
                "ViewerRTX default requests frame generation; generated frames not counted separately"
            )
        if self.frame_times:
            times = np.asarray(self.frame_times)
            count = max(1, math.ceil(len(times) * 0.01))
            report.update(
                median_ms=float(np.median(times)),
                p95_ms=float(np.percentile(times, 95)),
                p99_ms=float(np.percentile(times, 99)),
                max_ms=float(times.max()),
                one_percent_low_fps=float(1000 / np.sort(times)[-count:].mean()),
            )
        (self.directory / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))

    @staticmethod
    def create_parser():
        """Create the standalone prototype's command-line options."""
        parser = newton.examples.create_parser()
        parser.set_defaults(viewer="rtx", num_frames=180)
        parser.add_argument("--width", type=int, default=1280)
        parser.add_argument("--height", type=int, default=720)
        parser.add_argument("--radius", type=float, default=1000.0, help="Terrain radius [m].")
        parser.add_argument("--rings", type=int, default=256)
        parser.add_argument("--sectors", type=int, default=512)
        parser.add_argument("--rocks", type=int, default=4500)
        parser.add_argument("--seed", type=int, default=7)
        parser.add_argument("--sun-elevation", type=float, default=32.0, help="Sun elevation [degrees].")
        parser.add_argument("--g1", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--g1-asset", help="Optional local G1 USD file for posed mode.")
        parser.add_argument("--g1-mode", choices=["policy", "posed"], default="policy")
        parser.add_argument(
            "--use-cuda-graph",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture repeated rigid physics substeps on CUDA; policy inference remains outside.",
        )
        parser.add_argument("--config", default=str(Path(__file__).resolve().parents[3] / "config/g1.yaml"))
        parser.add_argument("--policy", help="TorchScript policy override.")
        parser.add_argument("--robot-asset", help="G1 MJCF or URDF override for policy mode.")
        parser.add_argument(
            "--command",
            nargs=3,
            type=float,
            default=(0.6, 0.0, 0.0),
            metavar=("VX", "VY", "WZ"),
            help="Initial commanded velocity [m/s, m/s, rad/s]; applied after 0.5 s settling.",
        )
        parser.add_argument("--collision-size", type=float, default=8.0, help="Local collision patch side length [m].")
        parser.add_argument("--collision-spacing", type=float, default=0.1, help="Local collision mesh spacing [m].")
        parser.add_argument("--camera", choices=["fixed", "orbit", "follow"], default="follow")
        parser.add_argument("--output-dir", default=str(Path(__file__).parent / "_generated"))
        parser.add_argument("--screenshot", help="Save an RTX screenshot at --capture-frame.")
        parser.add_argument("--video", help="Save RTX frames to an MP4 file; supports --headless.")
        parser.add_argument(
            "--video-duration",
            type=float,
            default=10.0,
            help="Recording length in simulation seconds; auto-exit afterward. Overrides --num-frames.",
        )
        parser.add_argument("--video-speed", type=float, default=1.0, help="Video playback speed multiplier.")
        parser.add_argument("--capture-frame", type=int, default=32)
        parser.add_argument("--warmup-frames", type=int, default=32)
        return parser


if __name__ == "__main__":
    args = Example.create_parser().parse_args()
    configure_video(args, fps=60)
    if args.viewer not in ("rtx", "null"):
        raise ValueError("Use --viewer rtx or --viewer null")
    if args.width <= 0 or args.height <= 0 or args.warmup_frames < 0:
        raise ValueError("Image dimensions must be positive and warm-up frames nonnegative")
    if args.headless and args.screenshot and args.capture_frame > args.num_frames:
        raise ValueError("--capture-frame must not exceed --num-frames in headless mode")
    if args.device:
        wp.set_device(args.device)
    viewer = (
        ViewerRTXVideo(args, fps=60) if args.viewer == "rtx" else newton.viewer.ViewerNull(num_frames=args.num_frames)
    )
    try:
        example = Example(viewer, args)
        newton.examples.run(example, args)
        example.write_report()
    finally:
        viewer.close()
