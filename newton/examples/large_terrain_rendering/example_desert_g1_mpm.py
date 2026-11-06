# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Initial alpha MPM desert demo with a terrain clipmap and adaptive bedrock collider.

Run: uv run --no-sync -m newton.examples desert_g1_mpm --config config/g1.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
import newton.viewer

from .clipmap import TerrainClipmap
from .example_desert_g1 import Example as RigidExample
from .mpm_simulation import MPMSimulation
from .scene import write_scene
from .video import ViewerRTXVideo, configure_video


class Example(RigidExample):
    """Keep physics centered on G1 and rendering centered on the independent camera."""

    def __init__(self, viewer, args):
        positive = [
            args.mpm_size,
            args.mpm_buffer,
            args.sand_depth,
            args.voxel_size,
            args.patch_shift,
            args.collider_spacing,
            args.clipmap_spacing,
        ]
        if not np.isfinite(positive).all() or min(positive) <= 0 or args.particles_per_cell < 1:
            raise ValueError("MPM dimensions, resolutions, and particles per cell must be positive")
        if args.mpm_iterations < 1 or args.use_cuda_graph:
            raise ValueError("Use positive MPM iterations and --no-use-cuda-graph for this initial demo")
        if args.g1_mode != "policy" or not args.g1:
            raise ValueError("The MPM demo requires G1 policy mode")
        if args.rocks < 0 or args.width <= 0 or args.height <= 0 or args.num_frames < 1 or args.warmup_frames < 0:
            raise ValueError("Invalid image dimensions, frame count, or rock count")
        if not 0 < args.sun_elevation <= 90 or args.capture_frame < 1:
            raise ValueError("Use a sun elevation in (0, 90] and a positive capture frame")
        if args.headless and args.screenshot and args.capture_frame > args.num_frames:
            raise ValueError("The capture frame must be within the requested run")
        self.viewer, self.args = viewer, args
        self.show_boundary = args.show_boundary
        self.show_particles = True
        self.fps, self.frame_dt = 50, 0.02
        self.sim_time, self.frame_count = 0.0, 0
        self.frame_times, self.step_times, self.total_frame_times = [], [], []
        self.last_step_ms = 0.0
        self.device = wp.get_device()
        self.directory = Path(args.output_dir).resolve()
        started = time.perf_counter()
        self.simulation = MPMSimulation(args, device=self.device)
        self.model, self.state_0 = self.simulation.model, self.simulation.state
        self.state_1 = self.simulation.next_state
        self.control, self.contacts = self.simulation.control, self.simulation.contacts
        self.clipmap = TerrainClipmap(
            cells=args.clipmap_cells,
            levels=args.clipmap_levels,
            spacing=args.clipmap_spacing,
            seed=args.seed,
            visual_noise=True,
            device=self.device,
        )
        viewer.set_model(self.model)
        viewer.show_particles = False
        self._set_camera()
        self._update_clipmap()
        self.terrain = (
            self.clipmap.points.numpy(),
            self.clipmap.normals.numpy(),
            self.clipmap.uvs.numpy(),
            self.clipmap.indices.numpy(),
        )
        self.scene_path, self.statistics = write_scene(
            self.directory,
            self.terrain,
            seed=args.seed,
            rock_count=args.rocks,
            device=self.device,
            sun_elevation=args.sun_elevation,
            include_terrain=False,
        )
        if isinstance(viewer, newton.viewer.ViewerRTX):
            viewer.add_background_usd(str(self.scene_path))
        self.statistics.update(
            demo="alpha_mpm_clipmap",
            clipmap_width_m=self.clipmap.extent,
            preparation_seconds=time.perf_counter() - started,
        )
        print(json.dumps(self.statistics, indent=2))

    def _update_clipmap(self):
        camera = getattr(self.viewer, "camera", None)
        center = tuple(camera.pos) if camera is not None else self.simulation.position
        patch = self.simulation.patch
        outer = patch.width / 2 if self.show_boundary else patch.size / 2
        inner = max(0.0, outer - 0.05)
        self.clipmap.update(center, hole=patch.center, inner=inner, outer=outer, depth=patch.depth)

    def gui(self, ui):
        """Expose camera tracking and boundary visibility with the actual region dimensions."""
        super().gui(ui)
        label = "Hide boundary terrain" if self.show_boundary else "Show boundary terrain"
        if ui.button(label):
            self.show_boundary = not self.show_boundary
            if isinstance(self.viewer, ViewerRTXVideo):
                self.viewer.reset_render_history(self.sim_time)
        label = "Hide particles" if self.show_particles else "Show particles"
        if ui.button(label):
            self.show_particles = not self.show_particles
            if isinstance(self.viewer, ViewerRTXVideo):
                self.viewer.reset_render_history(self.sim_time)
        patch = self.simulation.patch
        ui.text(f"Simulated terrain: {patch.size:g} x {patch.size:g} m")
        ui.text(f"Kinematic boundary: {patch.buffer:g} m per side (mass = 0)")
        ui.text(f"Total particle region: {patch.width:g} x {patch.width:g} m")

    def _render_particles(self):
        # Build both RTX batches initially with positive radii so they can be restored.
        dynamic_points, boundary_points = None, None
        if self.show_particles or self.frame_count == 0:
            dynamic_points, boundary_points = self.simulation.patch.render_points()
        self.viewer.log_points(
            "alpha_sand",
            dynamic_points,
            radii=self.simulation.patch.render_radius,
            colors=(0.72, 0.55, 0.34),
            hidden=not self.show_particles,
        )
        self.viewer.log_points(
            "boundary_sand",
            boundary_points if self.show_boundary or self.frame_count == 0 else None,
            radii=self.simulation.patch.render_radius,
            colors=(0.95, 0.08, 0.01),
            hidden=not (self.show_particles and self.show_boundary),
        )

    def render(self):
        started = time.perf_counter()
        self._update_clipmap()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_mesh(
            "desert_clipmap",
            self.clipmap.points,
            self.clipmap.indices,
            normals=self.clipmap.normals,
            uvs=self.clipmap.uvs,
            texture=str(self.directory / "sand_color.png"),
            roughness=0.9,
            dynamic=True,
        )
        self.viewer.log_state(self.state_0)
        self._render_particles()
        self.viewer.end_frame()
        elapsed = time.perf_counter() - started
        if self.frame_count == 0:
            self.statistics["first_render_seconds"] = elapsed
        self.frame_count += 1
        if self.frame_count > self.args.warmup_frames:
            self.frame_times.append(elapsed * 1000)
            self.total_frame_times.append(elapsed * 1000 + self.last_step_ms)
        self.last_step_ms = 0.0
        self.statistics["clipmap_updates"] = self.clipmap.updates
        if self.args.screenshot and self.frame_count == self.args.capture_frame:
            if not isinstance(self.viewer, newton.viewer.ViewerRTX):
                raise ValueError("Screenshots require the RTX viewer")
            path = Path(self.args.screenshot).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.viewer.save_screenshot(str(path))

    @staticmethod
    def create_parser():
        parser = RigidExample.create_parser()
        parser.set_defaults(
            use_cuda_graph=False, command=(0.6, 0.0, 0.0), output_dir=str(Path(__file__).parent / "_generated/mpm")
        )
        parser.add_argument("--clipmap-cells", type=int, default=32)
        parser.add_argument("--clipmap-levels", type=int, default=8)
        parser.add_argument("--clipmap-spacing", type=float, default=0.5)
        parser.add_argument("--collider-spacing", type=float, default=0.25)
        parser.add_argument("--mpm-size", type=float, default=3.0, help="Visible simulated sand width [m].")
        parser.add_argument("--mpm-buffer", type=float, default=0.5, help="Kinematic boundary width on each side [m].")
        parser.add_argument(
            "--show-boundary",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Show zero-mass boundary particles in orange; otherwise show only simulated terrain.",
        )
        parser.add_argument("--sand-depth", type=float, default=0.08)
        parser.add_argument("--voxel-size", type=float, default=0.05)
        parser.add_argument("--particles-per-cell", type=int, default=2)
        parser.add_argument("--patch-shift", type=float, default=0.5)
        parser.add_argument("--mpm-iterations", type=int, default=50)
        return parser


if __name__ == "__main__":
    args = Example.create_parser().parse_args()
    configure_video(args, fps=50)
    if args.device:
        wp.set_device(args.device)
    if args.viewer not in ("rtx", "null"):
        raise ValueError("Use --viewer rtx or --viewer null")
    viewer = (
        ViewerRTXVideo(args, fps=50) if args.viewer == "rtx" else newton.viewer.ViewerNull(num_frames=args.num_frames)
    )
    try:
        example = Example(viewer, args)
        newton.examples.run(example, args)
        example.write_report()
    finally:
        viewer.close()
