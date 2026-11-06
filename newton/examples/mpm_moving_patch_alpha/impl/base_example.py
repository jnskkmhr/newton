# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared alpha scene lifecycle with fresh terrain recycling and two-way coupling."""

from __future__ import annotations

import argparse
import math

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.examples.mpm_moving_patch_alpha.impl.controller.object_controller import ObjectController
from newton.examples.mpm_moving_patch_alpha.impl.moving_terrain_mpm import (
    MovingTerrainMPMConfig,
    MovingTerrainMPMWrapper,
)
from newton.examples.mpm_moving_patch_alpha.impl.terrain_generator import TerrainGenerator
from newton.solvers import SolverImplicitMPM, SolverMuJoCo


class MovingTerrainExample:
    frame_rate = 50

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = self.frame_rate
        self.frame_dt = 1.0 / self.fps
        self.sub_step = args.sub_step
        if not isinstance(self.sub_step, int) or isinstance(self.sub_step, bool) or self.sub_step < 1:
            raise ValueError("sub_step must be a positive integer")
        self.sim_dt = self.frame_dt / self.sub_step

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.5
        SolverImplicitMPM.register_custom_attributes(builder)

        rigid_body_start = builder.body_count
        tracked_body = self._emit_rigid_bodies(builder)
        rigid_body_end = builder.body_count
        if tracked_body is None:
            tracked_body = builder.body_label[rigid_body_start]
        self.setup_terrain(args, tracked_body=tracked_body)
        self.terrain_heightfield = self._emit_ground(builder)
        # Keep bounds available for robot placement and existing benchmark tooling.
        self.bed_lo = self.mpm_wrapper.bed_lo
        self.bed_hi = self.mpm_wrapper.bed_hi
        self.particles_per_cell = self.mpm_wrapper.particles_per_cell
        self.visible_ratio = 1.0
        self.show_buffer = args.show_boundary_particles
        self.shift_step = self.mpm_wrapper.config.shift_step
        self._emit_particles(builder, self.mpm_wrapper.voxel_size)
        self.model = builder.finalize()
        self.window_colors = wp.full(4, wp.vec3(0.1, 0.7, 1.0), dtype=wp.vec3, device=self.model.device)

        self.setup_solver(args, rigid_bodies=list(range(rigid_body_start, rigid_body_end)))
        self.mpm_solver = self.solver.solver("mpm")

        self.state_0 = self.model.state()
        self.rigid_collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.rigid_collision_pipeline.contacts()
        self.control = self.model.control()

        newton.examples.configure_coupled_view(self, args)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True
        self.show_impulses = False

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        # Articulated subclasses retain their own policy controllers.
        self.object_controller = None
        if all(joint_type == newton.JointType.FREE for joint_type in builder.joint_type):
            self.object_controller = ObjectController(
                self.model, self.state_0, list(range(rigid_body_start, rigid_body_end))
            )

        self.viewer.set_camera(pos=wp.vec3(3.0, -4.0, 2.5), pitch=-25.0, yaw=125.0)
        self.mpm_wrapper.bind(model=self.model, state=self.state_0, coupled_solver=self.solver)
        self.terrain_reference = self.mpm_wrapper.reference_positions
        self.particle_render_radii = self.mpm_wrapper.render_radii
        self.particle_render_colors = self.mpm_wrapper.render_colors
        self.window_center = self.mpm_wrapper.window_center
        self.window_shifts = self.mpm_wrapper.window_shifts
        self.window_starts = self.mpm_wrapper.window_starts
        self.window_ends = self.mpm_wrapper.window_ends
        self.show_window = False
        self.graph = None
        if not args.disable_cuda_graph:
            self.capture()

    def setup_terrain(self, args, *, tracked_body: str):
        """Set generated terrain, material parameters, and local particle dimensions."""
        self.terrain_gen = getattr(self, "terrain_gen", TerrainGenerator(type="flat"))
        self.mpm_wrapper = MovingTerrainMPMWrapper(
            config=MovingTerrainMPMConfig(shift_step=8.0 * args.voxel_size),
            tracked_body=tracked_body,
            terrain_gen=self.terrain_gen,
            voxel_size=args.voxel_size,
            particles_per_cell=args.particles_per_cell,
            density=3000.0,
            sand_depth=0.25,
            jitter=args.voxel_size / args.particles_per_cell,
            material_parameters={
                "mpm:friction": math.tan(math.radians(50.0)),
                "mpm:young_modulus": 40.0e6,
                "mpm:poisson_ratio": 0.3,
            },
        )

    def setup_solver(self, args, *, rigid_bodies: list[int], mpm_config: SolverImplicitMPM.Config | None = None):
        """Configure the local sparse MPM solve and two-way rigid-body coupling."""
        if mpm_config is None:
            mpm_config = SolverImplicitMPM.Config()
            mpm_config.voxel_size = self.mpm_wrapper.voxel_size
            mpm_config.grid_type = "sparse"
            mpm_config.grid_padding = 0
            mpm_config.max_active_cell_count = 1 << 16
            mpm_config.strain_basis = "P0"
            mpm_config.max_iterations = 50
            mpm_config.critical_fraction = 0.0
            mpm_config.warmstart_mode = "none"
            mpm_config.collider_velocity_mode = "forward"

        rigid_body_indices = wp.array(
            rigid_bodies,
            dtype=int,
            device=self.model.device,
        )
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mujoco",
                    solver=lambda v: SolverMuJoCo(
                        model=v,
                        use_mujoco_contacts=False,
                        njmax=args.rigid_njmax,
                        nconmax=args.rigid_nconmax,
                        cone="elliptic",
                        impratio=100,
                        iterations=100,
                    ),
                    bodies=[int(i) for i in rigid_body_indices.numpy()],
                    joints=list(range(self.model.joint_count)),
                    substeps=args.rigid_substeps,
                ),
                SolverCoupledProxy.Entry(
                    name="mpm",
                    solver=lambda v: SolverImplicitMPM(model=v, config=mpm_config),
                    particles=list(range(self.model.particle_count)),
                    in_place=True,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mujoco",
                        destination="mpm",
                        bodies=[int(i) for i in rigid_body_indices.numpy()],
                        mass_scale=getattr(args, "mass_scale", 1.0),
                        mode="lagged",
                        # MPM handles collider contact internally; no proxy
                        # collision pipeline should generate Contacts here.
                        collision_pipeline=lambda _model: None,
                    )
                ],
                iterations=args.proxy_iterations,
            ),
        )

    def _emit_ground(self, builder):
        self.mpm_wrapper.add_ground(builder)
        return self.mpm_wrapper.terrain

    def recycle_terrain(self):
        self.mpm_wrapper.update_terrain_layers()

    def capture(self):
        # Boundary mass changes require the solver's uncaptured property refresh.
        self.graph = None

    def simulate(self):
        self.recycle_terrain()
        self.state_0.clear_forces()
        if self.object_controller is not None:
            self.object_controller.apply(self.state_0, self.sim_dt)
        newton.examples.apply_coupled_viewer_forces(self, self.state_0)
        self.rigid_collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)

    def step(self):
        for _ in range(self.sub_step):
            self.simulate()
            self.sim_time += self.sim_dt

    def test_final(self):
        """Check finite coupled state and bounded terrain allocation above the ground."""
        assert self.model.particle_count == self.terrain_reference.shape[0]
        for array in (self.state_0.body_q, self.state_0.body_qd, self.state_0.particle_q, self.state_0.particle_qd):
            assert np.isfinite(array.numpy()).all()
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the sand",
            lambda q, qd: q[2] > 0.10,
        )
        voxel_size = self.mpm_solver.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

    def render(self):
        render_state = newton.examples.get_coupled_view_state(self)
        self.viewer.begin_frame(self.sim_time)
        self.mpm_wrapper.render_background(self.viewer, show_buffer=self.show_buffer)
        # if self.show_window:
        #     self.mpm_wrapper.update_window_outline()
        #     self.viewer.log_lines("/terrain_window", self.window_starts, self.window_ends, colors=self.window_colors)
        # else:
        #     self.viewer.log_lines("/terrain_window", None, None, None)
        show_particles = self.viewer.show_particles
        self.viewer.show_particles = False
        try:
            newton.examples.log_coupled_view(self, self.contacts)
        finally:
            self.viewer.show_particles = show_particles

        # adaptive particle rendering
        if render_state.particle_q is not None and render_state.particle_q.shape[0] > 0:
            self.mpm_wrapper.render_active_particle(
                positions=render_state.particle_q,
                visible_ratio=self.visible_ratio,
                show_buffer=self.show_buffer,
            )
            self.viewer.log_points(
                "/sand",
                points=render_state.particle_q,
                radii=self.particle_render_radii,
                colors=self.particle_render_colors,
                hidden=not self.viewer.show_particles,
            )

        if self.show_impulses and render_state.particle_q is not None:
            impulses, pos, _cid = self.mpm_solver.collect_collider_impulses(render_state)
            self.viewer.log_lines(
                "/impulses",
                starts=pos,
                ends=pos + impulses,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/impulses", None, None, None)

        self._render_overlays()
        self.viewer.end_frame()

    def _render_overlays(self):
        """Draw optional example overlays inside the viewer frame."""

    def render_ui(self, imgui):
        imgui.text("Right-click and drag the box to move the terrain window.")
        imgui.text("Blue outline: simulated boundary, including the settling buffer.")
        _changed, self.show_buffer = imgui.checkbox("Show boundary particles", self.show_buffer)
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)

    def _emit_rigid_bodies(self, builder: newton.ModelBuilder):
        """Create the tracked rigid body or articulated robot before terrain setup."""
        raise NotImplementedError("Scene examples must supply their rigid bodies")

    def _emit_particles(self, builder: newton.ModelBuilder, voxel_size: float):
        """Allocate particles through the shared wrapper lifecycle."""
        self.mpm_wrapper.allocate_particle_storage(builder)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument(
            "--sub-step",
            type=int,
            default=2,
            help="Coupled physics steps per rendered frame; physics rate is fps * sub_step",
        )
        parser.add_argument(
            "--show-boundary-particles",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Show zero-mass boundary particles and expand the background opening (rendering only).",
        )
        parser.add_argument("--voxel-size", type=float, default=0.05, help="MPM voxel size [m].")
        parser.add_argument("--particles-per-cell", type=int, default=3, help="Particle samples per voxel axis.")
        parser.add_argument("--rigid-njmax", type=int, default=300, help="MuJoCo constraint capacity per world.")
        parser.add_argument("--rigid-nconmax", type=int, default=150, help="MuJoCo contact capacity.")
        parser.add_argument(
            "--disable-cuda-graph",
            action="store_true",
            help="Compatibility flag; moving-boundary examples run without outer graph capture.",
        )
        parser.add_argument(
            "--mass-scale",
            "-pmr",
            help="Scale factor for source effective mass used by the MPM proxy collider",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--proxy-iterations",
            help="Number of proxy relaxation passes per coupled step",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--rigid-substeps",
            "--mujoco-substeps",
            dest="rigid_substeps",
            help="Number of rigid-solver substeps per coupled step",
            type=int,
            default=4,
        )
        parser.set_defaults(disable_cuda_graph=True)
        return parser
