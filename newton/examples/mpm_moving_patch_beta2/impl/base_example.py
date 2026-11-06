# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared scene lifecycle for beta2 moving-terrain examples."""

import math

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.examples.mpm_moving_patch_beta2.impl.moving_terrain_mpm import (
    MovingTerrainMPMConfig,
    MovingTerrainMPMWrapper,
)
from newton.examples.mpm_moving_patch_beta2.impl.particle_rendering import ParticleRenderer
from newton.examples.mpm_moving_patch_beta2.impl.terrain_generator import TerrainGenerator
from newton.examples.mpm_moving_patch_beta2.impl.terrain_slope_colors import TerrainSlopeColors
from newton.solvers import SolverImplicitMPM, SolverMuJoCo


class MovingTerrainExample:
    """Share terrain, two-way coupling, and rendering across rigid-body scenes."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sub_step = args.sub_step
        if not isinstance(self.sub_step, int) or isinstance(self.sub_step, bool) or self.sub_step < 1:
            raise ValueError("sub_step must be a positive integer")
        self.sim_dt = self.frame_dt / self.sub_step
        self.show_buffer = False
        self.particle_color = (0.72, 0.59, 0.34)
        self.boundary_color = (0.85, 0.38, 0.12)
        self.slope_colors = None
        self.color_by_slope = args.color_by_slope
        self.slope_color_range = args.slope_color_range
        if not math.isfinite(self.slope_color_range) or self.slope_color_range <= 0.0:
            raise ValueError("slope_color_range must be positive and finite")
        self.show_impulses = False
        self.visible_ratio = 1.0
        self.particle_lod_pixels = args.particle_lod_pixels
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.5
        SolverImplicitMPM.register_custom_attributes(builder)
        tracked_body = self._emit_rigid_bodies(builder)
        rigid_bodies = list(range(builder.body_count))
        self.setup_terrain(args, tracked_body=tracked_body)
        self._emit_ground(builder)
        self._place_rigid_bodies(builder)
        self.mpm_wrapper.allocate_particle_storage(builder)
        self.model = builder.finalize()
        self.setup_solver(args, rigid_bodies=rigid_bodies)
        self.mpm_solver = self.solver.solver("mpm")
        self.state_0 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.mpm_wrapper.bind(model=self.model, state=self.state_0, coupled_solver=self.solver)
        self.rigid_collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.rigid_collision_pipeline.contacts()
        self._create_controller(args)
        self.particle_renderer = ParticleRenderer(self.model.particle_count, self.model.device)
        self.particle_render_radii = self.mpm_wrapper.render_radii
        self.particle_render_colors = self.mpm_wrapper.render_colors
        newton.examples.configure_coupled_view(self, args)
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self.render_ui, position="side")
        viewer.show_particles = True
        viewer.set_camera(pos=wp.vec3(3.0, -4.0, 2.5), pitch=-25.0, yaw=125.0)
        self.camera_offset = wp.vec3(2.5, -2.5, 1.0)
        if isinstance(viewer, newton.viewer.ViewerGL):
            body = self.mpm_wrapper.body_index
            viewer.camera.look_at(self.state_0.body_q[body : body + 1].numpy()[0, :3])

    def setup_terrain(self, args, *, tracked_body: str):
        """Define the ground generator, particle material, and terrain dimensions.

        Called after rigid bodies are added, before ground or particles are added
        and before the model is finalized. Override this method to assign
        ``self.terrain_gen`` and ``self.mpm_wrapper`` with scene-specific settings.
        Particle allocation and wrapper binding remain in the base initializer.

        Args:
            args: Parsed example options.
            tracked_body: Name of the body followed by the terrain patch.
        """
        terrain_params = {
            "flat": {},
            "slope": {"angle_degrees": args.slope_angle},
        }
        self.terrain_gen = TerrainGenerator(type=args.terrain_type, params=terrain_params[args.terrain_type])
        voxel_size = args.voxel_size
        cell_size = voxel_size / args.particles_per_cell
        radius = 0.5 * cell_size
        jitter = 2.0 * radius  # Meters, as in ModelBuilder.add_particle_grid; 0 disables jitter.
        # Material choices belong to the example, not the moving-terrain wrapper.
        material_density = 2500.0  # kg/m^3
        material_parameters = {
            "mpm:friction": math.tan(math.radians(50)),
            "mpm:young_modulus": 40.0e6,  # Pa
            "mpm:poisson_ratio": 0.3,
        }
        self.mpm_wrapper = MovingTerrainMPMWrapper(
            config=MovingTerrainMPMConfig(
                background_height_spacing=voxel_size,
                background_terrain_size=(40, 40),
            ),
            tracked_body=tracked_body,
            material_parameters=material_parameters,
            density=material_density,
            voxel_size=voxel_size,
            particles_per_cell=args.particles_per_cell,
            jitter=jitter,
            terrain_gen=self.terrain_gen,
        )

    def setup_solver(
        self,
        args,
        *,
        rigid_bodies: list[int],
        mpm_config: SolverImplicitMPM.Config | None = None,
    ):
        """Create the MPM solver and two-way rigid-body coupling for the built model.

        Called after model finalization, before states and controllers are created.
        Override to assign ``self.solver``. To change only MPM options, pass a
        configured ``SolverImplicitMPM.Config`` to this base implementation.
        Defaults derive voxel size and capacity from ``self.mpm_wrapper``.

        Args:
            args: Parsed example options, including rigid and coupling settings.
            rigid_bodies: Model body indices participating in two-way coupling.
            mpm_config: Optional complete MPM configuration. If omitted, use the
                example's sparse-grid defaults based on the terrain definition.
        """
        if mpm_config is None:
            voxel_size = self.mpm_wrapper.voxel_size
            mpm_config = SolverImplicitMPM.Config()
            mpm_config.voxel_size = voxel_size
            mpm_config.grid_type = "sparse"
            mpm_config.grid_padding = 0
            # Reserve for the local particle bed, not the background height array.
            cells = np.ceil(self.mpm_wrapper.outer_size / voxel_size + 8)
            mpm_config.max_active_cell_count = max(
                65536, int(np.prod(cells) * (self.mpm_wrapper.sand_depth / voxel_size + 12))
            )
            mpm_config.transfer_scheme = "pic"
            mpm_config.strain_basis = "P0"
            # mpm_config.velocity_basis = "Q1"
            # mpm_config.collider_basis = "pic27"
            # mpm_config.air_drag = 1.0
            mpm_config.max_iterations = 50
            mpm_config.critical_fraction = 0.0
            # mpm_config.warmstart_mode = "auto"
            mpm_config.collider_velocity_mode = "forward"

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mujoco",
                    solver=lambda v: SolverMuJoCo(
                        v, njmax=args.rigid_njmax, nconmax=args.rigid_nconmax, impratio=100, iterations=100
                    ),
                    bodies=rigid_bodies,
                    joints=list(range(self.model.joint_count)),
                    substeps=args.rigid_substeps,
                ),
                SolverCoupledProxy.Entry(
                    name="mpm",
                    solver=lambda v: SolverImplicitMPM(model=v, config=mpm_config),
                    particles=list(range(self.model.particle_count)),
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mujoco",
                        destination="mpm",
                        bodies=rigid_bodies,
                        mass_scale=args.mass_scale,
                        mode="lagged",
                        collision_pipeline=lambda _model: None,
                    )
                ],
                iterations=args.proxy_iterations,
            ),
        )

    def _emit_ground(self, builder):
        """Add the wrapper's ground mesh; children may supply a texture."""
        self.mpm_wrapper.add_ground(builder)

    def _render_overlays(self):
        """Log optional scene visuals before the viewer finishes the frame."""

    def _place_rigid_bodies(self, builder):
        # Raise the entire robot by the local ground height, preserving its joint geometry.
        ground_z = float(self.mpm_wrapper.sample_ground_height(0.0))
        for body in range(builder.body_count):
            pose = builder.body_q[body]
            position = wp.transform_get_translation(pose) + wp.vec3(0.0, 0.0, ground_z)
            builder.body_q[body] = wp.transform(position, wp.transform_get_rotation(pose))
        for joint in range(builder.joint_count):
            if builder.joint_type[joint] == newton.JointType.FREE:
                builder.joint_q[builder.joint_q_start[joint] + 2] += ground_z

    def _emit_rigid_bodies(self, builder):
        """Add scene bodies and return the name of the body followed by the terrain."""
        raise NotImplementedError("Scene examples must supply their rigid bodies")

    def _create_controller(self, args):
        """Default to passive bodies; scenes may create their own controller."""
        self.object_controller = None

    def recycle_terrain(self):
        """Update terrain layers before the coupled physics step."""
        self.mpm_wrapper.update_terrain_layers()

    def simulate(self):
        """Update terrain and advance the two-way coupled solvers."""
        self.recycle_terrain()
        self.state_0.clear_forces()
        if self.object_controller is not None:
            self.object_controller.apply(self.state_0, self.sim_dt)
        newton.examples.apply_coupled_viewer_forces(self, self.state_0)
        self.rigid_collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)

    def step(self):
        """Advance one uncaptured frame and the simulation clock."""
        for _ in range(self.sub_step):
            self.simulate()
            self.sim_time += self.sim_dt

    def _follow_camera(self):
        """Follow the terrain's tracked body, preserving OpenGL orbit and zoom."""
        body = self.mpm_wrapper.body_index
        position = self.state_0.body_q[body : body + 1].numpy()[0, :3]
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.camera.follow(position)
        else:
            self.viewer.set_camera(pos=wp.vec3(*position) + self.camera_offset, pitch=-25.0, yaw=125.0)

    def render(self):
        """Render the bodies, terrain particles and optional coupling impulses."""
        self._follow_camera()
        render_state = newton.examples.get_coupled_view_state(self)
        self.viewer.begin_frame(self.sim_time)
        show_particles = self.viewer.show_particles
        self.viewer.show_particles = False
        try:
            newton.examples.log_coupled_view(self, self.contacts)
        finally:
            self.viewer.show_particles = show_particles

        if self.viewer.show_particles and render_state.particle_q is not None:
            self.mpm_wrapper.render_active_particle(
                positions=render_state.particle_q,
                visible_ratio=self.visible_ratio,
                show_buffer=self.show_buffer,
                color_by_height=False,
                particle_color=self.particle_color,
                boundary_color=self.boundary_color,
            )
            if self.color_by_slope:
                if self.slope_colors is None:
                    self.slope_colors = TerrainSlopeColors(self.mpm_wrapper)
                self.slope_colors.update(render_state.particle_q, slope_range=self.slope_color_range)
            positions, radii, colors = self.particle_renderer.prepare(
                render_state.particle_q,
                self.particle_render_radii,
                self.particle_render_colors,
                camera=getattr(self.viewer, "camera", None),
                target_pixels=self.particle_lod_pixels,
            )
            self.viewer.log_points(
                "/sand", points=positions, radii=radii, colors=colors, hidden=self.particle_renderer.count == 0
            )
        else:
            self.particle_renderer.count = 0
            self.viewer.log_points("/sand", points=self.particle_renderer.positions[:0], hidden=True)
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

    def render_ui(self, imgui):
        """Expose boundary particles and explain the beta2 scene controls."""
        imgui.text("Right-click drag the object; returning terrain uses GPU height memory.")
        _, self.show_buffer = imgui.checkbox("Show fixed boundary particles", self.show_buffer)
        _, self.show_impulses = imgui.checkbox("Show coupling impulses", self.show_impulses)
        self.render_particle_ui(imgui)

    def render_particle_ui(self, imgui):
        """Expose rendering detail without changing simulation particles."""
        _, self.particle_lod_pixels = imgui.slider_float(
            "Particle detail [pixels; 0 = all]", self.particle_lod_pixels, 0.0, 8.0
        )
        imgui.text(f"Rendered particles: {self.particle_renderer.count:,} / {self.model.particle_count:,}")
        imgui.text("Terrain cache: GPU measured height")
        _, self.color_by_slope = imgui.checkbox("Color particles by terrain slope", self.color_by_slope)
        if self.color_by_slope:
            _, self.slope_color_range = imgui.slider_float("Slope at full red [m/m]", self.slope_color_range, 0.05, 2.0)
        imgui.text("Blue: flat terrain; red: steep terrain")

    def test_final(self):
        """Check finite coupled state, saved heights and kinematic boundary masses."""
        for values in (
            self.state_0.body_q,
            self.state_0.particle_q,
            self.state_0.particle_qd,
            self.mpm_wrapper.background_heights,
        ):
            assert np.isfinite(values.numpy()).all()
        boundary = self.mpm_wrapper.layers.numpy() == 2
        assert np.all(self.model.particle_mass.numpy()[boundary] == 0.0)
        active = (self.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
        assert np.any(active & boundary)
        assert np.all(self.model.particle_mass.numpy()[~active] == 0.0)

    @staticmethod
    def create_parser():
        """Create viewer, terrain, and coupling command-line options."""
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument(
            "--sub-step",
            type=int,
            default=2,
            help="Coupled physics steps per rendered frame; physics rate is fps * sub_step",
        )
        parser.add_argument(
            "--disable-cuda-graph", action="store_true", help="Compatibility flag; beta2 always runs uncaptured"
        )
        parser.add_argument(
            "--mass-scale",
            "-pmr",
            type=float,
            default=1.0,
            help="Scale the effective rigid-body mass used by MPM proxy coupling",
        )
        parser.add_argument("--proxy-iterations", type=int, default=1, help="Proxy relaxation passes per coupled step")
        parser.add_argument(
            "--rigid-substeps",
            "--mujoco-substeps",
            dest="rigid_substeps",
            type=int,
            default=4,
            help="Rigid-solver substeps per coupled step",
        )
        parser.add_argument("--rigid-njmax", type=int, default=300, help="MuJoCo constraint capacity")
        parser.add_argument("--rigid-nconmax", type=int, default=150, help="MuJoCo contact capacity")
        parser.add_argument("--voxel-size", type=float, default=0.05, help="MPM voxel size [m]")
        parser.add_argument("--particles-per-cell", type=int, default=3, help="Particle samples per voxel axis")
        parser.add_argument("--terrain-type", choices=("flat", "slope"), default="flat")
        parser.add_argument("--slope-angle", type=float, default=20.0, help="Slope angle [degrees].")
        parser.add_argument(
            "--particle-lod-pixels",
            type=float,
            default=2.0,
            help="Approximate minimum rendered particle diameter [pixels]; 0 keeps all visible particles.",
        )
        parser.add_argument(
            "--color-by-slope",
            "--color-by-height",
            action="store_true",
            help="Color particles by live terrain gradient magnitude; flat is blue and steep is red.",
        )
        parser.add_argument(
            "--slope-color-range",
            "--height-color-range",
            type=float,
            default=1.0,
            help="Terrain slope [m/m] giving saturated red; flat terrain is blue.",
        )
        parser.set_defaults(disable_cuda_graph=True)
        return parser
