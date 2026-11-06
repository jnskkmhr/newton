# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Mouse-dragged box on local MPM terrain with tiled height memory.

Run ``uv run -m newton.examples moving_terrain_mpm_box_beta2``.
"""

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
from newton.solvers import SolverImplicitMPM, SolverMuJoCo


class Example:
    """Keep particle/grid work local while preserving departed terrain heights."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 50
        self.sub_step = args.sub_step
        if self.sub_step < 1:
            raise ValueError("sub_step must be positive")
        self.sim_dt = 1.0 / (self.fps * self.sub_step)
        self.sim_time = 0.0
        self.show_buffer = False
        self.color_by_height = args.color_by_height
        self.particle_lod_pixels = args.particle_lod_pixels
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        body = builder.add_body(label="box", xform=wp.transform(wp.vec3(0, 0, 0.8), wp.quat_identity()))
        builder.add_shape_box(
            body,
            hx=0.25,
            hy=0.35,
            hz=0.25,
            cfg=builder.ShapeConfig(density=75.0 / (0.5 * 0.7 * 0.5), mu=0.5),
            color=(0.2, 0.45, 0.75),
        )
        self.setup_terrain(args, tracked_body="box")
        self.mpm_wrapper.add_ground(builder)
        self.mpm_wrapper.allocate_particle_storage(builder)
        self.model = builder.finalize()
        self.setup_solver(args)
        self.mpm_solver = self.solver.solver("mpm")
        self.state_0 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.control = self.model.control()
        self.mpm_wrapper.bind(model=self.model, state=self.state_0, coupled_solver=self.solver)
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
        self.contacts = self.collision_pipeline.contacts()
        self.particle_renderer = ParticleRenderer(self.model.particle_count, self.model.device)
        newton.examples.configure_coupled_view(self, args)
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self.render_ui, position="side")
        viewer.show_particles = True
        viewer.set_camera(pos=wp.vec3(3, -4, 2.5), pitch=-25, yaw=125)
        if isinstance(viewer, newton.viewer.ViewerGL):
            viewer.camera.look_at(self.state_0.body_q[body : body + 1].numpy()[0, :3])

    def setup_terrain(self, args, *, tracked_body: str):
        """Define material, sampling, local region, and background tile dimensions."""
        self.terrain_gen = TerrainGenerator(
            type=args.terrain_type,
            params={"angle_degrees": args.slope_angle} if args.terrain_type == "slope" else {},
        )
        self.mpm_wrapper = MovingTerrainMPMWrapper(
            config=MovingTerrainMPMConfig(background_terrain_size=(40.0, 40.0)),
            tracked_body=tracked_body,
            voxel_size=args.voxel_size,
            particles_per_cell=args.particles_per_cell,
            sand_depth=0.5,
            density=2500.0,
            jitter=args.voxel_size / args.particles_per_cell,
            material_parameters={
                "mpm:young_modulus": 4.0e7,
                "mpm:poisson_ratio": 0.3,
                "mpm:friction": math.tan(math.radians(50)),
            },
            terrain_gen=self.terrain_gen,
        )

    def setup_solver(self, args):
        """Create a sparse local MPM grid and two-way MuJoCo coupling."""
        config = SolverImplicitMPM.Config()
        config.voxel_size = self.mpm_wrapper.voxel_size
        config.grid_type = "sparse"
        config.grid_padding = 0
        cells = np.ceil(self.mpm_wrapper.outer_size / config.voxel_size + 8)
        config.max_active_cell_count = max(
            65536, int(np.prod(cells) * (self.mpm_wrapper.sand_depth / config.voxel_size + 12))
        )
        config.transfer_scheme = "pic"
        config.strain_basis = "P0"
        config.max_iterations = 50
        config.critical_fraction = 0.0
        config.collider_velocity_mode = "forward"
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mujoco",
                    solver=lambda v: SolverMuJoCo(v, njmax=300, nconmax=150, impratio=100, iterations=100),
                    bodies=list(range(self.model.body_count)),
                    joints=list(range(self.model.joint_count)),
                    substeps=4,
                ),
                SolverCoupledProxy.Entry(
                    name="mpm",
                    solver=lambda v: SolverImplicitMPM(v, config=config),
                    particles=list(range(self.model.particle_count)),
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mujoco",
                        destination="mpm",
                        bodies=list(range(self.model.body_count)),
                        mass_scale=1.0,
                        mode="lagged",
                        collision_pipeline=lambda _model: None,
                    )
                ],
                iterations=1,
            ),
        )

    def simulate(self):
        """Maintain local terrain, then advance both coupled solvers."""
        self.mpm_wrapper.update_terrain_layers()
        self.state_0.clear_forces()
        newton.examples.apply_coupled_viewer_forces(self, self.state_0)
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)

    def step(self):
        for _ in range(self.sub_step):
            self.simulate()
            self.sim_time += self.sim_dt

    def render(self):
        position = self.state_0.body_q[self.mpm_wrapper.body_index : self.mpm_wrapper.body_index + 1].numpy()[0, :3]
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.camera.follow(position)
        else:
            self.viewer.set_camera(pos=wp.vec3(*position) + wp.vec3(3, -4, 2), pitch=-25, yaw=125)
        self.viewer.begin_frame(self.sim_time)
        show = self.viewer.show_particles
        self.viewer.show_particles = False
        try:
            newton.examples.log_coupled_view(self, self.contacts)
        finally:
            self.viewer.show_particles = show
        state = newton.examples.get_coupled_view_state(self)
        radii = self.mpm_wrapper.render_active_particle(
            positions=state.particle_q, show_buffer=self.show_buffer, color_by_height=self.color_by_height
        )
        if show:
            points, radii, colors = self.particle_renderer.prepare(
                state.particle_q,
                radii,
                self.mpm_wrapper.render_colors,
                camera=getattr(self.viewer, "camera", None),
                target_pixels=self.particle_lod_pixels,
            )
            self.viewer.log_points("/sand", points, radii, colors)
        else:
            self.viewer.log_points("/sand", self.particle_renderer.positions[:0], 0.0, hidden=True)
        self.viewer.end_frame()

    def render_ui(self, imgui):
        imgui.text("Right-click drag the box. Revisited terrain restores saved height.")
        _, self.show_buffer = imgui.checkbox("Show boundary terrain", self.show_buffer)
        _, self.color_by_height = imgui.checkbox("Color by lowering", self.color_by_height)
        cache = self.mpm_wrapper.height_cache
        imgui.text(f"Saved GPU tiles: {cache.saved_tile_count} / {cache.capacity}")
        imgui.text(f"GPU height cache: {cache.gpu_bytes / 2**20:.2f} MiB")

    def test_final(self):
        for a in (
            self.state_0.body_q,
            self.state_0.particle_q,
            self.state_0.particle_qd,
            self.mpm_wrapper.background_heights,
        ):
            assert np.isfinite(a.numpy()).all()
        boundary = self.mpm_wrapper.layers.numpy() == 2
        assert np.all(self.model.particle_mass.numpy()[boundary] == 0)
        assert np.any(boundary & ((self.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument("--sub-step", type=int, default=2)
        parser.add_argument("--voxel-size", type=float, default=0.05)
        parser.add_argument("--particles-per-cell", type=int, default=3)
        parser.add_argument("--terrain-type", choices=("flat", "slope"), default="flat")
        parser.add_argument("--slope-angle", type=float, default=20.0)
        parser.add_argument("--particle-lod-pixels", type=float, default=2.0)
        parser.add_argument("--color-by-height", action="store_true")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
