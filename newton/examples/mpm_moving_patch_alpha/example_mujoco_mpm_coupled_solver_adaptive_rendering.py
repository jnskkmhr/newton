# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Rigid-MPM Coupled Solver with Adaptive Rendering
#
# Rigid boxes driven by MuJoCo or Kamino fall into a granular MPM bed.  The rigid bodies
# are exposed to MPM as proxy colliders in the same shared model, and MPM
# grid impulses are harvested back into rigid-body wrenches.
#
# Only a central XY fraction of the bed is submitted for particle rendering.
# All particles remain in the simulation; there is no fading or recycling.
#
# Command: python -m newton.examples mujoco_mpm_coupled_solver_adaptive_rendering
#
###########################################################################

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
from newton.examples.mpm_moving_patch_alpha.impl.controller.object_controller import ObjectController
from newton.solvers import SolverImplicitMPM, SolverMuJoCo


def _capture_frame_graph(model: newton.Model, simulate: Callable[[], None], *, enabled: bool = True):
    if not enabled:
        return None

    with wp.ScopedDevice(model.device):
        with wp.ScopedCapture() as capture:
            simulate()

    if capture.graph is None:
        raise RuntimeError(f"Graph capture failed on device {model.device}")
    return capture.graph


def _launch_frame_graph(model: newton.Model, graph) -> bool:
    if graph is None:
        return False

    with wp.ScopedDevice(model.device):
        wp.capture_launch(graph)
    return True


@wp.kernel
def _compact_visible_particles(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    bed_lo: wp.vec3,
    bed_hi: wp.vec3,
    visible_ratio: float,
    visible_count: wp.array[int],
    visible_positions: wp.array[wp.vec3],
    visible_radii: wp.array[float],
):
    i = wp.tid()
    p = positions[i]
    center = (bed_lo + bed_hi) * 0.5
    half_width = (bed_hi - bed_lo) * (visible_ratio * 0.5)
    if visible_ratio > 0.0 and wp.abs(p[0] - center[0]) <= half_width[0] and wp.abs(p[1] - center[1]) <= half_width[1]:
        j = wp.atomic_add(visible_count, 0, 1)
        visible_positions[j] = p
        visible_radii[j] = radii[i]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 100
        self.frame_dt = 1.0 / self.fps

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 0.5
        SolverImplicitMPM.register_custom_attributes(builder)

        rigid_body_start = builder.body_count
        self._emit_rigid_bodies(builder)
        rigid_body_end = builder.body_count
        builder.add_ground_plane()

        # self.bed_lo = np.array([-1.5, -1.5, 0.0])
        # self.bed_hi = np.array([1.5, 1.5, 0.5])
        self.bed_lo = np.array([-3, -3, 0.0])
        self.bed_hi = np.array([3, 3, 0.5])
        # Fraction of each horizontal side length, not area or particle count.
        self.visible_ratio = 1 / 3
        if not 0.0 <= self.visible_ratio <= 1.0:
            raise ValueError("visible_ratio must be between 0 and 1")
        voxel_size = 0.05
        self._emit_particles(builder, voxel_size)

        self.model = builder.finalize()
        self.visible_positions = wp.empty_like(self.model.particle_q)
        self.visible_radii = wp.empty_like(self.model.particle_radius)
        self.visible_count = wp.zeros(1, dtype=int, device=self.model.device)

        mpm_config = SolverImplicitMPM.Config()
        mpm_config.voxel_size = voxel_size
        mpm_config.grid_type = "sparse"
        mpm_config.grid_padding = 0
        mpm_config.max_active_cell_count = 1 << 20
        mpm_config.strain_basis = "P0"
        mpm_config.max_iterations = 50
        mpm_config.critical_fraction = 0.0

        rigid_body_indices = wp.array(
            list(range(rigid_body_start, rigid_body_end)),
            dtype=int,
            device=self.model.device,
        )
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mujoco",
                    solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=100),
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

        self.particle_render_colors = wp.full(
            self.model.particle_count,
            value=wp.vec3(0.7, 0.6, 0.4),
            dtype=wp.vec3,
            device=self.model.device,
        )

        self.graph = None
        self.capture()

    def capture(self):
        self.graph = _capture_frame_graph(self.model, self.simulate)

    def simulate(self):
        self.state_0.clear_forces()
        if self.object_controller is not None:
            self.object_controller.apply(self.state_0, self.frame_dt)
        newton.examples.apply_coupled_viewer_forces(self, self.state_0)
        self.rigid_collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.frame_dt)

    def step(self):
        if not _launch_frame_graph(self.model, self.graph):
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the sand",
            lambda q, qd: q[2] > 0.45,
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
        # Avoid submitting the viewer's default, unfiltered particle set as well.
        show_particles = self.viewer.show_particles
        self.viewer.show_particles = False
        try:
            newton.examples.log_coupled_view(self, self.contacts)
        finally:
            self.viewer.show_particles = show_particles

        count = 0
        if show_particles and self.visible_ratio > 0.0 and render_state.particle_q is not None:
            self.visible_count.zero_()
            if render_state.particle_q.shape[0] > 0:
                wp.launch(
                    _compact_visible_particles,
                    dim=render_state.particle_q.shape[0],
                    inputs=[
                        render_state.particle_q,
                        self.model.particle_radius,
                        wp.vec3(self.bed_lo),
                        wp.vec3(self.bed_hi),
                        self.visible_ratio,
                        self.visible_count,
                        self.visible_positions,
                        self.visible_radii,
                    ],
                    device=self.model.device,
                )
                # The viewer needs a host-side draw count; this synchronization is
                # part of rendering overhead and must be included in benchmarks.
                count = int(self.visible_count.numpy()[0])
        self.viewer.log_points(
            "/sand",
            points=self.visible_positions[:count] if count else None,
            radii=self.visible_radii[:count] if count else None,
            colors=self.particle_render_colors[:count] if count else None,
            hidden=not show_particles or count == 0,
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

        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)

    def _emit_rigid_bodies(self, builder: newton.ModelBuilder):
        drop_z = 2.0
        offsets_xy = [
            (0.00, 0.00),
            (0.10, 0.00),
            (-0.10, 0.00),
            (0.00, 0.10),
            (0.00, -0.10),
            (0.10, 0.10),
            (-0.10, 0.10),
            (0.10, -0.10),
            (-0.10, -0.10),
            (0.15, 0.00),
            (-0.15, 0.00),
            (0.00, 0.15),
        ]
        boxes = [
            (0.25, 0.35, 0.25),
            # (0.25, 0.25, 0.25),
            # (0.3, 0.2, 0.2),
            # (0.25, 0.35, 0.25),
            # (0.25, 0.25, 0.25),
            # (0.3, 0.2, 0.2),
        ]

        for box_index, (hx, hy, hz) in enumerate(boxes):
            ox, oy = offsets_xy[box_index % len(offsets_xy)]
            pz = drop_z + float(box_index) * 0.6
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(float(ox), float(oy), pz), q=wp.quat_identity()),
                mass=75.0,
            )
            builder.add_shape_box(body, hx=float(hx), hy=float(hy), hz=float(hz))

    def _emit_particles(self, builder: newton.ModelBuilder, voxel_size: float):
        particles_per_cell = 3.0
        density = 2500.0
        bed_lo = self.bed_lo
        bed_hi = self.bed_hi
        bed_res = np.array(np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size), dtype=int)

        cell_size = (bed_hi - bed_lo) / bed_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_volume) * density)

        builder.add_particle_grid(
            pos=wp.vec3(bed_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=bed_res[0] + 1,
            dim_y=bed_res[1] + 1,
            dim_z=bed_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            custom_attributes={"mpm:friction": 0.75},
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
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
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
