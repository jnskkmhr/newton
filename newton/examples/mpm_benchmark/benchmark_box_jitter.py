# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure jitter/depth sensitivity with a dynamic box and a fixed sand container.

Run from the Newton repository with its simulation dependencies installed::

    uv run --extra sim python newton/examples/mpm_benchmark/benchmark_box_jitter.py \
        --sand-depth 0.2 --jitter 0.002 --output /tmp/box-j02-d20

Jitter is a total interval width [m], independently applied to X, Y, and Z.
The box has finite mass. Velocity is commanded; position is never prescribed.
No moving-patch code, particle recycling, or kinematic particles are used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import mujoco_warp as mjw
import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
from newton.solvers import SolverImplicitMPM, SolverMuJoCo


@wp.kernel
def command_box_velocity(speed: float, body_qd: wp.array[wp.spatial_vector]):
    body_qd[0] = wp.spatial_vector(0.0, 0.0, -speed, 0.0, 0.0, 0.0)


@wp.kernel
def sum_rigid_impulse(
    dt: float,
    geom_is_box: wp.array[int],
    geom: wp.array[wp.vec2i],
    active_count: wp.array[int],
    contact_force: wp.array[wp.spatial_vector],
    impulse: wp.array[wp.vec3],
):
    i = wp.tid()
    if i < active_count[0]:
        a, b = geom[i][0], geom[i][1]
        sign = float(geom_is_box[b] - geom_is_box[a])
        wp.atomic_add(impulse, 0, dt * sign * wp.spatial_top(contact_force[i]))


@wp.kernel
def record_box(
    index: int,
    time_s: float,
    speed: float,
    dt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    applied: wp.array[wp.spatial_vector],
    generated: wp.array[wp.spatial_vector],
    rigid_impulse: wp.array[wp.vec3],
    trace: wp.array2d[float],
):
    q = body_q[0]
    lowest_z = float(1.0e6)
    for corner in range(8):
        p = wp.vec3(
            0.085 * float(2 * (corner % 2) - 1),
            0.03 * float(2 * ((corner // 2) % 2) - 1),
            0.005 * float(2 * ((corner // 4) % 2) - 1),
        )
        lowest_z = wp.min(lowest_z, wp.transform_point(q, p)[2])
    trace[index, 0] = time_s
    trace[index, 1] = lowest_z
    trace[index, 2] = -speed
    for k in range(3):
        trace[index, 3 + k] = wp.transform_get_translation(q)[k]
        trace[index, 6 + k] = wp.spatial_top(body_qd[0])[k]
        trace[index, 9 + k] = rigid_impulse[0][k] / dt
    for k in range(6):
        trace[index, 12 + k] = applied[0][k]
        trace[index, 18 + k] = generated[0][k]


@wp.kernel
def record_sand_surface(
    index: int,
    cell_z: float,
    particle_ids: wp.array[int],
    particle_q: wp.array[wp.vec3],
    trace: wp.array[float],
):
    i = wp.tid()
    # Track the initial top layer away from the box; this is not a mesh surface query.
    height = particle_q[particle_ids[i]][2] + 0.5 * cell_z
    wp.atomic_add(trace, index, height / float(particle_ids.shape[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sand-depth", type=float, default=0.2)
    parser.add_argument("--jitter", type=float, default=0.002)
    parser.add_argument("--mass", type=float, default=5.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.sand_depth > 0 and args.mass > 0 and args.jitter >= 0 and args.seconds > args.settle >= 0):
        parser.error("Require positive depth/mass, nonnegative jitter, and seconds > settle >= 0")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    wp.init()
    wp.set_device(args.device)
    dt = 0.0025
    voxel_size = 0.02
    density = 1600.0
    half_width = 0.8
    builder = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(builder)
    box_size = np.array([0.17, 0.06, 0.01])
    inertia = (
        args.mass
        / 12
        * np.array(
            [
                box_size[1] ** 2 + box_size[2] ** 2,
                box_size[0] ** 2 + box_size[2] ** 2,
                box_size[0] ** 2 + box_size[1] ** 2,
            ]
        )
    )
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.065), wp.quat_identity()),
        mass=args.mass,
        inertia=wp.mat33(np.diag(inertia)),
        lock_inertia=True,
        label="box",
    )
    box_shape = builder.add_shape_box(
        body,
        hx=0.085,
        hy=0.03,
        hz=0.005,
        label="box_shape",
        cfg=builder.ShapeConfig(density=0.0, mu=0.5, margin=0.01875, gap=0.0, ke=160000.0, kd=1100.0),
    )
    floor_xform = wp.transform(wp.vec3(0.0, 0.0, -args.sand_depth - 0.05), wp.quat_identity())
    rigid_floor = builder.add_shape_box(
        -1,
        xform=floor_xform,
        hx=1.0,
        hy=1.0,
        hz=0.05,
        label="rigid_floor",
        cfg=builder.ShapeConfig(mu=0.5, margin=0.004, gap=0.002, has_particle_collision=False, ke=160000.0, kd=1100.0),
    )
    mpm_floor = builder.add_shape_box(
        -1,
        xform=floor_xform,
        hx=1.0,
        hy=1.0,
        hz=0.05,
        label="sand_floor",
        cfg=builder.ShapeConfig(mu=0.5, margin=0.0125, has_shape_collision=False),
    )
    sand_shapes = [mpm_floor]
    for axis in range(2):
        for sign in [-1, 1]:
            center = [0.0, 0.0, (0.05 - args.sand_depth) / 2]
            center[axis] = sign * (half_width + 0.05)
            half_size = [half_width + 0.1, half_width + 0.1, (args.sand_depth + 0.05) / 2]
            half_size[axis] = 0.05
            sand_shapes.append(
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                    hx=half_size[0],
                    hy=half_size[1],
                    hz=half_size[2],
                    label=f"sand_wall_{axis}_{sign}",
                    cfg=builder.ShapeConfig(mu=0.5, margin=0.0, has_shape_collision=False),
                )
            )
    extent = np.array([2 * half_width, 2 * half_width, args.sand_depth])
    resolution = np.ceil(1.25 * extent / voxel_size).astype(int)
    cell_size = extent / resolution
    cell_volume = float(np.prod(cell_size))
    lower = np.array([-half_width, -half_width, -args.sand_depth])
    template = (
        np.stack(np.meshgrid(*[np.arange(n) * s for n, s in zip(resolution, cell_size, strict=True)], indexing="xy"))
        .reshape(3, -1)
        .T
    )
    template += np.asarray(lower + 0.5 * cell_size, dtype=np.float32)
    positions = (template + (np.random.default_rng(args.seed).random(template.shape) - 0.5) * args.jitter).astype(
        np.float32
    )
    # Fixed physical volumes and masses: jitter only changes initial particle positions.
    material = {
        "young_modulus": 1.0e15,
        "poisson_ratio": 0.3,
        "friction": 0.48,
        "yield_pressure": 1.0e15,
        "tensile_yield_ratio": 0.0,
        "yield_stress": 0.0,
        "hardening": 0.0,
        "dilatancy": 0.0,
        "viscosity": 0.0,
    }
    count = len(positions)
    builder.add_particles(
        pos=positions.tolist(),
        vel=np.zeros_like(positions).tolist(),
        mass=[density * cell_volume] * count,
        radius=[0.5 * np.cbrt(cell_volume)] * count,
        custom_attributes={f"mpm:{k}": [v] * count for k, v in material.items()},
    )
    model = builder.finalize()
    config = SolverImplicitMPM.Config()
    config.voxel_size = voxel_size
    config.grid_type = "sparse"
    config.grid_padding = 0
    config.max_active_cell_count = 300000
    config.max_iterations = 25
    config.tolerance = 1.0e-4
    config.critical_fraction = 0.0
    config.strain_basis = "P0"
    config.collider_basis = "pic27"
    config.collider_velocity_mode = "forward"
    solver = SolverCoupledProxy(
        model,
        entries=[
            SolverCoupledProxy.Entry(
                name="box",
                solver=lambda v: SolverMuJoCo(v, njmax=256, nconmax=64, iterations=100, impratio=1.0),
                bodies=[body],
                joints=list(range(model.joint_count)),
                shapes=[box_shape, rigid_floor],
                substeps=2,
            ),
            SolverCoupledProxy.Entry(
                name="sand",
                solver=lambda v: SolverImplicitMPM(v, config=config),
                particles=list(range(count)),
                shapes=sand_shapes,
                in_place=True,
            ),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="box",
                    destination="sand",
                    bodies=[body],
                    mass_scale=1.0,
                    mode="lagged",
                    collision_pipeline=lambda _: None,
                )
            ],
            iterations=1,
        ),
    )
    state = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    collision = newton.CollisionPipeline(model, soft_contact_max=0)
    contacts = collision.contacts()
    rigid = solver.solver("box")
    sand = solver.solver("sand")
    feedback = solver._proxy_mappings[0].coupling_forces
    applied = wp.zeros_like(feedback)
    rigid_impulse = wp.zeros(1, dtype=wp.vec3)
    capacity = rigid.mjw_data.contact.geom.shape[0]
    contact_ids = wp.array(np.arange(capacity), dtype=int)
    contact_forces = wp.zeros(capacity, dtype=wp.spatial_vector)
    geom_is_box = wp.array((rigid.mj_model.geom_bodyid != 0).astype(np.int32), dtype=int)
    original_step = rigid.step

    def record_rigid_step(state_in, state_out, control, contacts, dt):
        original_step(state_in, state_out, control, contacts, dt)
        contact_forces.zero_()
        mjw.contact_force(rigid.mjw_model, rigid.mjw_data, contact_ids, True, contact_forces)
        wp.launch(
            sum_rigid_impulse,
            capacity,
            inputs=[dt, geom_is_box, rigid.mjw_data.contact.geom, rigid.mjw_data.nacon, contact_forces],
            outputs=[rigid_impulse],
        )

    rigid.step = record_rigid_step
    steps = round(args.seconds / dt)
    trace = wp.zeros((steps, 24), dtype=float)
    surface = wp.zeros(steps, dtype=float)
    radius_xy = np.max(np.abs(template[:, :2]), axis=1)
    top_ids = np.flatnonzero((template[:, 2] > -cell_size[2]) & (radius_xy >= 0.25) & (radius_xy <= 0.5))
    surface_ids = wp.array(top_ids, dtype=int)
    particle_volume = (2 * model.particle_radius.numpy()) ** 3
    np.testing.assert_allclose(model.particle_mass.numpy() / particle_volume, density, rtol=2e-6)
    assert int(model.body_flags.numpy()[body]) & int(newton.BodyFlags.KINEMATIC) == 0
    np.testing.assert_allclose(model.body_mass.numpy()[body], args.mass, rtol=1e-6)
    assert np.all(model.particle_mass.numpy() > 0)
    provenance = {
        "arguments": vars(args),
        "newton_path": newton.__file__,
        "dt_s": dt,
        "mpm_hz": 400,
        "mjwarp_hz": 800,
        "voxel_size_m": voxel_size,
        "particles_per_cell": 1.25,
        "resolution": resolution.tolist(),
        "cell_size_m": cell_size.tolist(),
        "particle_count": count,
        "particle_mass_kg": density * cell_volume,
        "density_kg_m3": density,
        "material": material,
        "mpm_config": vars(config),
        "surface_particle_count": len(top_ids),
        "mpm_collider_mass_kg": sand._mpm_model.collider_body_mass.numpy().tolist(),
        "versions": {p: importlib.metadata.version(p) for p in ["warp-lang", "mujoco-warp", "mujoco", "numpy"]},
        "newton_commit": subprocess.check_output(
            ["git", "-C", str(Path(newton.__file__).parents[1]), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    provenance["source_sha256"] = {
        str(p.relative_to(Path(newton.__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ["_src/solvers/implicit_mpm", "_src/solvers/coupled"]
        for p in (Path(newton.__file__).parent / folder).rglob("*.py")
    }
    (args.output / "config.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    for i in range(steps):
        speed = args.speed if i * dt >= args.settle else 0.0
        state.clear_forces()
        wp.launch(command_box_velocity, 1, inputs=[speed], outputs=[state.body_qd])
        newton.eval_ik(model, state, state.joint_q, state.joint_qd)
        applied.assign(feedback)
        rigid_impulse.zero_()
        collision.collide(state, contacts)
        solver.step(state, state, control, contacts, dt)
        wp.launch(
            record_box,
            1,
            inputs=[i, (i + 1) * dt, speed, dt, state.body_q, state.body_qd, applied, feedback, rigid_impulse],
            outputs=[trace],
        )
        wp.launch(
            record_sand_surface,
            len(top_ids),
            inputs=[i, float(cell_size[2]), surface_ids, state.particle_q],
            outputs=[surface],
        )
    data = trace.numpy().astype(np.float64)
    surface_height = surface.numpy()
    header = ["time_s", "box_sole_z_m", "command_vz_m_s"]
    header += [f"box_{x}_m" for x in "xyz"] + [f"box_v{x}_m_s" for x in "xyz"]
    header += [f"rigid_ground_f{x}_N" for x in "xyz"]
    for prefix in ["applied_mpm", "generated_mpm"]:
        header += [f"{prefix}_{component}" for component in ["fx_N", "fy_N", "fz_N", "tx_Nm", "ty_Nm", "tz_Nm"]]
    np.savetxt(
        args.output / "trace.csv",
        np.column_stack([data, surface_height]),
        delimiter=",",
        header=",".join([*header, "tracked_sand_surface_z_m"]),
        comments="",
    )
    assert np.isfinite(data).all() and np.isfinite(surface_height).all()
    assert np.isfinite(state.particle_q.numpy()).all() and np.isfinite(state.particle_qd.numpy()).all()
    np.testing.assert_allclose(data[1:, 12:18], data[:-1, 18:24], atol=1e-6, rtol=0)
    commanded = np.zeros((steps, 3))
    commanded[:, 2] = data[:, 2]
    momentum_rate = args.mass * (data[:, 6:9] - commanded) / dt
    forces = data[:, 9:12] + data[:, 12:15] + [0, 0, -args.mass * 9.81]
    error = np.linalg.norm(momentum_rate - forces, axis=1)
    scale = np.maximum(
        args.mass * 9.81, np.maximum(np.linalg.norm(momentum_rate, axis=1), np.linalg.norm(forces, axis=1))
    )
    steady = data[:, 0] >= 2 - 1e-6
    penetration = np.maximum(-data[:, 1], 0)
    result = {
        "jitter_width_mm": 1000 * args.jitter,
        "sand_depth_cm": 100 * args.sand_depth,
        "samples": steps,
        "duration_s": args.seconds,
        "wall_seconds": time.perf_counter() - start,
        "mean_penetration_cm": 100 * float(penetration[steady].mean()),
        "std_penetration_cm": 100 * float(penetration[steady].std()),
        "final_penetration_cm": 100 * float(penetration[-1]),
        "minimum_rigid_ground_clearance_cm": 100 * float(args.sand_depth + data[:, 1].min()),
        "mean_normal_force_N": float((data[:, 14] + data[:, 11])[steady].mean()),
        "std_normal_force_N": float((data[:, 14] + data[:, 11])[steady].std()),
        "peak_rigid_ground_normal_force_N": float(np.abs(data[:, 11]).max()),
        "sand_surface_at_0p5s_cm": 100 * float(surface_height[min(steps - 1, round(0.5 / dt) - 1)]),
        "sand_surface_at_end_cm": 100 * float(surface_height[-1]),
        "maximum_momentum_balance_error_N": float(error.max()),
        "maximum_relative_momentum_balance_error": float((error / scale).max()),
        "feedback_transfer_validated": True,
        "all_finite": True,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    assert np.all(error <= 1e-3 + 1e-3 * scale), "Box momentum balance failed"


if __name__ == "__main__":
    main()
