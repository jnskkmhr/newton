# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Isolated alpha terrain benchmarks. Run from the repository root with uv."""

# Delay simulation imports until the worker records pre-CUDA memory usage.
# ruff: noqa: PLC0415

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
METHODS = ("fixed", "sparse", "moving-patch")
SCENES = ("box", "rover", "g1")


def cases(background_margin=2.0):
    """Separate operating-space scaling from matched-particle-allocation overhead."""
    result = []
    for scene in SCENES:
        for size in range(3, 11):
            for method in METHODS:
                # Optional background margin beyond the listed full-bed size.
                terrain_size = float(size)
                result.append(
                    {
                        "name": f"scaling_{scene}_{method}_{size}",
                        "experiment": "scaling",
                        "scene": scene,
                        "method": method,
                        "size": size,
                        "terrain_size": terrain_size,
                        "background_size": size + background_margin,
                        "particle_width": 3.0 if method == "moving-patch" else terrain_size,
                        "boundary_width": 0.5,
                    }
                )
        for size in (1, 2, 4):
            for method in METHODS:
                result.append(
                    {
                        "name": f"overhead_{scene}_{method}_{size}",
                        "experiment": "overhead",
                        "scene": scene,
                        "method": method,
                        "size": size,
                        "terrain_size": float(size),
                        "background_size": size + 2.0,
                        "particle_width": float(size),
                        "boundary_width": 0.1,
                    }
                )
    for scene in SCENES:
        result.append(
            {
                "name": f"traversal_{scene}_moving-patch_6",
                "experiment": "traversal",
                "scene": scene,
                "method": "moving-patch",
                "size": 6,
                "terrain_size": 6.0,
                "background_size": 6.0 + background_margin,
                "particle_width": 3.0,
                "boundary_width": 0.5,
            }
        )
    return result


class MemorySampler:
    """Sample device memory via NVML; subtract the pre-CUDA process baseline."""

    class Info(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
            raise RuntimeError("NVML GPU lookup failed")
        self.baseline = self.read()
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def read(self):
        info = self.Info()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)) != 0:
            raise RuntimeError("NVML memory query failed")
        return int(info.used)

    def sample(self):
        while not self.stop.wait(0.05):
            self.samples.append((time.perf_counter(), self.read()))

    def finish(self, start):
        self.stop.set()
        self.thread.join()
        self.samples.append((time.perf_counter(), self.read()))
        steady = [max(0, used - self.baseline) for t, used in self.samples if start is not None and t >= start]
        return {
            "memory_baseline_bytes": self.baseline,
            "memory_samples": self.samples,
            "steady_memory_bytes": steady,
            "sampled_peak_bytes": max(x[1] for x in self.samples) - self.baseline,
        }


def make_scene(case):
    """Reuse robot/controller code; override only benchmark terrain and solver setup."""
    import numpy as np
    import warp as wp

    import newton
    import newton.examples
    from newton.examples.mpm_moving_patch_alpha.impl.controller.object_controller import ObjectController
    from newton.examples.mpm_moving_patch_alpha.impl.moving_terrain_mpm import (
        MovingTerrainMPMConfig,
        MovingTerrainMPMWrapper,
    )
    from newton.examples.mpm_moving_patch_alpha.impl.terrain_generator import TerrainGenerator
    from newton.solvers import SolverImplicitMPM

    suffix = "" if case["scene"] == "box" else "_" + case["scene"]
    original = importlib.import_module(
        f"newton.examples.mpm_moving_patch_alpha.example_mujoco_mpm_coupled_solver_moving_patch{suffix}"
    )
    moving = case["method"] == "moving-patch"

    class BenchmarkScene(original.Example):
        def _emit_rigid_bodies(self, builder):
            name = super()._emit_rigid_bodies(builder)
            index = builder.body_label.index(name)
            if case["scene"] == "box":
                # Start the 0.5 m-tall box 5 cm above the 0.5 m sand surface.
                builder.body_q[index] = wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity())
                for joint in range(builder.joint_count):
                    if builder.joint_child[joint] == index and builder.joint_type[joint] == newton.JointType.FREE:
                        builder.joint_q[builder.joint_q_start[joint] + 2] = 0.8
            self.start_xy = tuple(float(x) for x in builder.body_q[index][:2])
            return name

        def setup_terrain(self, args, *, tracked_body):
            width, boundary = case["particle_width"], case["boundary_width"]
            self.terrain_gen = TerrainGenerator(type="flat")
            background = case["background_size"]
            self.mpm_wrapper = MovingTerrainMPMWrapper(
                config=MovingTerrainMPMConfig(
                    moving_terrain_size=(width - 2 * boundary,) * 2,
                    boundary_terrain_size=boundary,
                    background_terrain_size=(background,) * 2,
                    background_terrain_origin=self.start_xy,
                    background_height_spacing=0.05,
                    shift_step=0.4,
                ),
                tracked_body=tracked_body,
                terrain_gen=self.terrain_gen,
                voxel_size=0.05,
                particles_per_cell=3,
                sand_depth=0.5,
                density=2500.0,
                jitter=0.05 / 3,
                material_parameters={
                    "mpm:friction": math.tan(math.radians(50)),
                    "mpm:young_modulus": 40.0e6,
                    "mpm:poisson_ratio": 0.3,
                },
            )

        def setup_solver(self, args, *, rigid_bodies, mpm_config=None):
            cfg = SolverImplicitMPM.Config()
            cfg.voxel_size = 0.05
            cfg.grid_type = "fixed" if case["method"] == "fixed" else "sparse"
            cfg.grid_padding = 4 if cfg.grid_type == "fixed" else 0
            cells = np.ceil(self.mpm_wrapper.outer_size / cfg.voxel_size + 8)
            cfg.max_active_cell_count = max(65536, int(np.prod(cells) * 22))
            cfg.transfer_scheme = "pic"
            cfg.strain_basis = "P0"
            cfg.max_iterations = 50
            cfg.critical_fraction = 0.0
            cfg.collider_velocity_mode = "forward"
            self.benchmark_config = cfg
            super().setup_solver(args, rigid_bodies=rigid_bodies, mpm_config=cfg)

        def _emit_ground(self, builder):
            self.mpm_wrapper.add_ground(builder)
            return self.mpm_wrapper.terrain

        def recycle_terrain(self):
            if moving:
                super().recycle_terrain()

    # No renderer or camera work is part of these headless physics measurements.
    args = BenchmarkScene.create_parser().parse_args(["--sub-step", "4", "--disable-cuda-graph"])
    if case["scene"] == "g1":
        args.auto_forward = True
        args.command = [0.2, 0.0, 0.0]
    elif case["scene"] == "rover":
        args.drive_speed = 0.2
        args.turn_rate = 0.0
    scene = BenchmarkScene(newton.viewer.ViewerNull(), args)
    # The box uses force tracking; the robots use their own actuators/policy.
    # Do not apply the base example's box force controller to either robot.
    scene.object_controller = None
    if case["scene"] == "box":
        scene.object_controller = ObjectController(scene.model, scene.state_0, [scene.mpm_wrapper.body_index])
        scene.object_controller.speed = 0.2
        scene.object_controller.settle_time = 0.1
        scene.object_controller.ramp_time = 0.2
    return scene


def worker(case, repeat, warmup, steps, output):
    """Time completed physics steps, retaining samples and state checks for audit."""
    if case["experiment"] == "traversal":
        steps = max(steps, 1200)
    memory = MemorySampler()
    result = {"case": case, "repeat": repeat, "status": "running", "warmup_steps": warmup, "measured_steps": steps}
    measurement_start = None
    started = time.perf_counter()
    try:
        import numpy as np
        import warp as wp

        wp.init()
        wp.set_device("cuda:0")
        scene = make_scene(case)
        wp.synchronize()
        result.update(
            particle_count=scene.model.particle_count,
            active_particles=int(np.count_nonzero(scene.model.particle_flags.numpy() & 1)),
            dynamic_particles=int(np.count_nonzero(scene.model.particle_mass.numpy() > 0)),
            cell_capacity=scene.mpm_solver.max_active_cell_count,
            rebuildable_sparse=bool(scene.mpm_solver._sparse_rebuildable),
            setup_seconds=time.perf_counter() - started,
            material_parameters=scene.mpm_wrapper.material_parameters,
            density=scene.mpm_wrapper.density,
            jitter=scene.mpm_wrapper.jitter,
            sub_step=scene.sub_step,
            rigid_substeps=4,
            warmstart_mode=scene.benchmark_config.warmstart_mode,
            physics_dt=scene.sim_dt,
            body_masses=scene.model.body_mass.numpy().tolist(),
            voxel_size=scene.mpm_wrapper.voxel_size,
            particles_per_cell=scene.mpm_wrapper.particles_per_cell,
            terrain_config={
                "moving_terrain_size": scene.mpm_wrapper.config.moving_terrain_size,
                "boundary_terrain_size": scene.mpm_wrapper.config.boundary_terrain_size,
                "background_terrain_size": scene.mpm_wrapper.config.background_terrain_size,
                "background_height_spacing": scene.mpm_wrapper.config.background_height_spacing,
                "shift_step": scene.mpm_wrapper.config.shift_step,
                "sand_depth": scene.mpm_wrapper.sand_depth,
            },
            solver_settings={
                k: getattr(scene.benchmark_config, k)
                for k in (
                    "grid_type",
                    "grid_padding",
                    "max_active_cell_count",
                    "max_leaf_node_count",
                    "max_lower_node_count",
                    "max_upper_node_count",
                    "transfer_scheme",
                    "strain_basis",
                    "velocity_basis",
                    "collider_basis",
                    "air_drag",
                    "max_iterations",
                    "warmstart_mode",
                )
            },
        )
        body = scene.mpm_wrapper.body_index
        initial_pose = scene.state_0.body_q[body : body + 1].numpy()[0].tolist()
        if case["scene"] == "box" and not np.isclose(initial_pose[2], 0.8):
            raise RuntimeError("Benchmark box initial height was not applied to its free joint")
        step_index = 0

        def advance():
            nonlocal step_index
            if case["scene"] == "rover" and step_index % scene.sub_step == 0:
                scene._apply_drive(0.2, 0.0)
            scene.simulate()
            scene.sim_time += scene.sim_dt
            step_index += 1

        for _ in range(warmup):
            advance()
        wp.synchronize()
        scene.mpm_solver.check_sparse_grid_rebuild_status()
        start_pose = scene.state_0.body_q[body : body + 1].numpy()[0].tolist()
        shifts_before = scene.mpm_wrapper.window_shifts.numpy().tolist() if case["method"] == "moving-patch" else 0
        timings = []
        measurement_start = time.perf_counter()
        for _ in range(steps):
            t0 = time.perf_counter()
            advance()
            wp.synchronize()
            timings.append((time.perf_counter() - t0) * 1000)
        end = time.perf_counter()
        scene.mpm_solver.check_sparse_grid_rebuild_status()
        state_finite = all(
            np.isfinite(a.numpy()).all()
            for a in (scene.state_0.body_q, scene.state_0.body_qd, scene.state_0.particle_q, scene.state_0.particle_qd)
        )
        if not state_finite:
            raise RuntimeError("Non-finite simulation state")
        partition = scene.mpm_solver._scratchpad.velocity_test.domain.geometry_partition
        active_cells = (
            int(np.count_nonzero(partition._partition_cells.numpy() >= 0))
            if hasattr(partition, "_partition_cells")
            else int(partition.cell_count())
        )
        if case["scene"] == "g1" and scene.policy_step != warmup + steps:
            raise RuntimeError("G1 policy physics-step counter mismatch")
        result.update(
            status="ok",
            wall_ms=timings,
            mean_ms=float(np.mean(timings)),
            std_ms=float(np.std(timings, ddof=1)),
            active_cells=active_cells,
            initial_pose=initial_pose,
            measurement_start_pose=start_pose,
            final_pose=scene.state_0.body_q[body : body + 1].numpy()[0].tolist(),
            simulated_seconds=scene.sim_time,
            measurement_seconds=end - measurement_start,
            patch_shifts_before=shifts_before,
            patch_shifts_after=scene.mpm_wrapper.window_shifts.numpy().tolist(),
            pool_current_bytes=wp.get_mempool_used_mem_current(),
            pool_peak_bytes=wp.get_mempool_used_mem_high(),
        )
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    result.update(memory.finish(measurement_start))
    result["elapsed_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: result.get(k) for k in ("status", "mean_ms", "particle_count", "elapsed_seconds")}), flush=True
    )


def sources():
    roots = [
        ROOT / "newton/examples/mpm_moving_patch_alpha",
        ROOT / "newton/_src/solvers/implicit_mpm",
        ROOT / "newton/_src/solvers/coupled",
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in roots
        for p in root.rglob("*.py")
        if "benchmark" not in p.parts
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--output", type=Path, default=OUT / "raw")
    parser.add_argument("--report-dir", type=Path, default=OUT)
    parser.add_argument("--background-margin", type=float, default=2.0)
    args = parser.parse_args()
    if args.background_margin < 0:
        parser.error("--background-margin must be nonnegative")
    selected = [c for c in cases(args.background_margin) if not args.only or c["name"] in args.only]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        case = next(c for c in cases(args.background_margin) if c["name"] == args.worker)
        worker(case, args.repeat, args.warmup, args.steps, args.output / f"{case['name']}_{args.repeat}.json")
        return
    snapshot = sources()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "source_hashes.json").write_text(json.dumps(snapshot, indent=2))
    (args.report_dir / "environment.json").write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "python": sys.version,
                "gpu": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True
                ),
                "warmup": args.warmup,
                "steps": args.steps,
                "repeats": args.repeats,
                "background_margin": args.background_margin,
            },
            indent=2,
        )
    )
    for repeat in range(args.repeats):
        # Alternate method order on the second pass to reduce ordering bias.
        ordered = selected if repeat % 2 == 0 else list(reversed(selected))
        for case in ordered:
            if sources() != snapshot:
                raise RuntimeError("Simulation source changed during benchmark; stopped to avoid mixing revisions")
            output = args.output / f"{case['name']}_{repeat}.json"
            if output.exists() and json.loads(output.read_text()).get("status") == "ok":
                continue
            # Wait for unrelated GPU workloads instead of contaminating timings.
            while subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip():
                print("WAIT: GPU is occupied by another process", flush=True)
                time.sleep(10)
            print(f"START {case['name']} repeat={repeat}", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                case["name"],
                "--repeat",
                str(repeat),
                "--warmup",
                str(args.warmup),
                "--steps",
                str(args.steps),
                "--output",
                str(args.output),
                "--background-margin",
                str(args.background_margin),
            ]
            try:
                foreign_pids = set()
                with (args.output / f"{case['name']}_{repeat}.log").open("w") as log:
                    child = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env={**os.environ, "PYTHONPATH": str(ROOT)},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    deadline = time.monotonic() + 600
                    while child.poll() is None:
                        if time.monotonic() > deadline:
                            child.kill()
                            child.wait()
                            raise subprocess.TimeoutExpired(command, 600)
                        running = subprocess.check_output(
                            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
                        )
                        foreign_pids.update(int(pid) for pid in running.split() if int(pid) != child.pid)
                        time.sleep(1)
                if foreign_pids and output.exists():
                    contaminated = json.loads(output.read_text())
                    contaminated.update(status="contended", foreign_gpu_pids=sorted(foreign_pids))
                    output.write_text(json.dumps(contaminated, indent=2))
                if not output.exists():
                    output.write_text(
                        json.dumps(
                            {"case": case, "repeat": repeat, "status": "failed", "error": f"exit {child.returncode}"}
                        )
                    )
            except subprocess.TimeoutExpired:
                output.write_text(
                    json.dumps({"case": case, "repeat": repeat, "status": "failed", "error": "600 s timeout"})
                )
            row = json.loads(output.read_text())
            log_path = args.output / f"{case['name']}_{repeat}.log"
            if log_path.exists():
                warnings = [
                    line
                    for line in log_path.read_text().splitlines()
                    if any(
                        term in line for term in ("overflow -", "exceeded MJWarp", "solver iterations limit reached")
                    )
                ]
                row["solver_warnings"] = warnings
                if warnings and row["status"] == "ok":
                    row["status"] = "solver_warning"
                output.write_text(json.dumps(row, indent=2))
            print(f"DONE {case['name']} repeat={repeat}: {row['status']} {row.get('mean_ms', '')}", flush=True)


if __name__ == "__main__":
    main()
