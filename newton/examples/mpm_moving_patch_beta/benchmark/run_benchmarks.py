# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Isolated beta terrain benchmarks. Run from the repository root with uv."""

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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
METHODS = ("fixed", "sparse", "moving-patch")
SCENES = ("box", "rover", "g1")


def cases():
    """Separate operating-space scaling from matched-particle-allocation overhead."""
    result = []
    for scene in SCENES:
        for size in range(3, 11):
            for method in METHODS:
                # The height map includes travel margin beyond the listed full-bed size.
                terrain_size = float(size)
                result.append(
                    {
                        "name": f"scaling_{scene}_{method}_{size}",
                        "experiment": "scaling",
                        "scene": scene,
                        "method": method,
                        "size": size,
                        "terrain_size": terrain_size,
                        "background_size": size + 2.0,
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
    from newton.examples.mpm_moving_patch_beta.impl import base_example
    from newton.examples.mpm_moving_patch_beta.impl.controller.object_controller import ObjectController
    from newton.examples.mpm_moving_patch_beta.impl.moving_terrain_mpm import (
        MovingTerrainMPMConfig,
        MovingTerrainMPMWrapper,
    )
    from newton.examples.mpm_moving_patch_beta.impl.terrain_generator import TerrainGenerator
    from newton.solvers import SolverImplicitMPM

    original = importlib.import_module(
        f"newton.examples.mpm_moving_patch_beta.example_moving_terrain_mpm_{case['scene']}"
    )
    moving = case["method"] == "moving-patch"

    class StaticTerrain(MovingTerrainMPMWrapper):
        def bind(self, *, model, state, coupled_solver=None, mpm_entry="mpm"):
            # Static baselines need neither a terrain-history cache nor recycling scratch arrays.
            self.model, self.state, self.device = model, state, model.device
            self.body_index = self._body_index(model.body_label)
            self.layers = wp.array(self.initial_layers, dtype=int, device=model.device)
            self.render_radii = wp.empty(0, dtype=float, device=model.device)
            self.render_colors = wp.empty(0, dtype=wp.vec3, device=model.device)
            self.background_heights = wp.empty((0, 0), dtype=float, device=model.device)
            self.particle_cache = None
            self.bound = True

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
            wrapper = MovingTerrainMPMWrapper if moving else StaticTerrain
            background = max(case["background_size"], width + 0.1)
            self.mpm_wrapper = wrapper(
                config=MovingTerrainMPMConfig(
                    moving_terrain_size=(width - 2 * boundary,) * 2,
                    boundary_terrain_size=boundary,
                    background_terrain_size=(background,) * 2,
                    background_terrain_origin=self.start_xy,
                    background_height_spacing=0.05,
                    preserve_particle_state=False,
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

        def _create_controller(self, args):
            super()._create_controller(args)
            if case["scene"] == "box":
                self.object_controller = ObjectController(self.model, self.state_0, [self.mpm_wrapper.body_index])
                self.object_controller.speed = 0.2
                self.object_controller.settle_time = 0.1
                self.object_controller.ramp_time = 0.2

        def recycle_terrain(self):
            if moving:
                super().recycle_terrain()

    # No renderer or camera work is part of these headless physics measurements.
    base_example.ParticleRenderer = lambda *_args: SimpleNamespace(count=0)
    args = BenchmarkScene.create_parser().parse_args(["--sub-step", "4"])
    if case["scene"] == "g1":
        args.auto_forward = True
        args.command = [0.2, 0.0, 0.0]
    elif case["scene"] == "rover":
        args.drive_speed = 0.2
        args.turn_rate = 0.0
    scene = BenchmarkScene(newton.viewer.ViewerNull(), args)
    if moving:
        # The base wrapper owns these display arrays; exclude their unused allocation.
        scene.mpm_wrapper.render_radii = wp.empty(0, dtype=float, device=scene.model.device)
        scene.mpm_wrapper.render_colors = wp.empty(0, dtype=wp.vec3, device=scene.model.device)
        scene.particle_render_radii = scene.mpm_wrapper.render_radii
        scene.particle_render_colors = scene.mpm_wrapper.render_colors
    return scene


def worker(case, repeat, warmup, steps, output):
    """Time completed physics steps, retaining samples and state checks for audit."""
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
            physics_dt=scene.sim_dt,
            body_masses=scene.model.body_mass.numpy().tolist(),
            voxel_size=scene.mpm_wrapper.voxel_size,
            particles_per_cell=scene.mpm_wrapper.particles_per_cell,
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
                scene.drive_controller.apply(scene.control, speed=0.2, yaw_rate=0.0)
            scene.simulate()
            scene.sim_time += scene.sim_dt
            step_index += 1

        for _ in range(warmup):
            advance()
        wp.synchronize()
        scene.mpm_solver.check_sparse_grid_rebuild_status()
        start_pose = scene.state_0.body_q[body : body + 1].numpy()[0].tolist()
        recycled_start = int(scene.mpm_wrapper.recycled_count.numpy()[0]) if case["method"] == "moving-patch" else 0
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
            recycled_during_measurement=int(scene.mpm_wrapper.recycled_count.numpy()[0]) - recycled_start
            if case["method"] == "moving-patch"
            else 0,
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
        ROOT / "newton/examples/mpm_moving_patch_beta",
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
    args = parser.parse_args()
    selected = [c for c in cases() if not args.only or c["name"] in args.only]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        case = next(c for c in cases() if c["name"] == args.worker)
        worker(case, args.repeat, args.warmup, args.steps, args.output / f"{case['name']}_{args.repeat}.json")
        return
    snapshot = sources()
    (OUT / "source_hashes.json").write_text(json.dumps(snapshot, indent=2))
    (OUT / "environment.json").write_text(
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
            ]
            try:
                with (args.output / f"{case['name']}_{repeat}.log").open("w") as log:
                    child = subprocess.run(
                        command,
                        cwd=ROOT,
                        env={**os.environ, "PYTHONPATH": str(ROOT)},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                        check=False,
                    )
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
            print(f"DONE {case['name']} repeat={repeat}: {row['status']} {row.get('mean_ms', '')}", flush=True)


if __name__ == "__main__":
    main()
