# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Isolated graph-enabled sparse MPM node-capacity sweep; raw results stay reproducible."""

import argparse
import ctypes
import importlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent

PROFILES = {
    "default": (-1, -1, -1),
    "leaf2": (2, -1, -1),
    "lower2": (2, 2, -1),
    "upper2": (2, -1, 2),
    "all2": (2, 2, 2),
    "all125": (1.25, 1.25, 1.25),
    "all4": (4, 4, 4),
}


class MemorySampler:
    """Sample total-device NVML memory without synchronizing CUDA (20 Hz)."""

    class Info(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
            raise RuntimeError("NVML device lookup failed")
        self.samples = []
        self.competing = set()
        self.last_process_check = 0.0
        self.stop = threading.Event()
        self.baseline = self.read()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def read(self):
        info = self.Info()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)) != 0:
            raise RuntimeError("NVML memory read failed")
        return int(info.used)

    def sample(self):
        while not self.stop.is_set():
            now = time.perf_counter()
            self.samples.append((now, self.read()))
            if now - self.last_process_check >= 0.5:
                processes = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
                )
                for process in processes.splitlines():
                    if process.strip() and int(process.split(",", 1)[0]) != os.getpid():
                        self.competing.add(process.strip())
                self.last_process_check = now
            self.stop.wait(0.05)

    def finish(self, measurement_start=None):
        self.stop.set()
        self.thread.join()
        self.samples.append((time.perf_counter(), self.read()))
        all_values = [v for _, v in self.samples]
        steady = [v for t, v in self.samples if measurement_start is not None and t >= measurement_start]
        return {
            "baseline_device_bytes": self.baseline,
            "peak_device_bytes": max(all_values),
            "peak_delta_bytes": max(all_values) - self.baseline,
            "steady_delta_bytes": float(sum(steady) / len(steady) - self.baseline) if steady else None,
            "sampling_interval_s": 0.05,
            "memory_samples": self.samples,
            "competing_compute_processes": sorted(self.competing),
        }


def worker(case, repeat, steps, warmup, output):
    # Keep CUDA imports out of the orchestration process to avoid GPU allocations.
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415
    import newton.examples  # noqa: PLC0415
    from newton.solvers import SolverImplicitMPM, SolverMuJoCo  # noqa: PLC0415

    memory = MemorySampler()
    result = {"case": case, "repeat": repeat, "status": "running", "measured_steps": steps, "warmup_steps": warmup}
    measurement_start = None
    start = time.perf_counter()
    try:
        wp.init()
        wp.set_device("cuda:0")
        width = float(case["width"])
        # 50% headroom above nominal bed cells, rounded to the next power of two.
        nominal = (width / 0.05) ** 2 * (0.5 / 0.05)
        capacity = 1 << math.ceil(math.log2(case.get("cell_ratio", 1.5) * nominal))
        suffix = {"original": "", "sparse": "_sparse", "patch": "_moving_patch"}[case["scene"]]
        module = importlib.import_module(
            "newton.examples.mpm_moving_patch_alpha.example_mujoco_mpm_coupled_solver" + suffix
        )

        class CommonMPM(SolverImplicitMPM):
            def __init__(self, model, config):
                config.voxel_size = 0.05
                config.grid_type = case["grid"]
                config.grid_padding = 50 if case["grid"] == "fixed" else 0
                config.max_active_cell_count = capacity
                initial = wp.Volume.allocate_by_voxels(voxel_points=model.particle_q, voxel_size=0.05)
                ijk = initial.get_voxels().numpy()
                counts = [len(np.unique(np.floor_divide(ijk, span), axis=0)) for span in (8, 128, 4096)]
                config.max_leaf_node_count = -1 if case["leaf"] < 0 else math.ceil(counts[0] * case["leaf"])
                config.max_lower_node_count = -1 if case["lower"] < 0 else math.ceil(counts[1] * case["lower"])
                config.max_upper_node_count = -1 if case["upper"] < 0 else math.ceil(counts[2] * case["upper"])
                result["initial_topology"] = dict(zip(("leaf", "lower", "upper"), counts, strict=True))
                result["initial_active_voxels"] = len(ijk)
                result["capacity"] = capacity
                result["node_limits"] = {
                    "leaf": config.max_leaf_node_count,
                    "lower": config.max_lower_node_count,
                    "upper": config.max_upper_node_count,
                }
                del initial, ijk
                config.strain_basis = "P0"
                config.max_iterations = 50
                config.critical_fraction = 0.0
                config.warmstart_mode = "none"
                config.collider_velocity_mode = "forward"
                super().__init__(model=model, config=config)

        class CommonMuJoCo(SolverMuJoCo):
            def __init__(self, model, **kwargs):
                super().__init__(model=model, use_mujoco_contacts=False, njmax=100)

        # Overrides are process-local; interactive source files are not edited.
        module.SolverImplicitMPM = CommonMPM
        module.SolverMuJoCo = CommonMuJoCo
        if hasattr(module.Example, "setup_terrain"):
            base = importlib.import_module("newton.examples.mpm_moving_patch_alpha.impl.base_example")
            base.SolverImplicitMPM = CommonMPM
            base.SolverMuJoCo = CommonMuJoCo

        class Scene(module.Example):
            def setup_terrain(self, args, *, tracked_body):
                super().setup_terrain(args, tracked_body=tracked_body)
                original = self.mpm_wrapper
                ratio = case.get("ratio", 2.0 / width)
                self.mpm_wrapper = type(original)(
                    config=replace(
                        original.config,
                        moving_terrain_size=(width * ratio, width * ratio),
                        boundary_terrain_size=width * (1.0 - ratio) / 2.0,
                    ),
                    tracked_body=tracked_body,
                    voxel_size=original.voxel_size,
                    particles_per_cell=original.particles_per_cell,
                    sand_depth=original.sand_depth,
                    density=2500.0,
                    jitter=original.jitter,
                    material_parameters={"mpm:friction": 0.75},
                    terrain_gen=original.terrain_gen,
                )

            def _emit_ground(self, builder):
                builder.add_ground_plane()
                return self.mpm_wrapper.terrain

            def _emit_rigid_bodies(self, builder):
                body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()), mass=75.0)
                builder.add_shape_box(body, hx=0.25, hy=0.35, hz=0.25)

            def _emit_particles(self, builder, voxel_size):
                if hasattr(self, "mpm_wrapper"):
                    self.mpm_wrapper.allocate_particle_storage(builder)
                    return
                self.bed_lo = np.array([-width / 2, -width / 2, 0.0])
                self.bed_hi = np.array([width / 2, width / 2, 0.5])
                self.visible_ratio = case.get("ratio", 2.0 / width)
                extent = self.bed_hi - self.bed_lo
                res = np.ceil(np.nextafter(3.0 * extent / voxel_size, -np.inf)).astype(int)
                spacing = extent / res
                builder.add_particle_grid(
                    pos=wp.vec3(self.bed_lo + np.array([spacing[0] / 2, spacing[1] / 2, 0])),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0.0),
                    dim_x=int(res[0]),
                    dim_y=int(res[1]),
                    dim_z=int(res[2]) + 1,
                    cell_x=float(spacing[0]),
                    cell_y=float(spacing[1]),
                    cell_z=float(spacing[2]),
                    mass=float(2500.0 * np.prod(spacing)),
                    radius_mean=float(np.max(spacing) / 2),
                    jitter=float(np.max(spacing)),
                    custom_attributes={"mpm:friction": 0.75},
                )

            def capture(self):
                if not case["motion"]:
                    self.object_controller.speed = 0.0
                super().capture()

        viewer = newton.viewer.ViewerNull()
        args = Scene.create_parser().parse_args([])
        args.rigid_substeps = 4
        args.proxy_iterations = 1
        args.mass_scale = 1.0
        scene = Scene(viewer, args)
        if not case["motion"]:
            scene.object_controller.speed = 0.0
        result["outer_cuda_graph"] = getattr(scene, "graph", None) is not None
        assert result["outer_cuda_graph"] == (case["scene"] != "sparse")
        assert scene.mpm_solver.max_active_cell_count == capacity
        assert scene.graph is not None, "Benchmark requires outer CUDA graph"
        wp.synchronize()
        result.update(
            particle_count=scene.model.particle_count,
            reserved_cell_capacity=capacity,
            grid_padding=50 if case["grid"] == "fixed" else 0,
            setup_seconds=time.perf_counter() - start,
            body_mass=scene.model.body_mass.numpy().tolist(),
            device=str(wp.get_device()),
            voxel_size=0.05,
            density=2500.0,
            frame_dt=scene.frame_dt,
        )
        for _ in range(warmup):
            scene.step()
        wp.synchronize()
        scene.mpm_solver.check_sparse_grid_rebuild_status()
        result["initial_measurement_pose"] = scene.state_0.body_q.numpy().tolist()
        active = []

        result["topology_samples"] = []

        def record_cells():
            volume = scene.mpm_solver._scratchpad.grid.cell_grid
            stats = volume.get_active_stats()
            result["topology_samples"].append(
                {
                    "active": stats.voxel_count,
                    "leaf": stats.leaf_node_count,
                    "lower": stats.lower_node_count,
                    "upper": stats.upper_node_count,
                }
            )
            result["resolved_limits"] = volume.get_rebuild_info()._asdict()
            partition = scene.mpm_solver._scratchpad.velocity_test.domain.geometry_partition
            if hasattr(partition, "_partition_cells"):
                indices = partition._partition_cells.numpy()
                active.append(int(np.count_nonzero(indices >= 0)))
            else:
                active.append(int(partition.cell_count()))

        record_cells()
        start_event = wp.Event(enable_timing=True)
        end_event = wp.Event(enable_timing=True)
        wall_ms, gpu_ms = [], []
        measurement_start = time.perf_counter()
        for _i in range(steps):
            wp.record_event(start_event)
            t0 = time.perf_counter()
            scene.step()
            wp.record_event(end_event)
            elapsed = wp.get_event_elapsed_time(start_event, end_event, synchronize=True)
            wall_ms.append((time.perf_counter() - t0) * 1000)
            gpu_ms.append(elapsed)

            scene.mpm_solver.check_sparse_grid_rebuild_status()
        wp.synchronize()
        record_cells()
        result.update(
            status="ok",
            wall_ms=wall_ms,
            gpu_ms=gpu_ms,
            mean_ms=float(np.mean(wall_ms)),
            std_ms=float(np.std(wall_ms, ddof=1)),
            gpu_mean_ms=float(np.mean(gpu_ms)),
            gpu_std_ms=float(np.std(gpu_ms, ddof=1)),
            active_cells_samples=active,
            final_pose=scene.state_0.body_q.numpy().tolist(),
            warp_pool_current_bytes=wp.get_mempool_used_mem_current(),
            warp_pool_peak_bytes=wp.get_mempool_used_mem_high(),
        )
        if hasattr(scene, "window_shifts"):
            result["window_shifts"] = int(scene.window_shifts.numpy()[0])
        result["finite_body_state"] = bool(np.isfinite(scene.state_0.body_q.numpy()).all())
        # Verification is outside timed sections; report rather than discard imperfect physics runs.
        try:
            scene.test_final()
            result["scene_validation"] = "passed"
        except Exception as exc:
            result["scene_validation"] = str(exc)
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        print(result["traceback"], flush=True)
    finally:
        result.update(memory.finish(measurement_start))
        if result["competing_compute_processes"] and result["status"] == "ok":
            result["status"] = "contended"
            result["error"] = (
                "Another GPU compute process ran during this case; exclude timing and device-memory results"
            )
        result["worker_seconds"] = time.perf_counter() - start
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result.get(k) for k in ("status", "mean_ms", "std_ms", "error")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--widths", nargs="+", type=int, default=[3])
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--tag", default="screen")
    args = parser.parse_args()
    raw = OUT / args.tag
    raw.mkdir(exist_ok=True)
    if args.worker:
        case = json.loads(args.worker)
        worker(case, args.repeat, args.steps, args.warmup, raw / f"{case['name']}_{args.repeat}.json")
        return
    (raw / "environment.json").write_text(
        json.dumps(
            {
                "command": sys.argv,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "gpu": subprocess.check_output(["nvidia-smi"], text=True),
                "warmup": args.warmup,
                "steps": args.steps,
                "repeats": args.repeats,
            },
            indent=2,
        )
    )
    for repeat in range(args.repeats):
        for width in args.widths:
            for profile in args.profiles:
                leaf, lower, upper = PROFILES[profile]
                case = {
                    "name": f"{width}m_{profile}",
                    "width": width,
                    "scene": "original",
                    "grid": "sparse",
                    "motion": True,
                    "leaf": leaf,
                    "lower": lower,
                    "upper": upper,
                }
                output = raw / f"{case['name']}_{repeat}.json"
                if output.exists():
                    continue
                waiting = False
                while True:
                    processes = subprocess.check_output(
                        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
                    ).strip()
                    if not processes:
                        break
                    if not waiting:
                        print("WAITING for other GPU workload:", processes, flush=True)
                        waiting = True
                    time.sleep(5)
                print("START", case["name"], repeat, flush=True)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    json.dumps(case),
                    "--repeat",
                    str(repeat),
                    "--warmup",
                    str(args.warmup),
                    "--steps",
                    str(args.steps),
                    "--tag",
                    args.tag,
                ]
                try:
                    with output.with_suffix(".log").open("w") as log:
                        run = subprocess.run(
                            command,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=600,
                            check=False,
                            env={**os.environ, "PYTHONPATH": str(ROOT)},
                        )
                    if not output.exists():
                        output.write_text(
                            json.dumps({"case": case, "status": "failed", "error": f"exit {run.returncode}"})
                        )
                except subprocess.TimeoutExpired:
                    output.write_text(json.dumps({"case": case, "status": "failed", "error": "600 s timeout"}))
                row = json.loads(output.read_text())
                print("DONE", case["name"], row.get("status"), row.get("mean_ms", row.get("error")), flush=True)


if __name__ == "__main__":
    main()
