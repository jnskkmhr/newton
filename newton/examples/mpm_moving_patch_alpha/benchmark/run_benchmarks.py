# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run isolated, normalized MPM benchmark cases; save per-step observations."""

from __future__ import annotations

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


def cases():
    result = []
    for width in (3, 6, 10):
        for grid in ("fixed", "sparse"):
            result.append(
                {"name": f"{grid}_{width}m", "scene": "original", "grid": grid, "width": width, "motion": True}
            )
    result.append({"name": "patch_3m", "scene": "patch", "grid": "sparse", "width": 3, "motion": True})
    for ratio in (1.0, 2.0 / 3.0, 0.0):
        result.append(
            {
                "name": f"render_{ratio:.3f}",
                "scene": "adaptive",
                "grid": "sparse",
                "width": 3,
                "motion": True,
                "ratio": ratio,
            }
        )
    for motion in (False, True):
        for scene in ("original", "patch"):
            result.append(
                {
                    "name": f"overhead_{scene}_{'moving' if motion else 'stationary'}",
                    "scene": scene,
                    "grid": "sparse",
                    "width": 2,
                    "motion": motion,
                }
            )
    return result


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
            self.samples.append((time.perf_counter(), self.read()))
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
        capacity = 1 << math.ceil(math.log2(1.5 * nominal))
        suffix = {"original": "", "adaptive": "_adaptive_rendering", "patch": "_moving_patch"}[case["scene"]]
        module = importlib.import_module(
            "newton.examples.mpm_moving_patch_alpha.example_mujoco_mpm_coupled_solver" + suffix
        )

        class CommonMPM(SolverImplicitMPM):
            def __init__(self, model, config):
                config.voxel_size = 0.05
                config.grid_type = case["grid"]
                config.grid_padding = 50 if case["grid"] == "fixed" else 0
                config.max_active_cell_count = capacity
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

        rendering = case["scene"] == "adaptive"
        viewer = (
            newton.viewer.ViewerGL(width=960, height=720, headless=True, vsync=False)
            if rendering
            else newton.viewer.ViewerNull()
        )
        args = Scene.create_parser().parse_args([])
        args.rigid_substeps = 4
        args.proxy_iterations = 1
        args.mass_scale = 1.0
        scene = Scene(viewer, args)
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
        if rendering:
            viewer.set_camera(pos=wp.vec3(4.0, -5.0, 3.0), pitch=-25, yaw=125)
            gl = viewer.renderer.gl
            for _ in range(20):
                scene.render()
            gl.glFinish()
            wp.synchronize()
        result["initial_measurement_pose"] = scene.state_0.body_q.numpy().tolist()
        active = []

        def record_cells():
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
        for i in range(steps):
            wp.record_event(start_event)
            t0 = time.perf_counter()
            if rendering:
                scene.render()
                gl.glFinish()
            else:
                scene.step()
            wp.record_event(end_event)
            elapsed = wp.get_event_elapsed_time(start_event, end_event, synchronize=True)
            wall_ms.append((time.perf_counter() - t0) * 1000)
            gpu_ms.append(elapsed)
            if i in (steps // 2, steps - 1):
                record_cells()
                scene.mpm_solver.check_sparse_grid_rebuild_status()
        wp.synchronize()
        result.update(
            status="ok",
            wall_ms=wall_ms,
            gpu_ms=gpu_ms,
            mean_ms=float(np.mean(wall_ms)),
            p95_ms=float(np.percentile(wall_ms, 95)),
            gpu_mean_ms=float(np.mean(gpu_ms)),
            gpu_p95_ms=float(np.percentile(gpu_ms, 95)),
            active_cells_samples=active,
            final_pose=scene.state_0.body_q.numpy().tolist(),
            warp_pool_current_bytes=wp.get_mempool_used_mem_current(),
            warp_pool_peak_bytes=wp.get_mempool_used_mem_high(),
        )
        if rendering:
            result["drawn_particles"] = int(scene.visible_count.numpy()[0]) if case["ratio"] else 0
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
        result["worker_seconds"] = time.perf_counter() - start
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result.get(k) for k in ("status", "mean_ms", "p95_ms", "error")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    selected = [case for case in cases() if not args.only or case["name"] in args.only]
    if args.worker:
        case = next(case for case in cases() if case["name"] == args.worker)
        worker(case, args.repeat, args.steps, args.warmup, OUT / "raw" / f"{case['name']}_{args.repeat}.json")
        return
    (OUT / "raw").mkdir(exist_ok=True)
    metadata = {
        "command": sys.argv,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True
        ),
        "python": sys.version,
        "repetitions": args.repeats,
        "steps": args.steps,
        "warmup": args.warmup,
    }
    (OUT / "environment.json").write_text(json.dumps(metadata, indent=2))
    for case in selected:
        for repeat in range(args.repeats):
            output = OUT / "raw" / f"{case['name']}_{repeat}.json"
            if output.exists():
                previous = json.loads(output.read_text())
                if previous["status"] != "ok":
                    break
                continue
            print(f"START {case['name']} repeat {repeat}", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                case["name"],
                "--repeat",
                str(repeat),
                "--steps",
                str(args.steps),
                "--warmup",
                str(args.warmup),
            ]
            with (OUT / "raw" / f"{case['name']}_{repeat}.log").open("w") as log:
                try:
                    child = subprocess.run(
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
                            json.dumps(
                                {
                                    "case": case,
                                    "repeat": repeat,
                                    "status": "failed",
                                    "error": f"worker exit {child.returncode}",
                                }
                            )
                        )
                except subprocess.TimeoutExpired:
                    output.write_text(
                        json.dumps({"case": case, "repeat": repeat, "status": "failed", "error": "600 s timeout"})
                    )
            row = json.loads(output.read_text())
            print(
                f"DONE {case['name']} repeat {repeat}: {row['status']} {row.get('mean_ms', row.get('error'))}",
                flush=True,
            )
            if row["status"] != "ok":
                break


if __name__ == "__main__":
    main()
