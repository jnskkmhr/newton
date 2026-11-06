# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Matched short beta/beta2 physics timings and particle-property refresh timings.

Run each mode in a fresh process, without other GPU workloads. This comparison
uses an identical static plane in both modes to isolate terrain maintenance from
changes to the ground collider. Long-distance terrain restoration is covered by coupled tests.
"""

import ctypes
import json
import threading
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.mpm_moving_patch_beta.impl.moving_terrain_mpm import (
    MovingTerrainMPMConfig,
    MovingTerrainMPMWrapper,
)
from newton.examples.mpm_moving_patch_beta2.example_moving_terrain_mpm_box import Example


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


@wp.kernel
def _drive_box(q: wp.array[wp.transform], qd: wp.array[wp.spatial_vector], f: wp.array[wp.spatial_vector], t: float):
    # A force controller, not a prescribed pose: the box still responds to sand.
    p = wp.transform_get_translation(q[0])
    target = 0.4 * wp.max(t - 0.25, 0.0)
    speed = float(0.0)
    if t > 0.25:
        speed = 0.4
    force = wp.clamp(1500.0 * (target - p[0]) + 400.0 * (speed - qd[0][0]), -1500.0, 1500.0)
    f[0] = wp.spatial_vector(wp.vec3(force, 0.0, 0.0), wp.vec3(0.0))


class BenchmarkExample(Example):
    def setup_terrain(self, args, *, tracked_body):
        super().setup_terrain(args, tracked_body=tracked_body)
        if args.mode == "beta":
            original = self.mpm_wrapper
            self.mpm_wrapper = MovingTerrainMPMWrapper(
                config=MovingTerrainMPMConfig(background_terrain_size=(40.0, 40.0)),
                tracked_body=tracked_body,
                voxel_size=original.voxel_size,
                particles_per_cell=original.particles_per_cell,
                sand_depth=original.sand_depth,
                density=original.density,
                jitter=original.jitter,
                material_parameters=original.material_parameters,
                terrain_gen=original.terrain_gen,
            )
        # Same collider, rigid-body count, physics settings, and particle sampling.
        self.mpm_wrapper.add_ground = lambda builder: builder.add_ground_plane()

    def simulate(self):
        self.mpm_wrapper.update_terrain_layers()
        self.state_0.clear_forces()
        wp.launch(
            _drive_box,
            dim=1,
            inputs=[self.state_0.body_q, self.state_0.body_qd, self.state_0.body_f, self.sim_time],
            device=self.model.device,
        )
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.sim_dt)


def main():
    parser = Example.create_parser()
    parser.set_defaults(sub_step=4)
    parser.add_argument("--mode", choices=("beta", "beta2"), required=True)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 2 or args.warmup < 0:
        parser.error("Require at least two measured steps and a nonnegative warmup")
    if args.device not in (None, "cuda", "cuda:0"):
        parser.error("The memory sampler measures GPU 0; use --device cuda:0")
    memory = MemorySampler()
    with wp.ScopedDevice(args.device or "cuda:0"):
        e = BenchmarkExample(newton.viewer.ViewerNull(), args)
        for _ in range(args.warmup):
            e.simulate()
            e.sim_time += e.sim_dt
        wp.synchronize_device(e.model.device)
        before = e.state_0.body_q.numpy()[0].tolist()
        times = []
        measurement_start = time.perf_counter()
        for _ in range(args.steps):
            start = time.perf_counter()
            e.simulate()
            wp.synchronize_device(e.model.device)
            times.append(1000 * (time.perf_counter() - start))
            e.sim_time += e.sim_dt
        measured_memory = memory.finish(measurement_start)
        e.test_final()
        result = {
            "mode": args.mode,
            "particle_count": e.model.particle_count,
            "voxel_size": e.mpm_wrapper.voxel_size,
            "particles_per_cell": args.particles_per_cell,
            "physics_dt": e.sim_dt,
            "warmup_steps": args.warmup,
            "measured_steps": args.steps,
            "physics_ms_mean": float(np.mean(times)),
            "physics_ms_std": float(np.std(times, ddof=1)),
            "physics_ms": times,
            "initial_measured_pose": before,
            "final_pose": e.state_0.body_q.numpy()[0].tolist(),
            "recycled_particles": int(e.mpm_wrapper.recycled_count.numpy()[0]),
        }
        result.update(measured_memory)
        # The optional targeted refresh API is absent from the reference solver.
        refresh_methods = [
            ("full_refresh", lambda: e.mpm_solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)),
        ]
        if hasattr(e.mpm_solver, "notify_particle_properties_changed"):
            refresh_methods.append(("targeted_refresh", e.mpm_solver.notify_particle_properties_changed))
        for name, refresh in refresh_methods:
            for _ in range(5):
                refresh()
            wp.synchronize_device(e.model.device)
            samples = []
            for _ in range(30):
                start = time.perf_counter()
                refresh()
                wp.synchronize_device(e.model.device)
                samples.append(1000 * (time.perf_counter() - start))
            result[name + "_ms_mean"] = float(np.mean(samples))
            result[name + "_ms_std"] = float(np.std(samples, ddof=1))
        w = e.mpm_wrapper
        if args.mode == "beta2":
            result["background_gpu_bytes"] = w.height_cache.gpu_bytes
            result["saved_gpu_tiles"] = int(w.height_cache.saved_tile_count)
            result["background_tile_capacity"] = w.height_cache.capacity
        else:
            result["background_gpu_bytes"] = sum(
                a.size * 4 for a in (w.ground_heights, w.background_heights, w.background_reference_depth)
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("physics_ms", "memory_samples", "steady_memory_bytes")},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
