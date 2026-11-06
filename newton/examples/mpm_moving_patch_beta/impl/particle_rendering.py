# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compact visible particles and reduce distant particle detail for either viewer."""

import math

import warp as wp


@wp.kernel
def _select_particles(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    camera: wp.vec3,
    focal_length: float,
    target_pixels: float,
    selected: wp.array[int],
    display_radii: wp.array[float],
):
    i = wp.tid()
    selected[i] = 0
    radius = radii[i]
    display_radii[i] = radius
    if radius > 0.0:
        probability = float(1.0)
        if target_pixels > 0.0:
            distance = wp.max(wp.length(positions[i] - camera), 1.0e-6)
            diameter_pixels = 2.0 * radius * focal_length / distance
            ratio = wp.min(diameter_pixels / target_pixels, 1.0)
            probability = ratio * ratio
        # The same particle ID uses the same random value every frame. Camera
        # motion changes the cutoff, rather than randomly resampling all grains.
        rng = wp.rand_init(42, i)
        if probability > 0.0 and wp.randf(rng) < probability:
            selected[i] = 1
            # Preserve approximate projected area, not particle volume. These
            # radii are exclusively for rendering and never enter the solver.
            display_radii[i] = radius / wp.sqrt(probability)


@wp.kernel
def _compact_particles(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    colors: wp.array[wp.vec3],
    selected: wp.array[int],
    offsets: wp.array[int],
    out_positions: wp.array[wp.vec3],
    out_radii: wp.array[float],
    out_colors: wp.array[wp.vec3],
):
    i = wp.tid()
    if selected[i] != 0:
        j = offsets[i] - 1
        out_positions[j] = positions[i]
        out_radii[j] = radii[i]
        out_colors[j] = colors[i]


class ParticleRenderer:
    """Render-only compaction with an adjustable approximate particle diameter.

    ``target_pixels=0`` preserves every visible particle at its original radius.
    A positive value thins distant particles whose estimated screen diameter is
    smaller than the target. Selected particles grow to preserve their expected
    projected area. This is a visual approximation, not a reconstructed surface:
    silhouettes can change and particles can pop when crossing the detail cutoff.
    """

    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self.device = device
        self.count = 0
        self.selected = wp.empty(capacity, dtype=int, device=device)
        self.offsets = wp.empty(capacity, dtype=int, device=device)
        self.display_radii = wp.empty(capacity, dtype=float, device=device)
        self.positions = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.radii = wp.empty(capacity, dtype=float, device=device)
        self.colors = wp.empty(capacity, dtype=wp.vec3, device=device)

    def prepare(self, positions, radii, colors, *, camera=None, target_pixels: float = 0.0):
        """Return compact arrays; only the draw count is read back from the GPU.

        The camera uses Newton's ``pos``, ``fov`` [degrees], and ``height`` [pixels].
        Without a camera, only visibility compaction is performed. No simulation
        arrays are modified. Output buffers are reused by the next call.
        """
        if not math.isfinite(target_pixels) or target_pixels < 0.0:
            raise ValueError("target_pixels must be finite and nonnegative")
        n = len(positions)
        if n > self.capacity or len(radii) != n or len(colors) != n:
            raise ValueError("Particle arrays must have matching lengths within the rendering capacity")
        if n == 0:
            self.count = 0
            return self.positions[:0], self.radii[:0], self.colors[:0]
        camera_pos = wp.vec3(0.0)
        focal_length = 1.0
        if camera is None:
            target_pixels = 0.0
        else:
            camera_pos = wp.vec3(*camera.pos)
            focal_length = camera.height / (2.0 * math.tan(math.radians(camera.fov) / 2.0))
        wp.launch(
            _select_particles,
            dim=n,
            inputs=[positions, radii, camera_pos, focal_length, target_pixels, self.selected, self.display_radii],
            device=self.device,
        )
        wp.utils.array_scan(self.selected[:n], self.offsets[:n], inclusive=True)
        # Both viewer APIs require a host-side count. This small read synchronizes
        # the selection kernels; particle attributes stay on the GPU here.
        self.count = int(self.offsets[n - 1 : n].numpy()[0])
        wp.launch(
            _compact_particles,
            dim=n,
            inputs=[
                positions,
                self.display_radii,
                colors,
                self.selected,
                self.offsets,
                self.positions,
                self.radii,
                self.colors,
            ],
            device=self.device,
        )
        return self.positions[: self.count], self.radii[: self.count], self.colors[: self.count]
