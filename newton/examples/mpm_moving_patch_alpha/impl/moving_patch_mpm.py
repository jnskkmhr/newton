# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Example-level moving MPM patch management, independent of the rigid solver."""

import numpy as np
import warp as wp

__all__ = ["HeightFieldTerrain", "MovingPatchMPMWrapper"]


@wp.struct
class _TerrainHeightField:
    heights: wp.array2d[float]
    origin: wp.vec3
    spacing: wp.vec2


@wp.func
def _sample_height(terrain: _TerrainHeightField, x: float, y: float):
    # Match the two triangles per cell in Mesh.create_heightfield (ij indexing).
    u = wp.clamp((x - terrain.origin[0]) / terrain.spacing[0], 0.0, float(terrain.heights.shape[0] - 1))
    v = wp.clamp((y - terrain.origin[1]) / terrain.spacing[1], 0.0, float(terrain.heights.shape[1] - 1))
    i = wp.min(int(wp.floor(u)), terrain.heights.shape[0] - 2)
    j = wp.min(int(wp.floor(v)), terrain.heights.shape[1] - 2)
    a = u - float(i)
    b = v - float(j)
    h00 = terrain.heights[i, j]
    h10 = terrain.heights[i + 1, j]
    h01 = terrain.heights[i, j + 1]
    h11 = terrain.heights[i + 1, j + 1]
    h = (1.0 - a - b) * h00 + a * h10 + b * h01
    if a + b > 1.0:
        h = (1.0 - b) * h10 + (1.0 - a) * h01 + (a + b - 1.0) * h11
    return terrain.origin[2] + h


class HeightFieldTerrain:
    """Regular height samples shared by spawning, recycling, and a static mesh.

    Heights use ij indexing (X, Y) and meters relative to origin.z. Sampling
    matches the triangulated mesh, rather than bilinear interpolation, so newly
    spawned particles lie at the same height as the collision surface. Outside
    the sampled domain, lookup clamps to its edge; callers must keep their full
    simulation window within the physical terrain. No mesh query or CPU transfer
    is required during recycling. One instance may be shared by multiple patches
    on the same device and in the same world coordinate system.
    """

    def __init__(self, heights, *, origin, spacing, device=None):
        samples = np.array(heights, dtype=np.float32, copy=True)
        if samples.ndim != 2 or min(samples.shape) < 2 or not np.isfinite(samples).all():
            raise ValueError("heights must be a finite array with at least two samples per axis")
        self.origin = np.asarray(origin, dtype=np.float32)
        self.spacing = np.asarray(spacing, dtype=np.float32)
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError("origin must contain three finite coordinates")
        if self.spacing.shape != (2,) or not np.isfinite(self.spacing).all() or np.any(self.spacing <= 0):
            raise ValueError("spacing must contain two finite positive values")
        self.samples = samples
        self.data = _TerrainHeightField()
        self.data.heights = wp.array(samples, dtype=float, device=device)
        self.data.origin = wp.vec3(*self.origin)
        self.data.spacing = wp.vec2(*self.spacing)

    @property
    def upper_xy(self):
        return self.origin[:2] + (np.array(self.samples.shape) - 1) * self.spacing

    def sample_height(self, xy):
        """Sample on the CPU for initial construction or diagnostics only."""
        xy = np.asarray(xy)
        uv = np.clip((xy - self.origin[:2]) / self.spacing, 0, np.array(self.samples.shape) - 1)
        ij = np.minimum(np.floor(uv).astype(int), np.array(self.samples.shape) - 2)
        a, b = (uv - ij).T
        i, j = ij.T
        h00, h10 = self.samples[i, j], self.samples[i + 1, j]
        h01, h11 = self.samples[i, j + 1], self.samples[i + 1, j + 1]
        return self.origin[2] + np.where(
            a + b <= 1.0,
            (1 - a - b) * h00 + a * h10 + b * h01,
            (1 - b) * h10 + (1 - a) * h01 + (a + b - 1) * h11,
        )


@wp.func
def _window_center(body_q: wp.array[wp.transform], shift_step: float):
    p = wp.transform_get_translation(body_q[0])
    return wp.vec2i(int(wp.floor(p[0] / shift_step + 0.5)), int(wp.floor(p[1] / shift_step + 0.5)))


@wp.kernel
def _recycle_terrain(
    body_q: wp.array[wp.transform],
    shift_step: float,
    bed_lo: wp.vec3,
    bed_hi: wp.vec3,
    reference: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    gradients: wp.array[wp.mat33],
    elastic: wp.array[wp.mat33],
    plastic: wp.array[float],
    stress: wp.array[wp.mat33],
    transforms: wp.array[wp.mat33],
    terrain: _TerrainHeightField,
    initial_plastic: wp.array[float],
):
    i = wp.tid()
    center = _window_center(body_q, shift_step)
    width = bed_hi - bed_lo
    lower = wp.vec2(float(center[0]) * shift_step + bed_lo[0], float(center[1]) * shift_step + bed_lo[1])
    p = positions[i]
    if p[0] < lower[0] or p[0] >= lower[0] + width[0] or p[1] < lower[1] or p[1] >= lower[1] + width[1]:
        # Wrap current coordinates, not original tile membership. Disturbed
        # particles inside the window retain their complete material history.
        x = p[0] - wp.floor((p[0] - lower[0]) / width[0]) * width[0]
        y = p[1] - wp.floor((p[1] - lower[1]) / width[1]) * width[1]
        layer_height = reference[i][2] - _sample_height(terrain, reference[i][0], reference[i][1])
        positions[i] = wp.vec3(x, y, _sample_height(terrain, x, y) + layer_height)
        velocities[i] = wp.vec3(0.0)
        gradients[i] = wp.mat33(0.0)
        elastic[i] = wp.identity(n=3, dtype=float)
        plastic[i] = initial_plastic[i]
        stress[i] = wp.mat33(0.0)
        transforms[i] = wp.identity(n=3, dtype=float)


@wp.kernel
def _particle_render_radii(
    body_q: wp.array[wp.transform],
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    bed_lo: wp.vec3,
    bed_hi: wp.vec3,
    visible_ratio: float,
    show_buffer: bool,
    render_radii: wp.array[float],
):
    i = wp.tid()
    p = positions[i] - wp.transform_get_translation(body_q[0])
    center = (bed_lo + bed_hi) * 0.5
    width = bed_hi - bed_lo
    distance = wp.max(wp.abs(p[0] - center[0]) / width[0], wp.abs(p[1] - center[1]) / width[1])
    # Visibility affects only rendering; physical radii stay unchanged.
    if show_buffer or distance <= visible_ratio * 0.5:
        render_radii[i] = radii[i]
    else:
        render_radii[i] = 0.0


@wp.kernel
def _record_window_center(
    body_q: wp.array[wp.transform],
    shift_step: float,
    center: wp.array[wp.vec2i],
    shifts: wp.array[int],
):
    updated = _window_center(body_q, shift_step)
    if updated[0] != center[0][0] or updated[1] != center[0][1]:
        shifts[0] += 1
    center[0] = updated


@wp.kernel
def _window_outline(
    center: wp.array[wp.vec2i],
    shift_step: float,
    bed_lo: wp.vec3,
    bed_hi: wp.vec3,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
    terrain: _TerrainHeightField,
):
    i = wp.tid()
    half_width = (bed_hi - bed_lo) * 0.5
    midpoint = (bed_hi + bed_lo) * 0.5
    corners = wp.vec4(-1.0, 1.0, 1.0, -1.0)
    y_corners = wp.vec4(-1.0, -1.0, 1.0, 1.0)
    c = wp.vec3(
        float(center[0][0]) * shift_step + midpoint[0], float(center[0][1]) * shift_step + midpoint[1], bed_hi[2] + 0.01
    )
    starts[i] = c + wp.vec3(corners[i] * half_width[0], y_corners[i] * half_width[1], 0.0)
    j = (i + 1) % 4
    ends[i] = c + wp.vec3(corners[j] * half_width[0], y_corners[j] * half_width[1], 0.0)
    start = starts[i]
    end = ends[i]
    starts[i] = wp.vec3(start[0], start[1], start[2] + _sample_height(terrain, start[0], start[1]))
    ends[i] = wp.vec3(end[0], end[1], end[2] + _sample_height(terrain, end[0], end[1]))


class MovingPatchMPMWrapper:
    """Manage a bounded, in-place MPM state without stepping its solver.

    ``state`` owns world particle positions/velocities and rigid poses.
    ``mpm_state`` owns material history (the MPM entry state for proxy coupling).
    This wrapper does not assume a particular rigid solver. Recreate it after
    reallocating either state. All per-step operations are CUDA-graph compatible.
    Bed Z bounds describe offsets above terrain, not absolute world heights.
    """

    def __init__(self, model, state, mpm_state, *, bed_lo, bed_hi, shift_step, terrain=None, body_index=0):
        self.device = model.device
        self.state = state
        self.mpm_state = mpm_state
        lo, hi = np.asarray(bed_lo), np.asarray(bed_hi)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
            raise ValueError("bed_hi must be finite and greater than bed_lo on all axes")
        if not np.isfinite(shift_step) or shift_step <= 0:
            raise ValueError("shift_step must be finite and positive")
        if not 0 <= body_index < state.body_q.shape[0]:
            raise ValueError("body_index must identify the tracked body")
        self.bed_lo, self.bed_hi = wp.vec3(*lo), wp.vec3(*hi)
        self.shift_step = float(shift_step)
        self.body_q = state.body_q[body_index : body_index + 1]
        self.terrain = terrain or HeightFieldTerrain(
            np.zeros((2, 2)), origin=(0, 0, 0), spacing=(1, 1), device=self.device
        )
        if self.terrain.data.heights.device != self.device:
            raise ValueError("Terrain and patch must be on the same device")
        self.reference_positions = wp.clone(model.particle_q)
        self.initial_plastic = wp.clone(mpm_state.mpm.particle_Jp)
        self.physical_radii = model.particle_radius
        self.render_radii = wp.empty_like(model.particle_radius)
        self.window_center = wp.zeros(1, dtype=wp.vec2i, device=self.device)
        self.window_shifts = wp.zeros(1, dtype=int, device=self.device)
        self.window_starts = wp.zeros(4, dtype=wp.vec3, device=self.device)
        self.window_ends = wp.zeros(4, dtype=wp.vec3, device=self.device)

    def reset_particle_background(self):
        """Wrap escaped particles and reset their material state before physics.

        Hidden particles still inside the outer simulation window are untouched.
        Recycled particles retain their original layer offset above local ground.
        """
        material = self.mpm_state.mpm
        wp.launch(
            _recycle_terrain,
            dim=self.state.particle_q.shape[0],
            inputs=[
                self.body_q,
                self.shift_step,
                self.bed_lo,
                self.bed_hi,
                self.reference_positions,
                self.state.particle_q,
                self.state.particle_qd,
                material.particle_qd_grad,
                material.particle_elastic_strain,
                material.particle_Jp,
                material.particle_stress,
                material.particle_transform,
                self.terrain.data,
                self.initial_plastic,
            ],
            device=self.device,
        )
        wp.launch(
            _record_window_center,
            dim=1,
            inputs=[self.body_q, self.shift_step, self.window_center, self.window_shifts],
            device=self.device,
        )

    def render_active_particle(self, *, positions=None, visible_ratio=1.0, show_buffer=False):
        """Return render-only radii; all particles remain physically simulated."""
        if not np.isfinite(visible_ratio) or not 0 < visible_ratio <= 1:
            raise ValueError("visible_ratio must be in (0, 1]")
        if positions is None:
            positions = self.state.particle_q
        wp.launch(
            _particle_render_radii,
            dim=positions.shape[0],
            inputs=[
                self.body_q,
                positions,
                self.physical_radii,
                self.bed_lo,
                self.bed_hi,
                visible_ratio,
                show_buffer,
                self.render_radii,
            ],
            device=self.device,
        )
        return self.render_radii

    def update_window_outline(self):
        """Update the optional outline, raised above the sampled ground."""
        wp.launch(
            _window_outline,
            dim=4,
            inputs=[
                self.window_center,
                self.shift_step,
                self.bed_lo,
                self.bed_hi,
                self.window_starts,
                self.window_ends,
                self.terrain.data,
            ],
            device=self.device,
        )
