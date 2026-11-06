# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render local terrain steepness from live particle columns."""

import math

import warp as wp

import newton

from .terrain_height_cache import ground_height


@wp.kernel
def _measure_columns(
    positions: wp.array[wp.vec3],
    flags: wp.array[int],
    fractions: wp.array[float],
    ground: wp.vec2,
    lower: wp.vec2,
    center: wp.array[wp.vec2],
    outer: wp.vec2,
    h: float,
    height_sum: wp.array2d[float],
    fraction_sum: wp.array2d[float],
):
    i = wp.tid()
    if flags[i] & int(newton.ParticleFlags.ACTIVE):
        p = positions[i]
        start = wp.vec2(
            wp.floor((center[0][0] - 0.5 * outer[0] - lower[0]) / h),
            wp.floor((center[0][1] - 0.5 * outer[1] - lower[1]) / h),
        )
        x = int(wp.floor((p[0] - lower[0]) / h + 0.5) - start[0])
        y = int(wp.floor((p[1] - lower[1]) / h + 0.5) - start[1])
        if x >= 0 and y >= 0 and x < height_sum.shape[0] and y < height_sum.shape[1]:
            z = wp.max(p[2] - ground_height(p[0], ground), 0.0)
            wp.atomic_add(height_sum, x, y, z)
            wp.atomic_add(fraction_sum, x, y, fractions[i])


@wp.kernel
def _column_heights(
    height_sum: wp.array2d[float],
    fraction_sum: wp.array2d[float],
    ground: wp.vec2,
    lower: wp.vec2,
    center: wp.array[wp.vec2],
    outer: wp.vec2,
    h: float,
    heights: wp.array2d[float],
):
    x, y = wp.tid()
    if fraction_sum[x, y] > 1.0e-6:
        start = wp.vec2(
            wp.floor((center[0][0] - 0.5 * outer[0] - lower[0]) / h),
            wp.floor((center[0][1] - 0.5 * outer[1] - lower[1]) / h),
        )
        world_x = lower[0] + (start[0] + float(x)) * h
        heights[x, y] = ground_height(world_x, ground) + height_sum[x, y] / fraction_sum[x, y]


@wp.kernel
def _color_slopes(
    positions: wp.array[wp.vec3],
    lower: wp.vec2,
    center: wp.array[wp.vec2],
    outer: wp.vec2,
    h: float,
    heights: wp.array2d[float],
    fraction_sum: wp.array2d[float],
    slope_range: float,
    colors: wp.array[wp.vec3],
):
    i = wp.tid()
    p = positions[i]
    start = wp.vec2(
        wp.floor((center[0][0] - 0.5 * outer[0] - lower[0]) / h),
        wp.floor((center[0][1] - 0.5 * outer[1] - lower[1]) / h),
    )
    x = int(wp.floor((p[0] - lower[0]) / h + 0.5) - start[0])
    y = int(wp.floor((p[1] - lower[1]) / h + 0.5) - start[1])
    slope = float(0.0)
    if x >= 0 and y >= 0 and x < heights.shape[0] and y < heights.shape[1]:
        if fraction_sum[x, y] > 1.0e-6:
            xl = wp.max(x - 1, 0)
            xr = wp.min(x + 1, heights.shape[0] - 1)
            yl = wp.max(y - 1, 0)
            yr = wp.min(y + 1, heights.shape[1] - 1)
            # Use one-sided differences at missing columns, never zero-height padding.
            if fraction_sum[xl, y] <= 1.0e-6:
                xl = x
            if fraction_sum[xr, y] <= 1.0e-6:
                xr = x
            if fraction_sum[x, yl] <= 1.0e-6:
                yl = y
            if fraction_sum[x, yr] <= 1.0e-6:
                yr = y
            dx = (heights[xr, y] - heights[xl, y]) / (h * float(wp.max(xr - xl, 1)))
            dy = (heights[x, yr] - heights[x, yl]) / (h * float(wp.max(yr - yl, 1)))
            slope = wp.sqrt(dx * dx + dy * dy)
    colors[i] = wp.lerp(wp.vec3(0.1, 0.3, 0.9), wp.vec3(0.9, 0.15, 0.1), wp.clamp(slope / slope_range, 0.0, 1.0))


class TerrainSlopeColors:
    """Estimate surface gradients on the background sample spacing for rendering only."""

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.height_sum = wp.zeros_like(wrapper.deposited_volume)
        self.fraction_sum = wp.zeros_like(wrapper.deposited_volume)
        self.heights = wp.zeros_like(wrapper.deposited_volume)

    def update(self, positions, *, slope_range: float):
        """Write blue-to-red colors; slope_range is the dimensionless slope at full red."""
        if not math.isfinite(slope_range) or slope_range <= 0.0:
            raise ValueError("slope_range must be positive and finite")
        w = self.wrapper
        self.height_sum.zero_()
        self.fraction_sum.zero_()
        geometry = [wp.vec2(0.0), w.center, wp.vec2(*w.outer_size), w.config.background_height_spacing]
        wp.launch(
            _measure_columns,
            dim=w.model.particle_count,
            inputs=[
                positions,
                w.model.particle_flags,
                w.fraction,
                w.ground_parameters,
                *geometry,
                self.height_sum,
                self.fraction_sum,
            ],
            device=w.device,
        )
        wp.launch(
            _column_heights,
            dim=self.heights.shape,
            inputs=[self.height_sum, self.fraction_sum, w.ground_parameters, *geometry, self.heights],
            device=w.device,
        )
        wp.launch(
            _color_slopes,
            dim=w.model.particle_count,
            inputs=[positions, *geometry, self.heights, self.fraction_sum, slope_range, w.render_colors],
            device=w.device,
        )
