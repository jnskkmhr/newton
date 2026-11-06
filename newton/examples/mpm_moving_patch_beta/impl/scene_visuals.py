# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Terrain appearance and velocity arrows shared with the reference robot scenes."""

import numpy as np
import warp as wp


def make_earth_texture(size: tuple[float, float]) -> np.ndarray:
    """Generate the reference earth texture at 32 texels per meter of terrain."""
    texels_per_meter = 32
    width = round(size[0] * texels_per_meter)
    height = round(size[1] * texels_per_meter)
    rng = np.random.default_rng(42)
    variation = np.zeros((height, width))
    for scale, weight in ((4.0, 0.5), (1.0, 0.3), (0.25, 0.2)):
        nx = max(2, round(size[0] / scale) + 1)
        ny = max(2, round(size[1] / scale) + 1)
        noise = rng.random((ny, nx))
        rows = np.array([np.interp(np.linspace(0, nx - 1, width), np.arange(nx), row) for row in noise])
        smooth = np.array([np.interp(np.linspace(0, ny - 1, height), np.arange(ny), column) for column in rows.T]).T
        variation += weight * smooth
    variation = np.clip(variation + rng.normal(0.0, 0.045, variation.shape), 0.0, 1.0)
    dark = np.array([65.0, 52.0, 36.0])
    light = np.array([170.0, 148.0, 108.0])
    return (dark + variation[..., None] * (light - dark)).astype(np.uint8)


@wp.kernel
def _velocity_arrow_points(
    q: wp.array[float],
    qd: wp.array[float],
    command: wp.vec3,
    yaw_only: bool,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    rotation = wp.quat(q[3], q[4], q[5], q[6])
    if yaw_only:
        yaw = wp.atan2(2.0 * (q[6] * q[5] + q[3] * q[4]), 1.0 - 2.0 * (q[4] * q[4] + q[5] * q[5]))
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    velocity = wp.quat_rotate(rotation, wp.vec3(command[0], command[1], 0.0))
    start = wp.vec3(q[0], q[1], q[2] + 0.5)
    starts[0] = start
    starts[1] = start + wp.vec3(0.0, 0.0, 0.08)
    # Common scale: 0.4 m of arrow length per m/s; ignore vertical velocity.
    ends[0] = starts[0] + 0.4 * wp.vec3(velocity[0], velocity[1], 0.0)
    ends[1] = starts[1] + 0.4 * wp.vec3(qd[0], qd[1], 0.0)


class VelocityArrows:
    """Display commanded and measured XY velocity for a free-root robot."""

    def __init__(self, device):
        self.starts = wp.empty(2, dtype=wp.vec3, device=device)
        self.ends = wp.empty(2, dtype=wp.vec3, device=device)
        self.device = device

    def render(self, viewer, state, command):
        """Log yaw-rotated command and world-frame measured velocity [m/s]."""
        wp.launch(
            _velocity_arrow_points,
            dim=1,
            inputs=[state.joint_q, state.joint_qd, wp.vec3(*command), True],
            outputs=[self.starts, self.ends],
            device=self.device,
        )
        viewer.log_arrows("/velocity_command", self.starts[:1], self.ends[:1], (0.0, 0.8, 0.0))
        viewer.log_arrows("/velocity_actual", self.starts[1:], self.ends[1:], (0.0, 0.2, 1.0))
