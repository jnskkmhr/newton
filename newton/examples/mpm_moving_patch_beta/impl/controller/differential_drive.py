# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Differential-drive wheel velocity commands for the beta rover."""

import math

import warp as wp

__all__ = ["DifferentialDriveController"]


@wp.kernel
def _set_wheel_targets(
    dofs: wp.array[int],
    sides: wp.array[float],
    speed: float,
    yaw_rate: float,
    track_width: float,
    radius: float,
    targets: wp.array[float],
):
    i = wp.tid()
    # Positive wheel rotation about +Y moves the rover along +X.
    targets[dofs[i]] = (speed - sides[i] * yaw_rate * track_width * 0.5) / radius


class DifferentialDriveController:
    """Set wheel velocity targets without prescribing chassis motion."""

    def __init__(self, *, wheel_dofs, wheel_sides, wheel_radius: float, track_width: float, device):
        """Configure wheel geometry and joint indices.

        Args:
            wheel_dofs: Velocity degree-of-freedom indices for the driven wheels.
            wheel_sides: +1 for left wheels and -1 for right wheels, in matching order.
            wheel_radius: Nominal rolling radius [m].
            track_width: Distance between left and right wheel centers [m].
            device: Device holding the model and control arrays.
        """
        if not wheel_dofs or len(wheel_dofs) != len(wheel_sides):
            raise ValueError("Wheel indices and sides must be nonempty and have matching lengths")
        if any(side not in (-1.0, 1.0) for side in wheel_sides):
            raise ValueError("Wheel sides must be +1 (left) or -1 (right)")
        if not all(math.isfinite(v) and v > 0 for v in (wheel_radius, track_width)):
            raise ValueError("Wheel radius and track width must be finite and positive")
        self.wheel_dofs = wp.array(wheel_dofs, dtype=int, device=device)
        self.wheel_sides = wp.array(wheel_sides, dtype=float, device=device)
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        self.device = device

    def apply(self, control, *, speed: float, yaw_rate: float):
        """Set forward speed [m/s] and counterclockwise yaw rate [rad/s]."""
        if not math.isfinite(speed) or not math.isfinite(yaw_rate):
            raise ValueError("Drive commands must be finite")
        wp.launch(
            _set_wheel_targets,
            dim=self.wheel_dofs.shape[0],
            inputs=[
                self.wheel_dofs,
                self.wheel_sides,
                speed,
                yaw_rate,
                self.track_width,
                self.wheel_radius,
                control.joint_target_qd,
            ],
            device=self.device,
        )
