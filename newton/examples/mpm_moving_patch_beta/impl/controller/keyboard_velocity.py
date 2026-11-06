# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Incremental keyboard velocity commands shared by policy-driven robots."""

import numpy as np


def update_keyboard_velocity(viewer, velocity, dt):
    """Integrate held keys over dt seconds; release retains the command, Backspace zeros it.

    Rates are 1 m/s² forward, 0.5 m/s² sideways, and 1 rad/s² yaw.
    Return the updated command and whether keyboard input was received.
    """
    command = np.array(velocity, dtype=np.float32, copy=True)
    if viewer.is_key_down("backspace"):
        return np.zeros(3, dtype=np.float32), True
    edited = False
    for positive, negative, axis, rate in (("i", "k", 0, 1.0), ("j", "l", 1, 0.5), ("u", "o", 2, 1.0)):
        up = viewer.is_key_down(positive)
        down = viewer.is_key_down(negative)
        command[axis] += (float(up) - float(down)) * rate * dt
        edited |= bool(up or down)
    return command, edited
