# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared dynamic box tracking for the MPM benchmark scenes."""

import warp as wp


@wp.kernel
def _track_reference(
    bodies: wp.array[int],
    initial_pose: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    time: wp.array[float],
    speed: float,
    settle_time: float,
    ramp_time: float,
    position_gain: float,
    velocity_gain: float,
    orientation_gain: float,
    angular_velocity_gain: float,
    max_force: float,
    max_torque: float,
    body_f: wp.array[wp.spatial_vector],
):
    body = bodies[wp.tid()]
    t = wp.max(time[0] - settle_time, 0.0)
    u = wp.min(t / ramp_time, 1.0)
    velocity = speed * u * u * (3.0 - 2.0 * u)
    distance = speed * ramp_time * (u * u * u - 0.5 * u * u * u * u)
    if t > ramp_time:
        distance = speed * (t - 0.5 * ramp_time)
    reference = wp.transform_point(initial_pose[body], body_com[body]) + wp.vec3(distance, 0.0, 0.0)
    position = wp.transform_point(body_q[body], body_com[body])
    linear_velocity = wp.spatial_top(body_qd[body])
    force = position_gain * (reference - position) + velocity_gain * (wp.vec3(velocity, 0.0, 0.0) - linear_velocity)
    # Leave sinking, gravity, and vertical sand reaction entirely dynamic.
    force[2] = 0.0
    force *= wp.min(1.0, max_force / wp.max(wp.length(force), 1.0e-8))

    rotation_error = wp.transform_get_rotation(initial_pose[body]) * wp.quat_inverse(
        wp.transform_get_rotation(body_q[body])
    )
    sign = float(1.0)
    if rotation_error[3] < 0.0:
        sign = -1.0
    torque = 2.0 * orientation_gain * sign * wp.vec3(rotation_error[0], rotation_error[1], rotation_error[2])
    torque -= angular_velocity_gain * wp.spatial_bottom(body_qd[body])
    torque *= wp.min(1.0, max_torque / wp.max(wp.length(torque), 1.0e-8))
    wp.atomic_add(body_f, body, wp.spatial_vector(force, torque))


@wp.kernel
def _advance_time(time: wp.array[float], dt: float):
    time[0] += dt


class ObjectController:
    """Track a straight +X reference with bounded forces and orientation torques.

    The reference starts after settling, accelerates smoothly, then moves at a
    constant speed. It does not prescribe the actual pose or vertical motion.
    Call once after clearing forces and before each coupled simulation step.
    """

    def __init__(self, model, state, bodies):
        self.device = model.device
        self.bodies = wp.array(bodies, dtype=int, device=self.device)
        self.initial_pose = wp.clone(state.body_q)
        self.body_com = model.body_com
        self.time = wp.zeros(1, device=self.device)
        # Shared settings for all three benchmark scenes (SI units).
        self.speed = 0.2  # m/s along world +X
        self.settle_time = 1.0  # s before translation begins
        self.ramp_time = 1.0  # s to reach cruising speed
        self.position_gain = 20000.0  # N/m
        self.velocity_gain = 2000.0  # N s/m
        self.orientation_gain = 2000.0  # N m per unit rotation error
        self.angular_velocity_gain = 300.0  # N m s/rad
        self.max_force = 5000.0  # N, horizontal magnitude
        self.max_torque = 2000.0  # N m, magnitude

    def apply(self, state, dt):
        wp.launch(
            _track_reference,
            dim=self.bodies.shape[0],
            inputs=[
                self.bodies,
                self.initial_pose,
                self.body_com,
                state.body_q,
                state.body_qd,
                self.time,
                self.speed,
                self.settle_time,
                self.ramp_time,
                self.position_gain,
                self.velocity_gain,
                self.orientation_gain,
                self.angular_velocity_gain,
                self.max_force,
                self.max_torque,
                state.body_f,
            ],
            device=self.device,
        )
        wp.launch(_advance_time, dim=1, inputs=[self.time, dt], device=self.device)
