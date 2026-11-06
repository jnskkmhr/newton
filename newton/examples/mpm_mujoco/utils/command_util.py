# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
from collections.abc import Sequence
import numpy as np
import torch

from .math_util import (
    yaw_quat,
    quat_from_euler_xyz,
    quat_unique,
    quat_mul,
)

"""
joint remapping
"""


def find_physx_mjwarp_mapping(mjwarp_joint_names, physx_joint_names):
    """
    Finds the mapping between PhysX and MJWarp joint names.
    Returns a tuple of two lists: (mjc_to_physx, physx_to_mjc).
    """
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    return mjc_to_physx, physx_to_mjc


"""
command generator
"""


class UniformVelocityGenerator:
    def __init__(
        self,
        num_envs: int,
        device: str,
        step_dt: float,
        vx_range: Tuple[float, float],
        vy_range: Tuple[float, float],
        wz_range: Tuple[float, float],
    ):
        self.num_envs = num_envs
        self.device = device
        self.step_dt = step_dt
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.wz_range = wz_range
        self.target_velocity = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)
        self.target_heading = torch.zeros(num_envs, 4, device=device, dtype=torch.float32)

        self.counter = 0
        self.initialization_step = int(0.1 / self.step_dt)

    def sample(self, current_quat: torch.Tensor):
        self.target_velocity[:, 0].uniform_(self.vx_range[0], self.vx_range[1])
        self.target_velocity[:, 1].uniform_(self.vy_range[0], self.vy_range[1])
        self.target_velocity[:, 2].uniform_(self.wz_range[0], self.wz_range[1])
        self.target_heading = yaw_quat(current_quat)  # keep the same heading

    def update_command(self):
        yaw_ang_vel_cmd = self.target_velocity[:, 2]
        delta_yaw = yaw_ang_vel_cmd * self.step_dt
        delta_quat = quat_from_euler_xyz(
            torch.zeros_like(delta_yaw),
            torch.zeros_like(delta_yaw),
            delta_yaw,
        )
        self.target_heading = quat_unique(quat_mul(delta_quat, self.target_heading))
        self.counter += 1

    @property
    def vel_command(self) -> torch.Tensor:
        if self.counter > self.initialization_step:
            return self.target_velocity
        else:
            v = self.target_velocity.clone()
            v *= 0.0
            return v

    @property
    def heading_command(self) -> torch.Tensor:
        return self.target_heading


class CircularVelocityGenerator:
    def __init__(
        self,
        num_envs: int,
        device: str,
        step_dt: float,
        radius: float,
        wz_range: Tuple[float, float],
    ):
        self.num_envs = num_envs
        self.device = device
        self.step_dt = step_dt
        self.radius = radius
        self.wz_range = wz_range
        self.target_velocity = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)
        self.target_heading = torch.zeros(num_envs, 4, device=device, dtype=torch.float32)

        self.counter = 0
        self.initialization_step = int(2.0 / self.step_dt)

    def sample(self, current_quat: torch.Tensor):
        self.counter = 0
        self.target_velocity[:, 2].uniform_(self.wz_range[0], self.wz_range[1])
        self.target_velocity[:, 0] = self.target_velocity[:, 2] * self.radius
        self.target_heading = yaw_quat(current_quat)  # keep the same heading

    def update_command(self):
        if self.counter > self.initialization_step:
            # self.target_velocity[:, 0] = 0
            yaw_ang_vel_cmd = self.target_velocity[:, 2]
            delta_yaw = yaw_ang_vel_cmd * self.step_dt
            delta_quat = quat_from_euler_xyz(
                torch.zeros_like(delta_yaw),
                torch.zeros_like(delta_yaw),
                delta_yaw,
            )
            self.target_heading = quat_unique(quat_mul(delta_quat, self.target_heading))
        else:
            self.counter += 1

    @property
    def vel_command(self) -> torch.Tensor:
        if self.counter > self.initialization_step:
            return self.target_velocity
        else:
            v = self.target_velocity.clone()
            v[:, 2] = 0.0
            return v

    @property
    def heading_command(self) -> torch.Tensor:
        return self.target_heading
