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

###########################################################################
# Example Robot control via keyboard
#
# Shows how to control robot pretrained in IsaacLab with RL.
# The policy is loaded from a file and the robot is controlled via keyboard.
#
# Press "p" to reset the robot.
# Press "i", "j", "k", "l", "u", "o" to move the robot.
# Run this example with:
# uv run newton/examples/mpm/example_mpm_robot.py 
# uv run newton/examples/mpm/example_mpm_robot.py --viewer usd --output-path **.usd
###########################################################################

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import warp as wp
import yaml

import newton

# Test: Disable CUDA-OpenGL interop to see if that fixes the issue
import newton._src.viewer.gl.opengl as opengl_module
import newton.examples
import newton.utils
from newton import State
from newton.solvers import SolverImplicitMPM

opengl_module.ENABLE_CUDA_INTEROP = False



@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: The quaternion in (x, y, z, w). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 3]  # w component is at index 3 for XYZW format
    q_vec = q[..., :3]  # xyz components are at indices 0, 1, 2
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def load_policy_and_setup_tensors(example: Any, policy_path: str, num_dofs: int, joint_pos_slice: slice):
    """Load policy and setup initial tensors for robot control.

    Args:
        example: Robot example instance
        policy_path: Path to the policy file
        num_dofs: Number of degrees of freedom
        joint_pos_slice: Slice for extracting joint positions from state
    """
    device = example.torch_device
    print("[INFO] Loading policy from:", policy_path)
    example.policy = torch.jit.load(policy_path, map_location=device)

    # Handle potential None state
    joint_q = example.state_0.joint_q if example.state_0.joint_q is not None else []
    example.joint_pos_initial = torch.tensor(joint_q[joint_pos_slice], device=device, dtype=torch.float32).unsqueeze(0)
    example.act = torch.zeros(1, num_dofs, device=device, dtype=torch.float32)
    example.rearranged_act = torch.zeros(1, num_dofs, device=device, dtype=torch.float32)


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


def _spawn_particles(builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, density):
    cell_size = (bounds_hi - bounds_lo) / res
    cell_volume = np.prod(cell_size)
    radius = np.max(cell_size) * 0.5
    mass = np.prod(cell_volume) * density

    builder.add_particle_grid(
        pos=wp.vec3(bounds_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=res[0] + 1,
        dim_y=res[1] + 1,
        dim_z=res[2] + 1,
        cell_x=cell_size[0],
        cell_y=cell_size[1],
        cell_z=cell_size[2],
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
    )


"""
circular buffer from IsaacLab
"""
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Sequence


class CircularBuffer:
    """Circular buffer for storing a history of batched tensor data.

    This class implements a circular buffer for storing a history of batched tensor data. The buffer is
    initialized with a maximum length and a batch size. The data is stored in a circular fashion, and the
    data can be retrieved in a LIFO (Last-In-First-Out) fashion. The buffer is designed to be used in
    multi-environment settings, where each environment has its own data.

    The shape of the appended data is expected to be (batch_size, ...), where the first dimension is the
    batch dimension. Correspondingly, the shape of the ring buffer is (max_len, batch_size, ...).
    """

    def __init__(self, max_len: int, batch_size: int, device: str):
        """Initialize the circular buffer.

        Args:
            max_len: The maximum length of the circular buffer. The minimum allowed value is 1.
            batch_size: The batch dimension of the data.
            device: The device used for processing.

        Raises:
            ValueError: If the buffer size is less than one.
        """
        if max_len < 1:
            raise ValueError(f"The buffer size should be greater than zero. However, it is set to {max_len}!")
        # set the parameters
        self._batch_size = batch_size
        self._device = device
        self._ALL_INDICES = torch.arange(batch_size, device=device)

        # max length tensor for comparisons
        self._max_len = torch.full((batch_size,), max_len, dtype=torch.int, device=device)
        # number of data pushes passed since the last call to :meth:`reset`
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        # the pointer to the current head of the circular buffer (-1 means not initialized)
        self._pointer: int = -1
        # the actual buffer for data storage
        # note: this is initialized on the first call to :meth:`append`
        self._buffer: torch.Tensor = None  # type: ignore

    """
    Properties.
    """

    @property
    def batch_size(self) -> int:
        """The batch size of the ring buffer."""
        return self._batch_size

    @property
    def device(self) -> str:
        """The device used for processing."""
        return self._device

    @property
    def max_length(self) -> int:
        """The maximum length of the ring buffer."""
        return int(self._max_len[0].item())

    @property
    def current_length(self) -> torch.Tensor:
        """The current length of the buffer. Shape is (batch_size,).

        Since the buffer is circular, the current length is the minimum of the number of pushes
        and the maximum length.
        """
        return torch.minimum(self._num_pushes, self._max_len)

    @property
    def buffer(self) -> torch.Tensor:
        """Complete circular buffer with most recent entry at the end and oldest entry at the beginning.
        Returns:
            Complete circular buffer with most recent entry at the end and oldest entry at the beginning of dimension 1. The shape is [batch_size, max_length, data.shape[1:]].
        """
        buf = self._buffer.clone()
        buf = torch.roll(buf, shifts=self.max_length - self._pointer - 1, dims=0)
        return torch.transpose(buf, dim0=0, dim1=1)

    """
    Operations.
    """

    def reset(self, batch_ids: Sequence[int] | None = None):
        """Reset the circular buffer at the specified batch indices.

        Args:
            batch_ids: Elements to reset in the batch dimension. Default is None, which resets all the batch indices.
        """
        # resolve all indices
        if batch_ids is None:
            batch_ids = slice(None)
        # reset the number of pushes for the specified batch indices
        self._num_pushes[batch_ids] = 0
        if self._buffer is not None:
            # set buffer at batch_id reset indices to 0.0 so that the buffer() getter returns the cleared circular buffer after reset.
            self._buffer[:, batch_ids, :] = 0.0

    def append(self, data: torch.Tensor):
        """Append the data to the circular buffer.

        Args:
            data: The data to append to the circular buffer. The first dimension should be the batch dimension.
                Shape is (batch_size, ...).

        Raises:
            ValueError: If the input data has a different batch size than the buffer.
        """
        # check the batch size
        if data.shape[0] != self.batch_size:
            raise ValueError(f"The input data has '{data.shape[0]}' batch size while expecting '{self.batch_size}'")

        # move the data to the device
        data = data.to(self._device)
        # at the first call, initialize the buffer size
        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty((self.max_length, *data.shape), dtype=data.dtype, device=self._device)
        # move the head to the next slot
        self._pointer = (self._pointer + 1) % self.max_length
        # add the new data to the last layer
        self._buffer[self._pointer] = data
        # Check for batches with zero pushes and initialize all values in batch to first append
        is_first_push = self._num_pushes == 0
        if torch.any(is_first_push):
            self._buffer[:, is_first_push] = data[is_first_push]
        # increment number of number of pushes for all batches
        self._num_pushes += 1

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        """Retrieve the data from the circular buffer in last-in-first-out (LIFO) fashion.

        If the requested index is larger than the number of pushes since the last call to :meth:`reset`,
        the oldest stored data is returned.

        Args:
            key: The index to retrieve from the circular buffer. The index should be less than the number of pushes
                since the last call to :meth:`reset`. Shape is (batch_size,).

        Returns:
            The data from the circular buffer. Shape is (batch_size, ...).

        Raises:
            ValueError: If the input key has a different batch size than the buffer.
            RuntimeError: If the buffer is empty.
        """
        # check the batch size
        if len(key) != self.batch_size:
            raise ValueError(f"The argument 'key' has length {key.shape[0]}, while expecting {self.batch_size}")
        # check if the buffer is empty
        if torch.any(self._num_pushes == 0) or self._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")

        # admissible lag
        valid_keys = torch.minimum(key, self._num_pushes - 1)
        # the index in the circular buffer (pointer points to the last+1 index)
        index_in_buffer = torch.remainder(self._pointer - valid_keys, self.max_length)
        # return output
        return self._buffer[index_in_buffer, self._ALL_INDICES]




class NewtonEnv:
    def __init__(
        self,
        viewer,
        config,
        mjc_to_physx: list[int],
        physx_to_mjc: list[int],
    ):
        # Setup simulation parameters first
        fps = 200
        self.frame_dt = 1.0e0 / fps
        self.decimation = 4
        self.cycle_time = 1 / fps * self.decimation

        # Group related attributes by prefix
        self.sim_time = 0.0
        self.sim_step = 0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.gait_period = config.get("gait_period", 0.4)
        
        # obs history length
        self.history_length = config.get("history_length", 10)

        # Save a reference to the viewer
        self.viewer = viewer

        # Store configuration
        self.use_mujoco = False
        self.config = config

        # Device setup
        self.device = wp.get_device()
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"

        # Build the model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        # add robot
        self.add_robot(builder)

        # add ground
        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        # -- set rigid body solver
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.use_mujoco,
            solver="newton",
            nconmax=30,
            njmax=100,
        )

        # Initialize state objects
        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # set up control policy
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Set model in viewer
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.vsync = True

        # Ensure FK evaluation (for non-MuJoCo solvers)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Store initial joint state for fast reset
        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        # Pre-compute tensors that don't change during simulation
        self.physx_to_mjc_indices = torch.tensor(physx_to_mjc, device=self.torch_device, dtype=torch.long)
        self.mjc_to_physx_indices = torch.tensor(mjc_to_physx, device=self.torch_device, dtype=torch.long)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self._reset_key_prev = False

        # create observation buffer
        self.create_obs_buffer()

        # Call capture at the end
        self.capture()

        # Load policy and setup tensors
        self.load_policy_and_setup_tensors()

    """
    initializations
    """

    def add_robot(self, builder: newton.ModelBuilder):
        # asset_path = "newton/examples/assets/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
        # asset_path = "newton/examples/assets/g1_29dof/g1.usd" # BUG: inertia eigen value error

        builder.add_usd(
            source=self.config["asset_path"], 
            xform=wp.transform(wp.vec3(*self.config["mjw_init_pos"])),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )

        # builder.add_urdf(
        #     source=self.config["asset_path"],
        #     xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)),
        #     floating=True,
        #     enable_self_collisions=False,
        #     collapse_fixed_joints=True,
        #     ignore_inertial_definitions=False,
        # )
        
        # builder.approximate_meshes("convex_hull")
        # builder.approximate_meshes("bounding_box")
        
        # -- set initial pose
        builder.joint_q[:3] = self.config["mjw_init_pos"]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = self.config["mjw_joint_pos"]

        # -- set joint gains
        for i in range(len(self.config["mjw_joint_stiffness"])):
            builder.joint_target_ke[i + 6] = self.config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[i + 6] = self.config["mjw_joint_damping"][i]
            builder.joint_armature[i + 6] = self.config["mjw_joint_armature"][i]


    def capture(self):
        """Put graph capture into it's own method."""
        self.graph = None
        self.use_cuda_graph = False
        if wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()):
            print("[INFO] Using CUDA graph")
            self.use_cuda_graph = True
            torch_tensor = torch.zeros(self.config["num_dofs"] + 6, device=self.torch_device, dtype=torch.float32)
            self.control.joint_target_pos = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.graph = capture.graph

    def create_obs_buffer(self):
        self.phase_time = torch.zeros((1), device=self.torch_device, dtype=torch.float32)
        self.clock = torch.zeros((1, 2), device=self.torch_device, dtype=torch.float32)
        self.base_ang_vel = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.projected_gravity = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.velocity_command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.joint_pos = torch.zeros((1, self.config["num_dofs"]), device=self.torch_device, dtype=torch.float32)
        self.joint_vel = torch.zeros((1, self.config["num_dofs"]), device=self.torch_device, dtype=torch.float32)
        # self.last_actions = torch.zeros((1, self.config["num_dofs"]), device=self.torch_device, dtype=torch.float32)
        self.last_actions = torch.zeros((1, self.config["num_policy_dofs"]), device=self.torch_device, dtype=torch.float32)

        self.group_obs_term_hisotry_buffer = {
            "clock": CircularBuffer(self.history_length, 1, self.torch_device),
            "base_ang_vel": CircularBuffer(self.history_length, 1, self.torch_device),
            "projected_gravity": CircularBuffer(self.history_length, 1, self.torch_device),
            "velocity_command": CircularBuffer(self.history_length, 1, self.torch_device),
            "joint_pos": CircularBuffer(self.history_length, 1, self.torch_device),
            "joint_vel": CircularBuffer(self.history_length, 1, self.torch_device),
            "last_actions": CircularBuffer(self.history_length, 1, self.torch_device),
        }
        obs_dim = self.history_length * (
            self.clock.shape[1] +
            self.base_ang_vel.shape[1] + 
            self.projected_gravity.shape[1] + 
            self.velocity_command.shape[1] +
            self.joint_pos.shape[1] + 
            self.joint_vel.shape[1] + 
            self.last_actions.shape[1])
        self.obs = torch.zeros(1, obs_dim, device=self.torch_device, dtype=torch.float32)


    """
    physics steps
    """

    def simulate_robot(self):
        """Simulate performs one frame's worth of updates."""
        state_0_dict = self.state_0.__dict__
        state_1_dict = self.state_1.__dict__
        state_temp_dict = self.state_temp.__dict__
        self.contacts = self.model.collide(self.state_0)
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states - handle CUDA graph case specially
            if i < self.sim_substeps - 1 or not self.use_cuda_graph:
                # We can just swap the state references
                self.state_0, self.state_1 = self.state_1, self.state_0
            elif self.use_cuda_graph:
                # Swap states by copying the state arrays for graph capture
                for key, value in state_0_dict.items():
                    if isinstance(value, wp.array):
                        if key not in state_temp_dict:
                            state_temp_dict[key] = wp.empty_like(value)
                        state_temp_dict[key].assign(value)
                        state_0_dict[key].assign(state_1_dict[key])
                        state_1_dict[key].assign(state_temp_dict[key])

    """
    mdp
    """

    def _get_obs_phase_time(self):
        """Calculate phase time for gait."""
        cur_time = time.perf_counter()
        phase_time = cur_time % self.gait_period / self.gait_period
        self.phase_time[:] = phase_time
        return self.phase_time
    
    def compute_obs(self):
        # Extract state information with proper handling
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        joint_qd = self.state_0.joint_qd if self.state_0.joint_qd is not None else []

        root_quat_w = torch.tensor(joint_q[3:7], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        # root_lin_vel_w = torch.tensor(joint_qd[:3], device=self.torch_device, dtype=torch.float32).unsqueeze(0) # maybe used 
        root_ang_vel_w = torch.tensor(joint_qd[3:6], device=self.torch_device, dtype=torch.float32).unsqueeze(0)

        joint_pos_current = torch.tensor(joint_q[7:], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        joint_vel_current = torch.tensor(joint_qd[6:], device=self.torch_device, dtype=torch.float32).unsqueeze(0)

        phase = self._get_obs_phase_time()
        self.clock[:, 0] = torch.sin(2.0 * np.pi * phase)
        self.clock[:, 1] = torch.cos(2.0 * np.pi * phase)

        self.base_ang_vel = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
        self.projected_gravity = quat_rotate_inverse(root_quat_w, self.gravity_vec)
        # # G1
        # self.joint_pos = torch.index_select(joint_pos_current - self.joint_pos_initial, 1, self.physx_to_mjc_indices)
        # T1
        self.joint_pos = torch.index_select(joint_pos_current, 1, self.physx_to_mjc_indices)

        self.joint_vel = torch.index_select(joint_vel_current, 1, self.physx_to_mjc_indices)
        self.velocity_command = self.command
        self.last_actions = self.act
        
        # add to history buffer
        self.group_obs_term_hisotry_buffer["clock"].append(self.clock)
        self.group_obs_term_hisotry_buffer["base_ang_vel"].append(self.base_ang_vel)
        self.group_obs_term_hisotry_buffer["projected_gravity"].append(self.projected_gravity)
        self.group_obs_term_hisotry_buffer["velocity_command"].append(self.velocity_command)
        self.group_obs_term_hisotry_buffer["joint_pos"].append(self.joint_pos)
        self.group_obs_term_hisotry_buffer["joint_vel"].append(self.joint_vel)
        self.group_obs_term_hisotry_buffer["last_actions"].append(self.last_actions)
        
        self.obs = torch.cat([
            self.group_obs_term_hisotry_buffer["clock"].buffer.reshape(1, -1), # (1, history_length * 2)
            self.group_obs_term_hisotry_buffer["base_ang_vel"].buffer.reshape(1, -1), # (1, history_length * 3)
            self.group_obs_term_hisotry_buffer["projected_gravity"].buffer.reshape(1, -1), # (1, history_length * 3)
            self.group_obs_term_hisotry_buffer["velocity_command"].buffer.reshape(1, -1), # (1, history_length * 3)
            self.group_obs_term_hisotry_buffer["joint_pos"].buffer.reshape(1, -1), # (1, history_length * num_dofs)
            self.group_obs_term_hisotry_buffer["joint_vel"].buffer.reshape(1, -1), # (1, history_length * num_dofs)
            self.group_obs_term_hisotry_buffer["last_actions"].buffer.reshape(1, -1), # (1, history_length * num_dofs)
        ], dim=1)

    
    def apply_control(self):
        self.compute_obs()
        with torch.no_grad():
            # pass
            self.act = self.policy(self.obs)
            # add upper body dof zeros if needed
            if self.act.shape[1] != self.config["num_dofs"]:
                act = torch.cat([
                    torch.zeros(1, self.config["num_dofs"] - self.act.shape[1], device=self.torch_device, dtype=torch.float32), 
                    self.act], dim=1)
                self.rearranged_act = torch.index_select(act, 1, self.mjc_to_physx_indices)
            else:
                self.rearranged_act = torch.index_select(self.act, 1, self.mjc_to_physx_indices)
            a = self.joint_pos_initial + self.config["action_scale"] * self.rearranged_act
            # add extra base dof zeros
            a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
            a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            wp.copy(self.control.joint_target_pos, a_wp)

    """
    step
    """
    def step(self):
        """
        process interactive keyboard
        """
        # Build command from viewer keyboard
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)
            # Reset when 'P' is pressed (edge-triggered)
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down
        
        """
        apply control from RL policy
        """
        self.apply_control()

        """
        step rigid body physics
        """
        for _ in range(self.decimation):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate_robot()
        

        self.sim_time += self.frame_dt

    """
    reset
    """

    def reset(self):
        print("[INFO] Resetting example")
        # Restore initial joint positions and velocities in-place.
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        # Recompute forward kinematics to refresh derived state.
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


    """
    utilities
    """
    def load_policy_and_setup_tensors(self):
        """Load policy and setup initial tensors for robot control.
        """
        print("[INFO] Loading policy from:", self.config["policy_path"])
        self.policy = torch.jit.load(self.config["policy_path"], map_location=self.torch_device)

        # Handle potential None state
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        self.joint_pos_initial = torch.tensor(joint_q[7:], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.act = torch.zeros(1, self.config["num_policy_dofs"], device=self.torch_device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(1, self.config["num_dofs"], device=self.torch_device, dtype=torch.float32)

    def test(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds
    # example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--config", type=str, 
                        default="newton/examples/assets/g1_29dof_rev_1_0/g1_29dof.yaml", # G1
                        # default="newton/examples/assets/t1_29dof/t1_29dof.yaml", # T1
                        help="Path to robot config YAML file.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)
    # viewer = newton.viewer.ViewerFile('humanoid_recording.mp4', auto_save=False)

    # Load robot configuration from YAML file in the downloaded assets
    yaml_file_path = args.config
    try:
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Robot config file not found: {yaml_file_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file: {e}")
        exit(1)

    print(f"[INFO] Loaded config with {config['num_dofs']} DOFs")

    mjc_to_physx = list(range(config["num_dofs"]))
    physx_to_mjc = list(range(config["num_dofs"]))

    # if args.physx:
    if "physx_joint_names" in config.keys():
        # when importing policy trained in IsaacLab
        mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping(config["mjw_joint_names"], config["physx_joint_names"])
        # print(f"[INFO] Using PhysX policy with mapping: mjc_to_physx={mjc_to_physx}, physx_to_mjc={physx_to_mjc}")

    env = NewtonEnv(viewer, config, mjc_to_physx, physx_to_mjc)

    # Main simulation loop
    total_sim_time = 20.0  # seconds
    while env.sim_time < total_sim_time:
        if not env.viewer.is_paused():
            with wp.ScopedTimer("step", active=False):
                env.step()

        with wp.ScopedTimer("render", active=False):
            env.render()
    print("[INFO] Simulation completed")
    env.viewer.close()