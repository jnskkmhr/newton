# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""G1 TorchScript observation history and action processing, local to beta examples."""

import torch

from .rl_util import CircularBuffer


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


class G1Policy:
    """Keep TorchScript inference separate from the coupled physics lifecycle."""

    def __init__(self, config, policy_path, device):
        self.config = config
        self.device = device
        self.num_dofs = config["num_dofs"]
        self.history_length = config.get("history_length", 10)
        self.policy = torch.jit.load(str(policy_path), map_location=device).eval()
        self.default_q = torch.tensor(config["mjw_joint_pos"], device=device, dtype=torch.float32).unsqueeze(0)
        self.action = torch.zeros((1, self.num_dofs), device=device)
        self.gravity = torch.tensor([[0.0, 0.0, -1.0]], device=device)
        self.history = {
            name: CircularBuffer(self.history_length, 1, device)
            for name in ("ang_vel", "projected_gravity", "command", "joint_pos", "joint_vel", "last_action")
        }
        self.obs = torch.zeros((1, self.history_length * (13 + 3 * self.num_dofs)), device=device)
        with torch.inference_mode():
            self._infer(self.obs)

    def _infer(self, obs):
        action = self.policy(obs)
        if not isinstance(action, torch.Tensor) or action.shape != (1, self.num_dofs):
            raise ValueError(f"G1 policy must return a tensor of shape (1, {self.num_dofs})")
        if not torch.isfinite(action).all():
            raise ValueError("G1 policy returned non-finite actions")
        return action

    @torch.inference_mode()
    def targets(self, root_quat, root_angular_velocity, joint_q, joint_qd, command):
        terms = (
            quat_rotate_inverse(root_quat, root_angular_velocity),
            quat_rotate_inverse(root_quat, self.gravity),
            command,
            joint_q - self.default_q,
            joint_qd,
            self.action,
        )
        for (name, history), value in zip(self.history.items(), terms, strict=True):
            history.append(value * self.config.get(f"{name}_scale", 1.0))
        # Match the original policy: term-major, oldest-to-newest history.
        self.obs = torch.cat([history.buffer.reshape(1, -1) for history in self.history.values()], dim=1)
        self.action = self._infer(self.obs)
        return self.default_q + self.config["action_scale"] * self.action
