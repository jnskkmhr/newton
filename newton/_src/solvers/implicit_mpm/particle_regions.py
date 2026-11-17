# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU coarse-region filtering for native implicit-MPM particle activity."""

import numpy as np
import warp as wp

from .particle_activity import _patch_coordinates, _patch_world, _PatchTable


@wp.kernel(enable_backward=False)
def _filter_region_requests(
    table: _PatchTable,
    body_q: wp.array[wp.transform],
    body_world: wp.array[int],
    body_indices: wp.array[int],
    half_extents: wp.array[wp.vec3],
    offsets: wp.array[wp.vec3],
    material_requests: wp.array[int],
    keep_material_motion: bool,
):
    slot = wp.tid()
    particle = table.slots[slot]
    if particle < 0 or table.requested[slot] == 0:
        return
    # Particle-driven requests include force, motion, changing stress, and quiet grace periods.
    # Preserve their swept neighbor requests too, before adding the ordinary support halo.
    if keep_material_motion and material_requests[slot] != 0:
        return
    cell = _patch_coordinates(table, particle)
    center = (wp.vec3(float(cell[0]), float(cell[1]), float(cell[2])) + wp.vec3(0.5)) * table.size
    world = _patch_world(table, particle)
    allowed = bool(False)
    for region in range(body_indices.shape[0]):
        body = body_indices[region]
        if not table.worlds or body_world[body] == -1 or body_world[body] == world:
            origin = wp.transform_get_translation(body_q[body]) + offsets[region]
            extent = half_extents[region] + wp.vec3(0.5 * table.size)
            delta = center - origin
            if wp.abs(delta[0]) <= extent[0] and wp.abs(delta[1]) <= extent[1] and wp.abs(delta[2]) <= extent[2]:
                allowed = True
                break
    if not allowed:
        table.requested[slot] = 0
        table.urgent[slot] = 0


class ParticleRegions:
    """Own fixed region descriptors; read live body poses on every captured activity update."""

    def __init__(self, model, activity, body_q, body_world, body_indices, half_extents, offsets, keep_material_motion):
        if body_q.dtype != wp.transform or body_world.dtype != wp.int32:
            raise ValueError("Region poses must be transform arrays and world IDs must be int32 arrays.")
        if body_q.ndim != 1 or body_world.shape != body_q.shape:
            raise ValueError("Region pose/world arrays must have the same one-dimensional shape.")
        if body_q.device != model.device or body_world.device != model.device:
            raise ValueError("Region body arrays must be on the MPM model device.")
        indices = np.asarray(body_indices)
        if indices.ndim != 1 or len(indices) == 0 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Provide a nonempty sequence of integer tracked body indices.")
        if np.any(indices < 0) or np.any(indices >= body_q.shape[0]):
            raise ValueError("Tracked body index is outside the provided pose array.")
        worlds = body_world.numpy()[indices]
        if np.any(worlds < -1) or np.any(worlds >= model.world_count):
            raise ValueError("Tracked bodies must belong to an MPM world, or global world -1.")
        count = len(indices)
        extents = np.broadcast_to(np.asarray(half_extents, dtype=np.float32), (count, 3)).copy()
        offsets = np.broadcast_to(np.asarray(offsets, dtype=np.float32), (count, 3)).copy()
        if not np.isfinite(extents).all() or np.any(extents <= 0) or not np.isfinite(offsets).all():
            raise ValueError("Region half extents must be positive and finite; offsets must be finite.")
        self.body_q = body_q
        self.body_world = body_world
        self.body_indices = wp.array(indices, dtype=wp.int32, device=model.device)
        self.half_extents = wp.array(extents, dtype=wp.vec3, device=model.device)
        self.offsets = wp.array(offsets, dtype=wp.vec3, device=model.device)
        self.keep_material_motion = keep_material_motion
        self.material_requests = wp.empty_like(activity.table.requested)

    def filter(self, table):
        wp.launch(
            _filter_region_requests,
            dim=table.slots.shape[0],
            inputs=[
                table,
                self.body_q,
                self.body_world,
                self.body_indices,
                self.half_extents,
                self.offsets,
                self.material_requests,
                self.keep_material_motion,
            ],
            device=self.body_q.device,
        )
