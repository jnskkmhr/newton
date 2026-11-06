# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""GPU-only height memory indexed by world-space terrain tiles."""

import math

import warp as wp

__all__ = ["TerrainHeightCache"]

_EMPTY_KEY = wp.constant(wp.int64(9223372036854775807))


@wp.struct
class TerrainHeightData:
    keys: wp.array[wp.int64]
    heights: wp.array3d[float]
    reference_depth: wp.array3d[float]
    valid: wp.array3d[wp.uint8]
    stats: wp.array[int]
    spacing: float
    sand_depth: float
    ground: wp.vec2


@wp.func
def ground_height(x: float, ground: wp.vec2):
    return ground[0] + ground[1] * x


@wp.func
def _tile_index(sample: int, count: int):
    # Integer division truncates toward zero; negative coordinates need floor.
    return int(wp.floor(float(sample) / float(count)))


@wp.func
def _find_tile(data: TerrainHeightData, tx: int, ty: int, create: bool):
    key = wp.int64(tx) * wp.int64(4294967296) + wp.int64(wp.uint32(ty))
    if key == _EMPTY_KEY:
        wp.atomic_max(data.stats, 0, 2)
        return -1
    hashed = (wp.uint32(tx) * wp.uint32(73856093)) ^ (wp.uint32(ty) * wp.uint32(19349663))
    start = int(hashed % wp.uint32(data.keys.shape[0]))
    for offset in range(data.keys.shape[0]):
        slot = (start + offset) % data.keys.shape[0]
        stored = data.keys[slot]
        if stored == key:
            return slot
        if stored == _EMPTY_KEY:
            if not create:
                return -1
            stored = wp.atomic_cas(data.keys, slot, _EMPTY_KEY, key)
            if stored == _EMPTY_KEY:
                wp.atomic_add(data.stats, 1, 1)
                return slot
            if stored == key:
                return slot
    if create:
        wp.atomic_max(data.stats, 0, 1)
    return -1


@wp.func
def save_height(data: TerrainHeightData, ix: int, iy: int, height: float, reference_depth: float):
    n = data.heights.shape[1]
    tx = _tile_index(ix, n)
    ty = _tile_index(iy, n)
    slot = _find_tile(data, tx, ty, True)
    if slot >= 0:
        x = ix - tx * n
        y = iy - ty * n
        data.heights[slot, x, y] = height
        data.reference_depth[slot, x, y] = reference_depth
        data.valid[slot, x, y] = wp.uint8(1)


@wp.func
def read_height(data: TerrainHeightData, ix: int, iy: int):
    # Missing samples are untouched terrain; a saved zero height is not missing.
    result = wp.vec2(ground_height(float(ix) * data.spacing, data.ground) + data.sand_depth, data.sand_depth)
    n = data.heights.shape[1]
    tx = _tile_index(ix, n)
    ty = _tile_index(iy, n)
    slot = _find_tile(data, tx, ty, False)
    if slot >= 0:
        x = ix - tx * n
        y = iy - ty * n
        if data.valid[slot, x, y] != wp.uint8(0):
            result = wp.vec2(data.heights[slot, x, y], data.reference_depth[slot, x, y])
    return result


class TerrainHeightCache:
    """Save and restore heights directly on the GPU, without a CPU terrain cache.

    A fixed-capacity hash table maps signed world tile coordinates to GPU tiles.
    Untouched samples are generated analytically. No tile is evicted: exhausting
    capacity raises an error instead of discarding visited terrain. Heights and
    reference volumes never leave the device during terrain updates. Only two
    integer diagnostic counters are read by ``check_capacity()``.
    """

    def __init__(self, *, spacing: float, tile_size: float, capacity: int, sand_depth: float, ground: wp.vec2):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("background_tile_capacity must be a positive integer")
        if not all(math.isfinite(v) and v > 0 for v in (spacing, tile_size, sand_depth)):
            raise ValueError("Height spacing, tile size, and sand depth must be positive and finite")
        self.samples_per_tile = round(tile_size / spacing)
        if self.samples_per_tile < 1 or not math.isclose(self.samples_per_tile * spacing, tile_size):
            raise ValueError("Tile size must be a multiple of height spacing")
        self.shape = (capacity, self.samples_per_tile, self.samples_per_tile)
        self.capacity = capacity
        self.spacing, self.sand_depth, self.ground = spacing, sand_depth, ground
        self.data = None
        self.saved_tile_count = 0

    @property
    def gpu_bytes(self) -> int:
        """Bytes of allocated GPU height data, keys, validity, and counters."""
        return self.capacity * self.samples_per_tile**2 * 9 + self.capacity * 8 + 8

    def bind(self, device):
        """Allocate once; saving and reconstruction never replace these arrays."""
        if self.data is not None:
            raise RuntimeError("The terrain height cache is already bound")
        data = TerrainHeightData()
        data.keys = wp.full(self.capacity, _EMPTY_KEY, dtype=wp.int64, device=device)
        data.heights = wp.zeros(self.shape, dtype=float, device=device)
        data.reference_depth = wp.zeros(self.shape, dtype=float, device=device)
        data.valid = wp.zeros(self.shape, dtype=wp.uint8, device=device)
        data.stats = wp.zeros(2, dtype=int, device=device)
        data.spacing, data.sand_depth, data.ground = self.spacing, self.sand_depth, self.ground
        self.data = data
        self.heights = data.heights
        self.reference_depth = data.reference_depth

    def check_capacity(self):
        """Read diagnostic counters, never terrain samples; fail before recycling."""
        status, self.saved_tile_count = self.data.stats.numpy()
        if status == 2:
            raise RuntimeError("Terrain coordinates exceed the GPU tile-key range")
        if status:
            raise RuntimeError(
                "GPU terrain height cache is full. Increase background_tile_capacity; "
                "saved terrain is not evicted or transferred to the CPU."
            )
