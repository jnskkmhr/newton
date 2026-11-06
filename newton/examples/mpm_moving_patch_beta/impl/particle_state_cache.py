# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""GPU particle-state storage for visited terrain outside the moving MPM patch.

Python tracks horizontal tiles and free simulation slots. Particle attributes stay
on the GPU; only outgoing slot indices, tile keys, and counts are read back.
"""

import math

import numpy as np
import warp as wp

import newton


@wp.struct
class ParticleData:
    q: wp.array[wp.vec3]
    qd: wp.array[wp.vec3]
    grad: wp.array[wp.mat33]
    elastic: wp.array[wp.mat33]
    plastic: wp.array[float]
    stress: wp.array[wp.mat33]
    transform: wp.array[wp.mat33]
    mass: wp.array[float]
    radius: wp.array[float]
    flags: wp.array[int]
    fraction: wp.array[float]


_FIELDS = {
    "q": wp.vec3,
    "qd": wp.vec3,
    "grad": wp.mat33,
    "elastic": wp.mat33,
    "plastic": float,
    "stress": wp.mat33,
    "transform": wp.mat33,
    "mass": float,
    "radius": float,
    "flags": int,
    "fraction": float,
}


def _allocate(n, device):
    data = ParticleData()
    for name, dtype in _FIELDS.items():
        setattr(data, name, wp.empty(n, dtype=dtype, device=device))
    return data


@wp.func
def _copy(source: ParticleData, i: int, destination: ParticleData, j: int):
    destination.q[j] = source.q[i]
    destination.qd[j] = source.qd[i]
    destination.grad[j] = source.grad[i]
    destination.elastic[j] = source.elastic[i]
    destination.plastic[j] = source.plastic[i]
    destination.stress[j] = source.stress[i]
    destination.transform[j] = source.transform[i]
    destination.mass[j] = source.mass[i]
    destination.radius[j] = source.radius[i]
    destination.flags[j] = source.flags[i]
    destination.fraction[j] = source.fraction[i]


@wp.kernel
def _initialize_frozen(live: ParticleData, frozen: ParticleData, density: float):
    i = wp.tid()
    _copy(live, i, frozen, i)
    # Initial boundary mass is zero in the simulation, but not in the cache.
    if live.flags[i] & int(newton.ParticleFlags.ACTIVE):
        r = live.radius[i]
        frozen.mass[i] = density * 8.0 * r * r * r


@wp.func
def _dynamic(p: wp.vec3, center: wp.vec2, size: wp.vec2):
    return wp.abs(p[0] - center[0]) < size[0] * 0.5 and wp.abs(p[1] - center[1]) < size[1] * 0.5


@wp.func
def _fix_boundary(live: ParticleData, i: int):
    live.mass[i] = 0.0
    live.qd[i] = wp.vec3(0.0)
    live.grad[i] = wp.mat33(0.0)


@wp.kernel
def _update_layers(
    live: ParticleData,
    frozen: ParticleData,
    layers: wp.array[int],
    inv_mass: wp.array[float],
    center: wp.vec2,
    moving_size: wp.vec2,
    origin: wp.vec2,
    spacing: float,
    tile_begin: wp.vec2i,
    tile_end: wp.vec2i,
    outgoing_indices: wp.array[int],
    outgoing_tiles: wp.array[wp.vec2i],
    outgoing_count: wp.array[int],
    changed: wp.array[int],
):
    i = wp.tid()
    old_layer = layers[i]
    if old_layer == 0:
        return
    # A zero-mass MPM step can overwrite stress. Restore the saved boundary state
    # before deciding whether the particle remains fixed or becomes dynamic.
    if old_layer == 2:
        _copy(frozen, i, live, i)
    p = live.q[i]
    tile = wp.vec2i(int(wp.floor((p[0] - origin[0]) / spacing)), int(wp.floor((p[1] - origin[1]) / spacing)))
    outside = tile[0] < tile_begin[0] or tile[1] < tile_begin[1] or tile[0] >= tile_end[0] or tile[1] >= tile_end[1]
    layer = int(2)
    if _dynamic(p, center, moving_size):
        layer = 1
    if outside:
        layer = 0
    if old_layer == 1 and layer != 1:
        _copy(live, i, frozen, i)
    if layer != old_layer:
        wp.atomic_add(changed, 0, 1)
    if outside:
        j = wp.atomic_add(outgoing_count, 0, 1)
        outgoing_indices[j] = i
        outgoing_tiles[j] = tile
        live.flags[i] = 0
        live.q[i] = wp.vec3(center[0], center[1], p[2])
        live.mass[i] = 0.0
        live.qd[i] = wp.vec3(0.0)
        live.grad[i] = wp.mat33(0.0)
    elif layer == 2:
        _fix_boundary(live, i)
    inv_mass[i] = 0.0
    if live.mass[i] > 0.0:
        inv_mass[i] = 1.0 / live.mass[i]
    layers[i] = layer


@wp.kernel
def _gather(source: ParticleData, indices: wp.array[int], target: ParticleData):
    i = wp.tid()
    _copy(source, indices[i], target, i)


@wp.kernel
def _restore(
    source: ParticleData,
    slots: wp.array[int],
    live: ParticleData,
    frozen: ParticleData,
    layers: wp.array[int],
    inv_mass: wp.array[float],
    center: wp.vec2,
    moving_size: wp.vec2,
):
    i = wp.tid()
    j = slots[i]
    _copy(source, i, live, j)
    _copy(source, i, frozen, j)
    layers[j] = 2
    if _dynamic(live.q[j], center, moving_size):
        layers[j] = 1
    else:
        _fix_boundary(live, j)
    inv_mass[j] = 0.0
    if live.mass[j] > 0.0:
        inv_mass[j] = 1.0 / live.mass[j]


@wp.kernel
def _fresh_column(
    data: ParticleData,
    tile: wp.vec2i,
    origin: wp.vec2,
    spacing: float,
    per_column: int,
    nz: int,
    depth: float,
    jitter: float,
    mass: float,
    radius: float,
    initial_plastic: float,
    ground: wp.array2d[float],
    ground_lower: wp.vec2,
):
    i = wp.tid()
    iz = i % nz
    iy = (i // nz) % per_column
    ix = i // (nz * per_column)
    # Stable per-world-column jitter: first visits do not depend on travel order.
    seed = wp.rand_init(tile[0] * 73856093 ^ tile[1] * 19349663, i)
    x = origin[0] + (float(tile[0]) + (float(ix) + 0.5) / float(per_column)) * spacing
    y = origin[1] + (float(tile[1]) + (float(iy) + 0.5) / float(per_column)) * spacing
    z = (float(iz) + 0.5) * depth / float(nz)
    x += (wp.randf(seed) - 0.5) * jitter
    y += (wp.randf(seed) - 0.5) * jitter
    z += (wp.randf(seed) - 0.5) * jitter
    gx = wp.clamp((x - ground_lower[0]) / spacing, 0.0, float(ground.shape[0] - 1))
    gi = wp.min(int(wp.floor(gx)), ground.shape[0] - 2)
    ground_z = wp.lerp(ground[gi, 0], ground[gi + 1, 0], gx - float(gi))
    data.q[i] = wp.vec3(x, y, ground_z + z)
    data.qd[i] = wp.vec3(0.0)
    data.grad[i] = wp.mat33(0.0)
    data.elastic[i] = wp.identity(n=3, dtype=float)
    data.plastic[i] = initial_plastic
    data.stress[i] = wp.mat33(0.0)
    data.transform[i] = wp.identity(n=3, dtype=float)
    data.mass[i] = mass
    data.radius[i] = radius
    data.flags[i] = int(newton.ParticleFlags.ACTIVE)
    data.fraction[i] = z / depth


class ParticleStateCache:
    """Freeze full particle state in GPU tiles and restore it into free slots.

    Tiles contain variable particle counts and are keyed by world XY, not recycled
    simulation indices. Empty visited tiles remain empty. Cached data is removed
    when restored, so a particle has exactly one simulation/cache owner.
    """

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.device = wrapper.device
        self.spacing = wrapper.config.background_height_spacing
        self.origin = wrapper.initial_center - wrapper.outer_size / 2.0
        self.tile_shape = np.rint(wrapper.outer_size / self.spacing).astype(int)
        self.loaded = self._tiles(np.zeros(2, dtype=int))
        self.visited = set(self.loaded)
        self.pages = {}  # tile -> list of GPU ParticleData batches (including incoming transported material)
        self.free_slots = list(np.flatnonzero(wrapper.initial_layers == 0))
        material = wrapper.mpm_state.mpm
        self.live = ParticleData()
        for name, values in {
            "q": wrapper.state.particle_q,
            "qd": wrapper.state.particle_qd,
            "grad": material.particle_qd_grad,
            "elastic": material.particle_elastic_strain,
            "plastic": material.particle_Jp,
            "stress": material.particle_stress,
            "transform": material.particle_transform,
            "mass": wrapper.model.particle_mass,
            "radius": wrapper.model.particle_radius,
            "flags": wrapper.model.particle_flags,
            "fraction": wrapper.fraction,
        }.items():
            setattr(self.live, name, values)
        n = wrapper.model.particle_count
        self.frozen = _allocate(n, self.device)
        self.outgoing_indices = wp.empty(n, dtype=int, device=self.device)
        self.outgoing_tiles = wp.empty(n, dtype=wp.vec2i, device=self.device)
        self.outgoing_count = wp.zeros(1, dtype=int, device=self.device)
        self.initial_plastic = float(wrapper.initial_plastic[:1].numpy()[0])
        wp.launch(
            _initialize_frozen,
            dim=n,
            inputs=[self.live, self.frozen, wrapper.density],
            device=self.device,
        )

    def _tiles(self, begin):
        return {
            (x, y)
            for x in range(int(begin[0]), int(begin[0] + self.tile_shape[0]))
            for y in range(int(begin[1]), int(begin[1] + self.tile_shape[1]))
        }

    def _fresh(self, tile):
        w = self.wrapper
        per_column = round(self.spacing / w.cell_xy)
        nz = max(1, math.ceil(w.sand_depth / (w.voxel_size / w.particles_per_cell)))
        n = per_column * per_column * nz
        data = _allocate(n, self.device)
        wp.launch(
            _fresh_column,
            dim=n,
            inputs=[
                data,
                wp.vec2i(*tile),
                wp.vec2(*self.origin),
                self.spacing,
                per_column,
                nz,
                w.sand_depth,
                w.jitter,
                w.density * w.base_volume,
                0.5 * w.base_volume ** (1 / 3),
                self.initial_plastic,
                w.ground_heights,
                wp.vec2(*w.lower),
            ],
            device=self.device,
        )
        return data

    def update(self, center):
        w = self.wrapper
        begin = np.rint((center - w.outer_size / 2.0 - self.origin) / self.spacing).astype(int)
        loaded = self._tiles(begin)
        self.outgoing_count.zero_()
        wp.launch(
            _update_layers,
            dim=w.model.particle_count,
            inputs=[
                self.live,
                self.frozen,
                w.layers,
                w.model.particle_inv_mass,
                wp.vec2(*center),
                wp.vec2(*w.moving_size),
                wp.vec2(*self.origin),
                self.spacing,
                wp.vec2i(*begin),
                wp.vec2i(*(begin + self.tile_shape)),
                self.outgoing_indices,
                self.outgoing_tiles,
                self.outgoing_count,
                w.changed,
            ],
            device=self.device,
        )
        outgoing_count = int(self.outgoing_count.numpy()[0])
        if outgoing_count:
            indices = self.outgoing_indices[:outgoing_count].numpy()
            keys = self.outgoing_tiles[:outgoing_count].numpy()
            unique_keys, groups = np.unique(keys, axis=0, return_inverse=True)
            for group, key in enumerate(unique_keys):
                ids = wp.array(indices[groups == group], dtype=int, device=self.device)
                data = _allocate(len(ids), self.device)
                wp.launch(_gather, dim=len(ids), inputs=[self.frozen, ids, data], device=self.device)
                self.pages.setdefault(tuple(key), []).append(data)
            self.free_slots.extend(indices.tolist())
        # Preserve empty visited tiles too; never refill a previously emptied area.
        for key in self.loaded - loaded:
            self.pages.setdefault(key, [])
        entering = sorted(loaded - self.loaded)
        incoming = []
        for key in entering:
            if key not in self.visited:
                incoming.append((key, self._fresh(key)))
            incoming.extend((key, data) for data in self.pages.get(key, []))
        needed = sum(len(data.q) for _, data in incoming)
        if needed > len(self.free_slots):
            raise RuntimeError(
                f"Saved terrain needs {needed} particle slots, but only {len(self.free_slots)} are free. "
                "The fixed simulation allocation cannot hold this accumulation. "
                "Increase the moving/boundary terrain size; no cached particles were discarded."
            )
        for _, data in incoming:
            n = len(data.q)
            slots = wp.array(self.free_slots[-n:], dtype=int, device=self.device)
            del self.free_slots[-n:]
            wp.launch(
                _restore,
                dim=n,
                inputs=[
                    data,
                    slots,
                    self.live,
                    self.frozen,
                    w.layers,
                    w.model.particle_inv_mass,
                    wp.vec2(*center),
                    wp.vec2(*w.moving_size),
                ],
                device=self.device,
            )
        for key in entering:
            self.pages.pop(key, None)
        self.visited.update(entering)
        self.loaded = loaded
        w.recycled_count.fill_(needed)
        return outgoing_count > 0 or needed > 0 or int(w.changed.numpy()[0]) > 0
