# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-side particle state, tile tables, allocators and transition kernels of the streaming particle cache.

Everything here runs with fixed launch sizes and without host synchronization, so the cache update can be recorded
in a CUDA graph with the rest of the physics step. Capacity problems set bits in a per-world error array that the
host checks on request.
"""

import warp as wp


@wp.func
def surface_height(mesh: wp.uint64, x: float, y: float, ray_z: float, ray_length: float):
    """Cast downward at world XY [m] and return whether the mesh was hit and its height [m]."""
    query = wp.mesh_query_ray(mesh, wp.vec3(x, y, ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
    height = ray_z - query.t
    return query.result, height


@wp.kernel(enable_backward=False)
def update_patch_center(
    body_q: wp.array[wp.transform],
    body_indices: wp.array2d[int],
    env_origin: wp.array[wp.vec3],
    background_terrain_center: wp.vec2,
    background_terrain_size: wp.vec2,
    total_patch_size: wp.vec2,
    patch_discretization_step: float,
    patch_center: wp.array2d[wp.vec2],
):
    """Quantize robot-centered patch positions [m] and keep each full patch inside the background."""
    i, j = wp.tid()
    if body_indices[i, j] < 0:
        patch_center[i, j] = wp.vec2(1.0e20)
        return
    p = wp.transform_get_translation(body_q[body_indices[i, j]])
    o = env_origin[i]
    allowed_patch_center_half = (background_terrain_size - total_patch_size) * 0.5
    x = wp.floor((p[0] - o[0]) / patch_discretization_step + 0.5) * patch_discretization_step
    y = wp.floor((p[1] - o[1]) / patch_discretization_step + 0.5) * patch_discretization_step
    patch_center[i, j] = wp.vec2(
        wp.clamp(
            o[0] + x,
            background_terrain_center[0] - allowed_patch_center_half[0],
            background_terrain_center[0] + allowed_patch_center_half[0],
        ),
        wp.clamp(
            o[1] + y,
            background_terrain_center[1] - allowed_patch_center_half[1],
            background_terrain_center[1] + allowed_patch_center_half[1],
        ),
    )


@wp.struct
class ParticleState:
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


@wp.func
def copy_particle(source: ParticleState, i: int, target: ParticleState, j: int):
    target.q[j] = source.q[i]
    target.qd[j] = source.qd[i]
    target.grad[j] = source.grad[i]
    target.elastic[j] = source.elastic[i]
    target.plastic[j] = source.plastic[i]
    target.stress[j] = source.stress[i]
    target.transform[j] = source.transform[i]
    target.mass[j] = source.mass[i]
    target.radius[j] = source.radius[i]
    target.flags[j] = source.flags[i]


@wp.kernel(enable_backward=False)
def plan_transitions(
    live: ParticleState,
    frozen: ParticleState,
    worlds: wp.array[int],
    centers: wp.array2d[wp.vec2],
    reset: wp.array[int],
    size: wp.vec2,
    tile_begins: wp.array2d[wp.vec2i],
    tile_shape: wp.vec2i,
    tile_size: float,
    layers: wp.array[int],
    next_layers: wp.array[int],
    outgoing_ids: wp.array[int],
    outgoing_tiles: wp.array[wp.vec2i],
    counts: wp.array[int],
):
    """Classify particles without mutating physics; collect departing slots for capacity checks."""
    i = wp.tid()
    world = worlds[i]
    layer = layers[i]
    new_layer = int(0)
    if layer != 0 and reset[world] == 0:
        p = live.q[i]
        if layer == 2:
            p = frozen.q[i]
        tile = wp.vec2i(int(wp.floor(p[0] / tile_size)), int(wp.floor(p[1] / tile_size)))
        for region in range(centers.shape[1]):
            delta = wp.vec2(p[0], p[1]) - centers[world, region]
            begin = tile_begins[world, region]
            if (
                tile[0] >= begin[0]
                and tile[0] < begin[0] + tile_shape[0]
                and tile[1] >= begin[1]
                and tile[1] < begin[1] + tile_shape[1]
            ):
                if new_layer == 0:
                    new_layer = 2
                if wp.abs(delta[0]) < size[0] * 0.5 and wp.abs(delta[1]) < size[1] * 0.5:
                    new_layer = 1
        if new_layer == 0:
            index = wp.atomic_add(counts, 0, 1)
            outgoing_ids[index] = i
            outgoing_tiles[index] = tile
    next_layers[i] = new_layer
    if layer != new_layer:
        wp.atomic_add(counts, 1, 1)


@wp.kernel(enable_backward=False)
def apply_transitions(
    live: ParticleState,
    frozen: ParticleState,
    inactive_position: wp.vec3,
    layers: wp.array[int],
    next_layers: wp.array[int],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    inv_mass: wp.array[float],
):
    """Freeze the original state before imposing kinematics; thaw from that original state."""
    i = wp.tid()
    old, new = layers[i], next_layers[i]
    if old == 2:
        copy_particle(frozen, i, live, i)
    if old == 1 and new == 2:
        copy_particle(live, i, frozen, i)
    if new == 0:
        # PicQuadrature bins positions before integrands inspect particle flags.
        # Coincident inactive slots inside the grid create expensive cell loops.
        live.q[i] = inactive_position
        live.flags[i] = 0
        live.mass[i] = 0.0
        live.qd[i] = wp.vec3(0.0)
        live.grad[i] = wp.mat33(0.0)
    elif new == 2:
        live.mass[i] = 0.0
        live.qd[i] = wp.vec3(0.0)
        live.grad[i] = wp.mat33(0.0)
    anchors[i] = live.q[i]
    layers[i] = new
    dynamic[i] = int(new == 1)
    inv_mass[i] = 0.0
    if new == 1:
        inv_mass[i] = 1.0 / live.mass[i]


@wp.func
def finish_loading(
    live: ParticleState,
    frozen: ParticleState,
    i: int,
    centers: wp.array2d[wp.vec2],
    world: int,
    size: wp.vec2,
    layers: wp.array[int],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    inv_mass: wp.array[float],
):
    copy_particle(live, i, frozen, i)
    p = live.q[i]
    is_dynamic = bool(False)
    for region in range(centers.shape[1]):
        center = centers[world, region]
        if wp.abs(p[0] - center[0]) < 0.5 * size[0] and wp.abs(p[1] - center[1]) < 0.5 * size[1]:
            is_dynamic = True
    dynamic[i] = int(is_dynamic)
    layers[i] = wp.where(is_dynamic, 1, 2)
    anchors[i] = p
    inv_mass[i] = 0.0
    if is_dynamic:
        inv_mass[i] = 1.0 / live.mass[i]
    else:
        live.mass[i] = 0.0
        live.qd[i] = wp.vec3(0.0)
        live.grad[i] = wp.mat33(0.0)


@wp.kernel(enable_backward=False)
def refresh_cached_properties(
    mass: wp.array[float],
    radius: wp.array[float],
    flags: wp.array[int],
    volume: wp.array[float],
    density: wp.array[float],
    material_volume: wp.array[float],
    transfer_flags: wp.array[int],
    material_flags: wp.array[int],
):
    """Refresh ordinary material particles without downloading particle arrays or changing their history."""
    i = wp.tid()
    v = 8.0 * radius[i] * radius[i] * radius[i]
    volume[i] = v
    density[i] = mass[i] / v
    transfer_flags[i] = flags[i]
    material_flags[i] = flags[i]
    material_volume[i] = wp.where((flags[i] & 1) != 0, v, 0.0)


TILE_OFFSET = wp.constant(16384)
TILE_RANGE = wp.constant(32768)
"""Tile coordinates are packed as ``(x + OFFSET) * RANGE + (y + OFFSET)``, which needs ``|coordinate| < OFFSET``."""
EMPTY = wp.constant(-1)
NO_TILE = wp.constant(1_000_000_000)
"""Begin coordinate of an unused or reset tile window; no tile lies inside it."""
ERROR_RECORDS = wp.constant(1)
ERROR_TILES = wp.constant(2)
ERROR_SLOTS = wp.constant(4)

FLAG_OUTGOING = wp.constant(0)
FLAG_LAYER_CHANGES = wp.constant(1)
FLAG_FRESH = wp.constant(2)
FLAG_RESTORED = wp.constant(3)
FLAG_RESET = wp.constant(4)


@wp.func
def pack_tile(tx: int, ty: int) -> int:
    return (tx + TILE_OFFSET) * TILE_RANGE + (ty + TILE_OFFSET)


@wp.func
def tile_hash(key: int, capacity: int) -> int:
    h = wp.uint32(key) * wp.uint32(2654435761)
    return int(h >> wp.uint32(7)) & (capacity - 1)


@wp.func
def find_or_insert(keys: wp.array2d[int], visited_count: wp.array[int], world: int, key: int):
    """Return ``(entry, inserted)`` for a tile of one world; ``entry`` is -1 when the table is full."""
    capacity = keys.shape[1]
    start = tile_hash(key, capacity)
    entry = int(-1)
    inserted = int(0)
    for n in range(capacity):
        idx = (start + n) & (capacity - 1)
        old = wp.atomic_cas(keys, world, idx, EMPTY, key)
        if old == EMPTY:
            wp.atomic_add(visited_count, world, 1)
            entry = idx
            inserted = 1
            break
        if old == key:
            entry = idx
            break
    return entry, inserted


@wp.func
def alloc_record(
    bump: wp.array[int], free_count: wp.array[int], free_stack: wp.array[int], world: int, capacity: int
) -> int:
    """Pop a recycled record of this world, else take the next never-used one; -1 when the pool is exhausted."""
    top = wp.atomic_add(free_count, world, -1) - 1
    if top >= 0:
        return free_stack[world * capacity + top]
    wp.atomic_add(free_count, world, 1)
    b = wp.atomic_add(bump, world, 1)
    if b < capacity:
        return world * capacity + b
    wp.atomic_add(bump, world, -1)
    return -1


@wp.func
def alloc_slot(
    bump: wp.array[int],
    free_count: wp.array[int],
    free_stack: wp.array[int],
    slot_table: wp.array[int],
    world: int,
    capacity: int,
) -> int:
    """Pop a recycled simulation slot of this world, else the next never-used one; -1 when none is free."""
    top = wp.atomic_add(free_count, world, -1) - 1
    if top >= 0:
        return free_stack[world * capacity + top]
    wp.atomic_add(free_count, world, 1)
    b = wp.atomic_add(bump, world, 1)
    if b < capacity:
        return slot_table[world * capacity + b]
    wp.atomic_add(bump, world, -1)
    return -1


@wp.func
def inside_window(tile: wp.vec2i, begin: wp.vec2i, shape: wp.vec2i) -> bool:
    return (
        tile[0] >= begin[0] and tile[0] < begin[0] + shape[0] and tile[1] >= begin[1] and tile[1] < begin[1] + shape[1]
    )


@wp.kernel(enable_backward=False)
def compute_tile_begins(
    body_counts: wp.array[int],
    tile_size: wp.float64,
    shape: wp.vec2i,
    lower: wp.vec2i,
    upper: wp.vec2i,
    centers: wp.array2d[wp.vec2],
    begins: wp.array2d[wp.vec2i],
):
    """Snap each tracked body's patch to the tile grid and keep it inside the background terrain.

    The arithmetic mirrors the host cache exactly (float32 quotient, float64 offset and product) so that patch centers
    and the layer of boundary particles are bit-identical.
    """
    w, r = wp.tid()
    half_x = wp.float64(shape[0]) * wp.float64(0.5)
    half_y = wp.float64(shape[1]) * wp.float64(0.5)
    tile_f32 = wp.float32(tile_size)
    if r < body_counts[w]:
        c = centers[w, r]
        qx = wp.float64(c[0] / tile_f32) - half_x
        qy = wp.float64(c[1] / tile_f32) - half_y
        bx = wp.clamp(int(wp.rint(qx)), lower[0], upper[0])
        by = wp.clamp(int(wp.rint(qy)), lower[1], upper[1])
        begins[w, r] = wp.vec2i(bx, by)
        cx = float((wp.float64(bx) + half_x) * tile_size)
        cy = float((wp.float64(by) + half_y) * tile_size)
        centers[w, r] = wp.vec2(cx, cy)
    else:
        begins[w, r] = wp.vec2i(NO_TILE, NO_TILE)
        cx = float((wp.float64(1.0e9) + half_x) * tile_size)
        cy = float((wp.float64(1.0e9) + half_y) * tile_size)
        centers[w, r] = wp.vec2(cx, cy)


@wp.kernel(enable_backward=False)
def reset_worlds(
    reset: wp.array[int],
    record_bump: wp.array[int],
    record_free_count: wp.array[int],
    slot_bump: wp.array[int],
    slot_free_count: wp.array[int],
    visited_count: wp.array[int],
    generation: wp.array[int],
    prev_begins: wp.array2d[wp.vec2i],
    flags: wp.array[int],
):
    """Forget the history of reset worlds in constant time: no pool is refilled."""
    w = wp.tid()
    if reset[w] != 0:
        record_bump[w] = 0
        record_free_count[w] = 0
        slot_bump[w] = 0
        slot_free_count[w] = 0
        visited_count[w] = 0
        generation[w] = generation[w] + 1
        flags[FLAG_RESET] = 1
        for r in range(prev_begins.shape[1]):
            prev_begins[w, r] = wp.vec2i(NO_TILE, NO_TILE)


@wp.kernel(enable_backward=False)
def clear_tile_tables(
    reset: wp.array[int],
    keys: wp.array2d[int],
    heads: wp.array2d[int],
    revisions: wp.array2d[int],
    loaded_once: wp.array2d[int],
):
    w, j = wp.tid()
    if reset[w] != 0:
        keys[w, j] = EMPTY
        heads[w, j] = EMPTY
        revisions[w, j] = 0
        loaded_once[w, j] = 0


@wp.kernel(enable_backward=False)
def save_outgoing(
    live: ParticleState,
    frozen: ParticleState,
    layers: wp.array[int],
    worlds: wp.array[int],
    outgoing_ids: wp.array[int],
    outgoing_tiles: wp.array[wp.vec2i],
    flags: wp.array[int],
    saved: ParticleState,
    record_capacity: int,
    record_bump: wp.array[int],
    record_free_count: wp.array[int],
    record_free: wp.array[int],
    record_next: wp.array[int],
    keys: wp.array2d[int],
    heads: wp.array2d[int],
    revisions: wp.array2d[int],
    visited_count: wp.array[int],
    max_visited: int,
    tile_counter: wp.array[int],
    slot_capacity: int,
    slot_free_count: wp.array[int],
    slot_free: wp.array[int],
    errors: wp.array[int],
):
    """Move each departing particle into a record of its tile's list and free its simulation slot."""
    k = wp.tid()
    if k >= flags[FLAG_OUTGOING]:
        return
    slot = outgoing_ids[k]
    w = worlds[slot]
    tile = outgoing_tiles[k]
    record = alloc_record(record_bump, record_free_count, record_free, w, record_capacity)
    if record < 0:
        wp.atomic_or(errors, w, ERROR_RECORDS)
        return
    if layers[slot] == 2:
        copy_particle(frozen, slot, saved, record)
    else:
        copy_particle(live, slot, saved, record)
    entry, _inserted = find_or_insert(keys, visited_count, w, pack_tile(tile[0], tile[1]))
    if entry < 0 or visited_count[w] > max_visited:
        wp.atomic_or(errors, w, ERROR_TILES)
    if entry >= 0:
        old = wp.atomic_exch(heads, w, entry, record)
        record_next[record] = old
        revisions[w, entry] = wp.atomic_add(tile_counter, 0, 1) + 1
    top = wp.atomic_add(slot_free_count, w, 1)
    slot_free[w * slot_capacity + top] = slot


@wp.kernel(enable_backward=False)
def restore_entering_tiles(
    body_counts: wp.array[int],
    begins: wp.array2d[wp.vec2i],
    prev_begins: wp.array2d[wp.vec2i],
    shape: wp.vec2i,
    saved: ParticleState,
    live: ParticleState,
    frozen: ParticleState,
    keys: wp.array2d[int],
    heads: wp.array2d[int],
    revisions: wp.array2d[int],
    loaded_once: wp.array2d[int],
    visited_count: wp.array[int],
    max_visited: int,
    tile_counter: wp.array[int],
    record_capacity: int,
    record_free_count: wp.array[int],
    record_free: wp.array[int],
    record_next: wp.array[int],
    slot_capacity: int,
    slot_bump: wp.array[int],
    slot_free_count: wp.array[int],
    slot_free: wp.array[int],
    slot_table: wp.array[int],
    centers: wp.array2d[wp.vec2],
    size: wp.vec2,
    layers: wp.array[int],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    fresh: wp.array[wp.vec3i],
    flags: wp.array[int],
    errors: wp.array[int],
):
    """Restore the saved particles of each tile that entered a window, or queue a never-visited tile."""
    tid = wp.tid()
    tiles_per_window = shape[0] * shape[1]
    regions = begins.shape[1]
    w = tid // (regions * tiles_per_window)
    r = (tid // tiles_per_window) % regions
    t = tid % tiles_per_window
    if r >= body_counts[w]:
        return
    begin = begins[w, r]
    tile = wp.vec2i(begin[0] + t // shape[1], begin[1] + t % shape[1])
    # A tile covered by several windows is handled once, by the first window that holds it.
    for other in range(r):
        if inside_window(tile, begins[w, other], shape):
            return
    # A tile that was already inside a window of the previous update is not entering.
    for other in range(regions):
        if inside_window(tile, prev_begins[w, other], shape):
            return
    entry, _inserted = find_or_insert(keys, visited_count, w, pack_tile(tile[0], tile[1]))
    if entry < 0 or visited_count[w] > max_visited:
        wp.atomic_or(errors, w, ERROR_TILES)
    if entry < 0:
        return
    # A tile can hold saved particles pushed there before it was ever loaded. It still needs its undeformed terrain,
    # so "has saved particles" and "has been loaded" are tracked separately, as the host cache does.
    if wp.atomic_exch(loaded_once, w, entry, 1) == 0:
        index = wp.atomic_add(flags, FLAG_FRESH, 1)
        fresh[index] = wp.vec3i(w, tile[0], tile[1])
    record = heads[w, entry]
    while record >= 0:
        slot = alloc_slot(slot_bump, slot_free_count, slot_free, slot_table, w, slot_capacity)
        if slot < 0:
            wp.atomic_or(errors, w, ERROR_SLOTS)
            break
        following = record_next[record]
        copy_particle(saved, record, live, slot)
        finish_loading(live, frozen, slot, centers, w, size, layers, dynamic, anchors, inv_mass)
        top = wp.atomic_add(record_free_count, w, 1)
        record_free[w * record_capacity + top] = record
        wp.atomic_add(flags, FLAG_RESTORED, 1)
        record = following
    heads[w, entry] = record
    revisions[w, entry] = wp.atomic_add(tile_counter, 0, 1) + 1


@wp.kernel(enable_backward=False)
def generate_fresh_tiles(
    fresh: wp.array[wp.vec3i],
    flags: wp.array[int],
    template: wp.array[wp.vec3],
    template_elastic: wp.array[wp.mat33],
    template_transform: wp.array[wp.mat33],
    template_stress: wp.array[wp.mat33],
    template_plastic: wp.array[float],
    template_count: int,
    tile_size: float,
    mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
    masses: wp.array[float],
    radius: float,
    live: ParticleState,
    frozen: ParticleState,
    centers: wp.array2d[wp.vec2],
    size: wp.vec2,
    layers: wp.array[int],
    dynamic: wp.array[int],
    anchors: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    missed: wp.array[int],
    slot_capacity: int,
    slot_bump: wp.array[int],
    slot_free_count: wp.array[int],
    slot_free: wp.array[int],
    slot_table: wp.array[int],
    errors: wp.array[int],
):
    """Generate the undeformed template particles of each never-visited tile from the background mesh."""
    k = wp.tid()
    f = k // template_count
    if f >= flags[FLAG_FRESH]:
        return
    j = k % template_count
    entry3 = fresh[f]
    w = entry3[0]
    tile = wp.vec2i(entry3[1], entry3[2])
    i = alloc_slot(slot_bump, slot_free_count, slot_free, slot_table, w, slot_capacity)
    if i < 0:
        wp.atomic_or(errors, w, ERROR_SLOTS)
        return
    entry = w * template_count + j
    p = template[entry]
    # Alternating mirror tiles retain matching wall states (Felix's mirror_tensor construction).
    dx = wp.where(tile[0] % 2 == 0, 1.0, -1.0)
    dy = wp.where(tile[1] % 2 == 0, 1.0, -1.0)
    mirror = wp.diag(wp.vec3(dx, dy, 1.0))
    p = wp.vec3(wp.where(dx < 0.0, tile_size - p[0], p[0]), wp.where(dy < 0.0, tile_size - p[1], p[1]), p[2])
    x = float(tile[0]) * tile_size + p[0]
    y = float(tile[1]) * tile_size + p[1]
    hit, height = surface_height(mesh, x, y, ray_z, ray_length)
    if not hit:
        wp.atomic_add(missed, 0, 1)
    live.q[i] = wp.vec3(x, y, height + p[2])
    live.qd[i] = wp.vec3(0.0)
    live.grad[i] = wp.mat33(0.0)
    live.elastic[i] = mirror * template_elastic[entry] * mirror
    live.plastic[i] = template_plastic[entry]
    live.stress[i] = mirror * template_stress[entry] * mirror
    live.transform[i] = mirror * template_transform[entry] * mirror
    live.mass[i] = masses[i]
    live.radius[i] = radius
    live.flags[i] = 1
    finish_loading(live, frozen, i, centers, w, size, layers, dynamic, anchors, inv_mass)


@wp.kernel(enable_backward=False)
def finalize_update(
    flags: wp.array[int],
    begins: wp.array2d[wp.vec2i],
    prev_begins: wp.array2d[wp.vec2i],
    changed: wp.array[int],
    revision: wp.array[int],
):
    """Remember this update's windows and publish whether any particle changed state."""
    w, r = wp.tid()
    prev_begins[w, r] = begins[w, r]
    if w == 0 and r == 0:
        total = (
            flags[FLAG_OUTGOING]
            + flags[FLAG_LAYER_CHANGES]
            + flags[FLAG_FRESH]
            + flags[FLAG_RESTORED]
            + flags[FLAG_RESET]
        )
        changed[0] = wp.where(total > 0, 1, 0)
        if total > 0:
            revision[0] = revision[0] + 1
