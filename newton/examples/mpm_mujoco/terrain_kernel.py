import warp as wp


@wp.kernel
def sample_sinusoidal_terrain(
    particle_q: wp.array(dtype=wp.vec3f),
    wl: float,
    amp: float,
):
    i = wp.tid()
    x = particle_q[i].x
    z = particle_q[i].z + amp * wp.sin(2.0 * wp.PI * x / wl)
    particle_q[i] = wp.vec3f(particle_q[i].x, particle_q[i].y, z)
