from dataclasses import MISSING, dataclass


@dataclass
class MPMMaterialCfg:
    bulk_density: float = MISSING
    friction_angle: float = MISSING
    youngs_modulus: float = MISSING
    poissons_ratio: float = MISSING
    critical_fraction: float = MISSING
    particles_per_cell: float = 3.0
    voxel_size: float = 0.025
    voxel_size_vis: float = 0.04
    transfer_scheme: str = "pic"
    collider_basis: str = "pic27"
    collider_velocity_mode: str = "finite_difference"
    particle_color: tuple[float, float, float] = (0.0, 0.0, 0.0)