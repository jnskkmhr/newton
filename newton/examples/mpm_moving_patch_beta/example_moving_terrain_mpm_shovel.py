# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Mouse-dragged shovel on beta moving MPM terrain.

Run ``uv run -m newton.examples moving_terrain_mpm_box`` and right-click drag.
"""

from typing import NamedTuple

import newton.examples
from newton.examples.mpm_moving_patch_beta.impl.base_example import MovingTerrainExample
from newton.examples.mpm_moving_patch_beta.impl.moving_terrain_mpm import (
    MovingTerrainMPMConfig,
    MovingTerrainMPMWrapper,
)
from newton.examples.mpm_moving_patch_beta.impl.shovel import add_shovel
from newton.examples.mpm_moving_patch_beta.impl.terrain_generator import TerrainGenerator


class StripMaterialPreset(NamedTuple):
    """Material values for one particle strip."""

    label: str
    density: float
    young_modulus: float
    friction: float
    yield_pressure: float
    tensile_yield_ratio: float
    yield_stress: float
    hardening: float
    dilatancy: float
    viscosity: float


MATERIAL_PRESETS = {
    # Values start from the sand, snow, and mud rows of Table 5 in Daviet's
    # mixed-MPM paper. A finite 1 PPa value represents the tabulated rigid
    # elastic limit while remaining valid input to Newton's schema.
    "sand": StripMaterialPreset("dry sand", 1600.0, 1.0e15, 0.48, 1.0e15, 0.0, 0.0, 0.0, 0.0, 0.0),
    "snow": StripMaterialPreset(
        "snow",
        250.0,
        1.0e15,
        0.30,
        2.0e6,
        0.05,
        0.0,
        1.0,
        1.0,
        0.0,
    ),
    "clay": StripMaterialPreset(
        "cohesive clay",
        1500.0,
        1.0e15,
        0.0,
        1.0e15,
        1.0,
        200.0,
        0.0,
        0.1,
        100.0,
    ),
}


class Example(MovingTerrainExample):
    """Supply a passive shovel to the shared moving-terrain scene."""

    def setup_terrain(self, args, *, tracked_body: str):
        """Define the ground generator, particle material, and terrain dimensions.

        Called after rigid bodies are added, before ground or particles are added
        and before the model is finalized. Override this method to assign
        ``self.terrain_gen`` and ``self.mpm_wrapper`` with scene-specific settings.
        Particle allocation and wrapper binding remain in the base initializer.

        Args:
            args: Parsed example options.
            tracked_body: Name of the body followed by the terrain patch.
        """
        terrain_params = {
            "flat": {},
            "slope": {"angle_degrees": args.slope_angle},
            "sinusoidal": {
                "amplitude": args.wave_amplitude,
                "wavelength": args.wave_wavelength,
                "phase_degrees": args.wave_phase,
            },
        }
        self.terrain_gen = TerrainGenerator(type=args.terrain_type, params=terrain_params[args.terrain_type])
        voxel_size = args.voxel_size
        cell_size = voxel_size / args.particles_per_cell
        radius = 0.5 * cell_size
        jitter = 2.0 * radius  # Meters, as in ModelBuilder.add_particle_grid; 0 disables jitter.
        # Material choices belong to the example, not the moving-terrain wrapper.
        material = MATERIAL_PRESETS["sand"]
        material_parameters = {
            f"mpm:{name}": value for name, value in material._asdict().items() if name not in ("label", "density")
        }
        # material_parameters["mpm:poisson_ratio"] = 0.3
        self.mpm_wrapper = MovingTerrainMPMWrapper(
            config=MovingTerrainMPMConfig(
                background_height_spacing=voxel_size,
                background_terrain_size=(20, 20),
                preserve_particle_state=args.full_state_cache,
            ),
            tracked_body=tracked_body,
            material_parameters=material_parameters,
            density=material.density,
            voxel_size=voxel_size,
            particles_per_cell=args.particles_per_cell,
            jitter=jitter,
            terrain_gen=self.terrain_gen,
        )

    def _emit_rigid_bodies(self, builder):
        """Create a mouse-draggable shovel without an automatic controller."""
        return add_shovel(builder)


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
