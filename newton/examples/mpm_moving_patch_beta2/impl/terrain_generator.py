# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Procedural height functions for the moving-patch terrain examples."""

from collections.abc import Mapping
from enum import Enum

import numpy as np

__all__ = ["TerrainGenerator", "TerrainType"]


class TerrainType(str, Enum):
    """Supported procedural ground profiles."""

    FLAT = "flat"
    SLOPE = "slope"
    SINUSOIDAL = "sinusoidal"


class TerrainGenerator:
    """Sample a procedural ground surface using broadcastable world X/Y arrays.

    All profiles are constant along Y. Parameters use meters and degrees:

    * Flat: ``height`` (default 0).
    * Slope: ``angle_degrees`` (default 30) and ``height`` (default 0).
    * Sinusoidal: ``amplitude`` (default 0.3), ``wavelength`` (default 4),
      ``phase_degrees`` (default 0), and ``height`` (default 0).

    Initial particles sample this generator during construction. The GPU cache
    evaluates the corresponding flat or sloped ground analytically.
    The beta2 wrapper currently supports flat and sloped supporting colliders.
    """

    def __init__(self, *, type: TerrainType | str = TerrainType.SLOPE, params: Mapping[str, float] | None = None):
        self.type = TerrainType(type)
        defaults = {
            TerrainType.FLAT: {"height": 0.0},
            TerrainType.SLOPE: {"angle_degrees": 30.0, "height": 0.0},
            TerrainType.SINUSOIDAL: {"amplitude": 0.3, "wavelength": 4.0, "phase_degrees": 0.0, "height": 0.0},
        }
        supplied = dict(params or {})
        unknown = supplied.keys() - defaults[self.type].keys()
        if unknown:
            raise ValueError(f"Unsupported {self.type.value} terrain parameters: {sorted(unknown)}")
        self.params = {key: float(value) for key, value in (defaults[self.type] | supplied).items()}
        if not all(np.isfinite(value) for value in self.params.values()):
            raise ValueError("Terrain parameters must be finite")
        if self.type == TerrainType.SLOPE and abs(self.params["angle_degrees"]) >= 90.0:
            raise ValueError("Slope angle must be strictly between -90 and 90 degrees")
        if self.type == TerrainType.SINUSOIDAL:
            if self.params["wavelength"] <= 0.0:
                raise ValueError("Wave wavelength must be positive")
            if self.params["amplitude"] < 0.0:
                raise ValueError("Wave amplitude must be nonnegative")

    def sample_height(self, x, y=0.0) -> np.ndarray:
        """Return ground heights [m] at broadcastable world coordinates [m].

        For an ij-indexed heightfield, pass ``x[:, None]`` and ``y[None, :]``.
        Scalars are also supported and produce a zero-dimensional array.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Terrain sample coordinates must be finite")
        height = self.params["height"]
        if self.type == TerrainType.FLAT:
            return np.full(x.shape, height)
        if self.type == TerrainType.SLOPE:
            return np.asarray(height + np.tan(np.deg2rad(self.params["angle_degrees"])) * x)
        phase = 2.0 * np.pi * x / self.params["wavelength"] + np.deg2rad(self.params["phase_degrees"])
        return np.asarray(height + self.params["amplitude"] * np.sin(phase))
