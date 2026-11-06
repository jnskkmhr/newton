# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Generate a CPU-only illustration of the planned three terrain layers."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle


def main():
    """Draw the terrain layout and distinguish physics from background storage."""
    out = Path(__file__).resolve().parent / "images" / "three_terrain_layers.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ink, dynamic, boundary, background = "#243447", "#1677a6", "#b86818", "#687b56"
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    fig.text(0.055, 0.94, "Three terrain layers", fontsize=25, weight="bold", color=ink)
    fig.text(0.055, 0.90, "Top view • boundary terrain surrounds the simulated terrain", fontsize=13, color=ink)
    ax = fig.add_axes((0.055, 0.30, 0.58, 0.55))
    ax.set(xlim=(-5.3, 5.3), ylim=(-4.05, 4.05), aspect="equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-5, -3.8), 10, 7.6, facecolor="#eef2e8", edgecolor=background, lw=1.8))
    # Background lines denote saved surface-height samples, not MPM particles.
    for y in np.linspace(-3.5, 3.5, 13):
        x = np.linspace(-4.8, 4.8, 180)
        ax.plot(x, y + 0.07 * np.sin(x * 2 + y), color="#c5d0b9", lw=0.8)
    ax.add_patch(Rectangle((-3.5, -2.65), 7, 5.3, facecolor="#fff0d8", edgecolor=boundary, lw=2))
    ax.add_patch(Rectangle((-2.55, -1.7), 5.1, 3.4, facecolor="#e3f3fa", edgecolor=dynamic, lw=2))
    rng = np.random.default_rng(8)
    x, y = np.meshgrid(np.arange(-3.3, 3.4, 0.23), np.arange(-2.45, 2.5, 0.23))
    x, y = x.ravel(), y.ravel()
    x += rng.uniform(-0.055, 0.055, x.size)
    y += rng.uniform(-0.055, 0.055, y.size)
    inside = (np.abs(x) < 2.5) & (np.abs(y) < 1.65)
    ax.scatter(x[~inside], y[~inside], s=9, color=boundary, alpha=0.75)
    ax.scatter(x[inside], y[inside], s=9, color=dynamic, alpha=0.75)
    ax.add_patch(FancyBboxPatch((-0.8, -0.48), 1.6, 0.96, boxstyle="round,pad=0.05", facecolor=ink, edgecolor="white", lw=2, zorder=4))
    ax.text(0, 0, "Robot", color="white", weight="bold", ha="center", va="center", fontsize=14, zorder=5)
    for yy in (-0.66, 0.50):
        for xx in (-0.67, 0.37):
            ax.add_patch(Rectangle((xx, yy), 0.3, 0.16, facecolor=ink, zorder=4))
    for yy, label, color in [(1.24, "1  Simulated terrain", dynamic), (2.22, "2  Boundary terrain", boundary), (3.3, "3  Background terrain", background)]:
        ax.text(0, yy, label, ha="center", va="center", fontsize=13, weight="bold", color=color, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 4}, zorder=6)
    ax.text(0, -3.45, "Schematic • not to scale", ha="center", fontsize=10, color=background, bbox={"facecolor": "#eef2e8", "edgecolor": "none"})

    cards = [
        (0.71, dynamic, "1  Simulated terrain", "Dynamic MPM particles\nPositive mass\nRetained particles keep their state"),
        (0.52, boundary, "2  Boundary terrain", "Fixed MPM particles\nZero mass and zero velocity\nProvides support through the grid"),
        (0.33, background, "3  Background terrain", "Surface heights stored on GPU\nNo MPM simulation\nSaved shape used on return visits"),
    ]
    for yy, color, title, detail in cards:
        fig.text(0.66, yy + 0.07, title, color=color, fontsize=15, weight="bold")
        fig.text(0.66, yy - 0.025, detail, color=ink, fontsize=12, linespacing=1.6)
    fig.text(0.055, 0.23, "As the robot moves", fontsize=16, color=ink, weight="bold")
    fig.text(0.055, 0.177, "Outgoing terrain  →  save surface heights before fixing or recycling particles", fontsize=13, color=ink)
    fig.text(0.055, 0.133, "Incoming terrain  →  generate particles from saved heights; initialize material state", fontsize=13, color=ink)
    fig.text(0.055, 0.077, "Fixed boundary particles stay in world coordinates; the selection of terrain moves with the robot.", fontsize=11, color=ink)
    fig.savefig(out, dpi=180, facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
