# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Create a CPU-only schematic of particle/grid transfers and NanoVDB indexing."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

OUT = Path(__file__).resolve().parent / "images"
BLUE = "#2466a1"
ORANGE = "#d98227"
GREEN = "#15836e"
INK = "#263648"
GRAY = "#b7c2ce"
STAGES = [
    (
        "1. Locate particles and rebuild topology",
        "Particle positions identify occupied grid cells. NanoVDB stores the spatial index for those cells.",
    ),
    (
        "2. Particle to grid (P2G)",
        "Particles contribute mass, momentum, and material terms to grid fields using the grid basis.",
    ),
    (
        "3. Solve on the grid",
        "Newton solves grid motion and material response, including gravity and collider interaction.",
    ),
    ("4. Grid to particles (G2P)", "Interpolate solved grid velocity and velocity gradients back to the particles."),
    (
        "5. Advance material state",
        "Update particle positions, velocity, deformation, and stress. Rebuild around the new positions next step.",
    ),
]
RNG = np.random.default_rng(12)
POINTS = RNG.uniform((0.45, 0.5), (2.5, 1.45), (32, 2))


def arrow(ax, start, end, color, label=None, label_pos=None, both=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<->" if both else "-|>", mutation_scale=15, lw=2, color=color))
    if label:
        ax.text(*label_pos, label, ha="center", va="center", fontsize=10, color=color, linespacing=1.4)


def panel(ax, x, y, w, h, title, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", facecolor="white", edgecolor=color, lw=2
        )
    )
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", weight="bold", fontsize=12, color=INK)


def render(stage, fraction=0.5, static=False):
    fig, ax = plt.subplots(figsize=(13.2, 7.6), dpi=110)
    fig.patch.set_facecolor("#f7f9fc")
    ax.set(xlim=(0, 15.5), ylim=(0, 8.6))
    ax.axis("off")
    ax.text(
        7.75,
        8.35,
        "MPM data flow: particles, sparse grid, and NanoVDB",
        ha="center",
        fontsize=19,
        weight="bold",
        color=INK,
    )

    def active(n):
        return static or stage == n

    panel(ax, 0.25, 4.9, 3.1, 2.45, "Particles at time t", ORANGE)
    ax.scatter(POINTS[:, 0] + 0.25, POINTS[:, 1] + 5.0, s=18, c=ORANGE)
    ax.text(1.8, 5.14, "Material state lives here", ha="center", fontsize=10, color=INK)
    panel(ax, 5.65, 3.95, 4.0, 3.4, "Sparse MPM grid", BLUE)
    # A 2D illustration of occupied cells. Grid fields have their own DOFs.
    origin = np.array([6.38, 4.47])
    spacing = 0.36
    for i in range(7):
        for j in range(5):
            occupied = j in (1, 2) and i in range(1, 6)
            ax.add_patch(
                Rectangle(
                    origin + spacing * np.array([i, j]),
                    spacing,
                    spacing,
                    facecolor="#dbeaf6" if occupied else "#f4f6f8",
                    edgecolor="#c1cbd5",
                    lw=0.6,
                )
            )
            if occupied and (static or stage >= 2):
                p = origin + spacing * np.array([i + 0.5, j + 0.7])
                arrow(ax, p, p + np.array([0.035, -0.18]), GREEN)
    ax.text(7.65, 6.65, "Numerical fields and solver", ha="center", fontsize=10, color=INK)
    ax.text(7.65, 4.15, "Blue cells: occupied region", ha="center", fontsize=10, color=BLUE)
    panel(ax, 11.6, 3.95, 3.55, 3.4, "NanoVDB index", BLUE)
    nodes = [
        (13.35, 6.65, "Root"),
        (13.35, 6.0, "Upper"),
        (13.35, 5.35, "Lower"),
        (12.5, 4.65, "Leaf"),
        (14.2, 4.65, "Leaf"),
    ]
    for a, b in ((0, 1), (1, 2), (2, 3), (2, 4)):
        x, y, _ = nodes[a]
        xx, yy, _ = nodes[b]
        ax.plot([x, xx], [y, yy], color=BLUE, lw=1.3)
    for x, y, text in nodes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "fc": "#deebf7", "ec": BLUE},
        )
    arrow(ax, (9.72, 5.52), (11.53, 5.52), BLUE, "Cell coordinates\n↔ grid indices", (10.6, 6.13), both=True)
    ax.text(13.35, 3.47, "Indexes the grid;\ndoes not solve the physics", ha="center", fontsize=11, color=INK)
    if stage in (0, 1, 3) and not static:
        query_progress = 2 * fraction if fraction <= 0.5 else 2 * (1 - fraction)
        ax.scatter([9.87 + 1.45 * query_progress], [5.52], s=55, c=BLUE, zorder=10)
    color = ORANGE if active(0) or active(1) else GRAY
    arrow(ax, (3.42, 6.18), (5.57, 6.18), color, "1. Positions → topology\n2. P2G contributions", (4.5, 6.84))
    if stage in (0, 1) and not static:
        ax.scatter([3.55 + 1.85 * fraction], [6.18], s=65, c=ORANGE, zorder=10)
    panel(ax, 0.25, 1.15, 3.1, 2.45, "Particles at time t + Δt", ORANGE)
    moved = POINTS.copy()
    moved[:, 0] += 0.08
    moved[:, 1] -= 0.14 if static or stage >= 4 else 0.0
    ax.scatter(moved[:, 0] + 0.25, moved[:, 1] + 1.4, s=18, c=ORANGE if active(4) or static else "#d6b590")
    ax.text(1.8, 1.38, "5. Motion + material update", ha="center", fontsize=9.5, color=INK)
    color = GREEN if active(3) else GRAY
    arrow(ax, (6.55, 3.88), (3.44, 2.55), color)
    ax.text(5.45, 2.46, "4. G2P: solved velocity\nand velocity gradient", ha="center", fontsize=10, color=color)
    if stage == 3 and not static:
        start, end = np.array([6.55, 3.88]), np.array([3.44, 2.55])
        point = start + (end - start) * fraction
        ax.scatter(*point, s=65, c=GREEN, zorder=10)
    arrow(ax, (1.8, 3.67), (1.8, 4.83), ORANGE if active(4) else GRAY, "Next step", (2.5, 4.22))
    ax.text(
        7.65, 3.45, "3. Implicit grid solve", ha="center", fontsize=11, weight="bold", color=GREEN if active(2) else INK
    )
    if static:
        title = "Data flow: particle state → grid solve → updated particle state"
        detail = "NanoVDB supplies topology and indexing throughout grid construction and access. It is not a separate transfer stage."
    else:
        title, detail = STAGES[stage]
    ax.text(7.75, 0.65, title, ha="center", fontsize=13, weight="bold", color=INK)
    ax.text(7.75, 0.15, detail, ha="center", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.04, top=0.98)
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return image


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    render(0, static=True).save(OUT / "05_particle_grid_data_flow.png")
    frames = [render(stage, frame / 7) for stage in range(5) for frame in range(8)]
    frames[0].save(
        OUT / "05_particle_grid_data_flow.gif",
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0,
        optimize=False,
    )
