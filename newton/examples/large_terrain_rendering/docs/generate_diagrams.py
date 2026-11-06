# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Generate the README's conceptual PNG figures with Matplotlib.

Run from the repository root:
uv run --no-sync python newton/examples/large_terrain_rendering/docs/generate_diagrams.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUTPUT = Path(__file__).resolve().parent
INK = "#203345"
SAND = "#c18a3e"
BLUE = "#237cab"
GREEN = "#258777"
PALE = "#f5f2eb"


def canvas(title, subtitle):
    fig = plt.figure(figsize=(12, 7), facecolor=PALE)
    fig.text(0.05, 0.94, title, fontsize=22, weight="bold", color=INK)
    fig.text(0.05, 0.895, subtitle, fontsize=11, color=INK)
    return fig


def panel(fig, bounds, xlim, ylim):
    ax = fig.add_axes(bounds)
    ax.set(xlim=xlim, ylim=ylim)
    ax.axis("off")
    return ax


def box(ax, xy, width, height, text, color=BLUE):
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            edgecolor=color,
            facecolor="white",
            linewidth=1.8,
        )
    )
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=11, color=INK)


def arrow(ax, start, end, color=INK, both=False):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="<->" if both else "->", mutation_scale=15, color=color, linewidth=1.6)
    )


def save(fig, name):
    fig.savefig(OUTPUT / name, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def rendering():
    fig = canvas(
        "Rendering: fixed radial mesh today",
        "Top views with fewer cells for clarity. Right: a possible future design, not implemented.",
    )
    left = panel(fig, [0.045, 0.23, 0.43, 0.59], (-1.2, 1.2), (-1.2, 1.2))
    left.set_aspect("equal")
    rings, sectors = 14, 48
    theta = np.arange(sectors) * 2 * np.pi / sectors
    points = np.array(
        [[0, 0], *[(r * np.cos(t), r * np.sin(t)) for r in (np.arange(1, rings + 1) / rings) ** 2 for t in theta]]
    )
    faces = [[0, 1 + s, 1 + (s + 1) % sectors] for s in range(sectors)]
    for ring in range(rings - 1):
        for s in range(sectors):
            a = 1 + ring * sectors + s
            b = 1 + ring * sectors + (s + 1) % sectors
            faces.extend([[a, a + sectors, b + sectors], [a, b + sectors, b]])
    left.triplot(points[:, 0], points[:, 1], faces, color=SAND, linewidth=0.65)
    left.plot(0, 0, "o", color=INK, ms=5)
    left.annotate(
        "Fixed origin", (0, 0), (0.24, -0.1), fontsize=10, color=INK, arrowprops={"arrowstyle": "->", "color": INK}
    )
    left.set_title("CURRENT • one connected disk", color=INK, fontsize=13, pad=8)
    right = panel(fig, [0.53, 0.23, 0.43, 0.59], (-9, 9), (-9, 9))
    right.set_aspect("equal")
    for extent, spacing, color in [(8, 2, SAND), (4, 1, BLUE), (2, 0.5, GREEN)]:
        right.add_patch(
            Rectangle((-extent, -extent), 2 * extent, 2 * extent, facecolor=PALE, edgecolor=color, zorder=extent * -1)
        )
        segments = []
        for v in np.arange(-extent, extent + spacing / 2, spacing):
            segments.extend([[(v, -extent), (v, extent)], [(-extent, v), (extent, v)]])
        right.add_collection(LineCollection(segments, colors=color, linewidths=0.75, zorder=-extent + 0.1))
    right.plot(0, 0, "o", color=INK)
    right.annotate("Camera", (0, 0), (2.4, -1), fontsize=10, color=INK, arrowprops={"arrowstyle": "->", "color": INK})
    right.set_title("FUTURE OPTION • geometric clipmap", color=INK, fontsize=13, pad=8)
    fig.text(
        0.06,
        0.13,
        "Spacing grows with distance from the origin.\nBuilt once; does not follow the camera.\nDefault: 1 km radius, 131,073 vertices.",
        fontsize=12,
        color=INK,
        linespacing=1.5,
    )
    fig.text(
        0.55,
        0.13,
        "Nested regular grids follow the camera.\nGrid spacing doubles at each level.\nNew strips update; level edges need stitching.",
        fontsize=12,
        color=INK,
        linespacing=1.5,
    )
    save(fig, "rendering_lod.png")


def collision():
    fig = canvas(
        "Collision: only the nearby ground enters physics",
        "Implemented today • render coverage and collision coverage have independent resolutions.",
    )
    ax = panel(fig, [0.04, 0.35, 0.49, 0.48], (-5.2, 8), (-5, 5))
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((-4, -4), 8, 8, facecolor="none", edgecolor=SAND, linestyle="--", linewidth=2))
    ax.add_patch(Rectangle((-1, -4), 8, 8, facecolor="#e1eff5", edgecolor=BLUE, linewidth=2))
    for v in np.arange(-4, 4.1, 1):
        ax.plot([-1, 7], [v, v], color=BLUE, alpha=0.25, lw=0.7)
        ax.plot([v + 3, v + 3], [-4, 4], color=BLUE, alpha=0.25, lw=0.7)
    ax.plot(0, 0, "x", color=SAND, ms=9)
    ax.plot(3, 0, "o", color=GREEN, ms=10)
    arrow(ax, (0.3, 0), (2.5, 0))
    ax.text(3, 0.7, "G1 / new center", ha="center", color=INK, fontsize=11)
    ax.text(-3.9, -4.8, "Dashed: old patch", color=SAND, fontsize=10)
    ax.text(2, -4.8, "Blue: new 8 x 8 m patch", color=BLUE, fontsize=10)
    fig.text(0.55, 0.73, "Recenter before a simulation frame", fontsize=14, weight="bold", color=INK)
    fig.text(
        0.55,
        0.43,
        "1. G1 moves > 2 m along X or Y.\n2. Snap patch center to the 0.1 m lattice.\n3. Resample all 6,561 vertices.\n4. Refit BVH and update shape bounds.\n5. Notify the rigid solver.",
        fontsize=12,
        color=INK,
        linespacing=1.7,
    )
    flow = panel(fig, [0.05, 0.12, 0.90, 0.20], (0, 12), (0, 2))
    box(flow, (0.05, 0.4), 3.1, 1.2, "G1 collision shapes\n+ local terrain triangles")
    box(flow, (4.3, 0.4), 3.1, 1.2, "Newton collision pipeline\nContact points / normals")
    box(flow, (8.6, 0.4), 3.1, 1.2, "MuJoCo rigid solver\nNext G1 state")
    arrow(flow, (3.15, 1), (4.3, 1))
    arrow(flow, (7.4, 1), (8.6, 1))
    fig.text(
        0.06,
        0.055,
        "Physics: 600 Hz • policy: 50 Hz • rocks: visual only • diagram grid simplified; terrain outside patch still renders",
        fontsize=10,
        color=INK,
    )
    save(fig, "collision_patch.png")


def integration():
    fig = canvas(
        "Planned: one terrain, separate rendering and physics",
        "Concept only • the desert prototype does not yet run beta2 MPM. Zone sizes are illustrative.",
    )
    ax = panel(fig, [0.045, 0.31, 0.46, 0.52], (-6, 6), (-5, 5))
    ax.set_aspect("equal")
    for xy, width, height, color in [
        ((-5.7, -4.7), 11.4, 9.4, "#ead8ba"),
        ((-3.4, -3.4), 6.8, 6.8, "#c3ded8"),
        ((-2.4, -2.4), 4.8, 4.8, "#8ebeb3"),
    ]:
        ax.add_patch(Rectangle(xy, width, height, facecolor=color, edgecolor="white", linewidth=2))
    for x in np.linspace(-2, 2, 11):
        for y in np.linspace(-2, 2, 11):
            ax.add_patch(Circle((x, y), 0.035, color=GREEN))
    ax.plot(0, 0, "o", color=INK, ms=9)
    ax.text(0, 0.45, "G1", ha="center", color=INK, weight="bold")
    ax.text(0, -1.6, "Simulated MPM terrain", ha="center", color=INK, fontsize=11)
    ax.text(0, -2.95, "Fixed supporting terrain", ha="center", color=INK, fontsize=11)
    ax.text(0, 4, "Background terrain / saved heights", ha="center", color=INK, fontsize=11)
    fig.text(0.54, 0.73, "Physics follows G1", fontsize=14, weight="bold", color=INK)
    fig.text(
        0.54,
        0.61,
        "MPM handles deformable sand contact.\nKeep rigid bedrock / selected rock contacts.\nRemove duplicate rigid sand-top contact.",
        fontsize=12,
        color=INK,
        linespacing=1.5,
    )
    fig.text(0.54, 0.51, "Rendering follows the camera", fontsize=14, weight="bold", color=INK)
    fig.text(
        0.54,
        0.36,
        "Near G1: reconstructed MPM surface.\nOutside: base terrain + saved heights.\nMatch edges; hide the old sand surface\nwhere the MPM surface is drawn.",
        fontsize=12,
        color=INK,
        linespacing=1.5,
    )
    flow = panel(fig, [0.05, 0.105, 0.90, 0.18], (0, 12), (0, 2))
    box(flow, (0.05, 0.3), 3.1, 1.35, "G1 rigid dynamics\n+ policy targets", BLUE)
    box(flow, (4.3, 0.3), 3.1, 1.35, "Moving MPM patch\nParticle state", GREEN)
    box(flow, (8.6, 0.3), 3.1, 1.35, "Terrain tile cache\nHeight + reference depth", SAND)
    arrow(flow, (3.15, 0.98), (4.3, 0.98), both=True)
    arrow(flow, (7.4, 0.98), (8.6, 0.98), both=True)
    fig.text(
        0.06,
        0.045,
        "Retained particles keep state. Reconstructed terrain restores height / volume, not full stress history.",
        fontsize=11,
        color=INK,
    )
    save(fig, "mpm_integration.png")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    rendering()
    collision()
    integration()
