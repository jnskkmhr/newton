# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render explanatory sparse-MPM diagrams with Matplotlib; no GPU is used."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

OUT = Path(__file__).resolve().parent / "images"
BLUE = "#2563a6"
LIGHT = "#deebf7"
ORANGE = "#de8c32"
INK = "#203044"
GREEN = "#29856a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK, "axes.titlecolor": INK})


def box(ax, x, y, w, h, title, detail, color=LIGHT):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=color, edgecolor=BLUE, linewidth=1.4
        )
    )
    ax.text(x + w / 2, y + h * 0.73, title, ha="center", va="center", weight="bold", fontsize=14)
    ax.text(x + w / 2, y + h * 0.35, detail, ha="center", va="center", fontsize=11, linespacing=1.5)


def arrow(ax, start, end, color=BLUE):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17, linewidth=1.8, color=color))


def pipeline():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set(xlim=(0, 14), ylim=(0, 7))
    ax.axis("off")
    ax.set_title("Where NanoVDB fits in one MPM step", fontsize=21, pad=18)
    box(ax, 0.25, 3.8, 3.1, 1.8, "1. Material particles", "Position, velocity, mass\nDeformation and stress", "#fbecd8")
    box(
        ax,
        5.05,
        3.8,
        3.7,
        1.8,
        "2. Sparse grid topology",
        "Particle positions identify cells\nNanoVDB indexes occupied regions",
    )
    box(
        ax,
        10.15,
        3.8,
        3.6,
        1.8,
        "3. Grid physics",
        "Particle-to-grid transfer\nImplicit material solve\nCollider interaction",
        "#deeee8",
    )
    arrow(ax, (3.4, 4.7), (5.0, 4.7))
    arrow(ax, (8.8, 4.7), (10.1, 4.7))
    ax.text(4.2, 5.05, "Rebuild", ha="center", fontsize=11)
    ax.text(9.45, 5.05, "Index + bin", ha="center", fontsize=11)
    box(
        ax,
        4.5,
        0.6,
        5.0,
        1.65,
        "4. Update particles",
        "Transfer grid results back\nUpdate motion and material state",
        "#fbecd8",
    )
    arrow(ax, (11.95, 3.75), (9.55, 1.5), GREEN)
    arrow(ax, (4.45, 1.5), (1.8, 3.75), GREEN)
    ax.text(2.3, 1.45, "Next step", ha="center", fontsize=12, color=GREEN)
    ax.text(
        7,
        6.4,
        "Particles store material history. The background grid supports the numerical solve.",
        ha="center",
        fontsize=13,
    )
    ax.text(
        7,
        0.15,
        "NanoVDB is the spatial index. MPM field arrays and particles consume memory too.",
        ha="center",
        fontsize=12,
    )
    fig.savefig(OUT / "01_mpm_pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def hierarchy():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set(xlim=(0, 14), ylim=(0, 8))
    ax.axis("off")
    ax.set_title("NanoVDB: a hierarchy of spatial blocks", fontsize=21, pad=16)
    levels = [
        (6.4, "Root", "Directory of occupied large regions", "No separate root limit in this MPM config"),
        (
            4.8,
            "Upper internal node",
            "32 x 32 x 32 lower-node slots",
            "max_upper_node_count\nSpan: 4096 voxels per axis = 204.8 m at h = 0.05 m",
        ),
        (
            3.2,
            "Lower internal node",
            "16 x 16 x 16 leaf-node slots",
            "max_lower_node_count\nSpan: 128 voxels per axis = 6.4 m at h = 0.05 m",
        ),
        (
            1.6,
            "Leaf node",
            "8 x 8 x 8 voxel positions",
            "max_leaf_node_count\nSpan: 8 voxels per axis = 0.4 m at h = 0.05 m",
        ),
    ]
    for i, (y, title, detail, explanation) in enumerate(levels):
        box(ax, 0.4, y, 5.2, 1.15, title, detail)
        ax.text(6.25, y + 0.58, explanation, va="center", fontsize=12, linespacing=1.65)
        if i < len(levels) - 1:
            arrow(ax, (3, y - 0.02), (3, y - 0.43))
    ax.text(3, 0.75, "Active voxel / cell", ha="center", color=ORANGE, weight="bold", fontsize=14)
    arrow(ax, (3, 1.57), (3, 1.02), ORANGE)
    ax.text(
        6.25,
        0.8,
        "max_active_cell_count: active voxels, NOT leaf blocks\nVoxel width h sets the spatial resolution.",
        va="center",
        fontsize=12,
        linespacing=1.6,
    )
    fig.savefig(OUT / "02_nanovdb_hierarchy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def occupancy():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.7))
    compact = [(x, y) for x in range(10, 14) for y in range(10, 14)]
    scattered = [(x, y) for x in (4, 12, 20, 28) for y in (4, 12, 20, 28)]
    for ax, points, title in zip(
        axes,
        (compact, scattered),
        ("Compact: 16 active voxels, 1 leaf", "Scattered: 16 active voxels, 16 leaves"),
        strict=True,
    ):
        leaves = {(x // 8, y // 8) for x, y in points}
        for x, y in leaves:
            ax.add_patch(Rectangle((x * 8, y * 8), 8, 8, facecolor=LIGHT))
        for x, y in points:
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=ORANGE))
        for n in range(33):
            ax.plot([0, 32], [n, n], color=BLUE if n % 8 == 0 else "#cbd5df", lw=1.1 if n % 8 == 0 else 0.3)
            ax.plot([n, n], [0, 32], color=BLUE if n % 8 == 0 else "#cbd5df", lw=1.1 if n % 8 == 0 else 0.3)
        ax.set(xlim=(0, 32), ylim=(0, 32), aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=13, pad=12)
    fig.suptitle("Equal cell counts can require different node capacities", fontsize=19, y=0.98)
    fig.legend(
        handles=[
            Patch(facecolor=ORANGE, label="Active voxel"),
            Patch(facecolor=LIGHT, edgecolor=BLUE, label="Leaf block containing active voxels"),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.06),
    )
    fig.text(
        0.5,
        0.025,
        "2D slice for illustration: each drawn leaf is 8 x 8 cells; a 3D leaf is 8 x 8 x 8.",
        ha="center",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.84, bottom=0.18, wspace=0.15)
    fig.savefig(OUT / "03_compact_vs_scattered.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def capacity():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")
    ax.set_title("Used topology and reserved capacity are different", fontsize=20, pad=15)
    scenarios = [
        (3.25, 5, 8, "Fits: 5 leaves used, 8 reserved"),
        (1.6, 9, 8, "Overflow: 9 leaves needed, only 8 reserved"),
    ]
    for y, used, reserved, label in scenarios:
        ax.text(0.2, y + 0.9, label, weight="bold", fontsize=13)
        for i in range(max(used, reserved)):
            color = BLUE if i < min(used, reserved) else ("#dc665a" if i >= reserved else "#edf0f4")
            ax.add_patch(Rectangle((0.3 + i * 1.18, y), 1.0, 0.62, facecolor=color, edgecolor=INK, linewidth=0.8))
        if used > reserved:
            ax.text(10.3, y - 0.35, "Capacity exceeded", ha="center", color="#af352a", fontsize=11)
    ax.text(
        6,
        0.4,
        "Each box is one leaf-node slot, not one voxel. The same budget principle applies to all four limits.",
        ha="center",
        fontsize=11,
    )
    fig.savefig(OUT / "04_capacity_vs_usage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pipeline()
    hierarchy()
    occupancy()
    capacity()
