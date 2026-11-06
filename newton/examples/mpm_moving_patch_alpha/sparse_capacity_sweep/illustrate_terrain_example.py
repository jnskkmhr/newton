# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Illustrate block-boundary effects for an idealized terrain bed, without a GPU."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

OUT = Path(__file__).resolve().parent / "images" / "06_concrete_terrain_example.png"
CASES = [
    ("A. Positive coordinates", (0, 0, 0), (59, 59, 9)),
    ("B. Centered in X and Y", (-30, -30, 0), (29, 29, 9)),
    ("C. Also crosses Z = 0", (-30, -30, -1), (29, 29, 8)),
]
BLUE = "#28659b"
ORANGE = "#e5a350"
RED = "#b74549"
INK = "#243448"


def counts(lo, hi, block):
    return np.floor_divide(hi, block) - np.floor_divide(lo, block) + 1


def view(ax, lo, hi, vertical_axis, title):
    ymin, ymax = (-40, 72) if vertical_axis == 1 else (-8, 16)
    ax.set_xlim(-40, 72)
    ax.set_ylim(ymin, ymax)
    ax.add_patch(Rectangle((-40, ymin), 40, ymax - ymin, facecolor="#eaf0f8", zorder=0))
    ax.add_patch(Rectangle((0, ymin), 72, ymax - ymin, facecolor="#f3f6fa", zorder=0))
    ax.add_patch(
        Rectangle(
            (lo[0], lo[vertical_axis]),
            hi[0] - lo[0] + 1,
            hi[vertical_axis] - lo[vertical_axis] + 1,
            facecolor=ORANGE,
            edgecolor="#9c671c",
            linewidth=1.8,
            alpha=0.9,
            zorder=2,
        )
    )
    for value in range(-40, 73, 8):
        ax.axvline(value, color=BLUE, lw=0.6, alpha=0.45, zorder=3)
    for value in range(ymin, ymax + 1, 8):
        ax.axhline(value, color=BLUE, lw=0.6, alpha=0.45, zorder=3)
    ax.axvline(0, color=RED, lw=2.2, zorder=4)
    ax.axhline(0, color=RED, lw=2.2, zorder=4)
    ax.set_xticks([-32, 0, 32, 64])
    ax.set_yticks([-32, 0, 32, 64] if vertical_axis == 1 else [-8, 0, 8, 16])
    ax.set_xlabel("X voxel coordinate", fontsize=10)
    ax.set_ylabel("Y voxel coordinate" if vertical_axis == 1 else "Z voxel coordinate", fontsize=10)
    ax.set_title(title, fontsize=12, color=INK)
    ax.tick_params(labelsize=9)
    if vertical_axis == 1:
        ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color("#a9b7c5")


def main():
    fig = plt.figure(figsize=(15, 10.5), facecolor="white")
    grid = fig.add_gridspec(3, 3, height_ratios=[3.1, 1.15, 1.25], hspace=0.62, wspace=0.26)
    fig.suptitle("Same bed size, different NanoVDB node counts", fontsize=22, weight="bold", color=INK, y=0.985)
    fig.text(
        0.5,
        0.947,
        "3 m x 3 m x 0.5 m bed  |  0.05 m voxels  |  60 x 60 x 10 = 36,000 occupied cells in every case",
        ha="center",
        fontsize=12,
        color=INK,
    )
    for col, (title, low, high) in enumerate(CASES):
        lo, hi = np.array(low), np.array(high)
        view(fig.add_subplot(grid[0, col]), lo, hi, 1, title + "\nTop view: X-Y")
        view(fig.add_subplot(grid[1, col]), lo, hi, 2, "Side view: X-Z (vertical scale enlarged)")
        ax = fig.add_subplot(grid[2, col])
        ax.axis("off")
        leaves, lower, upper = (counts(lo, hi, block) for block in (8, 128, 4096))
        expected = ((128, 1, 1), (128, 4, 4), (192, 8, 8))[col]
        assert tuple(int(np.prod(n)) for n in (leaves, lower, upper)) == expected
        lines = [f"x: {lo[0]}..{hi[0]}   y: {lo[1]}..{hi[1]}   z: {lo[2]}..{hi[2]}"]
        for label, n in (("Leaves", leaves), ("Lower nodes", lower), ("Upper nodes", upper)):
            lines.append(f"{label}: {n[0]} x {n[1]} x {n[2]} = {np.prod(n)}")
        ax.text(
            0.5,
            0.9,
            "\n".join(lines),
            ha="center",
            va="top",
            fontsize=12,
            color=INK,
            linespacing=1.7,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f2f6fa", "edgecolor": "#d5dee8"},
        )
    fig.legend(
        handles=[
            Patch(facecolor=ORANGE, label="Occupied bed"),
            Patch(facecolor=BLUE, label="Leaf boundaries: every 8 voxels"),
            Patch(facecolor=RED, label="Zero: shared lower/upper block boundary"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.025,
        "Lower blocks span 128 voxels; upper blocks span 4096. Their other boundaries lie outside these views.",
        ha="center",
        fontsize=11,
        color=INK,
    )
    fig.text(
        0.5,
        0.006,
        "C shifts the idealized bed down by one voxel, preserving its size. This illustrates indexing, not simulated floor penetration.",
        ha="center",
        fontsize=10,
        color=INK,
    )
    fig.subplots_adjust(top=0.885, bottom=0.14, left=0.06, right=0.985)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
