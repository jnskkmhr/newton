# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Generate CPU-only diagrams for the implicit-MPM walkthrough."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "images"
INK = "#25364a"
BLUE = "#286ba1"
GREEN = "#207f69"
ORANGE = "#cc7927"
plt.rcParams.update({"font.size": 12, "text.color": INK, "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, title, detail, color="#e8f1f8"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", facecolor=color, edgecolor=BLUE, lw=1.4
        )
    )
    ax.text(x + w / 2, y + h * 0.7, title, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, detail, ha="center", va="center", fontsize=10, linespacing=1.4)


def arrow(ax, start, end, both=False):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="<->" if both else "-|>", mutation_scale=16, lw=1.8, color=GREEN)
    )


def pipeline():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set(xlim=(0, 14), ylim=(0, 10))
    ax.axis("off")
    ax.set_title("One implicit-MPM physics step", fontsize=22, pad=15)
    stages = [
        (9.0, "1. Geometry and particle bins", "Active particles -> cells -> sparse topology -> particle quadrature"),
        (7.55, "2. Spaces and work arrays", "Velocity, strain, and collision spaces on the active material domain"),
        (6.1, "3. Rasterize colliders", "Distance, normal, collider velocity, friction, adhesion, collider ID"),
        (
            4.65,
            "4. Unconstrained grid velocity",
            "Particle mass and momentum + gravity; APIC contribution when enabled",
        ),
        (
            3.2,
            "5. Material and rigid-coupling operators",
            "Elastic compliance + plastic yield/strain system + collider rigidity",
        ),
        (
            1.75,
            "6. Coupled iterative solve",
            "Contact update <-> rheology update; repeat until tolerance / iteration limit",
        ),
        (
            0.3,
            "7. Update particles and save state",
            "Grid-to-particle motion, APIC gradient, elastic/plastic strain, stress",
        ),
    ]
    for i, (y, title, detail) in enumerate(stages):
        box(ax, 0.5, y, 9.0, 0.95, title, detail, "#fbecd9" if i == 2 else "#e8f1f8")
        if i < len(stages) - 1:
            arrow(ax, (5, y - 0.04), (5, y - 0.44))
    ax.text(10.1, 8.9, "Selection rule:\nmaterial occupancy", fontsize=12, va="center", weight="bold", color=BLUE)
    ax.text(
        10.1,
        6.5,
        "Adds contact data.\nDoes not select the\nentire solve region.",
        fontsize=12,
        va="center",
        color=ORANGE,
    )
    ax.text(10.1, 2.2, "Material response\nexists away from\nrobot contact too.", fontsize=12, va="center", color=GREEN)
    fig.savefig(OUT / "01_implicit_mpm_pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def masks():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set(xlim=(-0.7, 13.5), ylim=(-1.3, 6.1), aspect="equal")
    ax.axis("off")
    ax.set_title("Material-active cells and contact-active nodes are different", fontsize=19, pad=17)
    for i in range(10):
        for j in range(4):
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor="#dfedf8", edgecolor=BLUE, lw=0.7))
    ax.add_patch(Rectangle((4.0, 4.0), 2.0, 1.1, facecolor="#708399", edgecolor=INK))
    ax.text(5, 4.55, "Robot foot", ha="center", va="center", color="white", weight="bold")
    ax.plot([-0.3, 10.3], [-0.08, -0.08], color=INK, lw=4)
    ax.scatter(np.arange(11), np.zeros(11), s=55, color=GREEN, zorder=5)
    ax.scatter([4, 5, 6], [4, 4, 4], s=90, color=ORANGE, zorder=5)
    ax.annotate(
        "Robot contact nodes",
        xy=(6, 4),
        xytext=(10.6, 4.4),
        color=ORANGE,
        fontsize=11,
        arrowprops={"arrowstyle": "->", "color": ORANGE},
        va="center",
    )
    ax.annotate(
        "No robot contact here.\nStill part of material solve.",
        xy=(1.5, 2.5),
        xytext=(0.1, 5.55),
        fontsize=11,
        arrowprops={"arrowstyle": "->", "color": BLUE},
        ha="left",
    )
    ax.annotate(
        "Ground contact nodes",
        xy=(9, 0),
        xytext=(10.6, 0.9),
        fontsize=11,
        color=GREEN,
        arrowprops={"arrowstyle": "->", "color": GREEN},
    )
    ax.text(10.6, 2.7, "Blue cells:\nmaterial-active\nparticle region", fontsize=11, color=BLUE)
    ax.text(
        5,
        -0.65,
        "Static ground is a collider too. Removing robot contact does not remove ground contact.",
        ha="center",
        fontsize=10,
    )
    ax.text(
        5,
        -1.05,
        "Schematic 2D layout; actual node positions and activation distances depend on the configured spaces.",
        ha="center",
        fontsize=9.5,
    )
    fig.savefig(OUT / "02_material_vs_contact.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def constraints():
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax in axes:
        ax.set(xlim=(0, 7), ylim=(0, 8))
        ax.axis("off")
    axes[0].set_title("No collider-contact constraints", fontsize=16, pad=16)
    axes[1].set_title("With collider-contact constraints", fontsize=16, pad=16)
    for ax in axes:
        box(ax, 0.5, 6.25, 6, 1, "Same material domain", "Particle occupancy, mass, gravity, material parameters")
        arrow(ax, (3.5, 6.2), (3.5, 5.65))
        box(ax, 0.5, 4.6, 6, 1, "Material response", "Elasticity + yield / plastic flow + grid momentum")
    box(
        axes[0],
        0.5,
        2.65,
        6,
        1.2,
        "Material iteration",
        "Update stresses / strains and velocities\nNo collider impulse is imposed",
    )
    arrow(axes[0], (3.5, 4.55), (3.5, 3.9))
    box(
        axes[1],
        0.5,
        2.65,
        6,
        1.2,
        "Contact update",
        "Constrain relative motion near colliders\nNormal support + friction / adhesion",
        "#fbecd9",
    )
    arrow(axes[1], (3.5, 4.55), (3.5, 3.9), both=True)
    axes[1].text(5.1, 4.22, "Iterate together", fontsize=10, color=GREEN)
    for ax in axes:
        arrow(ax, (3.5, 2.6), (3.5, 1.95))
        box(ax, 0.5, 0.7, 6, 1.2, "Grid-to-particle update", "Every active material region still participates")
    fig.suptitle("Contact changes the equations, not the particle-occupancy selection", fontsize=19, y=0.985)
    fig.text(
        0.5,
        0.035,
        "Contact can change stresses and motion beyond touching nodes through the shared material equations.",
        ha="center",
        fontsize=12,
        color=INK,
    )
    fig.subplots_adjust(top=0.86, bottom=0.09, wspace=0.08)
    fig.savefig(OUT / "03_with_without_contact.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pipeline()
    masks()
    constraints()
