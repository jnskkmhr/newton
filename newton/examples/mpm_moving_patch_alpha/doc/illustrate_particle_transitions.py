# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Generate the planned moving-patch particle transition flow on the CPU."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    """Draw the seven steps performed between coupled physics steps."""
    steps = [
        ("1. Move the patch center", "Follow the configured robot body.\nKeep simulated terrain size and boundary width fixed."),
        ("2. Keep particles that remain in simulated terrain", "Preserve their positions, velocities and material state."),
        ("3. Save outgoing terrain heights", "Update background heights on the GPU\nbefore fixing or recycling outgoing particles."),
        ("4. Fix outgoing particles in boundary terrain", "Set mass, velocity and APIC velocity gradient to zero.\nRecord the motion removed by fixing these particles."),
        ("5. Activate boundary particles entering simulated terrain", "Restore positive material mass.\nInitialize material state using the configured defaults."),
        ("6. Recycle particles outside both particle layers", "Generate incoming particles from saved heights.\nInitialize velocity and material state using configured defaults."),
        ("7. Refresh solver data", "Update derived data and invalidate stale numerical caches.\nPreserve material history of retained dynamic particles."),
    ]
    fig, ax = plt.subplots(figsize=(11, 12.5), facecolor="white")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.91, bottom=0.035)
    ax.set(xlim=(0, 10), ylim=(-0.7, 14))
    ax.axis("off")
    fig.text(0.07, 0.965, "Particle transitions as the robot moves", fontsize=21, weight="bold", color="#243447")
    fig.text(0.07, 0.934, "Run between complete coupled physics steps", fontsize=13, color="#526779")
    for i, (title, detail) in enumerate(steps):
        y = 12.1 - i * 1.85
        ax.add_patch(FancyBboxPatch((0.3, y), 9.4, 1.42, boxstyle="round,pad=0.04,rounding_size=0.12", facecolor="#eaf4fa" if i != 2 else "#edf3e5", edgecolor="#527c96", lw=1.4))
        ax.text(0.55, y + 1.09, title, fontsize=13, weight="bold", color="#243447", va="center")
        ax.text(0.55, y + 0.47, detail, fontsize=11.5, color="#243447", va="center", linespacing=1.5)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((5, y - 0.055), (5, y - 0.39), arrowstyle="-|>", mutation_scale=17, color="#527c96", lw=1.7))
    ax.text(5, -0.32, "Next coupled physics step", ha="center", fontsize=13, weight="bold", color="#243447")
    ax.add_patch(FancyArrowPatch((5, 0.94), (5, 0.02), arrowstyle="-|>", mutation_scale=17, color="#527c96", lw=1.7))
    out = Path(__file__).resolve().parent / "images" / "particle_transitions.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
