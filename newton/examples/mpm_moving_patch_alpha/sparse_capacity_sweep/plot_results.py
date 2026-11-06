# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Summarize measured capacity sweeps without masking overflow failures."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    rows = []
    for group in ("scaling", "node_ablation", "validation"):
        for path in sorted((ROOT / group).glob("*.json")):
            data = json.loads(path.read_text())
            if "case" not in data:
                continue
            case = data["case"]
            profile = case["name"].split("_", 1)[1]
            limits = data.get("resolved_limits", {})
            rows.append(
                {
                    "group": group,
                    "width_m": case["width"],
                    "profile": profile,
                    "repeat": data.get("repeat", 0),
                    "status": data["status"],
                    "mean_ms": data.get("mean_ms"),
                    "std_ms": data.get("std_ms"),
                    "steady_gib": data.get("steady_delta_bytes", 0) / 2**30 if data.get("steady_delta_bytes") else None,
                    "peak_gib": data.get("peak_delta_bytes", 0) / 2**30,
                    "particles": data.get("particle_count"),
                    "active_capacity": data.get("capacity"),
                    "leaf_capacity": limits.get("max_leaf_node_count"),
                    "lower_capacity": limits.get("max_lower_node_count"),
                    "upper_capacity": limits.get("max_upper_node_count"),
                    "scene_validation": data.get("scene_validation"),
                    "error": data.get("error", ""),
                }
            )
    if not rows:
        raise RuntimeError("No sweep results")
    with (ROOT / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for profile, label in (
        ("default", "Default node limits"),
        ("all2", "2x initial nodes"),
        ("all4", "4x initial nodes"),
    ):
        samples = sorted(
            [r for r in rows if r["group"] == "scaling" and r["profile"] == profile and r["status"] == "ok"],
            key=lambda r: r["width_m"],
        )
        if not samples:
            continue
        x = [r["width_m"] for r in samples]
        axes[0].errorbar(
            x, [r["mean_ms"] for r in samples], yerr=[r["std_ms"] for r in samples], marker="o", capsize=3, label=label
        )
        axes[1].plot(x, [r["steady_gib"] for r in samples], marker="o", label=label)
    for ax in axes:
        ax.set_xlabel("Terrain side length [m]")
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Physics step time [ms]")
    axes[1].set_ylabel("VRAM above baseline [GiB]")
    fig.suptitle("Sparse MPM node-capacity tuning")
    fig.savefig(ROOT / "terrain_scaling.png", dpi=180)
    plt.close(fig)
    ablation = [r for r in rows if r["group"] == "node_ablation" and r["status"] == "ok"]
    if ablation:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
        labels = []
        for i, profile in enumerate(("default", "leaf2", "lower2", "upper2", "all2", "all4")):
            samples = [r for r in ablation if r["profile"] == profile]
            if not samples:
                continue
            labels.append(profile)
            axes[0].errorbar(
                i,
                np.mean([r["mean_ms"] for r in samples]),
                yerr=np.std([r["mean_ms"] for r in samples]),
                fmt="o",
                capsize=4,
            )
            axes[1].plot(i, np.mean([r["steady_gib"] for r in samples]), "o")
        for ax in axes:
            ax.set_xticks(range(len(labels)), labels, rotation=25)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Mean step time ± run SD [ms]")
        axes[1].set_ylabel("VRAM above baseline [GiB]")
        fig.suptitle("Node-capacity sweep: 3 m terrain")
        fig.savefig(ROOT / "node_ablation.png", dpi=180)
        plt.close(fig)
    print(f"Saved {len(rows)} cases to summary.csv")


if __name__ == "__main__":
    main()
