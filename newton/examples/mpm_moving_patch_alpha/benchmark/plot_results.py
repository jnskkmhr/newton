# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Aggregate raw benchmark observations and generate comparison figures."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
GIB = 1024**3
COLORS = {"fixed": "#4477aa", "sparse": "#ee7733", "patch": "#228833"}


def load():
    groups = {}
    for path in sorted((ROOT / "raw").glob("*.json")):
        r = json.loads(path.read_text())
        groups.setdefault(r["case"]["name"], []).append(r)
    summaries = {}
    for name, runs in groups.items():
        good = [r for r in runs if r["status"] == "ok"]
        row = {**runs[0]["case"], "status": "ok" if good else "failed", "repeats": len(good)}
        if not good:
            row["error"] = runs[0].get("error", "unknown failure")
        else:
            timings = np.concatenate([r["wall_ms"] for r in good])
            gpu = np.concatenate([r["gpu_ms"] for r in good])
            row.update(
                mean_ms=float(np.mean(timings)),
                p95_ms=float(np.percentile(timings, 95)),
                gpu_mean_ms=float(np.mean(gpu)),
                run_mean_min_ms=min(r["mean_ms"] for r in good),
                run_mean_max_ms=max(r["mean_ms"] for r in good),
                steady_gib=float(np.mean([r["steady_delta_bytes"] / GIB for r in good])),
                peak_gib=max(r["peak_delta_bytes"] / GIB for r in good),
                warp_pool_peak_gib=max(r["warp_pool_peak_bytes"] / GIB for r in good),
                particles=good[0]["particle_count"],
                capacity=good[0]["reserved_cell_capacity"],
                active_min=min(min(r["active_cells_samples"]) for r in good),
                active_max=max(max(r["active_cells_samples"]) for r in good),
                drawn_particles=good[0].get("drawn_particles"),
                final_x_mean=float(np.mean([r["final_pose"][0][0] for r in good])),
                final_z_mean=float(np.mean([r["final_pose"][0][2] for r in good])),
                window_shifts=[r.get("window_shifts") for r in good],
                validations=[r["scene_validation"] for r in good],
            )
        summaries[name] = row
    return summaries


def finish(fig, filename, note):
    fig.text(0.02, 0.015, note, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(ROOT / filename, dpi=180, facecolor="white")
    plt.close(fig)


def bars(ax, rows, labels, memory=False):
    x = np.arange(len(rows))
    fields = ("steady_gib", "peak_gib") if memory else ("mean_ms", "p95_ms")
    names = ("Steady-state", "Sampled peak") if memory else ("Mean", "p95")
    for i, (field, label) in enumerate(zip(fields, names, strict=True)):
        values = [r.get(field, 0) for r in rows]
        bars = ax.bar(x + (i - 0.5) * 0.34, values, width=0.34, label=label, color=("#4477aa", "#ee7733")[i])
        ax.bar_label(
            bars,
            labels=[f"{v:.2f}" if r["status"] == "ok" else "Failed" for v, r in zip(values, rows, strict=True)],
            fontsize=9,
            padding=3,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("VRAM above pre-run baseline [GiB]" if memory else "Completed call time [ms]")
    maximum = max(row.get(field, 0) for row in rows for field in fields)
    ax.set_ylim(0, max(maximum * 1.35, 1.0))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)


def main():
    data = load()
    if not data:
        raise RuntimeError("No raw benchmark results")
    (ROOT / "summary.json").write_text(json.dumps(data, indent=2))
    fields = [
        "name",
        "status",
        "repeats",
        "grid",
        "width",
        "motion",
        "ratio",
        "mean_ms",
        "p95_ms",
        "gpu_mean_ms",
        "run_mean_min_ms",
        "run_mean_max_ms",
        "steady_gib",
        "peak_gib",
        "warp_pool_peak_gib",
        "particles",
        "capacity",
        "active_min",
        "active_max",
        "drawn_particles",
        "final_x_mean",
        "final_z_mean",
        "error",
    ]
    with (ROOT / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data.values())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    rows = [data[f"{grid}_3m"] for grid in ("fixed", "sparse")]
    bars(axes[0], rows, ["Fixed", "Sparse"])
    bars(axes[1], rows, ["Fixed", "Sparse"], memory=True)
    fig.suptitle("1. Fixed vs sparse: identical 3 x 3 x 0.5 m particle bed", fontsize=15)
    finish(
        fig,
        "01_grid_comparison.png",
        "Headless, CUDA graphs; 200 warmup + 200 measured steps/run. p95 pools completed steps across runs.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    rows = [data[f"render_{ratio:.3f}"] for ratio in (1.0, 2 / 3, 0.0)]
    labels = ["Full footprint\nratio 1", "2 x 2 m visible\nratio 2/3", "No sand draw\nratio 0"]
    bars(axes[0], rows, labels)
    bars(axes[1], rows, labels, memory=True)
    fig.suptitle("2. Rendering only: identical sparse-MPM snapshot, 960 x 720", fontsize=15)
    finish(
        fig,
        "02_rendering_overhead.png",
        "Includes compaction, count readback, and GPU/OpenGL completion. Rigid bodies and ground are drawn in every case.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for grid in ("fixed", "sparse"):
        widths, times, p95, memories = [], [], [], []
        for width in (3, 6, 10):
            row = data[f"{grid}_{width}m"]
            if row["status"] == "ok":
                widths.append(width)
                times.append(row["mean_ms"])
                p95.append(row["p95_ms"])
                memories.append(row["steady_gib"])
            else:
                for ax in axes:
                    ax.annotate(
                        f"{grid}: failed at {width} m",
                        xy=(0.04, 0.93 if grid == "fixed" else 0.84),
                        xycoords="axes fraction",
                        fontsize=9,
                        color=COLORS[grid],
                    )
        axes[0].plot(widths, times, "o-", color=COLORS[grid], label=f"{grid}: mean")
        axes[0].plot(widths, p95, "o--", alpha=0.65, color=COLORS[grid], label=f"{grid}: p95")
        axes[1].plot(widths, memories, "o-", color=COLORS[grid], label=grid)
    patch = data["patch_3m"]
    if patch["status"] == "ok":
        axes[0].plot([3, 6, 10], [patch["mean_ms"]] * 3, "-", color=COLORS["patch"], label="3 m patch: mean")
        axes[0].plot([3, 6, 10], [patch["p95_ms"]] * 3, "--", color=COLORS["patch"], label="3 m patch: p95")
        axes[1].plot([3, 6, 10], [patch["steady_gib"]] * 3, "-", color=COLORS["patch"], label="3 m patch")
    axes[0].set_ylabel("Completed physics step [ms]")
    axes[1].set_ylabel("Steady VRAM above baseline [GiB]")
    for ax in axes:
        ax.set_xlabel("Full-terrain / conceptual operating-area side length [m]")
        ax.set_xticks([3, 6, 10])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, frameon=False)
    fig.suptitle("3. Terrain-size scaling at fixed 0.05 m voxel resolution", fontsize=15)
    finish(
        fig,
        "03_terrain_scaling.png",
        "Patch line reuses one constant-size workload; it is not three measured traversals. All cases use the same short +X reference.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = [f"overhead_{scene}_{motion}" for motion in ("stationary", "moving") for scene in ("original", "patch")]
    rows = [data[name] for name in names]
    labels = ["Sparse\nstationary", "Patch\nstationary", "Sparse\nmoving", "Patch\nmoving"]
    bars(axes[0], rows, labels)
    bars(axes[1], rows, labels, memory=True)
    fig.suptitle("4. Moving-patch overhead: identical 2 x 2 x 0.5 m allocation", fontsize=15)
    finish(
        fig,
        "04_moving_patch_overhead.png",
        "Controller speed is zero for stationary cases. Moving cases use the shared +X controller; physics and recycling are timed together.",
    )

    lines = [
        "# MPM benchmark results",
        "",
        "Generated from the raw JSON files by `plot_results.py`.",
        "",
        "| Case | Runs | Mean ms | p95 ms | Steady GiB | Peak GiB | Particles | Active cells (sampled range) | Capacity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data.values():
        if row["status"] == "ok":
            lines.append(
                f"| {row['name']} | {row['repeats']} | {row['mean_ms']:.2f} | {row['p95_ms']:.2f} | "
                f"{row['steady_gib']:.2f} | {row['peak_gib']:.2f} | {row['particles']:,} | "
                f"{row['active_min']:,}-{row['active_max']:,} | {row['capacity']:,} |"
            )
        else:
            lines.append(f"| {row['name']} | 0 | Failed | - | - | - | - | - | - |")
    lines += [
        "",
        "## Validation and tracking",
        "",
        "The force-controlled reference is shared; actual motion is not prescribed.",
        "",
    ]
    for row in data.values():
        if row["status"] != "ok":
            lines.append(f"- **{row['name']} failed:** {row.get('error')}")
            continue
        lines.append(
            f"- {row['name']}: final mean x={row['final_x_mean']:.3f} m, z={row['final_z_mean']:.3f} m; "
            f"window shifts={row['window_shifts']}; validation={row['validations']}."
        )
    findings = ["## Main observations", ""]
    fixed, sparse, patch = data["fixed_3m"], data["sparse_3m"], data["patch_3m"]
    findings.append(
        f"- At 3 m, fixed averaged {fixed['mean_ms']:.2f} ms and sparse {sparse['mean_ms']:.2f} ms per completed step; neither has a large advantage here."
    )
    full, partial, off = [data[f"render_{r:.3f}"] for r in (1.0, 2 / 3, 0.0)]
    findings.append(
        f"- Rendering the full footprint averaged {full['mean_ms']:.2f} ms, partial visibility {partial['mean_ms']:.2f} ms, and no sand drawing {off['mean_ms']:.2f} ms. Full/partial render-call ratio: {full['mean_ms'] / partial['mean_ms']:.2f}x."
    )
    for grid in ("fixed", "sparse"):
        large = data[f"{grid}_6m"]
        findings.append(
            f"- Compared with 6 m {grid}, the 3 m patch was {large['mean_ms'] / patch['mean_ms']:.2f}x faster per step and used {large['steady_gib'] / patch['steady_gib']:.2f}x less steady VRAM. This compares different simulated terrain sizes."
        )
    for motion in ("stationary", "moving"):
        base, moving = data[f"overhead_original_{motion}"], data[f"overhead_patch_{motion}"]
        findings.append(
            f"- At equal 2 m allocation ({motion}), moving patch added {moving['mean_ms'] - base['mean_ms']:.2f} ms, or {(moving['mean_ms'] / base['mean_ms'] - 1) * 100:.1f}% net step time."
        )
    findings.append(
        "- Both 10 m full-bed attempts exhausted GPU memory during graph creation with 1,048,576 reserved cells. This is a limit of the tested capacity/configuration, not proof that every 10 m configuration is infeasible."
    )
    findings.append(
        "- Validation and final-pose differences below limit claims of physical equivalence; these are performance observations for this prototype."
    )
    findings.append("")
    lines[4:4] = findings
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("Saved summary.csv, summary.json, RESULTS.md, and four PNG figures.")


if __name__ == "__main__":
    main()
