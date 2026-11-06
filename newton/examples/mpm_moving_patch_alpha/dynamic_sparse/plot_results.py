# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot method means and standard deviations from the dynamic-sparse rerun."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = {
    "fixed": ("fixed", "#4477aa", "o"),
    "sparse": ("sparse", "#ee7733", "s"),
    "moving-patch": ("moving-patch", "#228833", "^"),
}


def aggregate():
    groups = {}
    for path in sorted((ROOT / "raw").glob("*.json")):
        row = json.loads(path.read_text())
        groups.setdefault(row["case"]["name"], []).append(row)
    data = {}
    for name, runs in groups.items():
        good = [r for r in runs if r["status"] == "ok"]
        row = {**runs[0]["case"], "status": "ok" if good else "failed", "repeats": len(good)}
        if good:
            times = np.concatenate([r["wall_ms"] for r in good])
            steady = np.array([r["steady_delta_bytes"] for r in good]) / 1024**3
            peaks = np.array([r["peak_delta_bytes"] for r in good]) / 1024**3
            row.update(
                mean_ms=float(np.mean(times)),
                std_ms=float(np.std(times, ddof=1)),
                steady_gib=float(np.mean(steady)),
                steady_std_gib=float(np.std(steady, ddof=1)) if len(good) > 1 else 0.0,
                peak_gib=float(np.mean(peaks)),
                peak_std_gib=float(np.std(peaks, ddof=1)) if len(good) > 1 else 0.0,
                max_peak_gib=float(np.max(peaks)),
                particles=good[0]["particle_count"],
                capacity=good[0]["reserved_cell_capacity"],
                outer_cuda_graph=good[0]["outer_cuda_graph"],
                active_min=min(min(r["active_cells_samples"]) for r in good),
                active_max=max(max(r["active_cells_samples"]) for r in good),
                final_x_mean=float(np.mean([r["final_pose"][0][0] for r in good])),
                final_z_mean=float(np.mean([r["final_pose"][0][2] for r in good])),
                window_shifts=[r.get("window_shifts") for r in good],
                validations=[r.get("scene_validation") for r in good],
            )
        else:
            row["error"] = runs[0].get("error", "worker failure")
        data[name] = row
    return data


def plot(data, series, x, labels, title, xlabel, filename):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    metrics = (
        ("mean_ms", "std_ms", "Completed physics step [ms]"),
        ("steady_gib", "steady_std_gib", "Steady-state VRAM above baseline [GiB]"),
        ("peak_gib", "peak_std_gib", "Sampled peak VRAM above baseline [GiB]"),
    )
    for method, names in series:
        label, color, marker = METHODS[method]
        good = [(xx, data[name]) for xx, name in zip(x, names, strict=True) if data[name]["status"] == "ok"]
        for ax, (mean, std, ylabel) in zip(axes, metrics, strict=True):
            if good:
                ax.errorbar(
                    [a for a, _ in good],
                    [r[mean] for _, r in good],
                    yerr=[r[std] for _, r in good],
                    label=label,
                    color=color,
                    marker=marker,
                    markersize=6,
                    linewidth=1.6,
                    capsize=5,
                )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_xticks(x, labels)
            ax.grid(alpha=0.2)
    for ax in axes:
        ax.set_ylim(bottom=0)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=len(series), frameon=False)
    fig.suptitle(title, fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(ROOT / filename, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    data = aggregate()
    (ROOT / "summary.json").write_text(json.dumps(data, indent=2))
    fields = [
        "name",
        "method",
        "status",
        "repeats",
        "width",
        "motion",
        "mean_ms",
        "std_ms",
        "steady_gib",
        "steady_std_gib",
        "peak_gib",
        "peak_std_gib",
        "max_peak_gib",
        "particles",
        "capacity",
        "outer_cuda_graph",
        "active_min",
        "active_max",
        "final_x_mean",
        "final_z_mean",
        "error",
    ]
    with (ROOT / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data.values())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    widths = [3, 6, 10]
    series = [(method, [f"{method}_{w}m" for w in widths]) for method in ("fixed", "sparse")]
    plot(
        data,
        series,
        widths,
        [str(w) for w in widths],
        "Performance with fixed/sparse MPM grid",
        "Full-bed side length [m]",
        "01_grid_comparison.png",
    )
    plot(
        data,
        [*series, ("moving-patch", ["patch_3m"] * 3)],
        widths,
        [str(w) for w in widths],
        "Terrain size scaling",
        "Full-terrain / conceptual operating-area side [m]",
        "02_terrain_scaling.png",
    )
    series = [(method, [f"equal_{method}_{motion}" for motion in ("stationary", "moving")]) for method in METHODS]
    plot(
        data,
        series,
        [0, 1],
        ["Stationary reference", "Moving +X reference"],
        "Moving patch compute/memory overhead",
        "Box controller reference",
        "03_equal_size_comparison.png",
    )
    lines = [
        "# Dynamic sparse benchmark rerun",
        "",
        "No rendering benchmark is included. All cases are headless.",
        "",
        "Sparse uses per-step allocation (`max_active_cell_count=-1`) without an outer CUDA graph. Fixed and moving-patch use bounded capacity and graph replay.",
        "",
        "Timing SD pools measured step durations across successful runs. VRAM SD is across per-run means/peaks. These are standard deviations, not confidence intervals.",
        "",
        "| Case | Runs | Step mean +/- SD (ms) | Steady mean +/- SD (GiB) | Peak mean +/- SD (GiB) | Particles | Active cells | Capacity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in data.items():
        if row["status"] == "ok":
            capacity = "Dynamic (-1)" if row["capacity"] == -1 else f"{row['capacity']:,}"
            lines.append(
                f"| {name} | {row['repeats']} | {row['mean_ms']:.2f} +/- {row['std_ms']:.2f} | "
                f"{row['steady_gib']:.2f} +/- {row['steady_std_gib']:.2f} | {row['peak_gib']:.2f} +/- {row['peak_std_gib']:.2f} | "
                f"{row['particles']:,} | {row['active_min']:,}-{row['active_max']:,} | {capacity} |"
            )
        else:
            lines.append(f"| {name} | 0 | Failed | - | - | - | - | - |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Stationary means the controller has zero horizontal reference velocity. Gravity, settling, deformation, and corrective forces remain active. Moving cases follow the same straight +X reference at 0.2 m/s after settling and acceleration.",
        "",
        "The equal-size comparison measures net method performance, not pure recycling overhead. It now also changes sparse-grid allocation and graph execution. The earlier bounded-sparse experiment is preserved in the sibling `benchmark/` directory.",
        "",
        "## Validation and failures",
        "",
    ]
    for name, row in data.items():
        if row["status"] != "ok":
            lines.append(f"- {name}: {row['error']}")
        else:
            lines.append(
                f"- {name}: final mean x={row['final_x_mean']:.3f} m, z={row['final_z_mean']:.3f} m; "
                f"window shifts={row['window_shifts']}; checks={row['validations']}."
            )
    observations = ["## Main observations", ""]
    for width in (3, 6, 10):
        fixed, sparse = data[f"fixed_{width}m"], data[f"sparse_{width}m"]
        if fixed["status"] == "ok" and sparse["status"] == "ok":
            observations.append(
                f"- {width} m bed: fixed {fixed['mean_ms']:.2f} +/- {fixed['std_ms']:.2f} ms; dynamic sparse {sparse['mean_ms']:.2f} +/- {sparse['std_ms']:.2f} ms. Steady VRAM: {fixed['steady_gib']:.2f} vs {sparse['steady_gib']:.2f} GiB."
            )
        elif sparse["status"] == "ok":
            observations.append(
                f"- {width} m bed: dynamic sparse completed at {sparse['mean_ms']:.2f} +/- {sparse['std_ms']:.2f} ms with {sparse['steady_gib']:.2f} GiB steady VRAM; fixed failed under the tested configuration."
            )
    observations += [
        "- Relative method performance changes with terrain size and capacity. Do not generalize the smallest-bed timing to larger beds.",
        "- Equal-size results combine allocation, graph execution, recycling, and their effects on subsequent solver work; they do not isolate recycling overhead.",
        "",
    ]
    lines[8:8] = observations
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("Saved three mean +/- SD line figures, summary.csv, summary.json, and RESULTS.md")


if __name__ == "__main__":
    main()
