# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Create report figures and tables from completed benchmark measurements."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
METHODS = {
    "fixed": ("#4477aa", "o"),
    "sparse": ("#ee7733", "s"),
    "moving-patch": ("#228833", "^"),
}
SCENES = ("box", "rover", "g1")


def aggregate():
    """Pool measured steps; retain repeat means to expose run-to-run variation."""
    groups = {}
    for path in sorted((ROOT / "raw").glob("*.json")):
        row = json.loads(path.read_text())
        groups.setdefault(row["case"]["name"], []).append(row)
    summaries = []
    for _name, rows in sorted(groups.items()):
        good = [r for r in rows if r["status"] == "ok"]
        result = dict(rows[0]["case"])
        result.update(successful_repeats=len(good), failed_repeats=len(rows) - len(good))
        if good:
            timing = np.concatenate([r["wall_ms"] for r in good])
            memory = np.concatenate([r["steady_memory_bytes"] for r in good]) / 2**30
            result.update(
                mean_ms=float(np.mean(timing)),
                std_ms=float(np.std(timing, ddof=1)),
                steady_gib=float(np.mean(memory)),
                steady_std_gib=float(np.std(memory, ddof=1)),
                peak_gib=max(r["sampled_peak_bytes"] for r in good) / 2**30,
                particle_count=good[0]["particle_count"],
                dynamic_particles=good[0]["dynamic_particles"],
                active_cells_min=min(r["active_cells"] for r in good),
                active_cells_max=max(r["active_cells"] for r in good),
                cell_capacity=good[0]["cell_capacity"],
                measured_steps=len(timing),
                repeat_mean_ms=[r["mean_ms"] for r in good],
                recycled_particles=[r["recycled_during_measurement"] for r in good],
                final_body_position=[r["final_pose"][:3] for r in good],
            )
        if not good:
            for key in ("particle_count", "dynamic_particles", "cell_capacity"):
                available = [r[key] for r in rows if key in r]
                if available:
                    result[key] = available[0]
            peaks = [r["sampled_peak_bytes"] for r in rows if "sampled_peak_bytes" in r]
            if peaks:
                result["peak_gib"] = max(peaks) / 2**30
        result["errors"] = [r.get("error", "unknown error") for r in rows if r["status"] != "ok"]
        summaries.append(result)
    (ROOT / "summary.json").write_text(json.dumps(summaries, indent=2))
    scalar_keys = [
        "experiment",
        "scene",
        "method",
        "size",
        "particle_width",
        "background_size",
        "boundary_width",
        "successful_repeats",
        "failed_repeats",
        "mean_ms",
        "std_ms",
        "steady_gib",
        "steady_std_gib",
        "peak_gib",
        "particle_count",
        "dynamic_particles",
        "active_cells_min",
        "active_cells_max",
        "cell_capacity",
        "measured_steps",
    ]
    with (ROOT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def save_figure(figure, stem):
    figure.savefig(ROOT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def comparison(rows, experiment, methods, stem, title):
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.0), sharex=True)
    sizes = range(3, 11) if experiment == "scaling" else (1, 2, 4)
    for i, scene in enumerate(SCENES):
        for j, (mean_key, std_key, ylabel) in enumerate(
            [
                ("mean_ms", "std_ms", "Physics step [ms]"),
                ("steady_gib", "steady_std_gib", "GPU memory [GiB]"),
            ]
        ):
            ax = axes[i, j]
            for method in methods:
                selected = {
                    r["size"]: r
                    for r in rows
                    if r["experiment"] == experiment and r["scene"] == scene and r["method"] == method
                }
                means = [selected.get(s, {}).get(mean_key, np.nan) for s in sizes]
                stds = [selected.get(s, {}).get(std_key, np.nan) for s in sizes]
                color, marker = METHODS[method]
                ax.errorbar(
                    list(sizes),
                    means,
                    yerr=stds,
                    color=color,
                    marker=marker,
                    label=method,
                    capsize=2,
                    markersize=4,
                    linewidth=1.3,
                    elinewidth=0.8,
                )
            ax.set_title("G1" if scene == "g1" else scene.capitalize(), loc="left", fontsize=10)
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.set_xticks(list(sizes))
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Terrain side length [m]" if experiment == "scaling" else "Particle allocation side length [m]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(methods), frameon=False)
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.4, w_pad=1.5)
    save_figure(fig, stem)


def scene_images():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, scene in zip(axes, SCENES, strict=True):
        path = REPO / f"docs/images/examples/example_moving_terrain_mpm_{scene}.jpg"
        ax.imshow(plt.imread(path))
        ax.axis("off")
        ax.set_title("G1" if scene == "g1" else scene.capitalize())
    fig.tight_layout(pad=0.2)
    save_figure(fig, "00_scenes")


def result_text(rows):
    """Write numerical report paragraphs directly from the aggregated measurements."""

    def find(experiment, scene, method, size):
        return next(
            (
                r
                for r in rows
                if r["experiment"] == experiment
                and r["scene"] == scene
                and r["method"] == method
                and r["size"] == size
                and r["successful_repeats"]
            ),
            None,
        )

    def scene_name(scene):
        return "G1" if scene == "g1" else scene.capitalize()

    def time_text(row):
        return rf"${row['mean_ms']:.2f}\pm{row['std_ms']:.2f}$" if row else "---"

    def memory_text(row):
        return rf"${row['steady_gib']:.2f}\pm{row['steady_std_gib']:.2f}$" if row else "---"

    grid = [
        r"Sparse allocation does not eliminate work on particles throughout a filled sand bed. "
        r"Both full-terrain methods increase particle count from 972,000 at 3\,m to 10,800,000 at 10\,m. "
        r"The following table compares the largest size completed by both methods for each scene. "
        r"Times are mean $\pm$ standard deviation in milliseconds; memory is the measured mean in GiB.",
        r"\begin{center}\small\begin{tabular}{lrrrrr}\toprule",
        r"Scene & Side [m] & Fixed [ms] & Sparse [ms] & Fixed [GiB] & Sparse [GiB] \\\midrule",
    ]
    wins = 0
    pairs = 0
    for scene in SCENES:
        common = []
        for size in range(3, 11):
            a, b = (find("scaling", scene, method, size) for method in ("fixed", "sparse"))
            if a and b:
                common.append((size, a, b))
                pairs += 1
                wins += b["mean_ms"] < a["mean_ms"]
        if common:
            size, a, b = common[-1]
            grid.append(
                rf"{scene_name(scene)} & {size} & {time_text(a)} & {time_text(b)} & "
                rf"{a['steady_gib']:.2f} & {b['steady_gib']:.2f} \\"
            )
    grid += [
        r"\bottomrule\end{tabular}\end{center}",
        f"Sparse has a lower mean step time in {wins} of {pairs} matched size/scene comparisons. "
        "These results describe the tested reserved-capacity configuration; they do not establish an inherent "
        "advantage for either grid representation. No component-level profiling was performed to separate "
        "grid rebuilding, material solving, and capacity-related costs.",
    ]
    if pairs and wins == 0:
        grid.insert(
            0, f"Fixed has a lower mean physics-step time than sparse in all {pairs} matched size/scene comparisons."
        )
    elif pairs and wins == pairs:
        grid.insert(
            0, f"Sparse has a lower mean physics-step time than fixed in all {pairs} matched size/scene comparisons."
        )
    failed = [r for r in rows if r["failed_repeats"]]
    if failed:
        errors = [error for r in failed for error in r["errors"]]
        gpu_failures = sum("Failed to allocate" in error for error in errors)
        killed = sum(error == "exit -9" for error in errors)
        if gpu_failures:
            grid.append(f"{gpu_failures} attempts failed during GPU memory allocation with the tested settings.")
        if killed:
            grid.append(
                f"{killed} initialization workers were terminated by SIGKILL before measurements; "
                "the cause could not be confirmed because kernel logs were not accessible."
            )
        grid.append(
            "Failed measurements are excluded from means and shown as missing plot points. "
            "The affected cases and failure messages are listed in the accompanying summary data."
        )
        for scene in SCENES:
            for method in METHODS:
                sizes = [
                    str(r["size"])
                    for r in failed
                    if r["experiment"] == "scaling" and r["scene"] == scene and r["method"] == method
                ]
                if sizes:
                    grid.append(
                        f"{scene_name(scene)} {method}: failed runs at side lengths " + ", ".join(sizes) + r"\,m."
                    )
    (ROOT / "grid_results.tex").write_text("\n".join(grid) + "\n")

    scaling = [
        r"Moving-patch keeps 972,000 MPM particles for every background size in this sweep. "
        r"The measured ranges across 3--10\,m are:"
    ]
    for scene in SCENES:
        selected = [
            r
            for r in rows
            if r["experiment"] == "scaling"
            and r["scene"] == scene
            and r["method"] == "moving-patch"
            and r["successful_repeats"]
        ]
        if selected:
            times = [r["mean_ms"] for r in selected]
            memory = [r["steady_gib"] for r in selected]
            scaling.append(
                rf"\textbf{{{scene_name(scene)}:}} {min(times):.2f}--{max(times):.2f}\,ms per step "
                rf"and {min(memory):.2f}--{max(memory):.2f}\,GiB mean GPU memory."
            )
    scaling += [
        r"At the 10\,m endpoint, the comparison is:",
        r"\begin{center}\small\begin{tabular}{llrr}\toprule",
        r"Scene & Method & Time [ms] & Memory [GiB] \\\midrule",
    ]
    for scene in SCENES:
        for method in METHODS:
            r = find("scaling", scene, method, 10)
            scaling.append(rf"{scene_name(scene)} & {method} & {time_text(r)} & {memory_text(r)} \\")
    scaling += [
        r"\bottomrule\end{tabular}\end{center}",
        "The full-terrain methods retain all particles as terrain size grows. Moving-patch instead "
        "limits the local material simulation to a fixed allocation. Its background height map and ground "
        "still grow with terrain area, so approximately flat curves over this range do not imply "
        "constant memory for an arbitrarily large background.",
    ]
    (ROOT / "scaling_results.tex").write_text("\n".join(scaling) + "\n")

    overhead = [
        r"Matched allocations contain 108,000, 432,000, and 1,728,000 particles for 1, 2, and 4\,m "
        r"side lengths, respectively. At the 2\,m allocation:",
        r"\begin{center}\small\begin{tabular}{llrr}\toprule",
        r"Scene & Method & Time [ms] & Memory [GiB] \\\midrule",
    ]
    for scene in SCENES:
        for method in METHODS:
            row = find("overhead", scene, method, 2)
            overhead.append(rf"{scene_name(scene)} & {method} & {time_text(row)} & {memory_text(row)} \\")
    overhead += [r"\bottomrule\end{tabular}\end{center}"]
    for scene in SCENES:
        patch = find("overhead", scene, "moving-patch", 2)
        sparse = find("overhead", scene, "sparse", 2)
        fixed = find("overhead", scene, "fixed", 2)
        if patch and sparse and fixed:
            dt = patch["mean_ms"] - sparse["mean_ms"]
            dm = patch["steady_gib"] - sparse["steady_gib"]
            overhead.append(
                rf"For {scene_name(scene)}, moving-patch minus sparse is {dt:+.2f}\,ms "
                rf"({100 * dt / sparse['mean_ms']:+.1f}\%) and {dm:+.2f}\,GiB. "
                rf"Moving-patch minus fixed is {patch['mean_ms'] - fixed['mean_ms']:+.2f}\,ms "
                rf"and {patch['steady_gib'] - fixed['steady_gib']:+.2f}\,GiB."
            )
    overhead += [
        "The sparse comparison is the closer estimate of moving-patch overhead because it keeps the "
        "same grid representation. Differences against fixed include both grid implementation and "
        "moving-patch handling. The measurements include height-cache updates and particle recycling, "
        "but do not isolate those kernels; differences should not be attributed entirely to recycling."
    ]
    (ROOT / "overhead_results.tex").write_text("\n".join(overhead) + "\n")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = aggregate()
    result_text(rows)
    scene_images()
    comparison(rows, "scaling", ("fixed", "sparse"), "01_grid_comparison", "Performance with fixed/sparse MPM grid")
    comparison(rows, "scaling", tuple(METHODS), "02_terrain_scaling", "Terrain size scaling")
    comparison(rows, "overhead", tuple(METHODS), "03_equal_size_comparison", "Moving patch compute/memory overhead")


if __name__ == "__main__":
    main()
