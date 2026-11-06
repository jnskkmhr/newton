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
REPO = ROOT.parents[4]
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
                steps_per_second=float(1000.0 / np.mean(timing)),
                real_time_factor=float(good[0]["physics_dt"] * 1000.0 / np.mean(timing)),
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
                patch_shifts=[int(np.sum(r["patch_shifts_after"]) - np.sum(r["patch_shifts_before"])) for r in good],
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
        "steps_per_second",
        "real_time_factor",
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


def write_report(rows):
    """Write measured results and limitations without substituting historical data."""
    lookup = {(r["experiment"], r["scene"], r["method"], r["size"]): r for r in rows}
    raw = [json.loads(p.read_text()) for p in sorted((ROOT / "raw").glob("*.json"))]
    successful = sum(r["status"] == "ok" for r in raw)
    settings = json.loads((ROOT / "environment.json").read_text())
    expected = 102 * settings["repeats"]
    complete = len(raw) == expected
    lines = [
        "# Alpha physics speed and GPU memory benchmark",
        "",
        f"Status: {'complete' if complete else 'in progress'}; {len(raw)}/{expected} attempts recorded, "
        f"{successful} successful. Failed or contended attempts are excluded from plotted means.",
        "",
        "These runs use the current standalone alpha box, rover, and G1 examples. "
        "Alpha discards terrain history: recycled particles represent fresh sand. "
        "Each scene uses the same benchmark material and particle sampling. "
        "The default interactive material settings are overridden only in the benchmark subclass.",
        "",
        "## Conditions",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        "| GPU | NVIDIA GeForce RTX 5090, 32 GiB |",
        "| Physics | 200 Hz coupled steps; 4 rigid substeps per coupled step |",
        "| Rendering | Disabled (ViewerNull) |",
        "| Material | Density 2,500 kg/m³; Young's modulus 40 MPa; Poisson's ratio 0.3; friction tan(50°) |",
        "| Sampling | 0.05 m voxels; 3 particles per cell axis; 0.5 m depth; jitter 0.05/3 m |",
        "| MPM solve | PIC transfer; P0 strain basis; maximum 50 iterations |",
        "| Grid capacity | max(65,536, ceil(width/0.05 + 8)² x 22) cells |",
        "| Sparse allocation | Positive reserved capacity; rebuildable sparse grid |",
        "| Motion | Box force tracking; grouser-wheel rover differential drive; G1 policy; forward command 0.2 m/s |",
        "| Short runs | 150 warm-up steps + 150 measured steps; two fresh-process repeats |",
        "| Longer moving-patch runs | 150 warm-up steps + 1,200 measured steps; two repeats |",
        "",
        "The box mass is 250 kg; the grouser-wheel rover is approximately 38.11 kg; G1 is approximately 33.34 kg. "
        "G1 uses `config/g1.yaml`, `config/policy_3d_rft.pt`, and the improved-collision box-foot MJCF asset. "
        "The box force controller is applied only to the box; robot control comes from wheel actuators or the G1 policy.",
        "",
        "Physics time is synchronized wall time for one coupled step. It includes collision detection, "
        "rigid/MPM coupling, particle recycling, and control (including G1 policy inference at 50 Hz). "
        "It excludes scene construction, rendering, and final CPU state checks. "
        "No CUDA frame graph is used. Error bars show the sample standard deviation of individual "
        "steps pooled across repeats, not confidence intervals. Repeat means are retained in summary.json.",
        "",
        "Sparse uses a preallocated, rebuildable grid with the capacity formula above. "
        "This comparison does not benchmark negative-capacity dynamic allocation or search for optimal sparse capacities. "
        "Raw results include the reserved capacity and the occupied-cell count at the end of each run.",
        "",
        "GPU memory is device-wide NVML usage minus the pre-CUDA baseline, sampled every 50 ms. "
        "Steady memory covers the measured interval through final state checks. Sampled peak includes "
        "initialization and warm-up and can miss shorter peaks. Desktop graphics remain active and can "
        "introduce small fluctuations. Other compute processes are monitored; overlapping runs are excluded. "
        "VRAM includes arrays allocated by the example during construction even when rendering is disabled. "
        "Raw files also record Warp pool memory, which does not include all solver or PyTorch allocations.",
        "",
        "## Terrain size scaling",
        "",
        "Fixed and sparse simulate an L x L x 0.5 m bed, for L = 3 through 10 m. "
        "Moving-patch keeps a 3 x 3 x 0.5 m bed: a 2 x 2 m moving terrain area and a 0.5 m "
        "zero-mass active boundary on each side. Fixed and sparse use the same boundary width at "
        "the edges of their full beds. All methods have sampled background ground of side L + 2 m. "
        "Thus the horizontal axis describes particle-bed size for full-terrain methods and available "
        "terrain extent for moving-patch; it is not a common particle allocation.",
        "",
        "![Fixed and sparse grids](01_grid_comparison.png)",
        "",
        "![Terrain size scaling](02_terrain_scaling.png)",
        "",
        "| Scene | Method | Side [m] | Physics step [ms], mean ± std | Steady VRAM [GiB], mean ± std | Sampled peak [GiB] | Particles |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    table = []
    for scene in SCENES:
        for size in (3, 10):
            for method in METHODS:
                row = lookup.get(("scaling", scene, method, size), {})
                if "mean_ms" not in row:
                    continue
                lines.append(
                    f"| {scene} | {method} | {size} | {row['mean_ms']:.2f} ± {row['std_ms']:.2f} | "
                    f"{row['steady_gib']:.2f} ± {row['steady_std_gib']:.2f} | {row['peak_gib']:.2f} | "
                    f"{row['particle_count']:,} |"
                )
                table.append(
                    f"{scene} & {method} & {size} & "
                    rf"${row['mean_ms']:.2f}\pm{row['std_ms']:.2f}$ & "
                    rf"${row['steady_gib']:.2f}\pm{row['steady_std_gib']:.2f}$ & "
                    f"{row['peak_gib']:.2f}" + r" \\"
                )
    lines += [
        "",
        "## Equal particle allocation",
        "",
        "All three methods allocate the same initial 1, 2, or 4 m particle-bed width, "
        "with a 0.1 m zero-mass active boundary on each side. Differences include "
        "patch maintenance and the resulting material motion. This is an end-to-end comparison, "
        "not an isolated measurement of one recycling kernel.",
        "",
        "![Equal-allocation comparison](03_equal_size_comparison.png)",
        "",
        "## Longer moving-patch runs",
        "",
        "The short runs may not move far enough to shift a patch. These additional runs "
        "measure six seconds after warm-up with the default 0.4 m patch shift interval. "
        "The patch shift count below is the difference between the counters before and after measurement.",
        "",
        "| Scene | Repeat | Physics step [ms], mean ± std | Steady VRAM [GiB] | Patch shifts during measurement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in raw:
        if row["case"]["experiment"] == "traversal" and row["status"] == "ok":
            shifts = int(np.sum(row["patch_shifts_after"]) - np.sum(row["patch_shifts_before"]))
            memory = np.mean(row["steady_memory_bytes"]) / 2**30
            lines.append(
                f"| {row['case']['scene']} | {row['repeat']} | {row['mean_ms']:.2f} ± "
                f"{row['std_ms']:.2f} | {memory:.2f} | {shifts} |"
            )
    lines += ["", "## Interpretation", ""]
    findings = []
    for scene in SCENES:
        for method in METHODS:
            lo = lookup.get(("scaling", scene, method, 3), {})
            hi = lookup.get(("scaling", scene, method, 10), {})
            if "mean_ms" in lo and "mean_ms" in hi:
                findings.append(
                    f"For {scene}, {method} changes from {lo['mean_ms']:.2f} to {hi['mean_ms']:.2f} ms "
                    f"and from {lo['steady_gib']:.2f} to {hi['steady_gib']:.2f} GiB between 3 and 10 m."
                )
    for scene in SCENES:
        sparse = lookup.get(("overhead", scene, "sparse", 2), {})
        patch = lookup.get(("overhead", scene, "moving-patch", 2), {})
        fixed = lookup.get(("overhead", scene, "fixed", 2), {})
        if all("mean_ms" in r for r in (sparse, patch, fixed)):
            findings.append(
                f"At equal 2 m allocation for {scene}, fixed takes {fixed['mean_ms']:.2f} ms, "
                f"sparse {sparse['mean_ms']:.2f} ms, and moving-patch {patch['mean_ms']:.2f} ms. "
                f"Moving-patch differs from sparse by {100 * (patch['mean_ms'] / sparse['mean_ms'] - 1):.1f} percent."
            )
    findings.append(
        "Moving-patch versus fixed includes both the grid-method difference and patch maintenance. "
        "The equal-allocation comparison against sparse more closely measures added patch handling, "
        "although material motion can also differ."
    )
    lines += findings
    lines += [
        "",
        "Moving-patch keeps its particle count bounded, while full-terrain particle counts grow "
        "with horizontal area. Background height samples still grow with terrain extent; bounded "
        "MPM allocation does not mean all memory is independent of background size. Alpha also "
        "updates a local supporting collider and particle masses when boundary membership changes.",
        "",
        "These results are not a direct alpha-versus-beta comparison. Alpha preserves both vertical "
        "sampling endpoints (31 layers here, versus beta's 30). Robot assets, current solver source, "
        "and ground-collider handling can also differ. The benchmark uses alpha's current scene code.",
        "",
        "State checks reject non-finite rigid/particle positions and velocities; solver overflow "
        "warnings are excluded. Passing these checks does not establish long-distance locomotion "
        "stability or guarantee that every step of a G1 run is upright. The short scaling runs do "
        "not traverse the entire terrain. Initial, measurement-start, and final poses are saved.",
        "",
    ]
    failures = [r for r in raw if r["status"] != "ok"]
    if failures:
        lines += ["## Excluded attempts", ""]
        for row in failures:
            error = row.get("error", " ".join(row.get("solver_warnings", [])))
            lines.append(
                f"- {row['case']['name']}, repeat {row['repeat']}: {row['status']}; {error.splitlines()[-1] if error else ''}"
            )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run --no-sync python newton/examples/mpm_moving_patch_alpha/benchmark/physics_vram/run_benchmarks.py",
        "uv run --no-sync python newton/examples/mpm_moving_patch_alpha/benchmark/physics_vram/plot_results.py",
        "```",
        "",
        "The runner retains successful existing results. Use a clean output directory for "
        "a new experiment. Raw JSON/logs, summary CSV/JSON, source hashes, asset hashes, and environment "
        "metadata are saved here. No local LaTeX compiler was found; report.tex and PDF figures "
        "can be compiled elsewhere using `latexmk -pdf report.tex`.",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")
    (ROOT / "results_table.tex").write_text("\n".join(table) + "\n")
    (ROOT / "findings.tex").write_text("\n\n".join(findings) + "\n")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    rows = aggregate()
    comparison(rows, "scaling", ("fixed", "sparse"), "01_grid_comparison", "Performance with fixed/sparse MPM grid")
    comparison(rows, "scaling", tuple(METHODS), "02_terrain_scaling", "Terrain size scaling")
    comparison(rows, "overhead", tuple(METHODS), "03_equal_size_comparison", "Moving patch compute/memory overhead")
    write_report(rows)
    print(f"Wrote {len(rows)} case summaries and figures to {ROOT}")


if __name__ == "__main__":
    main()
