# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Generate the alpha box rerun report from fresh measurements only."""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = {"fixed": ("#4477aa", "o"), "sparse": ("#ee7733", "s"), "moving-patch": ("#228833", "^")}


def aggregate():
    """Use the existing benchmark's pooling convention and CSV schema."""
    spec = importlib.util.spec_from_file_location("physics_vram_plots", ROOT.parent / "physics_vram/plot_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module.aggregate()


def plot(rows, methods, title, stem):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, key, error, label in zip(
        axes,
        ("mean_ms", "steady_gib"),
        ("std_ms", "steady_std_gib"),
        ("Physics step [ms]", "VRAM above baseline [GiB]"),
        strict=True,
    ):
        for method in methods:
            points = {r["size"]: r for r in rows if r["method"] == method and r["experiment"] == "scaling"}
            sizes = list(range(3, 11))
            color, marker = METHODS[method]
            ax.errorbar(
                sizes,
                [points.get(size, {}).get(key, np.nan) for size in sizes],
                yerr=[points.get(size, {}).get(error, np.nan) for size in sizes],
                color=color,
                marker=marker,
                capsize=3,
                linewidth=1.6,
                label=method,
            )
        ax.set(xlabel="Terrain side length [m]", ylabel=label, xticks=range(3, 11))
        ax.grid(alpha=0.2)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle(title)
    for extension in ("png", "pdf"):
        fig.savefig(ROOT / f"{stem}.{extension}", dpi=250, facecolor="white")
    plt.close(fig)


def report(rows):
    raw = [json.loads(p.read_text()) for p in sorted((ROOT / "raw").glob("*.json"))]
    successes = [r for r in raw if r["status"] == "ok"]
    failed = [r for r in raw if r["status"] != "ok"]
    rows = sorted((r for r in rows if r["experiment"] == "scaling"), key=lambda r: (r["size"], r["method"]))
    settings = json.loads((ROOT / "environment.json").read_text())
    complete = len(raw) == 48
    lines = [
        "# Alpha box benchmark: 3-10 m terrain",
        "",
        f"Run date: {settings['date']}. Status: {'complete' if complete else 'in progress'}; "
        f"{len(raw)}/48 attempts recorded, {len(successes)} successful, {len(failed)} excluded.",
        "",
        "This rerun measures the updated alpha box example with fixed, sparse, and moving-patch terrain. "
        "Alpha discards terrain history when particles are recycled. The box remains dynamic and exchanges "
        "forces with the sand; a force controller follows a straight reference at 0.2 m/s.",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        "| Background terrain | 3 x 3 through 10 x 10 m, in 1 m increments |",
        "| Moving-patch simulated terrain | 2 x 2 m |",
        "| Boundary terrain | 0.5 m on each side; ACTIVE particles with zero mass |",
        "| Moving-patch total particle region | 3 x 3 x 0.5 m; 1,004,400 particles |",
        "| Fixed/sparse total particle region | L x L x 0.5 m; includes the 0.5 m boundary on each side |",
        "| Sampling | 0.05 m voxels; 3 particles per voxel axis; jitter 0.05/3 m |",
        "| Material | Density 2,500 kg/m³; Young's modulus 40 MPa; Poisson's ratio 0.3; friction tan(50°) |",
        "| Solver | PIC; P0 strain; Q1 velocity; S2 collider; max 50 iterations; warmstart auto |",
        "| Cell capacity setting | max(65,536, ceil(particle width / 0.05 + 8)² x 22) |",
        "| Physics | 50 Hz x 4 substeps = 200 Hz; four rigid substeps per coupled step |",
        "| Coupling | Two-way proxy coupling; one coupling iteration; mass scale 1 |",
        "| Box | Current example geometry: 0.5 x 0.7 x 0.5 m; finalized mass 250 kg |",
        "| Execution | ViewerNull; no rendering; no outer CUDA graph capture |",
        "| Measurements | 150 warm-up steps + 150 measured steps; two fresh-process repeats |",
        "| GPU | NVIDIA GeForce RTX 5090, 32 GiB |",
        "",
        "Material, sampling, solver settings, and the short-run protocol are retained from the previous "
        "physics/VRAM benchmark. They override the interactive example defaults (including density, sand depth, "
        "and warmstart). The updated box geometry and alpha implementation are used. "
        "Unlike the previous run, the background has exactly the plotted size; no extra 2 m margin is added.",
        "",
        "The alpha, implicit-MPM, and coupled-solver Python source hashes match the archived September 26 "
        "snapshot in physics_vram/source_hashes.json. These are fresh measurements of the current checkout; "
        "they do not establish a speedup from a source change relative to that snapshot. Software versions "
        "and the current source hashes are recorded alongside the results.",
        "",
        "Sparse uses a positive max_active_cell_count and rebuilds its reserved grid in place. NanoVDB "
        "node capacities retain their default automatic, worst-case reservations. This is the previous "
        "benchmark's capacity policy, not a sweep for the lowest valid capacities or a test of per-step "
        "dynamic allocation with max_active_cell_count=-1. The cell-capacity column records the configured "
        "limit; fixed and sparse can allocate different internal buffers despite the same setting.",
        "",
        "Physics time is wall time for one coupled physics step, synchronized with the GPU. It includes "
        "control, collision detection, coupling, MPM, and moving-patch maintenance. Error bars show mean ± "
        "sample standard deviation across the pooled step measurements, not uncertainty in the mean. "
        "VRAM is device-wide NVML usage minus the pre-CUDA baseline, sampled every 50 ms. Steady VRAM is "
        "averaged over the measured interval and final state checks; peak includes initialization and warm-up. "
        "Shorter memory peaks can be missed, and desktop graphics can affect the baseline.",
        "",
        "The runner waits for other GPU compute processes and excludes overlapping runs. It checks for "
        "sparse-capacity errors, solver warnings, and non-finite final rigid/particle positions and velocities. "
        "Passing these checks does not establish long-duration stability. Each short run covers 1.5 s total, "
        "including warm-up; this measures terrain-size scaling, not traversal of the entire background.",
        "",
        "## Results",
        "",
        "![Performance with fixed/sparse MPM grid](01_grid_comparison.png)",
        "",
        "![Terrain size scaling](02_terrain_scaling.png)",
        "",
        "Missing points are failed cases, not zero-cost simulations. See the exclusions below.",
        "",
        "| Side [m] | Method | Successful repeats | Physics step [ms], mean ± std | Steady VRAM [GiB], mean ± std | Sampled peak [GiB] |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    tex_rows = []
    for row in rows:
        if row["successful_repeats"]:
            timing = f"{row['mean_ms']:.2f} ± {row['std_ms']:.2f}"
            memory = f"{row['steady_gib']:.2f} ± {row['steady_std_gib']:.2f}"
            peak = f"{row['peak_gib']:.2f}"
        else:
            timing = memory = peak = "failed"
        lines.append(
            f"| {row['size']} | {row['method']} | {row['successful_repeats']}/2 | {timing} | {memory} | {peak} |"
        )
        tex_rows.append(
            f"{row['size']} & {row['method']} & {row['successful_repeats']}/2 & "
            + timing.replace("±", r"$\pm$")
            + " & "
            + memory.replace("±", r"$\pm$")
            + f" & {peak} "
            + r"\\"
        )
    findings = []
    for method in METHODS:
        good = sorted([r for r in rows if r["method"] == method and r["successful_repeats"]], key=lambda r: r["size"])
        if len(good) >= 2:
            first, last = good[0], good[-1]
            findings.append(
                f"For {method}, increasing the terrain side from {first['size']} to {last['size']} m changes "
                f"mean physics time from {first['mean_ms']:.2f} to {last['mean_ms']:.2f} ms and steady VRAM "
                f"from {first['steady_gib']:.2f} to {last['steady_gib']:.2f} GiB."
            )
    findings.append(
        "Only moving-patch keeps its particle region and reserved cell capacity constant as the background grows. "
        "Fixed and sparse both simulate the full particle bed; sparse allocation does not restrict the material "
        "solve to particles touching the box. The 3 m cases have equal particle allocation, so comparing "
        "moving-patch with sparse at 3 m estimates total patch-maintenance overhead under this short-run protocol. "
        "Comparing moving-patch with fixed also includes a grid-method difference. These results concern a flat, "
        "densely filled bed and this reservation policy; they are not a claim about all sparse-grid configurations."
    )
    moving = [r for r in successes if r["case"]["method"] == "moving-patch"]
    if moving and all(sum(r["patch_shifts_after"]) == sum(r["patch_shifts_before"]) for r in moving):
        findings.append(
            "The box moves during the size sweep, but does not travel far enough to shift the patch center (centers are spaced 0.4 m apart) "
            "before the short run ends. No patch shifts occur during the measured interval. Per-step patch "
            "checks are timed; particle recycling during a shift is covered by the longer run below."
        )
    traversal = [json.loads(p.read_text()) for p in sorted((ROOT / "traversal/raw").glob("*.json"))]
    traversal_text = []
    for r in traversal:
        if r["status"] == "ok":
            shifts = sum(r["patch_shifts_after"]) - sum(r["patch_shifts_before"])
            traversal_text.append(
                f"Repeat {r['repeat']}: {r['mean_ms']:.2f} ± {r['std_ms']:.2f} ms per step; "
                f"{np.mean(r['steady_memory_bytes']) / 2**30:.2f} GiB steady VRAM; "
                f"{shifts} patch shifts during the 6 s measured interval; final box X = {r['final_pose'][0]:.2f} m."
            )
        else:
            traversal_text.append(f"Repeat {r['repeat']} failed: see traversal/raw for the error and log.")
    lines += [
        "",
        "## Interpretation",
        "",
        "\n\n".join(findings),
        "",
        "## Particle allocation and patch movement",
        "",
        "| Side [m] | Method | Particles | Cell capacity setting | Active cells at end | Patch shifts during measurement, by repeat |",
        "| ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        if row["successful_repeats"]:
            lines.append(
                f"| {row['size']} | {row['method']} | {row['particle_count']:,} | {row['cell_capacity']:,} | "
                f"{row['active_cells_min']:,}-{row['active_cells_max']:,} | {row['patch_shifts']} |"
            )
    lines += [
        "",
        "## Longer moving-patch validation",
        "",
        "The 6 x 6 m background uses the same 2 x 2 m simulated terrain and 0.5 m boundary. "
        "Each repeat runs 150 warm-up steps followed by 1,200 measured steps. "
        "These measurements are separate from the terrain-size plots.",
        "",
        "\n\n".join(traversal_text) if traversal_text else "Pending.",
        "",
        "## Excluded attempts",
        "",
    ]
    for row in failed:
        reason = row.get("error", "").strip().splitlines()
        lines.append(
            f"- {row['case']['name']}, repeat {row['repeat']}: {row['status']}; "
            f"{reason[-1] if reason else row.get('solver_warnings', row.get('foreign_gpu_pids', 'see raw result'))}"
        )
    if not failed:
        lines.append("None.")
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run --no-sync python newton/examples/mpm_moving_patch_alpha/benchmark/box_2026_09_28/rerun.py",
        "uv run --no-sync python newton/examples/mpm_moving_patch_alpha/benchmark/box_2026_09_28/plot_results.py",
        "```",
        "",
        "The runner retains successful results. Use a new output directory for a different source revision. "
        "Raw JSON/logs, individual timing samples, summary CSV/JSON, source hashes, and environment metadata "
        "are saved beside this report. The figure PDFs and report.tex can be compiled with LaTeX elsewhere; "
        "no local pdflatex or latexmk executable was found.",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")
    tex = r"""% SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
% SPDX-License-Identifier: CC-BY-4.0
\documentclass[10pt]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{graphicx,booktabs,longtable,hyperref}
\title{Alpha box benchmark: 3--10 m terrain}
\date{September 28, 2026}
\begin{document}
\maketitle
The updated alpha box example uses two-way coupling and a dynamic box following a
0.2 m/s force-controlled reference. Background sizes are 3--10 m square.
Moving-patch uses 2 m square simulated terrain plus a 0.5 m boundary on each side.
Fixed and sparse use full-size particle beds. Sand depth is 0.5 m; voxel size is
0.05 m; sampling is 3 particles per voxel axis. Density is 2500 kg/m$^3$, Young's
modulus 40 MPa, Poisson's ratio 0.3, and friction $\tan(50^\circ)$.
PIC transfer, P0 strain, Q1 velocity, S2 collider, automatic warmstart and at most
50 iterations are used. These controlled settings match the earlier benchmark,
rather than the interactive defaults. Execution is headless at 200 Hz with four
rigid substeps, without outer CUDA graph capture, on an RTX 5090.

Two fresh-process repeats each use 150 warm-up and 150 measured steps.
Physics timings include a GPU synchronization. Error bars are mean plus/minus
sample standard deviation of pooled steps. VRAM is sampled device-wide memory
minus the pre-CUDA baseline, every 50 ms; desktop activity affects this estimate.
Peak includes initialization and warm-up. Shorter peaks can be missed.
Missing points are failed cases, not zero cost. Source hashes and full settings
are included with the data. This short-run benchmark does not establish
long-distance locomotion stability.
\begin{figure}[h]\centering
\includegraphics[width=\linewidth]{01_grid_comparison.pdf}
\end{figure}
\begin{figure}[h]\centering
\includegraphics[width=\linewidth]{02_terrain_scaling.pdf}
\end{figure}
\clearpage
"""
    tex += "\n\n".join(findings) + "\n\n"
    tex += r"\paragraph{Longer moving-patch validation.}" + "\n"
    tex += "On a 6 m background, two repeats use 150 warm-up and 1200 measured steps. "
    tex += "\n\n".join(traversal_text).replace("±", r"$\pm$") + "\n\n"
    tex += (
        r"""\begin{longtable}{rlrrrr}
\toprule
Side [m] & Method & Repeats & Step [ms] & Steady [GiB] & Peak [GiB] \\
\midrule\endhead
"""
        + "\n".join(tex_rows)
        + "\n"
        + r"\bottomrule\end{longtable}"
        + "\n"
    )
    tex += "Complete failure details and per-case particle and cell counts are in README.md and summary.csv.\n"
    tex += r"\end{document}" + "\n"
    (ROOT / "report.tex").write_text(tex)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    data = aggregate()
    plot(data, ("fixed", "sparse"), "Performance with fixed/sparse MPM grid", "01_grid_comparison")
    plot(data, tuple(METHODS), "Terrain size scaling", "02_terrain_scaling")
    report(data)
