# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate and plot the standalone fixed-container box sweep."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    data_dir = args.data
    execution = json.loads((data_dir / "execution.json").read_text())
    assert len(execution) == 63 and all(r["returncode"] == 0 for r in execution)
    rows = []
    traces = {}
    common_config = None
    for run in execution:
        folder = Path(run["folder"])
        cfg = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        width, depth = run["jitter_width_mm"], run["sand_depth_cm"]
        assert cfg["arguments"]["jitter"] == width / 1000 and cfg["arguments"]["sand_depth"] == depth / 100
        assert cfg["arguments"]["mass"] == 5 and cfg["arguments"]["speed"] == 0.05
        assert cfg["arguments"]["settle"] == 0.5 and cfg["arguments"]["seconds"] == 6
        assert cfg["mpm_config"]["max_iterations"] == 25 and cfg["mpm_config"]["collider_basis"] == "pic27"
        assert cfg["mpm_hz"] == 400 and cfg["mjwarp_hz"] == 800 and cfg["voxel_size_m"] == 0.02
        np.testing.assert_allclose(cfg["mpm_collider_mass_kg"], [5], rtol=1e-6)
        common = {k: cfg[k] for k in ["newton_path", "versions", "source_sha256", "material", "mpm_config"]}
        if common_config is None:
            common_config = common
        assert common == common_config, "Code, dependencies, material, or solver settings changed during the sweep"
        assert cfg["particle_count"] == np.prod(cfg["resolution"])
        np.testing.assert_allclose(cfg["particle_count"] * cfg["particle_mass_kg"], 1600 * 1.6**2 * depth / 100)
        trace = np.genfromtxt(folder / "trace.csv", delimiter=",", names=True)
        assert len(trace) == 2400 and all(np.isfinite(trace[n]).all() for n in trace.dtype.names)
        np.testing.assert_allclose(trace["time_s"], np.arange(1, 2401) / 400, atol=3e-7, rtol=0)
        for component in ["fx_N", "fy_N", "fz_N", "tx_Nm", "ty_Nm", "tz_Nm"]:
            np.testing.assert_allclose(
                trace[f"applied_mpm_{component}"][1:], trace[f"generated_mpm_{component}"][:-1], atol=1e-6, rtol=0
            )
        assert summary["maximum_relative_momentum_balance_error"] < 1e-3
        assert summary["all_finite"] and summary["feedback_transfer_validated"]
        mask = trace["time_s"] >= 2 - 1e-6
        pen = 100 * np.maximum(-trace["box_sole_z_m"], 0)
        np.testing.assert_allclose(summary["mean_penetration_cm"], pen[mask].mean())
        np.testing.assert_allclose(summary["std_penetration_cm"], pen[mask].std())
        early = mask & (trace["time_s"] < 4 - 1e-6)
        late = trace["time_s"] >= 4 - 1e-6
        contact = np.flatnonzero(trace["applied_mpm_fz_N"] > 1.0)
        rows.append(
            {
                **summary,
                "late_minus_early_penetration_cm": float(pen[late].mean() - pen[early].mean()),
                "mean_mpm_normal_force_N": float(trace["applied_mpm_fz_N"][mask].mean()),
                "mean_rigid_ground_normal_force_N": float(trace["rigid_ground_fz_N"][mask].mean()),
                "mean_tracked_surface_settlement_cm": float(-100 * trace["tracked_sand_surface_z_m"][mask].mean()),
                "mean_box_depth_below_tracked_surface_cm": float(
                    100 * (trace["tracked_sand_surface_z_m"] - trace["box_sole_z_m"])[mask].mean()
                ),
                "contact_onset_s": float(trace["time_s"][contact[0]]) if len(contact) else None,
                "peak_mpm_force_before_approach_N": float(
                    np.abs(trace["applied_mpm_fz_N"][trace["time_s"] <= 0.5]).max()
                ),
            }
        )
        traces[width, depth] = trace
    with (data_dir / "statistics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(int(r["jitter_width_mm"]), int(r["sand_depth_cm"])): r for r in rows}
    validation = {
        "completed_runs": len(rows),
        "coupled_intervals": len(rows) * 2400,
        "all_finite": True,
        "momentum_balance_validated": True,
        "lagged_feedback_transfer_validated": True,
        "same_solver_source_and_dependencies": True,
        "maximum_relative_momentum_balance_error": max(r["maximum_relative_momentum_balance_error"] for r in rows),
        "rigid_ground_contact_cases": [
            [r["jitter_width_mm"], r["sand_depth_cm"]] for r in rows if r["peak_rigid_ground_normal_force_N"] > 1e-6
        ],
    }
    (data_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    widths = list(range(0, 17, 2))
    depths = list(range(20, 51, 5))
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, 7))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    footer = "5 kg box   |   MPM grid: 2 cm   |   MJWarp: 800 Hz   |   MPM: 400 Hz   |   25 MPM iterations"

    def save(fig, name):
        for ext in ["png", "svg"]:
            fig.savefig(data_dir / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    for key, title, ylabel, name in [
        (
            "mean_penetration_cm",
            "Box penetration: sensitivity to particle jitter",
            "Mean penetration below initial sand plane (cm)",
            "penetration_sensitivity",
        ),
        (
            "mean_mpm_normal_force_N",
            "MPM normal force applied to the box",
            "Mean upward MPM force over 2-6 s (N)",
            "normal_force_sensitivity",
        ),
        (
            "sand_surface_at_0p5s_cm",
            "Sand settling before the box approaches",
            "Tracked sand-surface height at 0.5 s (cm)",
            "sand_settling",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.7))
        for depth, color in zip(depths, colors, strict=True):
            ax.plot(widths, [lookup[w, depth][key] for w in widths], "o-", color=color, label=f"{depth} cm", ms=4)
        ax.set(xlabel="Total jitter width per coordinate (mm)", xticks=widths, ylabel=ylabel)
        ax.grid(alpha=0.2)
        fig.suptitle(title, fontsize=16, y=0.99)
        fig.legend(
            *ax.get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=7,
            frameon=False,
            title="Sand-bed thickness",
        )
        fig.text(0.075, 0.03, footer, fontsize=9, ha="left")
        fig.tight_layout(rect=(0, 0.09, 1, 0.87))
        save(fig, name)
    fig, ax = plt.subplots(figsize=(12, 6))
    means = np.array([[lookup[w, d]["mean_penetration_cm"] for w in widths] for d in depths])
    im = ax.imshow(means, origin="lower", aspect="auto", cmap="viridis")
    for i, depth in enumerate(depths):
        for j, width in enumerate(widths):
            r = lookup[width, depth]
            ax.text(
                j,
                i,
                f"{r['mean_penetration_cm']:.2f}\n±{r['std_penetration_cm']:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if means[i, j] < (means.min() + means.max()) / 2 else "black",
            )
    ax.set(
        xticks=range(9),
        xticklabels=widths,
        yticks=range(7),
        yticklabels=depths,
        xlabel="Total jitter width per coordinate (mm)",
        ylabel="Sand-bed thickness (cm)",
        title="Box penetration over 2-6 s: mean ± temporal std (cm)",
    )
    fig.colorbar(im, ax=ax, label="Mean penetration below initial sand plane (cm)")
    fig.text(0.075, 0.03, footer, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "penetration_matrix")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row")
    for col, width in enumerate([0, 16]):
        for depth, color in zip(depths, colors, strict=True):
            trace = traces[width, depth]
            axes[0, col].plot(
                trace["time_s"], 100 * np.maximum(-trace["box_sole_z_m"], 0), color=color, label=f"{depth} cm"
            )
            axes[1, col].plot(trace["time_s"], trace["applied_mpm_fz_N"], color=color)
        axes[0, col].set_title(f"{width} mm jitter")
        axes[1, col].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Box penetration below\ninitial sand plane (cm)")
    axes[1, 0].set_ylabel("MPM normal force\napplied to box (N)")
    for ax in axes.flat:
        ax.axvspan(2, 6, color="grey", alpha=0.07)
        ax.grid(alpha=0.2)
    fig.suptitle("Box motion and MPM force: zero and largest jitter", fontsize=16, y=0.99)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=7,
        frameon=False,
        title="Sand-bed thickness",
    )
    fig.text(0.075, 0.03, footer, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.86))
    save(fig, "time_traces")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
