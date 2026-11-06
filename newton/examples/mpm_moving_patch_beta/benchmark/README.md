# Moving terrain MPM benchmark

The scaling sweep starts at **3 × 3 m** and ends at **10 × 10 m**, in 1 m increments. It compares fixed, sparse, and moving-patch methods for box, rover, and G1. Equal-allocation cases use 1, 2, and 4 m side lengths.

- Report: [report.tex](report.tex).
- Figures: `00_scenes`, `01_grid_comparison`, `02_terrain_scaling`, and `03_equal_size_comparison`, each as PNG and vector PDF.
- Measurements: `raw/*.json` and corresponding logs; `summary.csv` and `summary.json` combine repeats.
- Provenance: `environment.json`, `software.json`, `source_hashes.json`, `asset_hashes.json`, and `gpu_process_history.jsonl`.
- `validation/` retains preliminary and setup-validation runs. Aggregation reads **only `raw/`**. The successful `validation/low_box_validation/` run uses the final setup and was copied into `raw/`.

The benchmark overrides `setup_terrain()` for **all three scenes**, so it does not use their different interactive material presets:

| Shared terrain parameter | Value |
| --- | --- |
| Density | 2,500 kg/m³ |
| Young's modulus | 40 MPa |
| Poisson's ratio | 0.3 |
| MPM friction coefficient | tan(50°), approximately 1.19 |
| Voxel size | 0.05 m |
| Particles per cell | 3 per axis, 27 total |
| Sand depth | 0.5 m |
| Jitter distance | 0.05 / 3 m |

Other MPM material attributes use the same Newton defaults. Robot masses, geometry, and controllers retain their scene-specific settings. The interactive example files are unchanged.

Run from the repository root:

```bash
uv run --no-sync python newton/examples/mpm_moving_patch_beta/benchmark/run_benchmarks.py
uv run --no-sync python newton/examples/mpm_moving_patch_beta/benchmark/plot_results.py
```

The default run performs two independent repeats, each with 150 warm-up and 150 measured physics steps. Successful existing results are retained when restarting. Use a clean output directory for a new experiment. Each worker uses a fresh CUDA process; the parent waits while another compute process occupies the GPU. Do not change simulation sources while the sweep runs.

The plotted size is the full particle-bed side length for fixed and sparse. Moving-patch keeps a 3 × 3 × 0.5 m allocation, including a 0.5 m kinematic boundary on each side. All methods use a background ground extent 2 m wider than the listed size to allow movement. Equal-allocation cases use a 0.1 m boundary on each side for all methods. The sparse baseline uses **positive reserved capacity**, matching beta moving-patch, rather than dynamic allocation with `max_active_cell_count = -1`.

The benchmark box starts with its center at 0.8 m, 0.05 m above the sand at its bottom, to avoid measuring the example's initial high-drop bounce. Body pose and free-joint coordinates are both initialized consistently.

Timing includes synchronized physics and controller work, with no rendering. Time error bars are the sample standard deviation of individual steps pooled across repeats. Memory error bars are the sample standard deviation of NVML samples from measurement start through final state checks. Sampled peak memory is recorded separately and includes initialization. Sizes with failed runs are not assigned invented timings. Active-cell counts are measured at the end of each run; summary minima and maxima compare those endpoints across repeats, not every simulation step.

No `pdflatex`, `latexmk`, `lualatex`, or `tectonic` executable was installed locally when checked. On a machine with LaTeX, compile from this directory:

```bash
latexmk -pdf report.tex
```

The `.tex` source uses the accompanying PDF figures and result paragraphs. Scene images are existing example illustrations; their appearance is not part of the timing measurement.
