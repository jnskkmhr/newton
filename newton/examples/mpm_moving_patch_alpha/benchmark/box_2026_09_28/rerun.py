# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Rerun the alpha box size sweep, preserving earlier benchmark results."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent

if __name__ == "__main__":
    cases = [f"scaling_box_{method}_{size}" for size in range(3, 11) for method in ("fixed", "sparse", "moving-patch")]
    subprocess.run(
        [
            sys.executable,
            str(OUT.parent / "physics_vram" / "run_benchmarks.py"),
            "--background-margin",
            "0",
            "--report-dir",
            str(OUT),
            "--output",
            str(OUT / "raw"),
            "--warmup",
            "150",
            "--steps",
            "150",
            "--repeats",
            "2",
            "--only",
            *cases,
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(OUT.parent / "physics_vram" / "run_benchmarks.py"),
            "--background-margin",
            "0",
            "--report-dir",
            str(OUT / "traversal"),
            "--output",
            str(OUT / "traversal/raw"),
            "--warmup",
            "150",
            "--steps",
            "1200",
            "--repeats",
            "2",
            "--only",
            "traversal_box_moving-patch_6",
        ],
        cwd=ROOT,
        check=True,
    )
