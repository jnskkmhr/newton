# SPDX-FileCopyrightText: Copyright (c) 2026 Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the fixed-container box benchmark over jitter widths and sand depths."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    benchmark = Path(__file__).with_name("benchmark_box_jitter.py").resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(benchmark.parents[3])
    env["PYTHONUNBUFFERED"] = "1"
    script_hash = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    plans = []
    for width in range(0, 17, 2):
        for depth in range(20, 51, 5):
            folder = (args.output / f"j{width:02d}_d{depth}").resolve()
            command = [
                sys.executable,
                str(benchmark),
                "--jitter",
                str(width / 1000),
                "--sand-depth",
                str(depth / 100),
                "--output",
                str(folder),
            ]
            plans.append({"jitter_width_mm": width, "sand_depth_cm": depth, "folder": str(folder), "command": command})
    (args.output / "plan.json").write_text(
        json.dumps(
            {"benchmark_sha256": script_hash, "pythonpath": env["PYTHONPATH"], "jobs": args.jobs, "runs": plans},
            indent=2,
        )
        + "\n"
    )

    def run(plan):
        folder = Path(plan["folder"])
        summary = folder / "summary.json"
        start = time.perf_counter()
        reused = summary.exists()
        if reused:
            result = json.loads(summary.read_text())
            assert result["samples"] == 2400 and result["feedback_transfer_validated"]
            assert result["maximum_relative_momentum_balance_error"] < 0.001
            returncode = 0
        else:
            with folder.with_suffix(".log").open("w") as log:
                returncode = subprocess.run(plan["command"], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            result = json.loads(summary.read_text()) if summary.exists() else None
        return plan | {
            "returncode": returncode,
            "reused": reused,
            "wall_seconds": time.perf_counter() - start,
            "summary": result,
        }

    results = []
    with (args.output / "progress.jsonl").open("w") as progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run, plan) for plan in plans]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                progress.write(json.dumps(result) + "\n")
                progress.flush()
                print(
                    json.dumps(
                        {
                            "completed": len(results),
                            "width_mm": result["jitter_width_mm"],
                            "depth_cm": result["sand_depth_cm"],
                            "returncode": result["returncode"],
                            "wall_seconds": result["wall_seconds"],
                        }
                    ),
                    flush=True,
                )
    results.sort(key=lambda r: (r["jitter_width_mm"], r["sand_depth_cm"]))
    (args.output / "execution.json").write_text(json.dumps(results, indent=2) + "\n")
    assert len(results) == 63 and all(r["returncode"] == 0 for r in results), "Inspect failed cases before analysis"


if __name__ == "__main__":
    main()
