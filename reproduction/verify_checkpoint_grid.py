#!/usr/bin/env python3
"""Evaluate all 5 x 3 archived TimeMosaic checkpoints and summarize deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid


DATASETS = ("adftd", "tdbrain", "apava", "shimmer10", "pads11")
SEEDS = (42, 43, 44)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--visual-batch-size", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    arguments = parser.parse_args()
    if arguments.tolerance < 0:
        parser.error("--tolerance must be non-negative")

    project_root = arguments.project_root.expanduser().resolve()
    asset_root = arguments.asset_root.expanduser().resolve()
    runtime_root = arguments.runtime_root.expanduser().resolve()
    evaluator = project_root / "reproduction" / "evaluate_checkpoint.py"
    if not evaluator.is_file():
        parser.error(f"evaluator does not exist: {evaluator}")

    log_root = runtime_root / "verification" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    cache_namespace = f"grid_{uuid.uuid4().hex}"
    runs = []
    for dataset in DATASETS:
        for seed in SEEDS:
            command = [
                sys.executable,
                str(evaluator),
                "--project-root",
                str(project_root),
                "--asset-root",
                str(asset_root),
                "--runtime-root",
                str(runtime_root),
                "--dataset",
                dataset,
                "--seed",
                str(seed),
                "--visual-batch-size",
                str(arguments.visual_batch_size),
                "--tolerance",
                str(arguments.tolerance),
                "--cache-namespace",
                cache_namespace,
                "--reuse-verification-cache",
            ]
            log_path = log_root / f"{dataset}_seed_{seed}.log"
            print(f"Evaluating {dataset} seed {seed}", flush=True)
            with log_path.open("wb") as log:
                process = subprocess.run(
                    command,
                    cwd=project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            result_path = runtime_root / "verification" / f"{dataset}_seed_{seed}.json"
            result = None
            if result_path.is_file():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    result = None
            runs.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "exit_code": process.returncode,
                    "status": result.get("status") if result else "missing_result",
                    "max_abs_delta": max(
                        (abs(float(value)) for value in result.get("delta", {}).values()),
                        default=None,
                    )
                    if result
                    else None,
                    "result": str(result_path),
                    "log": str(log_path),
                }
            )

    verified = [run for run in runs if run["exit_code"] == 0 and run["status"] == "verified"]
    report = {
        "status": "verified" if len(verified) == 15 else "mismatch_or_failure",
        "tolerance": arguments.tolerance,
        "verified_runs": len(verified),
        "expected_runs": 15,
        "runs": runs,
    }
    output = runtime_root / "verification" / "checkpoint_grid_summary.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if len(verified) == 15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
