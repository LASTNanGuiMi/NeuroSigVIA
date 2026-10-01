#!/usr/bin/env python
"""Verify saved_results/provenance.csv against the files it points to.

Run from the repository root:

    python scripts/check_results_index.py

Every indexed metric file and model file must exist, and the six window-level
test metrics stored in the index must equal the values in the metric file after
rounding to two decimals in percent. The exit code is 0 only when all records
pass. The checkpoint-selection metric declared by each run is also tabulated;
records that do not declare it are reported as "undeclared", never guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

METRIC_COLUMNS = {
    "Accuracy": "accuracy",
    "Precision": "macro_precision",
    "Recall": "macro_recall",
    "F1": "macro_f1",
    "AUROC": "macro_auroc",
    "AUPRC": "macro_auprc",
}
# Result schemas in use: main/ablation runners, baseline runners, TiViT probes.
TEST_METRIC_LOCATIONS = (("test_metrics",), ("test",), ("tivit_numeric", "test"))
SELECTION_KEYS = ("checkpoint_metric_effective", "checkpoint_metric", "patch_checkpoint_metric")
SELECTION_FILES = ("protocol.json", "status.json", "args.json")
TOLERANCE = 0.006


def _lookup(payload, keys):
    for key in keys:
        if not isinstance(payload, dict) or key not in payload:
            return None
        payload = payload[key]
    return payload if isinstance(payload, dict) else None


def test_metrics(payload):
    for keys in TEST_METRIC_LOCATIONS:
        metrics = _lookup(payload, keys)
        if metrics is not None and all(name in metrics for name in METRIC_COLUMNS.values()):
            return metrics
    return None


def declared_selection_metric(payload, metric_path: Path):
    """Return the declared checkpoint metric, searching the run directory upward."""

    candidates = [payload]
    for directory in list(metric_path.parents)[:3]:
        for name in SELECTION_FILES:
            path = directory / name
            if path != metric_path and path.is_file():
                try:
                    candidates.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in SELECTION_KEYS:
            value = candidate.get(key)
            if isinstance(value, str) and value:
                return value.removeprefix("validation_")
    # Older protocols state the rule only as a sentence.
    for candidate in candidates:
        text = candidate.get("selection") if isinstance(candidate, dict) else None
        if isinstance(text, str):
            words = text.lower().replace("-", " ").replace("_", " ")
            for unit in ("subject", "window"):
                if f"validation {unit} macro f1" in words:
                    return f"{unit}_macro_f1"
    return "undeclared"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=Path("saved_results/provenance.csv"))
    parser.add_argument("--root", type=Path, default=Path("."), help="directory that indexed paths are relative to")
    args = parser.parse_args()

    with args.index.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    selection = Counter()
    for number, row in enumerate(rows, start=2):
        label = f"line {number} ({row['表']} / {row['方法']} / {row['数据集']} / seed {row['seed']})"
        metric_path = args.root / row["测试指标文件"]
        model_path = args.root / row["训练后模型参数文件"]
        if not model_path.is_file():
            errors.append(f"{label}: missing model file {row['训练后模型参数文件']}")
        if not metric_path.is_file():
            errors.append(f"{label}: missing metric file {row['测试指标文件']}")
            continue
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        metrics = test_metrics(payload)
        if metrics is None:
            errors.append(f"{label}: no test metrics found in {row['测试指标文件']}")
            continue
        for column, key in METRIC_COLUMNS.items():
            if abs(float(row[column]) - 100.0 * float(metrics[key])) > TOLERANCE:
                errors.append(f"{label}: {column} index={row[column]} file={100.0 * float(metrics[key]):.4f}")
        selection[(row["表"], declared_selection_metric(payload, metric_path))] += 1

    print(f"records: {len(rows)}")
    print("checkpoint-selection metric declared by each run:")
    for (table, metric), count in sorted(selection.items()):
        print(f"  {table}\t{metric}\t{count}")
    for message in errors:
        print("ERROR", message)
    print("PASSED" if not errors else f"FAILED: {len(errors)} problem(s)")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
