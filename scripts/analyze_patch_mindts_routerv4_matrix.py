#!/usr/bin/env python3
"""Summarize a partial or complete Patch-MindTS Router-v4 2x2 matrix.

Example
-------
python scripts/analyze_patch_mindts_routerv4_matrix.py \
  --cell single_v4=results/single_v4 \
  --cell single_uniform=results/single_uniform \
  --cell composite_v4=results/composite_v4 \
  --cell composite_uniform=results/composite_uniform \
  --output-prefix results/routerv4_matrix

The four contrasts use these signed directions:

* representation under uniform: single_uniform - composite_uniform
* router within single: single_v4 - single_uniform
* router within composite: composite_v4 - composite_uniform
* interaction: (single_v4 - single_uniform)
               - (composite_v4 - composite_uniform)

Validation measurements may be used to compare alternatives.  Test measurements
are deliberately labelled as frozen-post-selection reports and are never marked
selection eligible by this program.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 2
CELL_DESIGN = {
    "single_v4": ("single", "adaptive_v4"),
    "single_uniform": ("single", "uniform"),
    "composite_v4": ("composite", "adaptive_v4"),
    "composite_uniform": ("composite", "uniform"),
}
DEFAULT_DATASETS = ("tdbrain", "apava", "shimmer10", "pads11")
METRIC_NAMES = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_auroc",
    "macro_auprc",
    "macro_log_loss",
)
CONTRASTS = {
    "representation_single_minus_composite_under_uniform": {
        "single_uniform": 1.0,
        "composite_uniform": -1.0,
    },
    "router_v4_minus_uniform_within_single": {
        "single_uniform": -1.0,
        "single_v4": 1.0,
    },
    "router_v4_minus_uniform_within_composite": {
        "composite_uniform": -1.0,
        "composite_v4": 1.0,
    },
    "router_effect_single_minus_composite_interaction": {
        "single_uniform": -1.0,
        "single_v4": 1.0,
        "composite_uniform": 1.0,
        "composite_v4": -1.0,
    },
}
CONTRAST_BASELINES = {
    "representation_single_minus_composite_under_uniform": "composite_uniform",
    "router_v4_minus_uniform_within_single": "single_uniform",
    "router_v4_minus_uniform_within_composite": "composite_uniform",
    "router_effect_single_minus_composite_interaction": None,
}


def _finite_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _normalise_dataset(value: Any) -> str:
    raw = str(value or "").strip().lower()
    compact = "".join(character for character in raw if character.isalnum())
    if "tdbrain" in compact:
        return "tdbrain"
    if "apava" in compact:
        return "apava"
    if "shimmer" in compact:
        return "shimmer10"
    if "pads" in compact:
        return "pads11"
    return compact or "unknown"


def _read_json(path: Path, warnings: List[str]) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        warnings.append(f"cannot read JSON {path}: {error}")
        return None
    if not isinstance(payload, dict):
        warnings.append(f"expected a JSON object in {path}")
        return None
    return payload


def _read_csv_rows(path: Path, warnings: List[str]) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeError, csv.Error) as error:
        warnings.append(f"cannot read CSV {path}: {error}")
        return []


def _path_is_below(path: Path, ancestor: Path) -> bool:
    try:
        path.resolve().relative_to(ancestor.resolve())
    except (OSError, ValueError):
        return False
    return True


def _discover_cell(root: Path, warnings: List[str]) -> Dict[str, Any]:
    discovery: Dict[str, Any] = {
        "csv_rows": {},
        "artifact_dirs": {},
        "all_csv_paths": [],
    }
    if not root.exists():
        warnings.append(f"cell root does not exist: {root}")
        return discovery
    if not root.is_dir():
        warnings.append(f"cell root is not a directory: {root}")
        return discovery

    for csv_path in sorted(root.rglob("train_val.csv")):
        discovery["all_csv_paths"].append(csv_path)
        for row_index, row in enumerate(_read_csv_rows(csv_path, warnings)):
            dataset = _normalise_dataset(row.get("dataset", csv_path.parent.name))
            discovery["csv_rows"].setdefault(dataset, []).append(
                {
                    "row": row,
                    "path": csv_path,
                    "row_index": row_index,
                }
            )

    artifact_names = (
        "patch_atgs_summary.json",
        "patch_mindts_training_history.json",
        "patch_mindts_checkpoint.pt",
        "patch_atgs_diagnostics_validation.npz",
        "patch_atgs_diagnostics_test.npz",
    )
    artifact_dirs: Dict[Path, Dict[str, Path]] = {}
    for name in artifact_names:
        for path in sorted(root.rglob(name)):
            artifact_dirs.setdefault(path.parent, {})[name] = path
    for directory, paths in artifact_dirs.items():
        dataset = _normalise_dataset(directory.name)
        discovery["artifact_dirs"].setdefault(dataset, []).append(
            {"directory": directory, **paths}
        )
    return discovery


def _new_split() -> Dict[str, Any]:
    return {"window": {}, "subject": {}, "router": {}}


def _merge_metric(
    destination: MutableMapping[str, float],
    name: str,
    value: Any,
    source: str,
    warnings: List[str],
) -> None:
    parsed = _finite_float(value)
    if parsed is None:
        return
    previous = destination.get(name)
    if previous is not None and not math.isclose(
        previous, parsed, rel_tol=1e-8, abs_tol=1e-10
    ):
        warnings.append(
            f"metric mismatch for {name}: {previous:.12g} versus "
            f"{parsed:.12g} from {source}; summary value wins"
        )
    destination[name] = parsed


def _metrics_from_csv(
    row: Optional[Mapping[str, str]], warnings: List[str]
) -> Dict[str, Dict[str, Any]]:
    result = {"val": _new_split(), "test": _new_split()}
    if row is None:
        return result
    for column, value in row.items():
        split = None
        remainder = ""
        if column.startswith("val_"):
            split, remainder = "val", column[4:]
        elif column.startswith("test_"):
            split, remainder = "test", column[5:]
        if split is None:
            continue
        unit = "subject" if remainder.startswith("subject_") else "window"
        metric = remainder[8:] if unit == "subject" else remainder
        if metric in METRIC_NAMES:
            _merge_metric(
                result[split][unit], metric, value, f"train_val.csv:{column}", warnings
            )
    return result


def _numeric_list(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)):
        return None
    parsed = [_finite_float(item) for item in value]
    if any(item is None for item in parsed):
        return None
    return [float(item) for item in parsed if item is not None]


def _router_from_summary(split_payload: Mapping[str, Any]) -> Dict[str, Any]:
    hierarchy = split_payload.get("router_hierarchy", {})
    hierarchy = hierarchy if isinstance(hierarchy, Mapping) else {}
    prediction_tv = hierarchy.get("prediction_probability_total_variation", {})
    prediction_tv = prediction_tv if isinstance(prediction_tv, Mapping) else {}
    router = {
        "granularity_labels": list(split_payload.get("granularity_labels", [])),
        "mean_weights": _numeric_list(split_payload.get("mean_weights")),
        "mean_max_weight": _finite_float(split_payload.get("mean_max_weight")),
        "mean_normalized_entropy": _finite_float(
            split_payload.get("mean_normalized_entropy")
        ),
        "top1_time_weighted_fraction": _numeric_list(
            split_payload.get("top1_time_weighted_fraction")
        ),
        "mean_score_margin": _finite_float(split_payload.get("mean_score_margin")),
        "dataset_prior_weights": _numeric_list(
            hierarchy.get("mean_dataset_prior_weights")
        ),
        "sample_global_weights": _numeric_list(
            hierarchy.get("mean_sample_global_weights")
        ),
        "patch_local_weights": _numeric_list(
            hierarchy.get("mean_patch_local_weights")
        ),
        "candidate_evidence": _finite_float(hierarchy.get("candidate_evidence")),
        "global_mix": _finite_float(hierarchy.get("global_mix")),
        "local_mix": _finite_float(hierarchy.get("local_mix")),
        "effective_global_mix": _finite_float(
            hierarchy.get("effective_global_mix")
        ),
        "effective_local_mix": _finite_float(
            hierarchy.get("effective_local_mix")
        ),
        "dataset_prior_prediction_tv": _finite_float(
            prediction_tv.get("dataset_prior")
        ),
        "sample_global_prediction_tv": _finite_float(
            prediction_tv.get("sample_global")
        ),
        "patch_local_prediction_tv": _finite_float(
            prediction_tv.get("patch_local")
        ),
    }
    return {key: value for key, value in router.items() if value is not None}


def _merge_summary_metrics(
    metrics: MutableMapping[str, Dict[str, Any]],
    summary: Optional[Mapping[str, Any]],
    warnings: List[str],
) -> None:
    if summary is None:
        return
    splits = summary.get("splits", summary)
    if not isinstance(splits, Mapping):
        warnings.append("patch_atgs_summary.json has no split mapping")
        return
    for output_name, aliases in (("val", ("validation", "val")), ("test", ("test",))):
        split_payload = next(
            (splits[name] for name in aliases if isinstance(splits.get(name), Mapping)),
            None,
        )
        if split_payload is None:
            continue
        for unit, source_key in (("window", "window_metrics"), ("subject", "subject_metrics")):
            source = split_payload.get(source_key, {})
            if not isinstance(source, Mapping):
                continue
            for metric, value in source.items():
                _merge_metric(
                    metrics[output_name][unit],
                    str(metric),
                    value,
                    f"patch_atgs_summary.json:{output_name}.{source_key}",
                    warnings,
                )
        metrics[output_name]["router"] = _router_from_summary(split_payload)


def _merge_diagnostic_log_losses(
    metrics: MutableMapping[str, Dict[str, Any]],
    split: str,
    path: Optional[Path],
    warnings: List[str],
) -> None:
    """Recompute window/subject log loss from frozen diagnostic probabilities."""
    if path is None:
        return
    try:
        import numpy as np
    except ImportError:
        warnings.append(f"cannot compute diagnostic log loss without NumPy: {path}")
        return
    try:
        with np.load(path, allow_pickle=False) as payload:
            for unit, prefix in (("window", ""), ("subject", "subject_")):
                true_key = f"{prefix}y_true"
                score_key = f"{prefix}y_score"
                if true_key not in payload or score_key not in payload:
                    warnings.append(
                        f"diagnostics missing {true_key}/{score_key}: {path}"
                    )
                    continue
                y_true = np.asarray(payload[true_key])
                y_score = np.asarray(payload[score_key], dtype=np.float64)
                if y_true.ndim != 1 or y_score.ndim != 2:
                    warnings.append(
                        f"invalid diagnostic shapes for {unit} log loss in {path}: "
                        f"{y_true.shape}, {y_score.shape}"
                    )
                    continue
                if y_true.shape[0] == 0 or y_score.shape[0] != y_true.shape[0]:
                    warnings.append(
                        f"invalid diagnostic lengths for {unit} log loss in {path}: "
                        f"{y_true.shape[0]}, {y_score.shape[0]}"
                    )
                    continue
                if not np.all(np.isfinite(y_score)):
                    warnings.append(f"non-finite {unit} probabilities in {path}")
                    continue
                class_index = y_true.astype(np.int64, copy=False)
                if not np.all(class_index == y_true) or np.any(class_index < 0) or np.any(
                    class_index >= y_score.shape[1]
                ):
                    warnings.append(f"invalid {unit} labels for log loss in {path}")
                    continue
                selected = y_score[np.arange(y_true.shape[0]), class_index]
                epsilon = np.finfo(np.float64).eps
                per_sample_loss = -np.log(np.clip(selected, epsilon, 1.0))
                class_losses = [
                    per_sample_loss[class_index == class_id].mean()
                    for class_id in np.unique(class_index)
                ]
                log_loss = float(np.mean(class_losses))
                previous = metrics[split][unit].get("macro_log_loss")
                if previous is None:
                    metrics[split][unit]["macro_log_loss"] = log_loss
                elif not math.isclose(
                    float(previous), log_loss, rel_tol=1e-6, abs_tol=1e-8
                ):
                    warnings.append(
                        f"metric mismatch for macro_log_loss: {float(previous):.12g} "
                        f"versus {log_loss:.12g} from {path.name}:{unit}; "
                        "summary value retained"
                    )
    except (OSError, ValueError, KeyError) as error:
        warnings.append(f"cannot read diagnostic NPZ {path}: {error}")


def _safe_checkpoint_metadata(path: Path, warnings: List[str]) -> Dict[str, Any]:
    """Load only a trusted local checkpoint through PyTorch's restricted loader."""
    try:
        import torch  # Imported lazily; history JSON normally avoids this cost.
    except ImportError:
        warnings.append(f"cannot inspect checkpoint without PyTorch: {path}")
        return {}
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            warnings.append(
                f"PyTorch lacks safe weights_only loading; skipped checkpoint metadata: {path}"
            )
            return {}
    except Exception as error:  # PyTorch uses several specialised exception types.
        warnings.append(f"cannot safely inspect checkpoint {path}: {error}")
        return {}
    if not isinstance(payload, Mapping):
        warnings.append(f"checkpoint is not a mapping: {path}")
        return {}
    allowed = (
        "selected_epoch",
        "checkpoint_selection_requested",
        "checkpoint_selection_effective",
        "checkpoint_selection_unit",
        "validation_macro_f1",
        "validation_subject_macro_f1",
    )
    return {key: payload.get(key) for key in allowed if key in payload}


def _choose_latest(candidates: Sequence[Dict[str, Any]], path_key: str) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item[path_key].stat().st_mtime if item[path_key].exists() else -1.0,
            str(item[path_key]),
            int(item.get("row_index", -1)),
        ),
    )


def _choose_artifacts(
    discovery: Mapping[str, Any],
    dataset: str,
    csv_choice: Optional[Mapping[str, Any]],
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    candidates = list(discovery["artifact_dirs"].get(dataset, []))
    if not candidates and csv_choice is not None:
        csv_parent = csv_choice["path"].parent
        all_candidates = [
            candidate
            for values in discovery["artifact_dirs"].values()
            for candidate in values
            if _path_is_below(candidate["directory"], csv_parent)
        ]
        if len(all_candidates) == 1:
            candidates = all_candidates
    if len(candidates) > 1:
        warnings.append(
            f"multiple artifact directories for {dataset}; using the newest: "
            + ", ".join(str(item["directory"]) for item in candidates)
        )
    return _choose_latest(candidates, "directory")


def _build_record(
    cell: str,
    root: Path,
    dataset: str,
    discovery: Mapping[str, Any],
) -> Dict[str, Any]:
    representation, router = CELL_DESIGN[cell]
    warnings: List[str] = []
    csv_candidates = list(discovery["csv_rows"].get(dataset, []))
    if len(csv_candidates) > 1:
        warnings.append(
            f"multiple train_val.csv rows for {dataset}; using the newest row"
        )
    csv_choice = _choose_latest(csv_candidates, "path")
    artifacts = _choose_artifacts(discovery, dataset, csv_choice, warnings)
    csv_row = csv_choice["row"] if csv_choice is not None else None
    metrics = _metrics_from_csv(csv_row, warnings)

    summary_path = artifacts.get("patch_atgs_summary.json") if artifacts else None
    summary = _read_json(summary_path, warnings) if summary_path else None
    _merge_summary_metrics(metrics, summary, warnings)

    diagnostic_paths = {
        "val": (
            artifacts.get("patch_atgs_diagnostics_validation.npz")
            if artifacts
            else None
        ),
        "test": (
            artifacts.get("patch_atgs_diagnostics_test.npz") if artifacts else None
        ),
    }
    for split, diagnostic_path in diagnostic_paths.items():
        _merge_diagnostic_log_losses(metrics, split, diagnostic_path, warnings)

    history_path = (
        artifacts.get("patch_mindts_training_history.json") if artifacts else None
    )
    history = _read_json(history_path, warnings) if history_path else None
    selection: Dict[str, Any] = {}
    selection_source = None
    if history is not None:
        for key in (
            "selected_epoch",
            "checkpoint_selection_requested",
            "checkpoint_selection_effective",
            "checkpoint_selection_key",
        ):
            if key in history:
                selection[key] = history[key]
        selection_source = "history_json"

    checkpoint_path = artifacts.get("patch_mindts_checkpoint.pt") if artifacts else None
    required_selection = (
        "selected_epoch",
        "checkpoint_selection_effective",
    )
    if checkpoint_path and any(key not in selection for key in required_selection):
        checkpoint_metadata = _safe_checkpoint_metadata(checkpoint_path, warnings)
        for key, value in checkpoint_metadata.items():
            selection.setdefault(key, value)
        if checkpoint_metadata:
            selection_source = (
                "history_json+checkpoint" if selection_source else "checkpoint"
            )

    if "selected_epoch" in selection:
        try:
            selection["selected_epoch"] = int(selection["selected_epoch"])
        except (TypeError, ValueError):
            warnings.append("selected_epoch is not an integer")
            selection["selected_epoch"] = None

    missing_fields = []
    if csv_choice is None:
        missing_fields.append("train_val.csv")
    if summary_path is None:
        missing_fields.append("patch_atgs_summary.json")
    for split in ("val", "test"):
        if metrics[split]["window"].get("macro_f1") is None:
            missing_fields.append(f"{split}.window.macro_f1")
        if metrics[split]["subject"].get("macro_f1") is None:
            missing_fields.append(f"{split}.subject.macro_f1")
        route = metrics[split]["router"]
        for key in (
            "mean_weights",
            "mean_max_weight",
            "mean_normalized_entropy",
            "top1_time_weighted_fraction",
            "candidate_evidence",
            "global_mix",
            "local_mix",
            "effective_global_mix",
            "effective_local_mix",
            "dataset_prior_prediction_tv",
            "sample_global_prediction_tv",
            "patch_local_prediction_tv",
        ):
            if key not in route:
                missing_fields.append(f"{split}.router.{key}")
        if diagnostic_paths[split] is not None:
            for unit in ("window", "subject"):
                if metrics[split][unit].get("macro_log_loss") is None:
                    missing_fields.append(f"{split}.{unit}.macro_log_loss")
    for key in required_selection:
        if selection.get(key) is None:
            missing_fields.append(key)

    has_anything = csv_choice is not None or artifacts is not None
    status = "complete" if not missing_fields else "partial" if has_anything else "missing"
    return {
        "cell": cell,
        "representation": representation,
        "router": router,
        "dataset": dataset,
        "status": status,
        "root": str(root),
        "artifacts": {
            "train_val_csv": str(csv_choice["path"]) if csv_choice else None,
            "patch_atgs_summary_json": str(summary_path) if summary_path else None,
            "diagnostics_validation_npz": (
                str(diagnostic_paths["val"]) if diagnostic_paths["val"] else None
            ),
            "diagnostics_test_npz": (
                str(diagnostic_paths["test"]) if diagnostic_paths["test"] else None
            ),
            "training_history_json": str(history_path) if history_path else None,
            "checkpoint_pt": str(checkpoint_path) if checkpoint_path else None,
        },
        "selection": {**selection, "source": selection_source},
        "metrics": metrics,
        "test_role": "frozen_post_selection_report_only",
        "missing_fields": missing_fields,
        "warnings": warnings,
    }


def _measurement_map(record: Mapping[str, Any]) -> Dict[str, float]:
    measurements: Dict[str, float] = {}
    for split in ("val", "test"):
        split_metrics = record["metrics"][split]
        for unit in ("window", "subject"):
            for metric, value in split_metrics[unit].items():
                parsed = _finite_float(value)
                if parsed is not None:
                    measurements[f"{split}.{unit}.{metric}"] = parsed
        route = split_metrics["router"]
        for name, value in route.items():
            if name == "granularity_labels":
                continue
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    parsed = _finite_float(item)
                    if parsed is not None:
                        measurements[f"{split}.router.{name}.{index}"] = parsed
            else:
                parsed = _finite_float(value)
                if parsed is not None:
                    measurements[f"{split}.router.{name}"] = parsed
    return measurements


def _split_from_metric(metric: str) -> str:
    return metric.split(".", 1)[0]


def _contrast_rows(
    records: Sequence[Mapping[str, Any]], datasets: Sequence[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_key = {(record["cell"], record["dataset"]): record for record in records}
    measurements = {key: _measurement_map(record) for key, record in by_key.items()}
    availability = []
    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for contrast, coefficients in CONTRASTS.items():
            missing_cells = [
                cell
                for cell in coefficients
                if by_key.get((cell, dataset), {}).get("status") == "missing"
            ]
            availability.append(
                {
                    "dataset": dataset,
                    "contrast": contrast,
                    "available": not missing_cells,
                    "missing_cells": missing_cells,
                }
            )
            if missing_cells:
                continue
            metric_sets = [
                set(measurements.get((cell, dataset), {})) for cell in coefficients
            ]
            common_metrics = set.intersection(*metric_sets) if metric_sets else set()
            baseline_cell = CONTRAST_BASELINES[contrast]
            for metric in sorted(common_metrics):
                estimate = sum(
                    coefficient * measurements[(cell, dataset)][metric]
                    for cell, coefficient in coefficients.items()
                )
                baseline = (
                    measurements[(baseline_cell, dataset)][metric]
                    if baseline_cell is not None
                    else None
                )
                relative_percent = (
                    100.0 * estimate / abs(baseline)
                    if baseline is not None and abs(baseline) > 1e-12
                    else None
                )
                split = _split_from_metric(metric)
                rows.append(
                    {
                        "scope": "dataset",
                        "dataset": dataset,
                        "contrast": contrast,
                        "metric": metric,
                        "estimate": estimate,
                        "baseline_cell": baseline_cell,
                        "baseline_value": baseline,
                        "relative_percent": relative_percent,
                        "n_datasets": 1,
                        "expected_dataset_count": len(datasets),
                        "partial": False,
                        "reporting_role": (
                            "validation_comparison"
                            if split == "val"
                            else "frozen_post_selection_report_only"
                        ),
                        "selection_eligible": split == "val",
                    }
                )

    aggregate_rows = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["contrast"], row["metric"]), []).append(row)
    for (contrast, metric), items in sorted(grouped.items()):
        estimates = [float(item["estimate"]) for item in items]
        baselines = [
            float(item["baseline_value"])
            for item in items
            if item["baseline_value"] is not None
        ]
        mean_estimate = statistics.fmean(estimates)
        mean_baseline = statistics.fmean(baselines) if len(baselines) == len(items) else None
        split = _split_from_metric(metric)
        aggregate_rows.append(
            {
                "scope": "macro_across_datasets",
                "dataset": "__macro__",
                "contrast": contrast,
                "metric": metric,
                "estimate": mean_estimate,
                "std_across_datasets": (
                    statistics.stdev(estimates) if len(estimates) > 1 else None
                ),
                "baseline_cell": CONTRAST_BASELINES[contrast],
                "baseline_value": mean_baseline,
                "relative_percent": (
                    100.0 * mean_estimate / abs(mean_baseline)
                    if mean_baseline is not None and abs(mean_baseline) > 1e-12
                    else None
                ),
                "n_datasets": len(items),
                "expected_dataset_count": len(datasets),
                "partial": len(items) < len(datasets),
                "included_datasets": [item["dataset"] for item in items],
                "reporting_role": (
                    "validation_comparison"
                    if split == "val"
                    else "frozen_post_selection_report_only"
                ),
                "selection_eligible": split == "val",
            }
        )
    return rows + aggregate_rows, availability


def _flatten_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {
        "cell": record["cell"],
        "representation": record["representation"],
        "router": record["router"],
        "dataset": record["dataset"],
        "status": record["status"],
        "selected_epoch": record["selection"].get("selected_epoch"),
        "checkpoint_selection_requested": record["selection"].get(
            "checkpoint_selection_requested"
        ),
        "checkpoint_selection_effective": record["selection"].get(
            "checkpoint_selection_effective"
        ),
        "selection_metadata_source": record["selection"].get("source"),
        "test_role": record["test_role"],
        "missing_fields": json.dumps(record["missing_fields"], separators=(",", ":")),
        "warnings": json.dumps(record["warnings"], separators=(",", ":")),
    }
    for name, path in record["artifacts"].items():
        flat[f"artifact_{name}"] = path
    for split in ("val", "test"):
        for unit in ("window", "subject"):
            for metric, value in record["metrics"][split][unit].items():
                flat[f"{split}_{unit}_{metric}"] = value
        route = record["metrics"][split]["router"]
        for name, value in route.items():
            if isinstance(value, (list, tuple)):
                flat[f"{split}_router_{name}"] = json.dumps(
                    value, separators=(",", ":")
                )
                for index, item in enumerate(value):
                    flat[f"{split}_router_{name}_{index}"] = item
            else:
                flat[f"{split}_router_{name}"] = value
    return flat


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    preferred = [
        "scope",
        "cell",
        "representation",
        "router",
        "dataset",
        "status",
        "contrast",
        "metric",
        "estimate",
    ]
    fields = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in fields]
    fieldnames.extend(sorted(fields.difference(fieldnames)))
    if not fieldnames:
        fieldnames = ["status"]
        rows = [{"status": "no_rows"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                serialised = {}
                for key, value in _json_safe(dict(row)).items():
                    serialised[key] = (
                        json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (list, dict))
                        else value
                    )
                writer.writerow(serialised)
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def analyse(cells: Mapping[str, Path], datasets: Sequence[str]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    global_warnings: List[str] = []
    for cell in CELL_DESIGN:
        if cell not in cells:
            root = Path("<not-supplied>")
            discovery = {"csv_rows": {}, "artifact_dirs": {}, "all_csv_paths": []}
        else:
            root = cells[cell]
            cell_warnings: List[str] = []
            discovery = _discover_cell(root, cell_warnings)
            global_warnings.extend(f"{cell}: {warning}" for warning in cell_warnings)
        for dataset in datasets:
            records.append(_build_record(cell, root, dataset, discovery))

    contrasts, availability = _contrast_rows(records, datasets)
    completeness = {}
    for cell in CELL_DESIGN:
        cell_records = [record for record in records if record["cell"] == cell]
        completeness[cell] = {
            status: sum(record["status"] == status for record in cell_records)
            for status in ("complete", "partial", "missing")
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "patch_mindts_routerv4_candidate_router_2x2",
        "analysis_policy": {
            "validation": "comparison_and_checkpoint_selection_diagnostics",
            "test": "frozen_post_selection_report_only",
            "test_used_for_selection": False,
            "aggregate": "unweighted_macro_mean_across_available_datasets",
        },
        "contrast_directions": {
            name: coefficients for name, coefficients in CONTRASTS.items()
        },
        "cell_roots": {cell: str(path) for cell, path in cells.items()},
        "expected_datasets": list(datasets),
        "completeness": completeness,
        "records": records,
        "contrast_availability": availability,
        "contrasts": contrasts,
        "warnings": global_warnings,
    }


def _parse_cells(values: Sequence[str], parser: argparse.ArgumentParser) -> Dict[str, Path]:
    cells: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            parser.error(f"--cell expects CELL=ROOT, got {value!r}")
        cell, root = value.split("=", 1)
        if cell not in CELL_DESIGN:
            parser.error(
                f"unknown cell {cell!r}; choose from {', '.join(CELL_DESIGN)}"
            )
        if cell in cells:
            parser.error(f"duplicate --cell for {cell}")
        if not root.strip():
            parser.error(f"empty root for cell {cell}")
        cells[cell] = Path(root).expanduser().resolve()
    return cells


def _write_fixture_cell(root: Path, cell: str, datasets: Sequence[str]) -> None:
    representation, router = CELL_DESIGN[cell]
    representation_effect = 0.02 if representation == "composite" else 0.0
    router_effect = (
        0.05 if representation == "composite" else 0.03
    ) if router == "adaptive_v4" else 0.0
    for index, dataset in enumerate(datasets):
        run_dir = root / dataset
        artifact_dir = run_dir / "patch_atgs" / dataset
        artifact_dir.mkdir(parents=True, exist_ok=True)
        score = 0.50 + 0.01 * index + representation_effect + router_effect
        with (run_dir / "train_val.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "dataset",
                    "val_macro_f1",
                    "val_subject_macro_f1",
                    "test_macro_f1",
                    "test_subject_macro_f1",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "dataset": dataset,
                    "val_macro_f1": score,
                    "val_subject_macro_f1": score - 0.01,
                    "test_macro_f1": score - 0.02,
                    "test_subject_macro_f1": score - 0.03,
                }
            )
        split_payload = {
            "granularity_labels": ["4", "8", "16"],
            "window_metrics": {"macro_f1": score},
            "subject_metrics": {"macro_f1": score - 0.01},
            "mean_weights": [0.30, 0.40, 0.30],
            "mean_max_weight": 0.40,
            "mean_normalized_entropy": 0.95,
            "top1_time_weighted_fraction": [0.2, 0.6, 0.2],
            "mean_score_margin": 0.1,
            "router_hierarchy": {
                "mean_dataset_prior_weights": [0.33, 0.34, 0.33],
                "mean_sample_global_weights": [0.31, 0.38, 0.31],
                "mean_patch_local_weights": [0.30, 0.40, 0.30],
                "candidate_evidence": 0.2,
                "global_mix": 0.5,
                "local_mix": 0.1,
                "effective_global_mix": 0.1,
                "effective_local_mix": 0.02,
                "prediction_probability_total_variation": {
                    "dataset_prior": 0.01,
                    "sample_global": 0.02,
                    "patch_local": 0.03,
                },
            },
        }
        test_payload = json.loads(json.dumps(split_payload))
        test_payload["window_metrics"]["macro_f1"] = score - 0.02
        test_payload["subject_metrics"]["macro_f1"] = score - 0.03
        (artifact_dir / "patch_atgs_summary.json").write_text(
            json.dumps(
                {"schema_version": 5, "splits": {"validation": split_payload, "test": test_payload}}
            ),
            encoding="utf-8",
        )
        (artifact_dir / "patch_mindts_training_history.json").write_text(
            json.dumps(
                {
                    "schema_version": 5,
                    "selected_epoch": 3,
                    "checkpoint_selection_requested": "auto",
                    "checkpoint_selection_effective": "subject_macro_f1",
                    "epochs": [],
                }
            ),
            encoding="utf-8",
        )


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="routerv4_matrix_fixture_") as directory:
        base = Path(directory)
        cells = {}
        for cell in CELL_DESIGN:
            root = base / cell
            fixture_datasets = list(DEFAULT_DATASETS)
            if cell == "composite_v4":
                fixture_datasets = fixture_datasets[:-1]
            _write_fixture_cell(root, cell, fixture_datasets)
            cells[cell] = root
        report = analyse(cells, DEFAULT_DATASETS)
        assert report["completeness"]["single_v4"]["complete"] == 4
        assert report["completeness"]["composite_v4"]["missing"] == 1
        representation = next(
            row
            for row in report["contrasts"]
            if row["scope"] == "dataset"
            and row["dataset"] == "tdbrain"
            and row["contrast"]
            == "representation_single_minus_composite_under_uniform"
            and row["metric"] == "val.window.macro_f1"
        )
        assert math.isclose(representation["estimate"], -0.02, abs_tol=1e-12)
        interaction = next(
            row
            for row in report["contrasts"]
            if row["scope"] == "dataset"
            and row["dataset"] == "tdbrain"
            and row["contrast"] == "router_effect_single_minus_composite_interaction"
            and row["metric"] == "val.window.macro_f1"
        )
        assert math.isclose(interaction["estimate"], -0.02, abs_tol=1e-12)
        test_row = next(
            row
            for row in report["contrasts"]
            if row["scope"] == "dataset" and row["metric"] == "test.window.macro_f1"
        )
        assert test_row["selection_eligible"] is False
        assert test_row["reporting_role"] == "frozen_post_selection_report_only"
        partial_macro = next(
            row
            for row in report["contrasts"]
            if row["scope"] == "macro_across_datasets"
            and row["contrast"] == "router_effect_single_minus_composite_interaction"
            and row["metric"] == "val.window.macro_f1"
        )
        assert partial_macro["partial"] is True
        assert partial_macro["n_datasets"] == 3
        json_path = base / "analysis.json"
        cells_path = base / "analysis_cells.csv"
        contrasts_path = base / "analysis_contrasts.csv"
        _atomic_text(
            json_path,
            json.dumps(_json_safe(report), allow_nan=False),
        )
        _write_csv(cells_path, [_flatten_record(row) for row in report["records"]])
        _write_csv(contrasts_path, report["contrasts"])
        assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == 2
        assert len(_read_csv_rows(cells_path, [])) == 16
        assert _read_csv_rows(contrasts_path, [])
    print("self-test: ok")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only summarizer for single/composite x adaptive-v4/uniform "
            "Patch-MindTS result roots. Missing or unfinished cells are allowed."
        )
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        metavar="CELL=ROOT",
        help=(
            "repeat for any available cell; CELL is single_v4, single_uniform, "
            "composite_v4, or composite_uniform"
        ),
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="comma-separated expected datasets (default: %(default)s)",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="write PREFIX.json, PREFIX_cells.csv, and PREFIX_contrasts.csv",
    )
    parser.add_argument(
        "--json-out", type=Path, help="optional explicit JSON output path"
    )
    parser.add_argument(
        "--cells-csv-out", type=Path, help="optional explicit raw-cell CSV path"
    )
    parser.add_argument(
        "--contrasts-csv-out", type=Path, help="optional explicit contrasts CSV path"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a temporary synthetic 2x2/partial-matrix fixture and exit",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    cells = _parse_cells(args.cell, parser)
    if not cells:
        parser.error("provide at least one --cell CELL=ROOT (or use --self-test)")
    datasets = tuple(
        dict.fromkeys(
            _normalise_dataset(item)
            for item in args.datasets.split(",")
            if item.strip()
        )
    )
    if not datasets:
        parser.error("--datasets must contain at least one dataset")
    report = analyse(cells, datasets)
    json_text = json.dumps(
        _json_safe(report), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"

    json_out = args.json_out
    cells_csv_out = args.cells_csv_out
    contrasts_csv_out = args.contrasts_csv_out
    if args.output_prefix is not None:
        prefix = args.output_prefix
        json_out = json_out or Path(f"{prefix}.json")
        cells_csv_out = cells_csv_out or Path(f"{prefix}_cells.csv")
        contrasts_csv_out = contrasts_csv_out or Path(f"{prefix}_contrasts.csv")
    if json_out is not None:
        _atomic_text(json_out, json_text)
    if cells_csv_out is not None:
        _write_csv(cells_csv_out, [_flatten_record(record) for record in report["records"]])
    if contrasts_csv_out is not None:
        _write_csv(contrasts_csv_out, report["contrasts"])
    if json_out is None and cells_csv_out is None and contrasts_csv_out is None:
        sys.stdout.write(json_text)
    else:
        written = [
            str(path)
            for path in (json_out, cells_csv_out, contrasts_csv_out)
            if path is not None
        ]
        print("wrote " + ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
