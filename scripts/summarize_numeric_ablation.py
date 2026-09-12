"""Collect paired six-metric results with explicit evaluation and selection units."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auroc", "macro_auprc"]
NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC"]
DATASETS = ["shimmer10", "pads11", "apava", "tdbrain"]
CHECKPOINT_METRICS = {"window_macro_f1", "subject_macro_f1"}


def checkpoint_protocol(summary, summary_path, run_root=None):
    """Resolve actual selection provenance, never infer it from requested report units."""
    summary_path = Path(summary_path)
    sources = [("summary", summary)]
    declared = summary.get("protocol")
    if isinstance(declared, dict):
        sources.append(("summary.protocol", declared))
    else:
        path = summary_path.parent / (declared or "protocol.json")
        if path.is_file():
            sources.append((str(path), json.loads(path.read_text(encoding="utf-8"))))
        elif declared:
            raise FileNotFoundError(f"declared protocol is missing: {path}")
    found = []
    for name, source in sources:
        metric = source.get("checkpoint_metric_effective")
        if metric is not None:
            if metric not in CHECKPOINT_METRICS:
                raise ValueError(f"invalid effective checkpoint metric in {name}: {metric}")
            found.append((metric, name))
    if len({metric for metric, _ in found}) > 1:
        raise ValueError(f"conflicting checkpoint protocols: {summary_path}: {found}")
    if found:
        return dict(metric=found[0][0], source=found[0][1], legacy_inferred=False)
    # Recover historical selection only from actual run metadata. A requested
    # "auto" value is not proof of which metric became effective.
    legacy = []
    for name, source in sources:
        description = source.get("selection", "")
        if isinstance(description, str):
            for prefix, metric in [("validation subject macro-f1", "subject_macro_f1"),
                                   ("validation window macro-f1", "window_macro_f1")]:
                if description.lower().startswith(prefix):
                    legacy.append((metric, name + ".selection"))
        training_args = source.get("training_args", {})
        if isinstance(training_args, dict) and training_args.get("patch_checkpoint_metric") in CHECKPOINT_METRICS:
            legacy.append((training_args["patch_checkpoint_metric"], name + ".training_args.patch_checkpoint_metric"))
    boundary = Path(run_root).resolve() if run_root is not None else summary_path.parent.resolve()
    directory = summary_path.parent.resolve()
    if not directory.is_relative_to(boundary):
        raise ValueError(f"summary lies outside the specified run root: {summary_path}")
    while True:
        args_path = directory / "args.json"
        if args_path.is_file():
            metric = json.loads(args_path.read_text(encoding="utf-8")).get("patch_checkpoint_metric")
            if metric in CHECKPOINT_METRICS:
                legacy.append((metric, str(args_path) + ":patch_checkpoint_metric"))
            break  # Never use an unrelated higher-level args file instead.
        if directory == boundary:
            break
        directory = directory.parent
    if len({metric for metric, _ in legacy}) > 1:
        raise ValueError(f"conflicting historical checkpoint protocols: {summary_path}: {legacy}")
    if legacy:
        return dict(metric=legacy[0][0], source=legacy[0][1], legacy_inferred=True)
    return dict(metric="unknown", source="no_effective_or_historical_selection_evidence", legacy_inferred=False)


def metric_vector(summary, level):
    if level not in {"window", "subject"}:
        raise ValueError(f"unsupported metric level: {level}")
    prefix = "subject_" if level == "subject" else ""
    vector = np.asarray([summary["test_metrics"][prefix + key] for key in METRICS], dtype=float) * 100
    if not np.isfinite(vector).all():
        raise ValueError("non-finite result")
    return vector


def collect_report(run, reference_run, level="window"):
    run, reference_run = Path(run), Path(reference_run)
    if not run.is_dir() or not reference_run.is_dir():
        raise FileNotFoundError("both run directories must exist")
    architecture_labels = {
        "neurosigvia_numeric_only_zero_visual_slot_v1": "Numeric only (zero visual slot)",
        "neurosigvia_numeric_only_no_fusion_v2": "Numeric only (no fusion)",
    }
    architectures = {json.loads(path.read_text(encoding="utf-8"))["architecture"] for path in run.rglob("numeric_ablation_summary.json")}
    if len(architectures) > 1:
        raise ValueError("refusing to aggregate mixed numeric-only architectures")
    numeric_architecture = next(iter(architectures), None)
    if numeric_architecture is not None and numeric_architecture not in architecture_labels:
        raise ValueError(f"unknown numeric-only architecture: {numeric_architecture}")
    numeric_label = architecture_labels.get(numeric_architecture, "Numeric only (pending)")
    rows, grouped, contrasts, missing, warnings = [], [], [], [], []
    for dataset in DATASETS:
        paired, paired_protocols = {}, {}
        for label, root, pattern in [("Dual modality", reference_run, "adaptive_graph_summary.json"), (numeric_label, run, "numeric_ablation_summary.json")]:
            values, protocols = [], []
            for seed in [42, 43, 44]:
                paths = list((root / f"seed{seed}" / dataset).rglob(pattern))
                if not paths:
                    missing.append(f"{label}/{dataset}/seed{seed}")
                    continue
                if len(paths) != 1:
                    raise ValueError(f"ambiguous results: {paths}")
                summary = json.loads(paths[0].read_text(encoding="utf-8"))
                if label == numeric_label and summary.get("state") != "COMPLETED":
                    raise ValueError(f"incomplete summary: {paths[0]}")
                selection = checkpoint_protocol(summary, paths[0], root)
                vector = metric_vector(summary, level)
                values.append(vector)
                protocols.append(selection["metric"])
                rows.append(dict(method=label, dataset=dataset, seed=seed, metric_level=level,
                                 checkpoint_metric_effective=selection["metric"],
                                 checkpoint_protocol_source=selection["source"],
                                 legacy_protocol_inferred=selection["legacy_inferred"],
                                 **dict(zip(NAMES, vector)), summary=str(paths[0])))
            if len(set(protocols)) > 1:
                warnings.append(f"{label}/{dataset}: mixed checkpoint selection across seeds; aggregate and paired contrast withheld.")
                continue
            if len(values) == 3:
                matrix = np.stack(values)
                paired[label], paired_protocols[label] = matrix, protocols[0]
                grouped.append(dict(method=label, dataset=dataset, n=3, checkpoint_metric_effective=protocols[0],
                                    **{name: dict(mean=float(matrix[:, i].mean()), std=float(matrix[:, i].std(ddof=1))) for i, name in enumerate(NAMES)}))
        if len(paired) == 2:
            matched = (paired_protocols["Dual modality"] == paired_protocols[numeric_label]
                       and paired_protocols["Dual modality"] in CHECKPOINT_METRICS)
            if not matched:
                warnings.append(f"{dataset}: dual and numeric checkpoint selection is different or unverified; contrast is descriptive only, not a protocol-matched ablation.")
            delta = paired["Dual modality"] - paired[numeric_label]
            baseline = paired[numeric_label].mean(axis=0)
            contrasts.append(dict(dataset=dataset, direction="dual minus numeric", checkpoint_protocols_matched=matched,
                                  dual_checkpoint_metric=paired_protocols["Dual modality"], numeric_checkpoint_metric=paired_protocols[numeric_label],
                                  **{name: dict(delta_pp_mean=float(delta[:, i].mean()), delta_pp_std=float(delta[:, i].std(ddof=1)), relative_improvement_percent=float(100 * delta[:, i].mean() / baseline[i]) if baseline[i] != 0 else None, positive_seeds=int((delta[:, i] > 0).sum()), tied_seeds=int((delta[:, i] == 0).sum())) for i, name in enumerate(NAMES)}))
    if any(row["legacy_protocol_inferred"] for row in rows):
        warnings.append("Some archived runs have no effective-selection field; their historical selection is recovered from recorded protocol descriptions or explicit training arguments. Each run records that evidence source; reporting different units does not change the saved checkpoint's selection.")
    if any(row["checkpoint_metric_effective"] == "unknown" for row in rows):
        warnings.append("Some runs lack verifiable checkpoint-selection metadata and are marked unknown. They cannot be claimed to have matched selection or to implement window-selected validation.")
    if level == "window" and any(row["checkpoint_metric_effective"] != "window_macro_f1" for row in rows):
        warnings.append("This window table includes subject-selected or unverified checkpoints. Re-reporting their window scores is not a window-selected rerun or a strict Medformer-protocol reproduction.")
    report = dict(unit="percent", std_ddof=1, level=level, numeric_architecture=numeric_architecture,
                  seeds=[42, 43, 44], missing=missing, warnings=warnings, runs=rows,
                  aggregate=grouped, paired_contrasts=contrasts)
    return report


def render_markdown(report):
    lines = ["# Numeric-only ablation", "",
             f"{report['level'].capitalize()}-level metrics (%), seeds 42/43/44, mean ± sample SD (ddof=1). Fixed subject split; no window is reassigned between sets. Missing seeds are not imputed.",
             "", "## Checkpoint protocol notices", "", *(report["warnings"] or ["All listed checkpoint protocols are explicit and matched."]),
             "", "| Method | Dataset | Checkpoint selection | " + " | ".join(NAMES) + " |", "|---|---|---|" + "---|" * len(NAMES)]
    for row in report["aggregate"]:
        cells = [f"{row[name]['mean']:.2f} ± {row[name]['std']:.2f}" for name in NAMES]
        lines.append(f"| {row['method']} | {row['dataset']} | {row['checkpoint_metric_effective']} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired F1 difference (dual minus numeric)", ""]
    for row in report["paired_contrasts"]:
        stat = row["F1"]
        qualifier = "selection matched" if row["checkpoint_protocols_matched"] else "MIXED SELECTION: descriptive only"
        lines.append(f"- {row['dataset']} ({qualifier}): {stat['delta_pp_mean']:+.2f} ± {stat['delta_pp_std']:.2f} percentage points; positive in {stat['positive_seeds']}/3 seeds, tied in {stat['tied_seeds']}/3.")
    lines += ["", "These three-seed comparisons do not by themselves establish statistical significance or isolate the vision input from its associated auxiliary losses.",
              "", "AUPRC follows the stored code name and is macro Average Precision. Subject aggregation, when requested, averages window probabilities; it does not change the subject-disjoint split.",
              "", "Raw per-seed six metrics and selection provenance: `per_seed_metrics.csv`. Full precision, warnings and relative differences: `comparison.json`.",
              "", "## Missing", "", *(report["missing"] or ["None: all paired results available."])]
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--metric-level", choices=["window", "subject"], default="window")
    parser.add_argument("--output-dir", type=Path, help="default: RUN/report_WINDOW_OR_SUBJECT; source summaries are never modified")
    return parser


def main():
    args = build_parser().parse_args()
    report = collect_report(args.run, args.reference_run, args.metric_level)
    output = args.output_dir or args.run / f"report_{args.metric_level}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "dataset", "seed", "metric_level", "checkpoint_metric_effective", "checkpoint_protocol_source", "legacy_protocol_inferred", *NAMES, "summary"])
        writer.writeheader()
        writer.writerows(report["runs"])
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    (output / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
