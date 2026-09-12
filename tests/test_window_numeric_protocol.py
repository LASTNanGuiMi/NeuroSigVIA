"""Synthetic regression tests for window selection and historical-report labeling."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if torch is not None:
    from runners.numeric_ablation import (
        build_parser, checkpoint_selection_protocol, validation_selection,
    )
    from src.patch_mindts import _EarlyStoppingMonitor, _build_patch_lr_scheduler
from scripts.summarize_numeric_ablation import (
    METRICS, build_parser as report_parser, checkpoint_protocol,
    collect_report, metric_vector, render_markdown,
)


@unittest.skipUnless(torch is not None, "selection/scheduler tests require the training PyTorch environment")
class NumericWindowSelectionTests(unittest.TestCase):
    def test_cli_defaults_to_window_and_subject_mode_is_explicit(self):
        argv = ["--reference-run", "reference", "--dataset", "apava", "--seed", "42", "--output", "new-run"]
        self.assertEqual(build_parser().parse_args(argv).checkpoint_metric, "window_macro_f1")
        self.assertEqual(build_parser().parse_args(argv + ["--checkpoint-metric", "subject_macro_f1"]).checkpoint_metric, "subject_macro_f1")
        self.assertEqual(report_parser().parse_args(["--run", "r", "--reference-run", "p"]).metric_level, "window")

    def test_reversed_rankings_change_checkpoint_stop_and_scheduler(self):
        epochs = [
            dict(macro_f1=0.9, subject_macro_f1=0.2, subject_macro_log_loss=0.8),
            dict(macro_f1=0.7, subject_macro_f1=0.8, subject_macro_log_loss=0.1),
        ]
        outcomes = {}
        for metric in ("window_macro_f1", "subject_macro_f1"):
            parameter = torch.nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.AdamW([parameter], lr=0.01)
            scheduler = _build_patch_lr_scheduler(optimizer, "reduce_on_plateau", patience=0, factor=0.5, min_lr=0.0)
            stopping = _EarlyStoppingMonitor(strategy="raw_selection_key", patience=1)
            best_key, best_epoch = None, None
            for epoch, values in enumerate(epochs, 1):
                key, improved = validation_selection(values, metric, best_key)
                if improved:
                    best_key, best_epoch = key, epoch
                stop = stopping.update(key, epoch)
                scheduler.step(float(key[0]))
            outcomes[metric] = (best_epoch, stop["should_stop"], optimizer.param_groups[0]["lr"])
        self.assertEqual(outcomes["window_macro_f1"], (1, True, 0.005))
        self.assertEqual(outcomes["subject_macro_f1"], (2, False, 0.01))

    def test_window_ties_keep_earliest_subject_tie_breaker_is_preserved(self):
        first = dict(macro_f1=0.7, subject_macro_f1=0.8, subject_macro_log_loss=0.5)
        later = dict(macro_f1=0.7, subject_macro_f1=0.8, subject_macro_log_loss=0.1)
        for metric, expected_improvement in [("window_macro_f1", False), ("subject_macro_f1", True)]:
            first_key, _ = validation_selection(first, metric, None)
            _, improved = validation_selection(later, metric, first_key)
            self.assertEqual(improved, expected_improvement)
        self.assertEqual(checkpoint_selection_protocol("window_macro_f1")["metric_level"], "window")
        self.assertIn("negative subject macro-log-loss", checkpoint_selection_protocol("subject_macro_f1")["selection"])


class NumericReportProtocolTests(unittest.TestCase):
    def sample_summary(self, seed, numeric, metric):
        value = 0.5 + 0.1 * (seed - 42)
        metrics = {key: value for key in METRICS}
        metrics.update({"subject_" + key: 1 - value for key in METRICS})
        summary = dict(test_metrics=metrics, state="COMPLETED")
        if numeric:
            summary["architecture"] = "neurosigvia_numeric_only_no_fusion_v2"
        if metric:
            summary["checkpoint_metric_effective"] = metric
        elif numeric:
            summary["protocol"] = dict(selection="Validation subject macro-F1 then negative subject macro-log-loss; test evaluated only after checkpoint selection.")
        return summary

    def populate(self, root, numeric, metrics):
        for seed, metric in zip((42, 43, 44), metrics):
            folder = root / f"seed{seed}" / "apava"
            folder.mkdir(parents=True)
            filename = "numeric_ablation_summary.json" if numeric else "adaptive_graph_summary.json"
            (folder / filename).write_text(json.dumps(self.sample_summary(seed, numeric, metric)), encoding="utf-8")
            if metric is None and not numeric:
                (folder / "args.json").write_text(json.dumps(dict(patch_checkpoint_metric="subject_macro_f1")), encoding="utf-8")

    def test_report_units_use_different_columns(self):
        summary = self.sample_summary(42, True, "window_macro_f1")
        summary["test_metrics"]["macro_f1"] = 0.2
        summary["test_metrics"]["subject_macro_f1"] = 0.9
        self.assertEqual(metric_vector(summary, "window")[3], 20)
        self.assertEqual(metric_vector(summary, "subject")[3], 90)

    def test_legacy_window_report_remains_labeled_subject_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.populate(root / "numeric", True, [None] * 3)
            self.populate(root / "dual", False, [None] * 3)
            before = {path: path.read_bytes() for path in root.rglob("*.json")}
            report = collect_report(root / "numeric", root / "dual")
            self.assertEqual(report["level"], "window")
            self.assertEqual(report["std_ddof"], 1)
            self.assertTrue(all(row["legacy_protocol_inferred"] for row in report["runs"]))
            self.assertTrue(all(row["checkpoint_metric_effective"] == "subject_macro_f1" for row in report["runs"]))
            self.assertAlmostEqual(report["aggregate"][0]["F1"]["std"], 10)
            self.assertIn("not a window-selected rerun", render_markdown(report))
            self.assertEqual(before, {path: path.read_bytes() for path in root.rglob("*.json")})

    def test_mixed_pair_has_explicit_warning_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.populate(root / "numeric", True, ["window_macro_f1"] * 3)
            self.populate(root / "dual", False, ["subject_macro_f1"] * 3)
            report = collect_report(root / "numeric", root / "dual")
            self.assertFalse(report["paired_contrasts"][0]["checkpoint_protocols_matched"])
            self.assertIn("MIXED SELECTION", render_markdown(report))
            self.assertEqual(report["paired_contrasts"][0]["numeric_checkpoint_metric"], "window_macro_f1")

    def test_mixed_protocols_within_three_seeds_are_not_averaged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.populate(root / "numeric", True, ["window_macro_f1", "subject_macro_f1", "window_macro_f1"])
            self.populate(root / "dual", False, ["window_macro_f1"] * 3)
            report = collect_report(root / "numeric", root / "dual")
            self.assertEqual(len(report["aggregate"]), 1)
            self.assertEqual(report["aggregate"][0]["method"], "Dual modality")
            self.assertFalse(report["paired_contrasts"])

    def test_nested_protocol_and_conflicting_metadata(self):
        path = Path("not-a-real-run") / "numeric_ablation_summary.json"
        parsed = checkpoint_protocol(dict(protocol=dict(checkpoint_metric_effective="window_macro_f1")), path)
        self.assertEqual(parsed["metric"], "window_macro_f1")
        self.assertFalse(parsed["legacy_inferred"])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            checkpoint_protocol(dict(checkpoint_metric_effective="subject_macro_f1", protocol=dict(checkpoint_metric_effective="window_macro_f1")), path)

    def test_unknown_selection_is_not_assumed_subject_or_matched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.populate(root / "numeric", True, ["window_macro_f1"] * 3)
            self.populate(root / "dual", False, ["window_macro_f1"] * 3)
            for path in root.rglob("*summary.json"):
                summary = json.loads(path.read_text(encoding="utf-8"))
                summary.pop("checkpoint_metric_effective")
                path.write_text(json.dumps(summary), encoding="utf-8")
            report = collect_report(root / "numeric", root / "dual")
            self.assertTrue(all(row["checkpoint_metric_effective"] == "unknown" for row in report["runs"]))
            self.assertFalse(report["paired_contrasts"][0]["checkpoint_protocols_matched"])
            self.assertIn("marked unknown", render_markdown(report))

    def test_legacy_parent_args_are_bounded_to_run_and_preserve_window(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            folder = run / "seed42" / "apava" / "timestamp" / "artifacts"
            folder.mkdir(parents=True)
            path = folder / "adaptive_graph_summary.json"
            (folder.parent / "args.json").write_text(json.dumps(dict(patch_checkpoint_metric="window_macro_f1")), encoding="utf-8")
            parsed = checkpoint_protocol({}, path, run)
            self.assertEqual(parsed["metric"], "window_macro_f1")
            self.assertTrue(parsed["legacy_inferred"])
            # No args outside this run may be treated as its protocol.
            isolated = root / "isolated"
            isolated.mkdir()
            (root / "args.json").write_text(json.dumps(dict(patch_checkpoint_metric="subject_macro_f1")), encoding="utf-8")
            self.assertEqual(checkpoint_protocol({}, isolated / "summary.json", isolated)["metric"], "unknown")


if __name__ == "__main__":
    unittest.main()
