"""Small protocol checks; run with python -m unittest discover -s tests."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runners import baselines as baseline


@contextlib.contextmanager
def replace_modules(replacements):
    # Restore only the injected modules. Restoring the whole sys.modules dict
    # unloads torch's lazy modules while leaving C++ registrations alive.
    previous = {key: sys.modules.get(key) for key in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def bundle(length=16):
    result = SimpleNamespace()
    for offset, split in enumerate(("train", "vali", "test")):
        x = torch.arange(4 * 2 * length, dtype=torch.float32).reshape(4, 2, length) / 100
        dataset = TensorDataset(x)
        dataset.sample_subject_ids = np.array([10 * offset + 1, 10 * offset + 1,
                                               10 * offset + 2, 10 * offset + 2])
        setattr(result, split + "_loader", DataLoader(dataset, batch_size=2))
        setattr(result, split + "_labels", np.array([0, 0, 1, 1]))
    return result


class TinyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = torch.nn.Linear(config.enc_in, config.num_class)

    def forward(self, x, mask, _decoder, _decoder_mask):
        if not torch.all(mask == 1):
            raise AssertionError("Expected full valid mask")
        return self.projection(x.mean(dim=1))


class ProtocolTests(unittest.TestCase):
    def test_selection_default_and_tie_breaks(self):
        args = baseline.parser().parse_args([
            "--model", "Transformer", "--dataset", "apava", "--result_dir", "unused"])
        self.assertEqual(args.checkpoint_metric, "window_macro_f1")
        first = {"macro_f1": 0.75, "macro_log_loss": 0.8,
                 "subject_macro_f1": 0.5, "subject_macro_log_loss": 0.8}
        tied_better_log_loss = dict(first, macro_log_loss=0.1, subject_macro_log_loss=0.1)
        self.assertEqual(baseline.checkpoint_selection_key(first, "window_macro_f1"),
                         baseline.checkpoint_selection_key(tied_better_log_loss, "window_macro_f1"))
        self.assertGreater(baseline.checkpoint_selection_key(tied_better_log_loss, "subject_macro_f1"),
                           baseline.checkpoint_selection_key(first, "subject_macro_f1"))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            baseline.checkpoint_selection_key(first, "test_macro_f1")

    def test_window_subject_reverse_ranking_controls_checkpoint_scheduler_and_early_stop(self):
        # Epoch A gets 6/8 windows right but both subject means wrong. Epoch B
        # gets 2/8 windows right but both subject means right. These are actual
        # probabilities, so the ranking difference is not a renamed JSON key.
        y_true = np.repeat([0, 1], 4)
        scores_a = np.array([[0.51, 0.49]] * 3 + [[0.01, 0.99]] +
                            [[0.49, 0.51]] * 3 + [[0.99, 0.01]])
        scores_b = np.array([[0.49, 0.51]] * 3 + [[0.99, 0.01]] +
                            [[0.51, 0.49]] * 3 + [[0.01, 0.99]])

        def predictions(scores, ids):
            aggregated = baseline.aggregate_subjects(y_true, scores, ids)
            values = baseline.metrics(y_true, scores)
            values.update({"subject_" + key: value for key, value in baseline.metrics(
                aggregated["subject_y_true"], aggregated["subject_y_score"]).items()})
            arrays = {"y_true": y_true, "y_score": scores, "y_pred": scores.argmax(axis=1),
                      "sample_subject_id": ids, **aggregated,
                      "subject_y_pred": aggregated["subject_y_score"].argmax(axis=1)}
            return values, arrays

        rank_a, _ = predictions(scores_a, np.repeat([11, 12], 4))
        rank_b, _ = predictions(scores_b, np.repeat([11, 12], 4))
        self.assertGreater(rank_a["macro_f1"], rank_b["macro_f1"])
        self.assertLess(rank_a["subject_macro_f1"], rank_b["subject_macro_f1"])
        for mode, expected_best, expected_epochs in (("window_macro_f1", 1, 2),
                                                     ("subject_macro_f1", 2, 3)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                # Omitting the option in the window case verifies the CLI default
                # actually reaches the training loop and saved checkpoint.
                command = ["--model", "Transformer", "--dataset", "apava", "--device", "cpu",
                           "--result_dir", directory, "--train_epochs", "5", "--warmup_epochs", "0",
                           "--batch_size", "4", "--patience", "1"]
                if mode == "subject_macro_f1":
                    command += ["--checkpoint_metric", mode]
                args = baseline.parser().parse_args(command)
                baseline.validate_args(args)
                source = bundle()
                for split in ("train", "vali", "test"):
                    old = getattr(source, split + "_loader").dataset
                    dataset = TensorDataset(old.tensors[0].repeat_interleave(2, dim=0))
                    dataset.sample_subject_ids = np.repeat(old.sample_subject_ids, 2)
                    setattr(source, split + "_loader", DataLoader(dataset, batch_size=4))
                    setattr(source, split + "_labels", np.repeat(getattr(source, split + "_labels"), 2))
                common = ModuleType("data_loading.experiment")
                common.load_data = lambda key, smoke=False: (source, {"subject_overlap": [0, 0, 0]})
                datautils = ModuleType("src.datautils")
                def write_audit(_bundle, path):
                    target = Path(path) / "split.csv"
                    target.write_text("subject,split\n1,train\n11,vali\n21,test\n", encoding="utf-8")
                    return target
                datautils.write_eeg_medformer_split_audit = write_audit
                datautils.write_wearable_split_audit = write_audit
                validation_calls, test_calls, scheduler_values = [], [], []
                def controlled_evaluate(model, loader, ids, device, num_classes):
                    if np.min(ids) >= 21:
                        test_calls.append(True)
                        scores = scores_a if expected_best == 1 else scores_b
                    else:
                        validation_calls.append(True)
                        if len(validation_calls) <= expected_epochs:
                            scores = scores_a if len(validation_calls) % 2 else scores_b
                        else:
                            checkpoint = torch.load(Path(directory) / "best_checkpoint.pt",
                                                    map_location="cpu", weights_only=True)
                            scores = scores_a if checkpoint["epoch"] == 1 else scores_b
                    return predictions(scores, ids)
                with replace_modules({"data_loading.experiment": common, "src.datautils": datautils}), \
                     patch.object(baseline, "import_model", return_value=TinyModel), \
                     patch.object(baseline, "source_metadata", return_value={}), \
                     patch.object(baseline, "evaluate", side_effect=controlled_evaluate), \
                     patch.object(baseline.torch.optim.lr_scheduler, "ReduceLROnPlateau",
                                  return_value=SimpleNamespace(step=scheduler_values.append)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    baseline.run(args)
                checkpoint = torch.load(Path(directory) / "best_checkpoint.pt", map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["epoch"], expected_best)
                final = json.loads((Path(directory) / "metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(final["best_epoch"], expected_best)
                self.assertEqual(final["epochs_run"], expected_epochs)
                self.assertEqual(final["test_evaluation_count"], 1)
                self.assertEqual(len(test_calls), 1)
                np.testing.assert_allclose(scheduler_values, [0.75, 0.25] if mode == "window_macro_f1" else [0, 1, 0])
                for filename in ("metrics.json", "status.json", "protocol.json"):
                    value = json.loads((Path(directory) / filename).read_text(encoding="utf-8"))
                    self.assertEqual(value["evaluation_unit"], "window")
                    self.assertEqual(value["supplementary_evaluation_unit"], "subject")
                    self.assertEqual(value["checkpoint_metric"], "validation_" + mode)
                    self.assertEqual(value["selection_unit"], mode.split("_")[0])
                history = json.loads((Path(directory) / "history.json").read_text(encoding="utf-8"))
                self.assertEqual([row["checkpoint_improved"] for row in history],
                                 [True, False] if mode == "window_macro_f1" else [True, True, False])
                for split in ("validation", "test"):
                    with np.load(Path(directory) / f"{split}_predictions.npz", allow_pickle=False) as saved:
                        self.assertEqual(saved["y_score"].shape, (8, 2))
                        self.assertEqual(saved["subject_y_score"].shape, (2, 2))

    def test_subject_probability_mean_and_inconsistent_label_rejection(self):
        result = baseline.aggregate_subjects(np.array([0, 0, 0, 1]),
            np.array([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]]),
            np.array([1, 1, 1, 2]))
        np.testing.assert_allclose(result["subject_y_score"], [[0.6, 0.4], [0.1, 0.9]])
        np.testing.assert_array_equal(result["subject_window_count"], [3, 1])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            baseline.aggregate_subjects(np.array([0, 1]), np.array([[0.8, 0.2], [0.3, 0.7]]), np.array([1, 1]))

    def test_smoke_preserves_wearable_lengths_and_seed_changes_only_shuffle(self):
        for length in (976, 4096):
            source = bundle(length)
            a, ids_a, rows_a = baseline.build_loaders(source, 2, 42, smoke=True, smoke_samples_per_class=1)
            b, ids_b, rows_b = baseline.build_loaders(source, 2, 44, smoke=True, smoke_samples_per_class=1)
            self.assertEqual(rows_a, rows_b)
            for split in ("train", "vali", "test"):
                self.assertEqual(a[split].dataset.tensors[0].shape[-1], length)
                torch.testing.assert_close(a[split].dataset.tensors[0], b[split].dataset.tensors[0])
                np.testing.assert_array_equal(ids_a[split], ids_b[split])

    def test_final_test_once_and_smoke_never_evaluates_test(self):
        for smoke in (False, True):
            for use_swa in (False, True):
                with self.subTest(smoke=smoke, swa=use_swa), tempfile.TemporaryDirectory() as directory:
                    model_name = "Medformer" if use_swa else "Transformer"
                    extra = ["--swa"] if use_swa else []
                    args = baseline.parser().parse_args([
                        "--model", model_name, "--dataset", "apava", "--device", "cpu",
                        "--result_dir", directory, "--train_epochs", "2", "--warmup_epochs", "0",
                        "--batch_size", "2", *extra] + (["--smoke"] if smoke else []))
                    baseline.validate_args(args)
                    common = ModuleType("data_loading.experiment")
                    common.load_data = lambda key, smoke=False: (bundle(), {"subject_overlap": [0, 0, 0]})
                    datautils = ModuleType("src.datautils")
                    def write_audit(_bundle, path):
                        target = Path(path) / "split.csv"
                        target.write_text("subject,split\n1,train\n11,vali\n21,test\n", encoding="utf-8")
                        return target
                    datautils.write_eeg_medformer_split_audit = write_audit
                    datautils.write_wearable_split_audit = write_audit
                    seen_test = []
                    real_evaluate = baseline.evaluate
                    def counting_evaluate(model, loader, ids, device, num_classes):
                        if np.min(ids) >= 21:
                            seen_test.append(True)
                        return real_evaluate(model, loader, ids, device, num_classes)
                    with replace_modules({"data_loading.experiment": common, "src.datautils": datautils}), \
                         patch.object(baseline, "import_model", return_value=TinyModel), \
                         patch.object(baseline, "source_metadata", return_value={}), \
                         patch.object(baseline, "evaluate", side_effect=counting_evaluate), \
                         contextlib.redirect_stdout(io.StringIO()):
                        baseline.run(args)
                    self.assertEqual(len(seen_test), 0 if smoke else 1)
                    self.assertEqual((Path(directory) / "test_predictions.npz").exists(), not smoke)
                    checkpoint = torch.load(Path(directory) / "best_checkpoint.pt", map_location="cpu", weights_only=True)
                    self.assertEqual(checkpoint["swa"], use_swa)
                    self.assertFalse(any(name.startswith("module.") for name in checkpoint["model_state_dict"]))
                    self.assertEqual(checkpoint["swa_n_averaged"], checkpoint["epoch"] if use_swa else 0)
                    protocol = json.loads((Path(directory) / "protocol.json").read_text(encoding="utf-8"))
                    self.assertEqual(protocol["stochastic_weight_averaging"]["enabled"], use_swa)
                    restored = TinyModel(SimpleNamespace(enc_in=2, num_class=2)).eval()
                    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
                    source = bundle()
                    loaders, subject_ids, _ = baseline.build_loaders(source, 2, 42)
                    _, expected = baseline.evaluate(restored, loaders["vali"], subject_ids["vali"], torch.device("cpu"), 2)
                    with np.load(Path(directory) / "validation_predictions.npz") as saved:
                        np.testing.assert_allclose(expected["y_score"], saved["y_score"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
