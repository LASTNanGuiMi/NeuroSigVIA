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
