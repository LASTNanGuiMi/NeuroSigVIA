"""CPU contract tests; tiny encoders replace only the expensive visual model.

Run from the staged source directory with:
python -m unittest discover -s tests -p test_vision_ablation_training.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src import vision_ablation_training as training


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(2.0)

    def forward(self, image):
        return self.linear(image)


class TinyVisualOnly(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.renderer = nn.Module()
        self.renderer.gate = nn.Module()
        self.renderer.gate.region_cls = nn.Linear(1, 1)
        self.classifier = nn.Linear(1, config["num_classes"])

    def get_config(self):
        return dict(self.config)

    def forward(self, raw, line, mask, fraction, lengths, vision_model, **kwargs):
        gate = self.renderer.gate.region_cls(raw.mean((1, 2, 3))[:, None])
        features = vision_model(gate) + line.mean((1, 2))[:, None]
        return self.classifier(features), {"selector_balance_loss": gate.square().mean()}


class NoNumericAccess(dict):
    def __getitem__(self, key):
        if "mantis" in key or "temporal" in key:
            raise AssertionError("numeric branch must never be accessed")
        return super().__getitem__(key)


def tiny_splits():
    result = {}
    for index, name in enumerate(("train", "vali", "test")):
        labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        subjects = np.array([1, 1, 1, 2, 2, 3], dtype=np.int64) + index * 10
        raw = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1).expand(-1, 2, 2, 64).contiguous() / 10 + index
        result[name] = NoNumericAccess(
            raw_windows=raw, line_tokens=torch.ones(6, 2, 4) * 0.1,
            patch_mask=torch.ones(6, 2, dtype=torch.bool),
            valid_fraction=torch.ones(6, 2), valid_lengths=torch.full((6, 2), 64, dtype=torch.long),
            labels=labels, subject_ids=subjects,
        )
    return result


class VisualTrainingContract(unittest.TestCase):
    def test_full_loop_restores_earliest_tie_then_tests_once(self):
        splits, vision = tiny_splits(), TinyVision()
        initial_vision = training._state_sha256(vision)
        events = []
        real_evaluate = training.evaluate
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "admission.json").write_text('{"identity":"PASS"}')

            def observed_evaluate(model, loader, classes, subjects, encoder, device, encode_batch_size):
                phase = "test" if int(np.min(subjects)) > 20 else "vali"
                events.append(phase)
                if phase == "test":
                    history = json.loads((output / "training_history.json").read_text())
                    self.assertEqual(len(history), 3)
                    saved = torch.load(output / "best_checkpoint.pt", map_location="cpu", weights_only=False)
                    self.assertEqual(saved["best_epoch"], 1)
                    self.assertTrue(all(torch.equal(value, saved["model_state_dict"][key]) for key, value in model.state_dict().items()))
                metrics, details = real_evaluate(model, loader, classes, subjects, encoder, device, encode_batch_size)
                if phase == "vali":
                    # A forced tie isolates checkpoint control flow from prediction quality.
                    metrics["macro_f1"] = 0.25
                return metrics, details

            config = dict(mlp_epochs=3, batch_size=2, fusion_dim=4,
                          mlp_hidden_dim=4, mlp_lr_scheduler="none", random_seed=999)
            with patch.object(training, "VisionOnlyClassifier", TinyVisualOnly), patch.object(training, "evaluate", observed_evaluate):
                summary = training.train_vision_only(splits, vision, config, output, 42, "cpu")
            self.assertEqual(events, ["vali", "vali", "vali", "vali", "test"])
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(summary["test_evaluations"], 1)
            self.assertEqual(summary["training_args"]["random_seed"], 42)
            self.assertEqual(config["random_seed"], 999)
            self.assertEqual(summary["class_names"], ["HC", "FTD", "AD"])
            self.assertEqual(initial_vision, training._state_sha256(vision))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in vision.parameters()))
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertIsNone(protocol["gradient_clip_max_norm"])
            self.assertFalse(protocol["trained_main_weights_loaded"])
            self.assertFalse(protocol["uses_alignment"])
            for phase in ("validation", "test"):
                metrics = summary[f"{phase}_metrics"]
                for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auroc", "macro_auprc"):
                    self.assertTrue(np.isfinite(metrics[key]))
                with np.load(output / f"{phase}_predictions.npz", allow_pickle=False) as predictions:
                    self.assertEqual(predictions["y_score"].shape, (6, 3))
                    self.assertEqual(predictions["sample_subject_id"].shape, (6,))
                    self.assertEqual(predictions["subject_y_score"].shape, (3, 3))
            self.assertTrue((output / "admission.json").exists())

    def test_balanced_loss_matches_original_per_sample_mean_and_gate_gradients(self):
        splits, _ = training._prepare_splits(tiny_splits())
        model, vision = TinyVisualOnly(num_classes=3), TinyVision()
        for parameter in vision.parameters():
            parameter.requires_grad_(False)
        loader = training._loader(splits["train"], 6, False)
        weights = training._balanced_class_weights(splits["train"]["labels"], 3, torch.device("cpu"))
        torch.testing.assert_close(weights, torch.tensor([2/3, 1.0, 2.0]))
        criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
        logits, details, labels, _ = training._forward(model, next(iter(loader)), vision, "cpu", 4, True)
        expected = criterion(logits, labels).mean() + 0.001 * details["selector_balance_loss"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
        statistics = training._run_epoch(model, loader, vision, optimizer, criterion, "cpu", 0.001, 4)
        self.assertAlmostEqual(statistics["loss"], float(expected.detach()), places=6)
        self.assertTrue(any(bool(p.grad.abs().gt(0).any()) for p in model.renderer.gate.region_cls.parameters()))
        self.assertTrue(all(p.grad is None for p in vision.parameters()))

    def test_subject_overlap_is_rejected_before_training(self):
        splits = tiny_splits()
        splits["test"]["subject_ids"] = splits["train"]["subject_ids"].copy()
        with self.assertRaisesRegex(ValueError, "subject overlap"):
            training._prepare_splits(splits)

    def test_existing_results_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_checkpoint.pt"
            path.write_bytes(b"existing historical artifact")
            with self.assertRaises(FileExistsError):
                training.train_vision_only(tiny_splits(), TinyVision(), {}, directory, 42, "cpu")
            self.assertEqual(path.read_bytes(), b"existing historical artifact")


if __name__ == "__main__":
    unittest.main()
