"""Numeric batch overrides change loaders, not the paired immutable reference."""

import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runners.numeric_ablation import build_parser, loader_for, training_configuration


class NumericBatchOverrideTests(unittest.TestCase):
    argv = ["--reference-run", "reference", "--dataset", "apava", "--seed", "42", "--output", "new-run"]

    def reference_args(self):
        return dict(batch_size=8, patch_checkpoint_metric="subject_macro_f1",
                    mlp_lr=0.001, mlp_weight_decay=0.0001, nested_cache_config=dict(enabled=True))

    def test_default_inherits_reference_batch_size(self):
        parsed = build_parser().parse_args(self.argv)
        self.assertIsNone(parsed.batch_size)
        cfg, metadata = training_configuration(self.reference_args(), parsed.checkpoint_metric, parsed.batch_size)
        self.assertEqual(cfg["batch_size"], 8)
        self.assertEqual(metadata, dict(batch_size_requested=None, batch_size_effective=8, reference_batch_size=8))

    def test_override_updates_training_args_and_metadata_without_reference_mutation(self):
        reference = self.reference_args()
        original = copy.deepcopy(reference)
        parsed = build_parser().parse_args(self.argv + ["--batch-size", "16"])
        cfg, metadata = training_configuration(reference, parsed.checkpoint_metric, parsed.batch_size)
        self.assertEqual(reference, original)
        self.assertIsNot(cfg, reference)
        self.assertEqual(cfg["batch_size"], 16)
        self.assertEqual(cfg["patch_checkpoint_metric"], "window_macro_f1")
        self.assertEqual(metadata, dict(batch_size_requested=16, batch_size_effective=16, reference_batch_size=8))
        for key in ("mlp_lr", "mlp_weight_decay", "nested_cache_config"):
            self.assertEqual(cfg[key], reference[key])

    def test_cli_rejects_invalid_batch_sizes(self):
        for value in ("0", "-1", "1.5", "invalid"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(self.argv + ["--batch-size", value])

    def test_effective_batch_reaches_real_train_and_eval_loaders(self):
        split = dict(labels=np.zeros(35, dtype=np.int64),
                     mantis_channel_tokens=np.zeros((35, 1, 1, 2), dtype=np.float32),
                     patch_mask=np.ones((35, 1), dtype=bool),
                     valid_fraction=np.ones((35, 1), dtype=np.float32))
        cfg, metadata = training_configuration(self.reference_args(), "window_macro_f1", 16)
        for shuffle in (False, True):
            loader = loader_for(split, cfg["batch_size"], shuffle)
            self.assertEqual(loader.batch_size, metadata["batch_size_effective"])
            self.assertEqual([len(batch[3]) for batch in loader], [16, 16, 3])
        smoke_loader = loader_for(split, cfg["batch_size"], True, 2 * cfg["batch_size"])
        self.assertEqual([len(batch[3]) for batch in smoke_loader], [16, 16])


if __name__ == "__main__":
    unittest.main()
