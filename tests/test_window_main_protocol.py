"""Window-primary selection must not silently become subject-level selection."""
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.arguments import parse_args
from src.adaptive_graph_training import (
    _atomic_prediction_dump, _evaluation_protocol, train_neurosigvia_classifier,
)
from src.patch_fusion import _checkpoint_selection_key, _resolve_checkpoint_metric


class WindowMainProtocolTests(unittest.TestCase):
    def test_cli_and_training_default_to_window(self):
        with patch.object(sys, "argv", ["main", "--data_dir", "data", "--result_dir", "out"]):
            self.assertEqual(parse_args().patch_checkpoint_metric, "window_macro_f1")
        self.assertEqual(inspect.signature(train_neurosigvia_classifier)
                         .parameters["checkpoint_metric"].default, "window_macro_f1")

    def test_window_and_subject_choose_opposite_epochs(self):
        a = dict(macro_f1=0.9, subject_macro_f1=0.5, subject_macro_log_loss=0.7)
        b = dict(macro_f1=0.6, subject_macro_f1=1.0, subject_macro_log_loss=0.2)
        self.assertGreater(_checkpoint_selection_key(a, "window_macro_f1"),
                           _checkpoint_selection_key(b, "window_macro_f1"))
        self.assertLess(_checkpoint_selection_key(a, "subject_macro_f1"),
                        _checkpoint_selection_key(b, "subject_macro_f1"))
        tied = dict(a, subject_macro_f1=1.0, subject_macro_log_loss=0.01)
        self.assertEqual(_checkpoint_selection_key(a, "window_macro_f1"),
                         _checkpoint_selection_key(tied, "window_macro_f1"))

    def test_metadata_does_not_relabel_legacy_selection(self):
        for metric in ("window_macro_f1", "subject_macro_f1"):
            metadata = _evaluation_protocol(metric)
            self.assertEqual(metadata["checkpoint_metric_effective"], metric)
            self.assertEqual(metadata["evaluation_unit"], "window")
        with self.assertRaises(ValueError):
            _evaluation_protocol("unknown")

    def test_predictions_roundtrip_without_pickle(self):
        expected = dict(y_true=np.array([0, 1]), y_score=np.array([[.8, .2], [.1, .9]]),
                        sample_index=np.array([0, 1]), sample_subject_id=np.array([1, 17]))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "test_predictions.npz"
            _atomic_prediction_dump(target, expected)
            with np.load(target, allow_pickle=False) as saved:
                for key, value in expected.items():
                    np.testing.assert_array_equal(saved[key], value)
            with self.assertRaises(ValueError):
                _atomic_prediction_dump(target, dict(bad=np.array([{}], dtype=object)))


if __name__ == "__main__":
    unittest.main()
