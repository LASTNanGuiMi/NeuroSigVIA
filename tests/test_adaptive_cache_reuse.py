"""Exercise signature matching and actual sample/split alignment during reuse."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.adaptive_cache_reuse import promote_adaptive_static_caches
from src.adaptive_graph_training import (
    NEUROSIGVIA_CACHE_ARCHITECTURE, save_adaptive_graph_feature_cache,
    load_adaptive_graph_feature_cache,
)
from src.patch_mindts import make_temporal_patches


class StaticReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'old' / 'dataset' / 'known_key'
        self.target = self.root / 'new' / 'dataset' / 'new_key'
        self.signals = torch.arange(4 * 73 * 2).reshape(4, 2, 73).float()
        self.lengths = torch.tensor([73, 66, 64, 35])
        self.labels = np.asarray([0, 1, 0, 1])
        self.loader = DataLoader(TensorDataset(self.signals, self.lengths), batch_size=2, shuffle=False)
        temporal = make_temporal_patches(self.signals, window_size=64, stride=64, lengths=self.lengths)
        self.bundle = dict(raw_windows=temporal.patches.float(),
                           line_tokens=torch.ones(4, 2, 8).half(),
                           mantis_channel_tokens=torch.ones(4, 2, 2, 6).half(),
                           patch_mask=temporal.patch_mask, valid_lengths=temporal.valid_lengths,
                           valid_fraction=temporal.valid_fraction.half())
        self.old = json.dumps({'schema': 10, 'feature_code_identity': {'manifest_sha256': 'known'}})
        self.new = json.dumps({'schema': 11, 'feature_code_identity': {'manifest_sha256': 'current'}})

    def save(self, signature=None, labels=None, bundle=None):
        path = self.source / 'adaptive_graph_train.npz'
        save_adaptive_graph_feature_cache(path, self.bundle if bundle is None else bundle,
                                         self.labels if labels is None else labels,
                                         signature or self.old, window_size=64, stride=64)
        return path

    def promote(self, loader=None, signatures=None):
        return promote_adaptive_static_caches(self.target, self.new,
            {self.old: NEUROSIGVIA_CACHE_ARCHITECTURE} if signatures is None else signatures,
            self.root, {'train': (loader or self.loader, self.labels)}, 2, 64, 64)

    def test_legacy_promotion_preserves_arrays_and_source_bytes(self):
        path = self.save()
        before = hashlib.sha256(path.read_bytes()).digest()
        receipt = self.promote()
        self.assertEqual(receipt['reused_splits'], ['train'])
        self.assertTrue(receipt['verified_raw_against_loader'])
        actual = load_adaptive_graph_feature_cache(self.target / 'adaptive_graph_train.npz',
                    self.labels, self.new, window_size=64, stride=64, expected_channels=2)
        for key in self.bundle:
            torch.testing.assert_close(actual[key], self.bundle[key])
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before)

    def test_current_signature_reuses_across_run_directories_without_legacy_allowlist(self):
        self.save(self.new)
        self.assertEqual(self.promote(signatures={})['status'], 'reused')

    def test_unrecognized_signature_is_a_miss(self):
        self.save(json.dumps({'schema': 10, 'unknown': True}))
        self.assertEqual(self.promote()['status'], 'cache_miss')
        self.assertFalse(self.target.exists())

    def test_swapped_records_with_same_labels_fail_raw_comparison(self):
        corrupted = dict(self.bundle)
        corrupted['raw_windows'] = corrupted['raw_windows'].flip(0)
        self.save(bundle=corrupted)
        with self.assertRaisesRegex(ValueError, 'raw_windows does not match'):
            self.promote()
        self.assertFalse(self.target.exists())

    def test_labels_and_shuffled_loader_are_rejected(self):
        self.save(labels=self.labels[::-1])
        with self.assertRaisesRegex(ValueError, 'labels mismatch'):
            self.promote()
        self.save()
        shuffled = DataLoader(self.loader.dataset, batch_size=2, shuffle=True)
        with self.assertRaisesRegex(ValueError, 'sequential loader'):
            self.promote(loader=shuffled)


if __name__ == '__main__':
    unittest.main()
