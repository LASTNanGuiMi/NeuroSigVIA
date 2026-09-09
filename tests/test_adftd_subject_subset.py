import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from src.eeg_subject_subset import explicit_subject_ids, subject_subset_configuration


class SubjectSubsetTests(unittest.TestCase):
    def setUp(self):
        self.fixed = json.loads((Path(__file__).resolve().parents[1] / 'configs/ADFTD_subject13_seed42.json').read_text())
        self.ids = {s: row['subject_ids'] for s, row in self.fixed['splits'].items()}
        mapping = self.fixed['source_subject_label_mapping']
        self.inventory = SimpleNamespace(
            records=[SimpleNamespace(subject_id=int(s), label=c, window_count=10) for s,c in mapping.items()],
            split_ids=self.fixed['original_split_subject_ids'])

    def test_frozen_selection_preserves_original_order(self):
        selected = explicit_subject_ids(self.inventory, self.ids)
        self.assertEqual({s:list(ids) for s,ids in selected.items()}, self.ids)

    def test_reject_cross_split_subject(self):
        ids = copy.deepcopy(self.ids)
        ids['test'][0] = ids['train'][0]
        with self.assertRaises(ValueError):
            explicit_subject_ids(self.inventory, ids)

    def test_reject_window_count_change(self):
        with self.assertRaises(ValueError):
            explicit_subject_ids(self.inventory, self.ids, dict(train=69, vali=30, test=30))

    def test_reject_window_subsampling(self):
        self.fixed['eeg_train_fraction'] = .1
        with self.assertRaises(ValueError):
            subject_subset_configuration({'selection_manifest': self.fixed, 'selection_manifest_file_sha256': 'unused'})


if __name__ == '__main__':
    unittest.main()
