#!/usr/bin/env python3
import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datautils import (  # noqa: E402
    EEG_MEDFORMER_DATASET_NAMES,
    EEG_MEDFORMER_SPECS,
    _make_eeg_medformer_split_loader,
    get_eeg_medformer_inventory,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Medformer EEG sources, splits, and frozen TEST alignment."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[name.lower() for name in EEG_MEDFORMER_DATASET_NAMES],
        default=[name.lower() for name in EEG_MEDFORMER_DATASET_NAMES],
    )
    return parser.parse_args()


def natural_key(path):
    return tuple(
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", path.as_posix())
    )


def find_public_batches(frozen_root, dataset_name):
    public_root = Path(frozen_root) / dataset_name / "public"
    candidates = sorted(public_root.rglob("*.npy"), key=natural_key)
    if not candidates:
        raise FileNotFoundError(f"No frozen public NPY batches below {public_root}")
    return candidates


def align_source_to_frozen_manifest(
    frozen_root, dataset_name, inventory, test_loader, test_labels
):
    private_path = Path(frozen_root) / dataset_name / "private" / "manifest.json"
    private_manifest = json.loads(private_path.read_text(encoding="utf-8"))
    official = private_manifest["official_split_ids"]
    for inventory_split, manifest_split in (
        ("train", "train"),
        ("vali", "val"),
        ("test", "test"),
    ):
        assert list(inventory.split_ids[inventory_split]) == official[manifest_split]

    samples = sorted(private_manifest["samples"], key=lambda item: item["global_index"])
    subject_ids = test_loader.dataset.sample_subject_ids
    window_indices = test_loader.dataset.sample_window_indices
    row_by_source = {
        (int(subject_id), int(window_index)): row
        for row, (subject_id, window_index) in enumerate(
            zip(subject_ids, window_indices)
        )
    }
    assert len(row_by_source) == len(test_labels) == len(samples)

    record_by_id = {record.subject_id: record for record in inventory.records}
    ordered_rows = []
    manifest_labels = []
    for sample in samples:
        subject_id = int(sample["source_subject_id"])
        window_index = int(sample["source_window_index"])
        assert sample["source_feature_file"] == record_by_id[subject_id].feature_path.name
        ordered_rows.append(row_by_source[(subject_id, window_index)])
        manifest_labels.append(int(sample["class_id"]))
    assert len(set(ordered_rows)) == len(ordered_rows)
    ordered_rows = np.asarray(ordered_rows, dtype=np.int64)
    assert np.array_equal(
        test_labels[ordered_rows], np.asarray(manifest_labels, dtype=np.int64)
    )
    return ordered_rows


def main():
    args = parse_args()
    for requested in args.datasets:
        dataset_name = requested.upper()
        inventory = get_eeg_medformer_inventory(
            args.data_dir, dataset_name, verify_content=True
        )
        test_loader, test_labels = _make_eeg_medformer_split_loader(
            inventory, "test", batch_size=32
        )
        source_test = test_loader.dataset.tensors[0].numpy().transpose(0, 2, 1)
        ordered_rows = align_source_to_frozen_manifest(
            args.frozen_root,
            dataset_name,
            inventory,
            test_loader,
            test_labels,
        )
        source_test = source_test[ordered_rows]
        batch_paths = find_public_batches(args.frozen_root, dataset_name)
        frozen_test = np.concatenate(
            [np.load(path, allow_pickle=False) for path in batch_paths], axis=0
        )
        spec = EEG_MEDFORMER_SPECS[dataset_name]

        assert source_test.dtype == np.float32
        assert source_test.shape == frozen_test.shape
        assert source_test.shape[0] == spec["expected_window_counts"]["test"]
        assert len(test_labels) == source_test.shape[0]
        assert np.isfinite(source_test).all()
        assert np.array_equal(source_test, frozen_test), (
            dataset_name,
            float(np.max(np.abs(source_test - frozen_test))),
            int(np.count_nonzero(source_test != frozen_test)),
        )
        means = source_test.mean(axis=1)
        stds = source_test.std(axis=1, ddof=0)
        assert float(np.max(np.abs(means))) < 2e-6
        std_distance = np.minimum(np.abs(stds), np.abs(stds - 1.0))
        assert float(np.max(std_distance)) < 2e-6

        print(
            f"{dataset_name}: PASS test_shape={source_test.shape} "
            f"label_sha256={inventory.label_sha256} "
            f"content_sha256={inventory.content_sha256} exact_frozen_match=yes"
        )
        del test_loader, test_labels, source_test, frozen_test
        gc.collect()

    print("EEG MEDFORMER DATASET VALIDATION PASSED")


if __name__ == "__main__":
    main()
