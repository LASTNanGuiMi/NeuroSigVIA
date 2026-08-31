#!/usr/bin/env python3
import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.datautils as datautils
from src.datautils import (
    FALLTL_BODY_PART_CODES,
    FALLTL_FEATURE_COLUMNS,
    get_falltl_comparison_dataloaders,
    write_falltl_comparison_split_audit,
)


def as_numpy(tensor):
    return tensor.detach().cpu().numpy()


REORDERED_COLUMNS = [
    "timestamp",
    "EulerZ",
    "AccY",
    "GyrX",
    "AccX",
    "EulerY",
    "GyrZ",
    "AccZ",
    "EulerX",
    "GyrY",
    "Label",
]


def write_sequence(path, length, offset=0.0, missing_column=None):
    columns = [column for column in REORDERED_COLUMNS if column != missing_column]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        denominator = max(length - 1, 1)
        for point in range(length):
            row = {
                "timestamp": point / 100.0,
                "Label": "ignored-extra-column",
            }
            for channel, feature in enumerate(FALLTL_FEATURE_COLUMNS):
                row[feature] = offset + channel * 10.0 + point / denominator
            writer.writerow({column: row[column] for column in columns})


def expect_value_error(action, text):
    try:
        action()
    except ValueError as exc:
        if text.lower() not in str(exc).lower():
            raise AssertionError(
                f"Expected ValueError containing {text!r}, got {exc!r}"
            ) from exc
    else:
        raise AssertionError(f"Expected ValueError containing {text!r}")


def build_valid_dataset(root):
    falltl_dir = root / "FallTL"
    falltl_dir.mkdir(parents=True)
    activity_codes = ["D01"] * 5 + ["F01FR"] * 5
    body_codes = list(FALLTL_BODY_PART_CODES)
    for group_index, activity_code in enumerate(activity_codes):
        trial_no = group_index % 5 + 1
        for body_offset in range(2):
            body_code = body_codes[(2 * group_index + body_offset) % len(body_codes)]
            basename = f"{activity_code}_{body_code}_{trial_no}.csv"
            if group_index == 0 and body_offset == 0:
                basename = basename.lower()
            write_sequence(
                falltl_dir / basename,
                length=5 + ((group_index + body_offset) % 4),
            )
    return falltl_dir


def validate_group_split_and_preprocessing(root):
    falltl_dir = build_valid_dataset(root)
    args = SimpleNamespace(
        data_dir=str(root),
        batch_size=4,
        falltl_target_length=17,
    )
    bundle = get_falltl_comparison_dataloaders(args)

    group_sets = {
        "train": set(bundle.train_group_ids),
        "vali": set(bundle.vali_group_ids),
        "test": set(bundle.test_group_ids),
    }
    assert group_sets["train"].isdisjoint(group_sets["vali"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["vali"].isdisjoint(group_sets["test"])
    assert sum(map(len, group_sets.values())) == 10
    assert bundle.target_length == 17

    split_for_group = {}
    for split in ("train", "vali", "test"):
        tensor = getattr(bundle, f"{split}_loader").dataset.tensors[0]
        labels = getattr(bundle, f"{split}_labels")
        files = getattr(bundle, f"{split}_files")
        assert tensor.shape == (len(files), len(FALLTL_FEATURE_COLUMNS), 17)
        assert as_numpy(tensor).dtype == np.float32
        assert labels.dtype == np.int64
        assert np.isfinite(as_numpy(tensor)).all()
        for filename in files:
            activity_code, _, trial_no = datautils._parse_falltl_comparison_filename(
                filename
            )
            group_id = (activity_code, trial_no)
            assert group_id in group_sets[split]
            previous_split = split_for_group.setdefault(group_id, split)
            assert previous_split == split

    # Every synthetic event has two body recordings; neither may cross a split.
    assert len(split_for_group) == 10
    for group_id in split_for_group:
        count = 0
        for split in ("train", "vali", "test"):
            for filename in getattr(bundle, f"{split}_files"):
                activity_code, _, trial_no = (
                    datautils._parse_falltl_comparison_filename(filename)
                )
                count += (activity_code, trial_no) == group_id
        assert count == 2

    # Header order is deliberately unrelated to feature order, and timestamp /
    # Label are extras. The loader must still return the named feature ordering.
    sequences, _, source_files = datautils._load_falltl_comparison_arrays(
        str(root)
    )
    lower_index = source_files.tolist().index("d01_fc_1.csv")
    np.testing.assert_allclose(
        sequences[lower_index][0],
        np.arange(len(FALLTL_FEATURE_COLUMNS), dtype=np.float32) * 10.0,
    )
    assert datautils._parse_falltl_comparison_filename(
        source_files[lower_index]
    ) == ("D01", "FC", 1)

    sequence_by_file = dict(zip(source_files.tolist(), sequences))
    raw_train_points = np.concatenate(
        [sequence_by_file[filename] for filename in bundle.train_files], axis=0
    ).astype(np.float64)
    expected_mean = raw_train_points.mean(axis=0, keepdims=True)
    expected_std = raw_train_points.std(axis=0, keepdims=True)
    expected_std = np.where(expected_std < 1e-8, 1.0, expected_std)
    expected_first_train = datautils._resample_falltl_sequence(
        (sequence_by_file[bundle.train_files[0]] - expected_mean) / expected_std,
        bundle.target_length,
    )
    np.testing.assert_allclose(
        as_numpy(bundle.train_loader.dataset.tensors[0])[0],
        expected_first_train,
        rtol=1e-6,
        atol=1e-6,
    )

    train_tensor_before = as_numpy(
        bundle.train_loader.dataset.tensors[0]
    ).copy()
    train_files = set(bundle.train_files)
    for path in falltl_dir.glob("*.csv"):
        if path.name not in train_files:
            with path.open(encoding="utf-8", newline="") as handle:
                length = sum(1 for _ in csv.DictReader(handle))
            write_sequence(path, length=length, offset=1_000_000.0)

    # Held-out values must not alter the training statistics or training tensor.
    bundle_after = get_falltl_comparison_dataloaders(args)
    assert bundle_after.train_files == bundle.train_files
    np.testing.assert_allclose(
        as_numpy(bundle_after.train_loader.dataset.tensors[0]),
        train_tensor_before,
    )
    train_means = train_tensor_before.mean(axis=(0, 2))
    np.testing.assert_allclose(train_means, np.zeros_like(train_means), atol=1e-5)

    audit_path = write_falltl_comparison_split_audit(bundle, root / "audit")
    with Path(audit_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert set(rows[0]) == {
        "filename",
        "label_id",
        "activity_code",
        "body_part_code",
        "trial_no",
        "event_group",
        "split",
    }
    audit_splits = {}
    for row in rows:
        assert row["activity_code"] == row["activity_code"].upper()
        assert row["body_part_code"] in FALLTL_BODY_PART_CODES
        expected_group = f"{row['activity_code']}_{int(row['trial_no'])}"
        assert row["event_group"] == expected_group
        previous_split = audit_splits.setdefault(expected_group, row["split"])
        assert previous_split == row["split"]

    bad_target_args = SimpleNamespace(
        data_dir=str(root), batch_size=2, falltl_target_length=0
    )
    expect_value_error(
        lambda: get_falltl_comparison_dataloaders(bad_target_args),
        "positive integer",
    )


def validate_rejections(root):
    assert datautils._parse_falltl_comparison_filename(
        "F03LR_RA_15.csv"
    ) == ("F03LR", "RA", 15)
    assert datautils._parse_falltl_comparison_filename(
        "f06fw_rw_6.CSV"
    ) == ("F06FW", "RW", 6)

    invalid_dir = root / "invalid_name"
    invalid_dir.mkdir()
    write_sequence(invalid_dir / "X01_BW_1.csv", length=4)
    expect_value_error(
        lambda: datautils._load_falltl_comparison_arrays(str(invalid_dir)),
        "invalid falltl comparison filename",
    )

    missing_dir = root / "missing_column"
    missing_dir.mkdir()
    write_sequence(
        missing_dir / "D01_BW_1.csv", length=4, missing_column="EulerZ"
    )
    expect_value_error(
        lambda: datautils._load_falltl_comparison_arrays(str(missing_dir)),
        "missing columns",
    )

    duplicate_dir = root / "duplicate"
    duplicate_dir.mkdir()
    duplicate_path = duplicate_dir / "D01_BW_1.csv"
    write_sequence(duplicate_path, length=4)
    original_glob = datautils._glob_csv_files
    try:
        datautils._glob_csv_files = lambda *_: [
            str(duplicate_path),
            str(duplicate_path),
        ]
        expect_value_error(
            lambda: datautils._load_falltl_comparison_arrays(str(duplicate_dir)),
            "duplicate",
        )
    finally:
        datautils._glob_csv_files = original_glob


def main():
    with tempfile.TemporaryDirectory(prefix="falltl_comparison_test_") as temp_dir:
        root = Path(temp_dir)
        validate_group_split_and_preprocessing(root / "valid")
        rejection_root = root / "rejections"
        rejection_root.mkdir()
        validate_rejections(rejection_root)
    print("PASS FallTL comparison synthetic protocol checks")


if __name__ == "__main__":
    main()
