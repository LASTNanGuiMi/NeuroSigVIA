import csv
import hashlib
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aeon.datasets import load_classification
from torch.utils.data import DataLoader, TensorDataset


FALLTL_FEATURE_COLUMNS = [
    "AccX",
    "AccY",
    "AccZ",
    "GyrX",
    "GyrY",
    "GyrZ",
    "EulerX",
    "EulerY",
    "EulerZ",
]

FALLTL_BODY_PART_CODES = (
    "FC",  # Front chest
    "BW",  # Back waist
    "LW",  # Left forearm
    "RW",  # Right forearm
    "LT",  # Left thigh
    "RT",  # Right thigh
    "LA",  # Left shank
    "RA",  # Right shank
)
FALLTL_COMPARISON_FILENAME_RE = re.compile(
    r"^(?P<activity>D\d{2}|F\d{2}[FBL][RW])_"
    r"(?P<body>FC|BW|LW|RW|LT|RT|LA|RA)_"
    r"(?P<trial>\d+)\.csv$",
    flags=re.IGNORECASE,
)

FENG_PREFERRED_SENSORS = [
    "LowerBack",
    "RightThigh",
    "LeftThigh",
]

FENG_PREFERRED_SIGNALS = [
    "Acc",
    "Gyr",
]


UCI_HAR_SIGNAL_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

UCI_HAR_ACC_GYRO_SIGNAL_FILES = [
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

UCI_HAR_ACC_GYRO_INDICES = [6, 7, 8, 3, 4, 5]


WEARABLE_DATASET_NAMES = (
    "mPowerRest",
    "mPowerReturn",
    "mPowerOutbound",
    "PADS_09_task06_DrinkGlas",
    "PADS_10_task07_CrossArms",
    "PADS_11_task08_TouchIndex",
    "Shimmer_10_session10_AFC",
    "Shimmer_11_session11_DRINK",
    "Shimmer_12_session12_PICK",
)
WEARABLE_EXPECTED_SPLIT_SAMPLES = {
    "mPowerRest": (16124, 5399, 5368),
    "mPowerReturn": (9132, 2619, 2733),
    "mPowerOutbound": (15289, 5278, 5193),
    "PADS_09_task06_DrinkGlas": (280, 92, 97),
    "PADS_10_task07_CrossArms": (280, 92, 97),
    "PADS_11_task08_TouchIndex": (280, 92, 97),
    "Shimmer_10_session10_AFC": (69, 23, 25),
    "Shimmer_11_session11_DRINK": (77, 25, 28),
    "Shimmer_12_session12_PICK": (65, 21, 25),
}
WEARABLE_DYNAMIC_DATASET_SPECS = {
    "PADS_10_task07_CrossArms": {
        "sequence_length": 976,
        "label_names": {
            0: "Healthy",
            1: "Parkinson",
            2: "OtherMovementDisorders",
        },
        "expected_subject_counts": (280, 92, 97),
        "expected_sample_count": 469,
    },
    "PADS_11_task08_TouchIndex": {
        "sequence_length": 976,
        "label_names": {
            0: "Healthy",
            1: "Parkinson",
            2: "OtherMovementDisorders",
        },
        "expected_subject_counts": (280, 92, 97),
        "expected_sample_count": 469,
    },
    "Shimmer_10_session10_AFC": {
        "sequence_length": 4096,
        "label_names": {0: "HC", 1: "MildPD", 2: "ModeratePD"},
        "expected_subject_counts": (69, 23, 25),
        "expected_sample_count": 117,
    },
    "Shimmer_12_session12_PICK": {
        "sequence_length": 4096,
        "label_names": {0: "HC", 1: "MildPD", 2: "ModeratePD"},
        "expected_subject_counts": (65, 21, 25),
        "expected_sample_count": 111,
    },
}
WEARABLE_REFERENCE_DATASETS_SHA256 = (
    "0113d69736e9678a43b8e2c62b344bb34e6776c023085f1a80e7e81b0a512092"
)
_WEARABLE_MODULE_CACHE = {}


EEG_MEDFORMER_DATASET_NAMES = ("TDBRAIN", "APAVA", "ADFTD")
EEG_MEDFORMER_NORMALIZATION = "per_window_per_channel_standard_scaler_ddof0"
EEG_MEDFORMER_SPECS = {
    "APAVA": {
        "channels": 16,
        "subject_count": 23,
        "class_names": {0: "healthy", 1: "alzheimers"},
        "label_sha256": (
            "57d98d5207e300999428ed0951f54bb4ef29f9ab67a547feedd07d35ed6da1f7"
        ),
        "content_sha256": (
            "96984c06c5d5d41e62b2dc751733c78f67002e990f6c7dc421f155b32dfaf0d6"
        ),
        "protocol": "medformer_code_exact_apava_subject_split",
        "split_ids": {
            "train": tuple(range(3, 15)) + (21, 22, 23),
            "vali": (15, 16, 19, 20),
            "test": (1, 2, 17, 18),
        },
        "expected_subject_counts": {"train": 15, "vali": 4, "test": 4},
        "expected_window_counts": {"train": 3123, "vali": 1413, "test": 1431},
    },
    "ADFTD": {
        "channels": 19,
        "subject_count": 88,
        "class_names": {
            0: "healthy",
            1: "frontotemporal_dementia",
            2: "alzheimers",
        },
        "label_sha256": (
            "3b0be092cd886c0315c38120f3d244fa16f515e83608f30569e1a0ff508e4922"
        ),
        "content_sha256": (
            "46d083120f3c07a69628efa140ac31053da796d16871aaab1c813d86506f86f8"
        ),
        "protocol": "medformer_code_exact_label_row_order_60_20_20",
        "split_ids": {
            "train": (
                tuple(range(37, 54))
                + tuple(range(66, 79))
                + tuple(range(1, 22))
            ),
            "vali": (
                tuple(range(54, 60))
                + tuple(range(79, 84))
                + tuple(range(22, 29))
            ),
            "test": (
                tuple(range(60, 66))
                + tuple(range(84, 89))
                + tuple(range(29, 37))
            ),
        },
        "expected_subject_counts": {"train": 51, "vali": 18, "test": 19},
        "expected_window_counts": {
            "train": 40446,
            "vali": 14658,
            "test": 14648,
        },
    },
    "TDBRAIN": {
        "channels": 33,
        "subject_count": 72,
        "class_names": {0: "healthy", 1: "parkinsons_disease"},
        "label_sha256": (
            "12a23405ff73b720bc5075767d63c7e00b8a8e9d18fb9972407d10b025d1e069"
        ),
        "content_sha256": (
            "3ae88137c1bc4869ee4e5c265a409c0405932f3fec9e76d04aeca3538d7e3652"
        ),
        "protocol": "medformer_code_exact_official_legacy50_only",
        "split_ids": {
            "train": tuple(range(1, 18)) + tuple(range(29, 46)),
            "vali": (18, 19, 20, 21, 46, 47, 48, 49),
            "test": (22, 23, 24, 25, 50, 51, 52, 53),
        },
        "expected_subject_counts": {"train": 34, "vali": 8, "test": 8},
        "expected_window_counts": {"train": 4320, "vali": 960, "test": 960},
        "expected_excluded_ids": (26, 27, 28) + tuple(range(54, 73)),
    },
}


@dataclass
class WearableDataBundle:
    dataset_name: str
    data_root: Path
    reference_root: Path
    train_loader: DataLoader
    train_labels: np.ndarray
    vali_loader: DataLoader
    vali_labels: np.ndarray
    test_loader: DataLoader
    test_labels: np.ndarray
    train_dataset: Any
    vali_dataset: Any
    test_dataset: Any
    label_mode: str = "original"
    label_mapping: dict[int, int] | None = None


@dataclass(frozen=True)
class EEGMedformerSubjectRecord:
    subject_id: int
    label: int
    feature_path: Path
    window_count: int


@dataclass(frozen=True)
class EEGMedformerInventory:
    dataset_name: str
    data_root: Path
    records: tuple[EEGMedformerSubjectRecord, ...]
    split_ids: dict[str, tuple[int, ...]]
    excluded_ids: tuple[int, ...]
    protocol: str
    label_sha256: str
    content_sha256: str


@dataclass
class EEGMedformerBundle:
    inventory: EEGMedformerInventory
    train_loader: DataLoader
    train_labels: np.ndarray
    vali_loader: DataLoader
    vali_labels: np.ndarray
    test_loader: DataLoader
    test_labels: np.ndarray
    normalization: str = EEG_MEDFORMER_NORMALIZATION


@dataclass
class FallTLComparisonBundle:
    train_loader: DataLoader
    train_labels: np.ndarray
    vali_loader: DataLoader
    vali_labels: np.ndarray
    test_loader: DataLoader
    test_labels: np.ndarray
    train_files: list[str]
    vali_files: list[str]
    test_files: list[str]
    train_group_ids: list[tuple[str, int]]
    vali_group_ids: list[tuple[str, int]]
    test_group_ids: list[tuple[str, int]]
    target_length: int


@dataclass
class UCIHARSubjectBundle:
    train_loader: DataLoader
    train_labels: np.ndarray
    vali_loader: DataLoader
    vali_labels: np.ndarray
    test_loader: DataLoader
    test_labels: np.ndarray
    train_subjects: list[int]
    vali_subjects: list[int]
    test_subjects: list[int]
    har_channels: str


def linear_interpolation(data):
    n, d, l = data.shape
    result = data.copy()
    x = np.arange(l)

    for i in range(n):
        for j in range(d):
            y = data[i, j, :]
            nan_mask = np.isnan(y)
            if np.all(nan_mask):
                continue
            result[i, j, nan_mask] = np.interp(x[nan_mask], x[~nan_mask], y[~nan_mask])

    return result


def pad_samples(samples, padding_value=0, to_length=None):
    # Step 1: Find the maximum size of the second dimension
    if to_length is None:
        to_length = max([sample.shape[1] for sample in samples])

    output = np.zeros((len(samples), samples[0].shape[0], to_length))
    # Step 2: Pad each sample's second dimension using numpy.pad

    for i, sample in enumerate(samples):
        second_dim_len = sample.shape[1]

        # Pad the second dimension with the padding_value
        padded_sample = np.pad(
            sample,
            ((0, 0), (0, to_length - second_dim_len)),
            constant_values=padding_value,
        )

        # Stack the first dimension with the padded second dimension
        output[i] = padded_sample

    return output


def sample_equal_classes(train_data, train_labels, num_samples=1000):
    # Step 1: Get unique classes
    classes = np.unique(train_labels)

    # Step 2: Calculate the number of samples to be selected from each class
    num_classes = len(classes)
    samples_per_class = (
        num_samples // num_classes
    )  # Ensure total number of samples is exactly `num_samples`

    # Step 3: Sample equally from each class
    sampled_data = []
    sampled_labels = []

    for cls in classes:
        # Get indices of samples belonging to class `cls`
        class_indices = np.flatnonzero(train_labels == cls)

        # Randomly sample `samples_per_class` samples
        sampled_indices = np.random.choice(
            class_indices, samples_per_class, replace=False
        )

        # Ensure that `sampled_indices` is a flat array of integers for proper indexing
        sampled_indices = sampled_indices.astype(int)

        # Append the sampled data and labels
        sampled_data.append(train_data[sampled_indices])
        sampled_labels.append(train_labels[sampled_indices])

    # Combine the data and labels into single arrays
    sampled_data = np.vstack(sampled_data)
    sampled_labels = np.hstack(sampled_labels)

    return sampled_data, sampled_labels


def find_uci_har_dir(data_dir):
    candidates = [
        data_dir,
        os.path.join(data_dir, "UCI HAR Dataset"),
    ]

    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "train", "Inertial Signals")):
            return candidate

    raise FileNotFoundError(
        "Could not find UCI HAR Dataset. Expected either data_dir itself or "
        "data_dir/'UCI HAR Dataset' to contain train/Inertial Signals."
    )


def load_uci_har_split(data_dir, split, signal_files=UCI_HAR_SIGNAL_FILES):
    uci_dir = find_uci_har_dir(data_dir)
    signal_dir = os.path.join(uci_dir, split, "Inertial Signals")

    signals = []
    for signal_name in signal_files:
        path = os.path.join(signal_dir, f"{signal_name}_{split}.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing UCI HAR signal file: {path}")
        signals.append(np.loadtxt(path, dtype=np.float32))

    data = np.stack(signals, axis=1)
    labels_path = os.path.join(uci_dir, split, f"y_{split}.txt")
    labels = np.loadtxt(labels_path, dtype=np.int64) - 1

    return data, labels


def load_uci_har_subjects(data_dir, split):
    uci_dir = find_uci_har_dir(data_dir)
    subjects_path = os.path.join(uci_dir, split, f"subject_{split}.txt")
    if not os.path.isfile(subjects_path):
        raise FileNotFoundError(f"Missing UCI HAR subject file: {subjects_path}")
    subjects = np.loadtxt(subjects_path, dtype=np.int64)
    return np.atleast_1d(subjects)


def _resolve_uci_har_channel_spec(har_channels):
    if har_channels == "all":
        return UCI_HAR_SIGNAL_FILES, list(range(9)), 9
    if har_channels == "acc_gyro":
        return UCI_HAR_ACC_GYRO_SIGNAL_FILES, UCI_HAR_ACC_GYRO_INDICES, 6
    raise ValueError(f"Unsupported HAR channel subset: {har_channels}")


def _split_uci_har_train_subjects(subject_ids, val_ratio, split_seed=42):
    if not 0 < val_ratio < 1:
        raise ValueError(f"Validation ratio must be between 0 and 1, got {val_ratio}.")
    unique_subjects = np.unique(np.asarray(subject_ids, dtype=np.int64))
    if len(unique_subjects) < 2:
        raise ValueError("UCI HAR subject-level validation needs at least two subjects.")

    shuffled = unique_subjects.copy()
    np.random.default_rng(split_seed).shuffle(shuffled)
    validation_count = int(len(shuffled) * val_ratio)
    validation_count = min(max(validation_count, 1), len(shuffled) - 1)
    validation_subjects = np.sort(shuffled[:validation_count])
    train_subjects = np.sort(shuffled[validation_count:])
    return train_subjects, validation_subjects


def get_uci_har_official_dataloaders(args):
    har_channels = getattr(args, "har_channels", "all")
    signal_files, _, expected_channels = _resolve_uci_har_channel_spec(har_channels)
    uci_dir = find_uci_har_dir(args.data_dir)

    official_train_data, official_train_labels = load_uci_har_split(
        uci_dir, "train", signal_files=signal_files
    )
    test_data, test_labels = load_uci_har_split(
        uci_dir, "test", signal_files=signal_files
    )
    official_train_subjects = load_uci_har_subjects(uci_dir, "train")
    test_sample_subjects = load_uci_har_subjects(uci_dir, "test")
    if len(official_train_subjects) != len(official_train_data):
        raise ValueError("UCI HAR train subjects and samples have different lengths.")
    if len(test_sample_subjects) != len(test_data):
        raise ValueError("UCI HAR test subjects and samples have different lengths.")

    train_subjects, vali_subjects = _split_uci_har_train_subjects(
        official_train_subjects,
        getattr(args, "val_ratio", 0.25),
        split_seed=42,
    )
    test_subjects = np.unique(test_sample_subjects)
    if (
        np.intersect1d(train_subjects, vali_subjects).size
        or np.intersect1d(train_subjects, test_subjects).size
        or np.intersect1d(vali_subjects, test_subjects).size
    ):
        raise AssertionError("UCI HAR subject leakage detected between splits.")

    train_mask = np.isin(official_train_subjects, train_subjects)
    vali_mask = np.isin(official_train_subjects, vali_subjects)
    train_data = official_train_data[train_mask]
    train_labels = official_train_labels[train_mask]
    vali_data = official_train_data[vali_mask]
    vali_labels = official_train_labels[vali_mask]
    if train_data.shape[1] != expected_channels:
        raise ValueError(
            f"Expected UCI HAR {har_channels} data to contain {expected_channels} "
            f"channels, got {train_data.shape}."
        )

    train_loader, vali_loader = _make_tensor_loaders(
        train_data, vali_data, args.batch_size
    )
    _, test_loader = _make_tensor_loaders(
        train_data, test_data, args.batch_size
    )
    # Keep subject metadata on the CPU dataset object.  The raw training batch
    # intentionally remains a one-tensor tuple because downstream feature
    # extractors interpret additional tensors as sequence lengths.
    train_loader.dataset.sample_subject_ids = np.asarray(
        official_train_subjects[train_mask], dtype=np.int64
    )
    vali_loader.dataset.sample_subject_ids = np.asarray(
        official_train_subjects[vali_mask], dtype=np.int64
    )
    test_loader.dataset.sample_subject_ids = np.asarray(
        test_sample_subjects, dtype=np.int64
    )
    bundle = UCIHARSubjectBundle(
        train_loader=train_loader,
        train_labels=train_labels,
        vali_loader=vali_loader,
        vali_labels=vali_labels,
        test_loader=test_loader,
        test_labels=test_labels,
        train_subjects=train_subjects.tolist(),
        vali_subjects=vali_subjects.tolist(),
        test_subjects=test_subjects.tolist(),
        har_channels=har_channels,
    )
    print(
        f"UCI HAR official_subject: channels={har_channels}:{expected_channels}; "
        f"subjects=train:{len(bundle.train_subjects)}/vali:{len(bundle.vali_subjects)}"
        f"/test:{len(bundle.test_subjects)}; "
        f"samples=train:{len(train_labels)}/vali:{len(vali_labels)}"
        f"/test:{len(test_labels)}; split_seed=42; subject_leakage=PASS"
    )
    return bundle


def write_uci_har_subject_split_audit(bundle, result_dir):
    split_dir = Path(result_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / "UCIHAR_official_subject_split.csv"
    rows = []
    for split, subjects, official_partition in (
        ("train", bundle.train_subjects, "train"),
        ("vali", bundle.vali_subjects, "train"),
        ("test", bundle.test_subjects, "test"),
    ):
        rows.extend((int(subject_id), official_partition, split) for subject_id in subjects)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "official_partition", "split"])
        writer.writerows(sorted(rows))
    return output_path


def find_preprocessed_har_dir(data_dir, dirname, required=True):
    candidates = [
        data_dir,
        os.path.join(data_dir, dirname),
        os.path.join(data_dir, dirname, dirname),
        os.path.join(data_dir, "med_data", dirname),
        os.path.join(data_dir, "med_data", dirname, dirname),
    ]

    seen = set()
    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        feature_path = os.path.join(candidate, "Feature", "feature.npy")
        label_path = os.path.join(candidate, "Label", "label.npy")
        if os.path.isfile(feature_path) and os.path.isfile(label_path):
            return candidate

    if not required:
        return None

    raise FileNotFoundError(
        f"Could not find preprocessed {dirname}. Expected Feature/feature.npy "
        f"and Label/label.npy below {data_dir!r}, optionally under med_data/{dirname}."
    )


def load_preprocessed_har(data_dir, dirname, channel_indices=None):
    dataset_dir = find_preprocessed_har_dir(data_dir, dirname)
    feature_path = os.path.join(dataset_dir, "Feature", "feature.npy")
    label_path = os.path.join(dataset_dir, "Label", "label.npy")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)

    if features.ndim != 3:
        raise ValueError(
            f"Expected {dirname} features with shape (samples, time, channels), "
            f"got {features.shape}."
        )
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError(
            f"Expected one {dirname} label per sample, got features={features.shape}, "
            f"labels={labels.shape}."
        )

    if channel_indices is None:
        channel_indices = list(range(features.shape[2]))
    else:
        channel_indices = list(channel_indices)
    if not channel_indices or len(set(channel_indices)) != len(channel_indices):
        raise ValueError(f"Channel indices must be non-empty and unique: {channel_indices}")
    if min(channel_indices) < 0 or max(channel_indices) >= features.shape[2]:
        raise ValueError(
            f"Channel indices {channel_indices} are invalid for {dirname} shape "
            f"{features.shape}."
        )

    # Downloaded Medformer arrays use (N, T, C); NeuroSigViT expects (N, C, T).
    data = np.asarray(
        features[:, :, channel_indices], dtype=np.float32
    ).transpose(0, 2, 1)
    encoded_labels = _encode_labels(np.asarray(labels))

    return data, encoded_labels


def _split_array_data(data, labels, test_ratio, random_seed):
    train_indices, test_indices = _split_indices(labels, test_ratio, random_seed)
    return (
        data[train_indices],
        labels[train_indices],
        data[test_indices],
        labels[test_indices],
    )


def find_dataset_dir(data_dir, dirname_candidates, required_glob="*.csv"):
    candidates = [data_dir]
    candidates.extend(os.path.join(data_dir, dirname) for dirname in dirname_candidates)

    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        if required_glob is None:
            return candidate
        if _glob_csv_files(candidate, required_glob):
            return candidate

    joined = ", ".join(dirname_candidates)
    raise FileNotFoundError(
        f"Could not find dataset CSV files. Expected data_dir itself or one of "
        f"these subdirectories to contain {required_glob}: {joined}."
    )


def _glob_csv_files(data_dir, pattern):
    from glob import glob

    return sorted(
        path for path in glob(os.path.join(data_dir, pattern)) if os.path.isfile(path)
    )


def _validate_window_args(window_size, stride):
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")


def _read_csv(path):
    import pandas as pd

    return pd.read_csv(path)


def _check_columns(df, columns, csv_file):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_file}: {missing}")


def _numeric_values(df, feature_cols):
    import pandas as pd

    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32)
    if np.isnan(values).any():
        values = linear_interpolation(values.T[None, :, :])[0].T
        values = np.nan_to_num(values, nan=0.0)

    return values


def _encode_labels(labels):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}

    return np.asarray([label_to_idx[label] for label in labels], dtype=np.int64)


def _standardize_from_train(train_data, test_data):
    mean = np.nanmean(train_data, axis=(0, 2), keepdims=True)
    std = np.nanstd(train_data, axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    return (train_data - mean) / std, (test_data - mean) / std


def _split_indices(labels, test_ratio, random_seed):
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    _, counts = np.unique(labels, return_counts=True)
    test_count = int(np.ceil(len(labels) * test_ratio))
    train_count = len(labels) - test_count
    class_count = len(counts)
    stratify = (
        labels
        if np.all(counts >= 2) and test_count >= class_count and train_count >= class_count
        else None
    )

    if stratify is None:
        print(
            "Warning: at least one class has fewer than two windows; "
            "using a non-stratified train/test split."
        )

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratify,
    )

    return train_indices, test_indices


def _resplit_data(
    train_data,
    train_labels,
    test_data,
    test_labels,
    test_ratio,
    random_seed,
):
    labels = np.concatenate((np.asarray(train_labels), np.asarray(test_labels)))
    train_indices, test_indices = _split_indices(labels, test_ratio, random_seed)

    if isinstance(train_data, list):
        data = train_data + test_data
        new_train_data = [data[index] for index in train_indices]
        new_test_data = [data[index] for index in test_indices]
    else:
        data = np.concatenate((train_data, test_data), axis=0)
        new_train_data = data[train_indices]
        new_test_data = data[test_indices]

    return (
        new_train_data,
        labels[train_indices],
        new_test_data,
        labels[test_indices],
    )


def _make_windows_from_segment(values, label, window_size, stride, samples, labels):
    if len(values) < window_size:
        return

    for start in range(0, len(values) - window_size + 1, stride):
        samples.append(values[start : start + window_size].T)
        labels.append(label)


def _iter_contiguous_label_segments(df, label_column):
    label_changes = df[label_column].ne(df[label_column].shift()).cumsum()
    for _, segment in df.groupby(label_changes, sort=False):
        label = segment[label_column].iloc[0]
        if label != label:
            continue
        yield label, segment


def _build_custom_split(samples, labels, test_ratio, random_seed):
    if not samples:
        raise ValueError(
            "No windows were created. Check data paths, labels, window_size, and stride."
        )

    data = np.asarray(samples, dtype=np.float32)
    labels = _encode_labels(labels)
    train_indices, test_indices = _split_indices(labels, test_ratio, random_seed)

    train_data = data[train_indices]
    test_data = data[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    train_data, test_data = _standardize_from_train(train_data, test_data)

    return (
        train_data.astype(np.float32),
        train_labels,
        test_data.astype(np.float32),
        test_labels,
    )


def load_falltl_data(
    data_dir,
    test_ratio=0.2,
    random_seed=None,
    window_size=200,
    stride=100,
    max_windows_per_file=None,
):
    _validate_window_args(window_size, stride)
    falltl_dir = find_dataset_dir(data_dir, ["FallTL", "falltl"], "*.csv")
    csv_files = _glob_csv_files(falltl_dir, "*.csv")

    samples = []
    labels = []

    for csv_file in csv_files:
        df = _read_csv(csv_file)
        _check_columns(df, FALLTL_FEATURE_COLUMNS, csv_file)

        if "Label" in df.columns:
            label_segments = _iter_contiguous_label_segments(df, "Label")
        else:
            filename_parts = os.path.splitext(os.path.basename(csv_file))[0].split("_")
            label = filename_parts[1] if len(filename_parts) >= 2 else filename_parts[0]
            label_segments = [(label, df)]

        created_for_file = 0
        for label, segment in label_segments:
            values = _numeric_values(segment, FALLTL_FEATURE_COLUMNS)
            before = len(samples)
            _make_windows_from_segment(
                values, label, window_size, stride, samples, labels
            )
            created_for_file += len(samples) - before

            if max_windows_per_file and created_for_file >= max_windows_per_file:
                extra = created_for_file - max_windows_per_file
                if extra > 0:
                    del samples[-extra:]
                    del labels[-extra:]
                break

    return _build_custom_split(samples, labels, test_ratio, random_seed)


def _natural_path_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", os.path.basename(path))
    ]


def _parse_falltl_comparison_filename(filename):
    basename = os.path.basename(os.fspath(filename))
    match = FALLTL_COMPARISON_FILENAME_RE.fullmatch(basename)
    if match is None:
        raise ValueError(
            f"Invalid FallTL comparison filename {basename!r}. Expected "
            "<Dxx|Fxx[direction][ending]>_<body_part_code>_<trial_no>.csv, "
            "where direction is F/B/L, ending is R/W, and body_part_code is "
            f"one of {FALLTL_BODY_PART_CODES}."
        )

    activity_code = match.group("activity").upper()
    body_part_code = match.group("body").upper()
    trial_no = int(match.group("trial"))
    return activity_code, body_part_code, trial_no


def _interpolate_falltl_sequence(values, source_file):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"FallTL file {source_file} has no data rows.")
    positions = np.arange(len(values), dtype=np.float32)
    interpolated = values.copy()
    for channel in range(values.shape[1]):
        observed = np.isfinite(values[:, channel])
        if not np.any(observed):
            raise ValueError(
                f"FallTL file {source_file} has no finite values in channel {channel}."
            )
        interpolated[:, channel] = np.interp(
            positions,
            positions[observed],
            values[observed, channel],
        )
    return interpolated


def _validate_falltl_target_length(target_length):
    if (
        isinstance(target_length, bool)
        or not isinstance(target_length, (int, np.integer))
        or target_length <= 0
    ):
        raise ValueError(
            "falltl_target_length must be a positive integer, "
            f"got {target_length!r}."
        )
    return int(target_length)


def _resample_falltl_sequence(sequence, target_length):
    source_length, channels = sequence.shape
    source_positions = np.linspace(0.0, 1.0, source_length, dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    resampled = np.empty((channels, target_length), dtype=np.float32)
    for channel in range(channels):
        resampled[channel] = np.interp(
            target_positions,
            source_positions,
            sequence[:, channel],
        ).astype(np.float32)
    return resampled


def _standardize_and_resample_falltl(
    train_sequences, *other_splits, target_length
):
    target_length = _validate_falltl_target_length(target_length)
    if not train_sequences:
        raise ValueError("FallTL training split is empty.")

    # Fit channel statistics to every original training point before temporal
    # resampling. Validation and test points never influence these statistics.
    train_points = np.concatenate(train_sequences, axis=0).astype(
        np.float64, copy=False
    )
    mean = train_points.mean(axis=0, keepdims=True)
    std = train_points.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    resampled_splits = []
    for sequences in (train_sequences, *other_splits):
        if not sequences:
            raise ValueError("FallTL comparison split is empty.")
        resampled = np.stack(
            [
                _resample_falltl_sequence(
                    (sequence.astype(np.float64, copy=False) - mean) / std,
                    target_length,
                )
                for sequence in sequences
            ],
            axis=0,
        )
        resampled_splits.append(resampled.astype(np.float32, copy=False))
    return tuple(resampled_splits)


def _load_falltl_comparison_arrays(data_dir):
    falltl_dir = find_dataset_dir(data_dir, ["FallTL", "falltl"], "*.csv")
    csv_files = sorted(
        _glob_csv_files(falltl_dir, "*.csv"), key=_natural_path_key
    )
    sequences = []
    labels = []
    source_files = []
    seen_basenames = {}
    for csv_file in csv_files:
        source_file = os.path.basename(csv_file)
        activity_code, body_part_code, trial_no = (
            _parse_falltl_comparison_filename(source_file)
        )

        normalized_basename = (
            f"{activity_code}_{body_part_code}_{trial_no}.csv".casefold()
        )
        if normalized_basename in seen_basenames:
            raise ValueError(
                "Duplicate FallTL comparison basename after case/trial "
                f"normalization: {seen_basenames[normalized_basename]!r} and "
                f"{source_file!r}."
            )
        seen_basenames[normalized_basename] = source_file

        df = _read_csv(csv_file)
        _check_columns(df, FALLTL_FEATURE_COLUMNS, csv_file)
        import pandas as pd

        values = (
            df.loc[:, FALLTL_FEATURE_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        sequences.append(_interpolate_falltl_sequence(values, source_file))
        labels.append(1 if activity_code.startswith("F") else 0)
        source_files.append(source_file)

    if not sequences:
        raise FileNotFoundError(f"No FallTL CSV files found below {data_dir}")
    return (
        sequences,
        np.asarray(labels, dtype=np.int64),
        np.asarray(source_files),
    )


def get_falltl_comparison_dataloaders(args):
    from sklearn.model_selection import train_test_split

    sequences, labels, source_files = _load_falltl_comparison_arrays(args.data_dir)
    target_length = _validate_falltl_target_length(
        getattr(args, "falltl_target_length", 2048)
    )

    metadata = []
    for source_file in source_files:
        activity_code, body_part_code, trial_no = (
            _parse_falltl_comparison_filename(source_file)
        )
        metadata.append(
            (activity_code, body_part_code, trial_no, (activity_code, trial_no))
        )
    file_group_ids = [item[3] for item in metadata]
    group_to_label = {}
    for group_id, label in zip(file_group_ids, labels):
        label = int(label)
        existing_label = group_to_label.setdefault(group_id, label)
        if existing_label != label:
            raise ValueError(
                f"FallTL event group {group_id!r} contains conflicting labels."
            )

    unique_groups = sorted(group_to_label)
    group_labels = np.asarray(
        [group_to_label[group_id] for group_id in unique_groups], dtype=np.int64
    )
    group_indices = np.arange(len(unique_groups))
    try:
        train_group_indices, remainder_group_indices = train_test_split(
            group_indices,
            test_size=0.4,
            random_state=42,
            stratify=group_labels,
        )
        vali_group_indices, test_group_indices = train_test_split(
            remainder_group_indices,
            test_size=0.5,
            random_state=42,
            stratify=group_labels[remainder_group_indices],
        )
    except ValueError as exc:
        raise ValueError(
            "FallTL comparison_binary requires enough unique D/F event groups "
            "for a label-stratified 60/20/20 split."
        ) from exc

    train_groups = {unique_groups[index] for index in train_group_indices}
    vali_groups = {unique_groups[index] for index in vali_group_indices}
    test_groups = {unique_groups[index] for index in test_group_indices}
    assert train_groups.isdisjoint(vali_groups)
    assert train_groups.isdisjoint(test_groups)
    assert vali_groups.isdisjoint(test_groups)

    def expand_groups(groups):
        return np.asarray(
            [
                index
                for index, group_id in enumerate(file_group_ids)
                if group_id in groups
            ],
            dtype=np.int64,
        )

    train_indices = expand_groups(train_groups)
    vali_indices = expand_groups(vali_groups)
    test_indices = expand_groups(test_groups)

    train_data, vali_data, test_data = _standardize_and_resample_falltl(
        [sequences[index] for index in train_indices],
        [sequences[index] for index in vali_indices],
        [sequences[index] for index in test_indices],
        target_length=target_length,
    )
    train_loader, vali_loader = _make_tensor_loaders(
        train_data, vali_data, args.batch_size
    )
    _, test_loader = _make_tensor_loaders(
        train_data, test_data, args.batch_size
    )
    bundle = FallTLComparisonBundle(
        train_loader=train_loader,
        train_labels=labels[train_indices],
        vali_loader=vali_loader,
        vali_labels=labels[vali_indices],
        test_loader=test_loader,
        test_labels=labels[test_indices],
        train_files=source_files[train_indices].tolist(),
        vali_files=source_files[vali_indices].tolist(),
        test_files=source_files[test_indices].tolist(),
        train_group_ids=sorted(train_groups),
        vali_group_ids=sorted(vali_groups),
        test_group_ids=sorted(test_groups),
        target_length=target_length,
    )
    distributions = []
    for split, split_labels in (
        ("train", bundle.train_labels),
        ("vali", bundle.vali_labels),
        ("test", bundle.test_labels),
    ):
        values, counts = np.unique(split_labels, return_counts=True)
        distributions.append(
            f"{split}=" + "/".join(
                f"{int(value)}:{int(count)}"
                for value, count in zip(values, counts)
            )
        )
    print(
        "FallTL comparison_binary: one_sequence_per_csv; labels=D:0/F:1; "
        f"resampled_length={target_length}; "
        "group_split=(activity_code,trial_no), stratified=60/20/20, seed=42; "
        f"groups=train:{len(train_groups)}/vali:{len(vali_groups)}"
        f"/test:{len(test_groups)}; group_overlap=0/0/0; "
        + "; ".join(distributions)
    )
    return bundle


def write_falltl_comparison_split_audit(bundle, result_dir):
    split_dir = Path(result_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / "FallTL_comparison_binary_split.csv"
    rows = []
    for split, files, labels in (
        ("train", bundle.train_files, bundle.train_labels),
        ("vali", bundle.vali_files, bundle.vali_labels),
        ("test", bundle.test_files, bundle.test_labels),
    ):
        for filename, label in zip(files, labels):
            activity_code, body_part_code, trial_no = (
                _parse_falltl_comparison_filename(filename)
            )
            rows.append(
                (
                    filename,
                    int(label),
                    activity_code,
                    body_part_code,
                    trial_no,
                    f"{activity_code}_{trial_no}",
                    split,
                )
            )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "filename",
                "label_id",
                "activity_code",
                "body_part_code",
                "trial_no",
                "event_group",
                "split",
            ]
        )
        writer.writerows(sorted(rows))
    return output_path


def _find_feng_feature_columns(columns):
    available = set(columns)
    preferred = []

    for sensor in FENG_PREFERRED_SENSORS:
        for signal in FENG_PREFERRED_SIGNALS:
            for axis in ["X", "Y", "Z"]:
                candidates = [
                    f"{signal}_{axis}_{sensor}",
                    f"{signal}{axis}_{sensor}",
                    f"{sensor}_{signal}_{axis}",
                    f"{sensor}_{signal}{axis}",
                ]
                match = next((column for column in candidates if column in available), None)
                if match:
                    preferred.append(match)

    if preferred:
        return preferred

    excluded = {"Activity", "TimeStamp", "Timestamp", "Time", "Subject"}
    numeric_like = []
    for column in columns:
        if column in excluded:
            continue
        if any(token in column.lower() for token in ["acc", "gyr", "gyro", "quat"]):
            numeric_like.append(column)

    return numeric_like


def load_feng_data(
    data_dir,
    test_ratio=0.2,
    random_seed=None,
    window_size=200,
    stride=100,
    max_windows_per_file=None,
):
    _validate_window_args(window_size, stride)
    feng_dir = find_dataset_dir(
        data_dir,
        [
            "Feng et al.",
            os.path.join("Feng", "dataset"),
            "Feng",
            os.path.join("feng", "dataset"),
            "feng",
            "feng_et_al",
            "dataset",
        ],
        "P*.csv",
    )
    csv_files = _glob_csv_files(feng_dir, "P*.csv")

    samples = []
    labels = []

    for csv_file in csv_files:
        df = _read_csv(csv_file)
        _check_columns(df, ["Activity"], csv_file)
        feature_cols = _find_feng_feature_columns(df.columns)
        if not feature_cols:
            raise ValueError(
                f"Could not identify Feng feature columns in {csv_file}. "
                "Expected sensor columns containing Acc, Gyr/Gyro, or Quat."
            )

        created_for_file = 0
        for label, segment in _iter_contiguous_label_segments(df, "Activity"):
            values = _numeric_values(segment, feature_cols)
            before = len(samples)
            _make_windows_from_segment(
                values, label, window_size, stride, samples, labels
            )
            created_for_file += len(samples) - before

            if max_windows_per_file and created_for_file >= max_windows_per_file:
                extra = created_for_file - max_windows_per_file
                if extra > 0:
                    del samples[-extra:]
                    del labels[-extra:]
                break

    return _build_custom_split(samples, labels, test_ratio, random_seed)


def _make_tensor_loaders(train_data, test_data, batch_size):
    train_loader = DataLoader(
        TensorDataset(torch.Tensor(train_data).type(torch.float)),
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.Tensor(test_data).type(torch.float)),
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eeg_medformer_content_sha256(data_root):
    """Match `sha256sum Feature/*.npy Label/label.npy | sha256sum`."""
    data_root = Path(data_root)
    paths = sorted(
        (data_root / "Feature").glob("feature_*.npy"), key=lambda path: path.name
    )
    paths.append(data_root / "Label" / "label.npy")
    lines = "".join(
        f"{_sha256_file(path)}  {path.relative_to(data_root).as_posix()}\n"
        for path in paths
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def find_eeg_medformer_data_root(data_dir, dataset_name):
    dataset_name = str(dataset_name).upper()
    if dataset_name not in EEG_MEDFORMER_DATASET_NAMES:
        raise ValueError(f"Unsupported Medformer EEG dataset: {dataset_name}")

    base = Path(data_dir).expanduser()
    candidates = (
        base,
        base / dataset_name,
        base / "processed" / dataset_name,
    )
    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if (
            (normalized / "Feature").is_dir()
            and (normalized / "Label" / "label.npy").is_file()
        ):
            return normalized
    raise FileNotFoundError(
        f"Could not find processed Medformer dataset {dataset_name} below "
        f"{data_dir!r}. Expected Feature/feature_*.npy and Label/label.npy."
    )


def get_eeg_medformer_inventory(data_dir, dataset_name, verify_content=True):
    dataset_name = str(dataset_name).upper()
    if dataset_name not in EEG_MEDFORMER_SPECS:
        raise ValueError(f"Unsupported Medformer EEG dataset: {dataset_name}")
    spec = EEG_MEDFORMER_SPECS[dataset_name]
    data_root = find_eeg_medformer_data_root(data_dir, dataset_name)
    label_path = data_root / "Label" / "label.npy"

    label_sha256 = _sha256_file(label_path)
    if label_sha256 != spec["label_sha256"]:
        raise ValueError(
            f"{dataset_name} label SHA-256 mismatch: {label_sha256}; "
            f"expected {spec['label_sha256']}"
        )

    labels = np.load(label_path, allow_pickle=False)
    if labels.ndim != 2 or labels.shape != (spec["subject_count"], 2):
        raise ValueError(
            f"Expected {dataset_name} labels with shape "
            f"({spec['subject_count']}, 2), got {labels.shape}."
        )

    label_by_id = {}
    for raw_label, raw_subject_id in labels.tolist():
        if not float(raw_label).is_integer() or not float(raw_subject_id).is_integer():
            raise ValueError(
                f"{dataset_name} label rows must contain integer-valued IDs."
            )
        label = int(raw_label)
        subject_id = int(raw_subject_id)
        if label not in spec["class_names"]:
            raise ValueError(f"Unexpected {dataset_name} class ID: {label}")
        if subject_id in label_by_id:
            raise ValueError(f"Duplicate {dataset_name} subject ID: {subject_id}")
        label_by_id[subject_id] = label

    feature_by_id = {}
    pattern = re.compile(r"^feature_(\d+)\.npy$")
    for path in sorted((data_root / "Feature").glob("*.npy")):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected {dataset_name} feature file: {path.name}")
        subject_id = int(match.group(1))
        if subject_id in feature_by_id:
            raise ValueError(f"Duplicate {dataset_name} feature ID: {subject_id}")
        feature_by_id[subject_id] = path.resolve()

    if set(feature_by_id) != set(label_by_id):
        raise ValueError(
            f"{dataset_name} feature/label subject mismatch: "
            f"missing_features={sorted(set(label_by_id) - set(feature_by_id))}, "
            f"missing_labels={sorted(set(feature_by_id) - set(label_by_id))}"
        )

    records = []
    for subject_id in sorted(label_by_id):
        feature_path = feature_by_id[subject_id]
        array = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        expected_tail = (256, spec["channels"])
        if array.ndim != 3 or tuple(array.shape[1:]) != expected_tail:
            raise ValueError(
                f"Unexpected {dataset_name} feature shape for subject "
                f"{subject_id}: {array.shape}; expected [N,{expected_tail[0]},"
                f"{expected_tail[1]}]."
            )
        if array.dtype != np.float64:
            raise ValueError(
                f"Unexpected {dataset_name} source dtype for subject "
                f"{subject_id}: {array.dtype}; expected float64."
            )
        records.append(
            EEGMedformerSubjectRecord(
                subject_id=subject_id,
                label=label_by_id[subject_id],
                feature_path=feature_path,
                window_count=int(array.shape[0]),
            )
        )

    split_ids = {
        split: tuple(int(value) for value in ids)
        for split, ids in spec["split_ids"].items()
    }
    assigned = [value for ids in split_ids.values() for value in ids]
    if len(assigned) != len(set(assigned)):
        raise ValueError(f"{dataset_name} subject leakage detected between splits.")
    available = set(label_by_id)
    unknown = set(assigned) - available
    if unknown:
        raise ValueError(f"{dataset_name} split contains unknown subjects: {unknown}")
    excluded_ids = tuple(sorted(available - set(assigned)))
    expected_excluded = tuple(spec.get("expected_excluded_ids", ()))
    if excluded_ids != expected_excluded:
        raise ValueError(
            f"{dataset_name} excluded cohort mismatch: {excluded_ids}; "
            f"expected {expected_excluded}."
        )
    for split, expected in spec["expected_subject_counts"].items():
        if len(split_ids[split]) != expected:
            raise ValueError(
                f"{dataset_name} {split} subject count mismatch: "
                f"{len(split_ids[split])}; expected {expected}."
            )

    content_sha256 = spec["content_sha256"]
    if verify_content:
        content_sha256 = _eeg_medformer_content_sha256(data_root)
        if content_sha256 != spec["content_sha256"]:
            raise ValueError(
                f"{dataset_name} aggregate content SHA-256 mismatch: "
                f"{content_sha256}; expected {spec['content_sha256']}"
            )

    return EEGMedformerInventory(
        dataset_name=dataset_name,
        data_root=data_root,
        records=tuple(records),
        split_ids=split_ids,
        excluded_ids=excluded_ids,
        protocol=spec["protocol"],
        label_sha256=label_sha256,
        content_sha256=content_sha256,
    )


def _normalize_eeg_medformer_windows(windows):
    windows = np.asarray(windows)
    if windows.ndim != 3:
        raise ValueError(f"Expected EEG windows [N,T,C], got {windows.shape}.")
    if not np.isfinite(windows).all():
        raise ValueError("Medformer EEG source contains NaN or infinite values.")
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, ddof=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    normalized = ((windows - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Medformer EEG normalization produced non-finite values.")
    return normalized


def _make_eeg_medformer_split_loader(inventory, split, batch_size):
    spec = EEG_MEDFORMER_SPECS[inventory.dataset_name]
    record_by_id = {record.subject_id: record for record in inventory.records}
    subject_ids = inventory.split_ids[split]
    total_windows = sum(record_by_id[value].window_count for value in subject_ids)
    expected_windows = spec["expected_window_counts"][split]
    if total_windows != expected_windows:
        raise ValueError(
            f"{inventory.dataset_name} {split} window count mismatch: "
            f"{total_windows}; expected {expected_windows}."
        )

    inputs = np.empty(
        (total_windows, spec["channels"], 256), dtype=np.float32
    )
    labels = np.empty(total_windows, dtype=np.int64)
    sample_subject_ids = np.empty(total_windows, dtype=np.int64)
    sample_window_indices = np.empty(total_windows, dtype=np.int64)
    cursor = 0
    for subject_id in subject_ids:
        record = record_by_id[subject_id]
        source = np.load(record.feature_path, mmap_mode="r", allow_pickle=False)
        normalized = _normalize_eeg_medformer_windows(source)
        next_cursor = cursor + record.window_count
        inputs[cursor:next_cursor] = normalized.transpose(0, 2, 1)
        labels[cursor:next_cursor] = record.label
        sample_subject_ids[cursor:next_cursor] = subject_id
        sample_window_indices[cursor:next_cursor] = np.arange(
            record.window_count, dtype=np.int64
        )
        cursor = next_cursor

    tensor_dataset = TensorDataset(torch.from_numpy(inputs))
    tensor_dataset.sample_subject_ids = sample_subject_ids
    tensor_dataset.sample_window_indices = sample_window_indices
    tensor_dataset.split_name = split
    loader = DataLoader(
        tensor_dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
    )
    return loader, labels


def get_eeg_medformer_dataloaders(dataset_name, args):
    protocol = getattr(args, "eeg_protocol", "medformer_code_exact")
    if protocol != "medformer_code_exact":
        raise ValueError(f"Unsupported Medformer EEG protocol: {protocol}")
    normalization = getattr(
        args, "eeg_normalization", EEG_MEDFORMER_NORMALIZATION
    )
    if normalization != EEG_MEDFORMER_NORMALIZATION:
        raise ValueError(f"Unsupported Medformer EEG normalization: {normalization}")

    inventory = get_eeg_medformer_inventory(
        args.data_dir, dataset_name, verify_content=True
    )
    train_loader, train_labels = _make_eeg_medformer_split_loader(
        inventory, "train", args.batch_size
    )
    vali_loader, vali_labels = _make_eeg_medformer_split_loader(
        inventory, "vali", args.batch_size
    )
    test_loader, test_labels = _make_eeg_medformer_split_loader(
        inventory, "test", args.batch_size
    )
    bundle = EEGMedformerBundle(
        inventory=inventory,
        train_loader=train_loader,
        train_labels=train_labels,
        vali_loader=vali_loader,
        vali_labels=vali_labels,
        test_loader=test_loader,
        test_labels=test_labels,
    )

    distributions = []
    for split, labels in (
        ("train", train_labels),
        ("vali", vali_labels),
        ("test", test_labels),
    ):
        values, counts = np.unique(labels, return_counts=True)
        distribution = "/".join(
            f"{int(value)}:{int(count)}" for value, count in zip(values, counts)
        )
        distributions.append(f"{split}=[{distribution}]")
    print(
        f"EEG {inventory.dataset_name}: "
        f"train={len(train_labels)}, vali={len(vali_labels)}, "
        f"test={len(test_labels)}; {' '.join(distributions)}; "
        f"protocol={inventory.protocol}; normalization={normalization}; "
        "subject_split=PASS; metric_unit=window"
    )
    return bundle


def write_eeg_medformer_split_audit(bundle, result_dir):
    split_dir = Path(result_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    inventory = bundle.inventory
    output_path = split_dir / f"{inventory.dataset_name}_subject_split.csv"
    spec = EEG_MEDFORMER_SPECS[inventory.dataset_name]
    split_by_id = {}
    for split, subject_ids in inventory.split_ids.items():
        for subject_id in subject_ids:
            split_by_id[subject_id] = split

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_name",
                "protocol",
                "normalization",
                "metric_unit",
                "legacy_subject_id",
                "label_id",
                "label_name",
                "window_count",
                "split",
                "label_sha256",
                "aggregate_content_sha256",
            ]
        )
        for record in inventory.records:
            writer.writerow(
                [
                    inventory.dataset_name,
                    inventory.protocol,
                    bundle.normalization,
                    "processed_one_second_window",
                    record.subject_id,
                    record.label,
                    spec["class_names"][record.label],
                    record.window_count,
                    split_by_id.get(record.subject_id, "excluded"),
                    inventory.label_sha256,
                    inventory.content_sha256,
                ]
            )
    return output_path


def find_wearable_data_root(data_dir, dataset_name):
    if dataset_name not in WEARABLE_DATASET_NAMES:
        raise ValueError(f"Unsupported wearable dataset: {dataset_name}")

    base = Path(data_dir).expanduser()
    candidates = (
        base,
        base / "wearable",
        base / "Neuro" / "wearable",
        base / "Neuro",
        base / "med_data" / "wearable",
        base / "med_data",
    )
    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if not (
            (normalized / dataset_name / "Meta" / "subject_map.csv").is_file()
            and (normalized / dataset_name / "Feature").is_dir()
        ):
            continue
        return normalized

    raise FileNotFoundError(
        f"Could not find wearable/{dataset_name} below {data_dir!r}. "
        "Expected the dataset Feature/ and Meta/subject_map.csv files."
    )


def _find_wearable_reference_root(data_root):
    candidates = (
        Path(data_root) / "data_loading",
        Path(__file__).resolve().parents[1] / "data_loading",
    )
    for candidate in candidates:
        if (
            (candidate / "datasets.py").is_file()
            and (candidate / "split_reference_seed42.csv").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Missing wearable reference loader and split file. Expected "
        "data_loading/datasets.py and data_loading/split_reference_seed42.csv "
        "either beside the dataset root or in the repository."
    )


def _load_wearable_reference(reference_root):
    module_path = Path(reference_root) / "datasets.py"
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if digest != WEARABLE_REFERENCE_DATASETS_SHA256:
        raise ValueError(
            f"Unexpected SHA-256 for {module_path}: {digest}. "
            f"Expected {WEARABLE_REFERENCE_DATASETS_SHA256}."
        )

    cache_key = str(module_path.resolve())
    if cache_key in _WEARABLE_MODULE_CACHE:
        return _WEARABLE_MODULE_CACHE[cache_key]

    module_name = f"_wearable_reference_{digest[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load wearable reference module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _WEARABLE_MODULE_CACHE[cache_key] = module
    return module


def _make_wearable_tensor_loader(source_dataset, batch_size):
    if source_dataset.X is None or source_dataset.y is None:
        raise ValueError("Wearable source dataset did not load samples")

    # The reference interface is [N,T,6]; NeuroSigViT consumes [N,6,T].
    inputs = torch.from_numpy(source_dataset.X.transpose(0, 2, 1))
    tensor_dataset = TensorDataset(inputs)
    tensor_dataset.source_dataset = source_dataset
    tensor_dataset.sample_subject_ids = source_dataset.sample_subject_ids
    loader = DataLoader(
        tensor_dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
    )
    labels = np.asarray(source_dataset.y, dtype=np.int64)
    return loader, labels


def _standardize_wearable_bundle_from_train(bundle, batch_size):
    """Standardize selected task channels with training-split statistics only."""
    train_data = bundle.train_loader.dataset.tensors[0]
    mean = train_data.mean(dim=(0, 2), keepdim=True)
    std = train_data.std(dim=(0, 2), unbiased=False, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    for split in ("train", "vali", "test"):
        loader = getattr(bundle, f"{split}_loader")
        standardized = ((loader.dataset.tensors[0] - mean) / std).float()
        tensor_dataset = TensorDataset(standardized)
        tensor_dataset.source_dataset = loader.dataset.source_dataset
        tensor_dataset.sample_subject_ids = loader.dataset.sample_subject_ids
        setattr(
            bundle,
            f"{split}_loader",
            DataLoader(
                tensor_dataset,
                num_workers=0,
                batch_size=batch_size,
                shuffle=False,
            ),
        )


def _apply_wearable_label_protocol(bundle, label_mode, batch_size):
    family = "shimmer" if bundle.dataset_name.startswith("Shimmer_") else "pads"
    protocols = {
        "shimmer_hc_vs_pd": ("shimmer", {0: 0, 1: 1, 2: 1}),
        "pads_pd_vs_hc": ("pads", {0: 0, 1: 1}),
        "pads_pd_vs_omd": ("pads", {2: 0, 1: 1}),
    }

    if label_mode == "original":
        labels = bundle.train_dataset.label_names
        mapping = {int(label): int(label) for label in labels}
    else:
        if label_mode not in protocols:
            raise ValueError(f"Unsupported wearable label mode: {label_mode}")
        expected_family, mapping = protocols[label_mode]
        if family != expected_family:
            raise ValueError(
                f"Label mode {label_mode} is only valid for {expected_family} "
                f"datasets, not {bundle.dataset_name}."
            )

    for split in ("train", "vali", "test"):
        source_dataset = getattr(bundle, f"{split}_dataset")
        loader = getattr(bundle, f"{split}_loader")
        original_labels = np.asarray(source_dataset.y, dtype=np.int64)
        keep = np.isin(original_labels, np.asarray(list(mapping), dtype=np.int64))
        if not np.any(keep):
            raise ValueError(
                f"{bundle.dataset_name} split={split} has no samples for {label_mode}."
            )

        inputs = loader.dataset.tensors[0][torch.as_tensor(keep)]
        tensor_dataset = TensorDataset(inputs)
        tensor_dataset.source_dataset = source_dataset
        tensor_dataset.sample_subject_ids = np.asarray(
            source_dataset.sample_subject_ids
        )[keep]
        mapped_labels = np.asarray(
            [mapping[int(label)] for label in original_labels[keep]],
            dtype=np.int64,
        )
        setattr(
            bundle,
            f"{split}_loader",
            DataLoader(
                tensor_dataset,
                num_workers=0,
                batch_size=batch_size,
                shuffle=False,
            ),
        )
        setattr(bundle, f"{split}_labels", mapped_labels)

    bundle.label_mode = label_mode
    bundle.label_mapping = mapping


def _validate_wearable_bundle(bundle, reference_module):
    reference_csv = bundle.reference_root / "split_reference_seed42.csv"
    reference = reference_module._read_reference_csv(reference_csv)
    reference_status = "NOT_AVAILABLE"
    if bundle.dataset_name in reference:
        reference_module._verify_reference_assignment(bundle.train_dataset, reference)
        reference_status = "PASS"
    reference_module._validate_label_file(bundle.train_dataset)

    split_items = (
        ("train", bundle.train_dataset, bundle.train_loader, bundle.train_labels),
        ("vali", bundle.vali_dataset, bundle.vali_loader, bundle.vali_labels),
        ("test", bundle.test_dataset, bundle.test_loader, bundle.test_labels),
    )
    split_sample_counts = []
    for split, source_dataset, loader, labels in split_items:
        if source_dataset.split != split:
            raise AssertionError(
                f"{bundle.dataset_name}: expected split={split}, got {source_dataset.split}"
            )
        expected_shape = (
            len(source_dataset),
            source_dataset.num_channels,
            source_dataset.sequence_length,
        )
        actual_shape = tuple(loader.dataset.tensors[0].shape)
        if actual_shape != expected_shape:
            raise AssertionError(
                f"{bundle.dataset_name} split={split}: expected NeuroSigViT shape "
                f"{expected_shape}, got {actual_shape}"
            )
        if not np.array_equal(labels, source_dataset.y):
            raise AssertionError(
                f"{bundle.dataset_name} split={split}: adapter labels changed"
            )
        expected_labels = np.asarray(
            [
                source_dataset.subject_label[int(subject_id)]
                for subject_id in source_dataset.sample_subject_ids
            ],
            dtype=np.int64,
        )
        if not np.array_equal(labels, expected_labels):
            raise AssertionError(
                f"{bundle.dataset_name} split={split}: sample labels do not match subjects"
            )
        split_sample_counts.append(len(source_dataset))

    actual_split_samples = tuple(split_sample_counts)
    expected_split_samples = WEARABLE_EXPECTED_SPLIT_SAMPLES[bundle.dataset_name]
    if actual_split_samples != expected_split_samples:
        raise AssertionError(
            f"{bundle.dataset_name}: expected train/vali/test samples "
            f"{expected_split_samples}, got {actual_split_samples}"
        )
    total_samples = sum(actual_split_samples)
    if total_samples != bundle.train_dataset.expected_sample_count:
        raise AssertionError(
            f"{bundle.dataset_name}: expected "
            f"{bundle.train_dataset.expected_sample_count} total samples, "
            f"got {total_samples}"
        )
    return reference_status


def get_wearable_dataloaders(dataset_name, args):
    data_root = find_wearable_data_root(args.data_dir, dataset_name)
    reference_root = _find_wearable_reference_root(data_root)
    reference_module = _load_wearable_reference(reference_root)
    dataset_classes = {
        dataset_class.dataset_name: dataset_class
        for dataset_class in reference_module.DATASET_CLASSES
    }
    for dynamic_name, spec in WEARABLE_DYNAMIC_DATASET_SPECS.items():
        dataset_classes[dynamic_name] = type(
            f"{dynamic_name}Dataset",
            (reference_module.SubjectMapDataset,),
            {
                "dataset_name": dynamic_name,
                "relative_directories": (dynamic_name,),
                **spec,
            },
        )
    dataset_class = dataset_classes[dataset_name]

    split_datasets = {
        split: dataset_class(
            data_root=data_root,
            split=split,
            normalize=False,
            verbose=False,
        )
        for split in ("train", "vali", "test")
    }
    train_loader, train_labels = _make_wearable_tensor_loader(
        split_datasets["train"], args.batch_size
    )
    vali_loader, vali_labels = _make_wearable_tensor_loader(
        split_datasets["vali"], args.batch_size
    )
    test_loader, test_labels = _make_wearable_tensor_loader(
        split_datasets["test"], args.batch_size
    )
    bundle = WearableDataBundle(
        dataset_name=dataset_name,
        data_root=data_root,
        reference_root=reference_root,
        train_loader=train_loader,
        train_labels=train_labels,
        vali_loader=vali_loader,
        vali_labels=vali_labels,
        test_loader=test_loader,
        test_labels=test_labels,
        train_dataset=split_datasets["train"],
        vali_dataset=split_datasets["vali"],
        test_dataset=split_datasets["test"],
    )
    reference_status = _validate_wearable_bundle(bundle, reference_module)

    label_mode = getattr(args, "wearable_label_mode", "original")
    _apply_wearable_label_protocol(bundle, label_mode, args.batch_size)
    _standardize_wearable_bundle_from_train(bundle, args.batch_size)

    split_distributions = []
    for split, labels in (
        ("train", bundle.train_labels),
        ("vali", bundle.vali_labels),
        ("test", bundle.test_labels),
    ):
        values, counts = np.unique(labels, return_counts=True)
        distribution = "/".join(
            f"{int(value)}:{int(count)}" for value, count in zip(values, counts)
        )
        split_distributions.append(f"{split}=[{distribution}]")
    print(
        f"Wearable {dataset_name}: "
        f"train={len(bundle.train_labels)}, vali={len(bundle.vali_labels)}, "
        f"test={len(bundle.test_labels)}; "
        f"label_mode={label_mode}; {' '.join(split_distributions)}; "
        f"subject_split=PASS; reference={reference_status}"
    )
    return bundle


def write_wearable_split_audit(bundle, result_dir):
    split_dir = Path(result_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / f"{bundle.dataset_name}_subject_split.csv"

    assignments = []
    for split, source_dataset in (
        ("train", bundle.train_dataset),
        ("vali", bundle.vali_dataset),
        ("test", bundle.test_dataset),
    ):
        assignments.extend(
            (
                int(subject_id),
                int(source_dataset.subject_label[int(subject_id)]),
                split,
            )
            for subject_id in source_dataset.selected_subject_ids
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset_name",
            "label_protocol",
            "numeric_subject_id",
            "original_label_id",
            "label_id",
            "split",
        ])
        for subject_id, original_label_id, split in sorted(assignments):
            if original_label_id not in bundle.label_mapping:
                continue
            label_id = bundle.label_mapping[original_label_id]
            writer.writerow(
                [
                    bundle.dataset_name,
                    bundle.label_mode,
                    subject_id,
                    original_label_id,
                    label_id,
                    split,
                ]
            )
    return output_path


def get_dataloader(dataset, args):
    har_channels = getattr(args, "har_channels", "all")
    if har_channels not in {"all", "acc_gyro"}:
        raise ValueError(f"Unsupported HAR channel subset: {har_channels}")

    if args.datasets == "flaap":
        data, labels = load_preprocessed_har(
            args.data_dir,
            "FLAAP",
            channel_indices=range(6),
        )
        if data.shape[1] != 6:
            raise ValueError(f"Expected FLAAP to contain 6 channels, got {data.shape}.")
        train_data, train_labels, test_data, test_labels = _split_array_data(
            data,
            labels,
            args.custom_test_ratio,
            args.random_seed,
        )
        train_loader, test_loader = _make_tensor_loaders(
            train_data, test_data, args.batch_size
        )
        return train_loader, train_labels, test_loader, test_labels

    if args.datasets == "uci":
        uci_protocol = getattr(args, "uci_protocol", "official_subject")
        signal_files, channel_indices, expected_channels = (
            _resolve_uci_har_channel_spec(har_channels)
        )

        raw_uci_dir = None
        try:
            raw_uci_dir = find_uci_har_dir(args.data_dir)
        except FileNotFoundError:
            pass

        if uci_protocol == "official_subject":
            if raw_uci_dir is None:
                raise FileNotFoundError(
                    "The official UCI-HAR protocol requires the original "
                    "train/Inertial Signals and test/Inertial Signals directories; "
                    "a combined Feature/feature.npy file cannot recover the "
                    "subject-disjoint split. Use --uci_protocol legacy_resplit "
                    "only for auditing historical results."
                )
            train_data, train_labels = load_uci_har_split(
                raw_uci_dir, "train", signal_files=signal_files
            )
            test_data, test_labels = load_uci_har_split(
                raw_uci_dir, "test", signal_files=signal_files
            )
        elif uci_protocol == "legacy_resplit":
            if raw_uci_dir is not None:
                train_data, train_labels = load_uci_har_split(
                    raw_uci_dir, "train", signal_files=signal_files
                )
                test_data, test_labels = load_uci_har_split(
                    raw_uci_dir, "test", signal_files=signal_files
                )
                train_data, train_labels, test_data, test_labels = _resplit_data(
                    train_data,
                    train_labels,
                    test_data,
                    test_labels,
                    args.custom_test_ratio,
                    args.random_seed,
                )
            else:
                data, labels = load_preprocessed_har(
                    args.data_dir,
                    "UCI-HAR",
                    channel_indices=channel_indices,
                )
                train_data, train_labels, test_data, test_labels = _split_array_data(
                    data,
                    labels,
                    args.custom_test_ratio,
                    args.random_seed,
                )
        else:
            raise ValueError(f"Unsupported UCI-HAR protocol: {uci_protocol}")

        if train_data.shape[1] != expected_channels:
            raise ValueError(
                f"Expected UCI-HAR {har_channels} data to contain "
                f"{expected_channels} channels, got {train_data.shape}."
            )

        train_loader, test_loader = _make_tensor_loaders(
            train_data, test_data, args.batch_size
        )

        return train_loader, train_labels, test_loader, test_labels

    if args.datasets == "falltl":
        train_data, train_labels, test_data, test_labels = load_falltl_data(
            args.data_dir,
            test_ratio=args.custom_test_ratio,
            random_seed=args.random_seed,
            window_size=args.window_size,
            stride=args.window_stride,
            max_windows_per_file=args.max_windows_per_file,
        )

        train_loader, test_loader = _make_tensor_loaders(
            train_data, test_data, args.batch_size
        )

        return train_loader, train_labels, test_loader, test_labels

    if args.datasets == "feng":
        train_data, train_labels, test_data, test_labels = load_feng_data(
            args.data_dir,
            test_ratio=args.custom_test_ratio,
            random_seed=args.random_seed,
            window_size=args.window_size,
            stride=args.window_stride,
            max_windows_per_file=args.max_windows_per_file,
        )

        train_loader, test_loader = _make_tensor_loaders(
            train_data, test_data, args.batch_size
        )

        return train_loader, train_labels, test_loader, test_labels

    data_dir = f"{args.data_dir}/{str(args.datasets).upper()}"

    train_data, train_labels = load_classification(
        dataset,
        split="train",
        extract_path=data_dir,
        load_equal_length=(args.aeon or (args.datasets == "uea")),
        load_no_missing=(args.aeon or (args.datasets == "uea")),
    )
    test_data, test_labels = load_classification(
        dataset,
        split="test",
        extract_path=data_dir,
        load_equal_length=(args.aeon or (args.datasets == "uea")),
        load_no_missing=(args.aeon or (args.datasets == "uea")),
    )

    if dataset == "InsectWingbeat":
        # Downsample before recombining so the final split still follows 60/20/20.
        train_data, train_labels = sample_equal_classes(
            train_data, train_labels, num_samples=1000
        )
        test_data, test_labels = sample_equal_classes(
            test_data, test_labels, num_samples=1000
        )

    train_data, train_labels, test_data, test_labels = _resplit_data(
        train_data,
        train_labels,
        test_data,
        test_labels,
        args.custom_test_ratio,
        args.random_seed,
    )

    # Preprocessing
    if args.datasets == "ucr" and not args.aeon:
        # Padding if time series are of different length
        if isinstance(train_data, list):
            to_length = max(
                np.unique([sample.shape[1] for sample in train_data + test_data])
            )
            train_data = pad_samples(train_data, to_length=to_length)
            test_data = pad_samples(test_data, to_length=to_length)

        # Linear interpolation for missing values
        if np.isnan(train_data).any():
            train_data = linear_interpolation(train_data)

        if np.isnan(test_data).any():
            test_data = linear_interpolation(test_data)

        # Standard normalization
        if (np.abs(train_data.mean()) > 0.01) or (np.abs(train_data.std() - 1) > 0.01):
            mean = np.nanmean(train_data)
            std = np.nanstd(train_data)
            train_data = (train_data - mean) / std
            test_data = (test_data - mean) / std

    train_loader = DataLoader(
        TensorDataset(torch.Tensor(train_data).type(torch.float)),
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.Tensor(test_data).type(torch.float)),
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=False,
    )

    return train_loader, train_labels, test_loader, test_labels
