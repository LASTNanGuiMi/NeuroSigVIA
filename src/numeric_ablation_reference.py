"""Read-only, identity-checked inputs for a paired numeric-only ablation.

The completed multimodal run supplies its configuration and frozen Mantis
features, never trained model weights.  The real source loaders reconstruct the
sample order, labels and subject IDs.  Their complete input identities must
match the original cache signature before any features are returned.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


DATASET_NAMES = {
    "shimmer10": "Shimmer_10_session10_AFC",
    "pads11": "PADS_11_task08_TouchIndex",
    "apava": "APAVA",
    "tdbrain": "TDBRAIN",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _single(paths, description: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {description}; found {paths}")
    return paths[0]


def _manifest_path(reference_run_path) -> Path:
    path = Path(reference_run_path).expanduser().resolve()
    if path.is_file():
        if path.name != "manifest.tsv":
            raise ValueError("The reference file must be manifest.tsv")
        return path
    if (path / "manifest.tsv").is_file():
        return path / "manifest.tsv"
    if path.parent.name == "results":
        candidate = path.parent.parent / "status" / path.name / "manifest.tsv"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate a reference manifest for {path}")


def _verify_subjects(split_rows, split, labels, subject_ids):
    """Validate IDs and labels against the original subject-level audit CSV."""
    id_column = (
        "legacy_subject_id"
        if "legacy_subject_id" in split_rows[0]
        else "numeric_subject_id"
    )
    selected = [row for row in split_rows if row["split"] == split]
    expected_ids = [int(row[id_column]) for row in selected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError(f"Duplicate {split} subjects in reference audit")
    if set(expected_ids) != set(subject_ids.tolist()):
        raise ValueError(f"{split} subject IDs differ from reference audit")
    for row in selected:
        subject_id = int(row[id_column])
        selected_labels = labels[subject_ids == subject_id]
        if not np.all(selected_labels == int(row["label_id"])):
            raise ValueError(f"{split} subject {subject_id} labels differ")
        if "window_count" in row and len(selected_labels) != int(row["window_count"]):
            raise ValueError(f"{split} subject {subject_id} window count differs")


def load_reference(reference_run_path, seed: int, dataset: str) -> dict:
    """Return validated numeric features and exact reference configuration.

    ``reference_run_path`` accepts ``status/<run>``, its ``manifest.tsv``, or
    ``results/<run>``.  The returned keys are ``model_config``, ``args``,
    ``classes``, ``splits``, ``provenance`` and ``reference_summary``.  Each split
    holds CPU tensors ``mantis_channel_tokens``, ``patch_mask``,
    ``valid_fraction``, and NumPy ``labels`` (class indices) and ``subject_ids``.
    No files are created and no model is constructed or loaded from weights.
    """
    if dataset not in DATASET_NAMES or int(seed) not in (42, 43, 44):
        raise ValueError("Reference must use one of four datasets and seed 42/43/44")
    seed = int(seed)
    dataset_name = DATASET_NAMES[dataset]
    manifest_path = _manifest_path(reference_run_path)
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matched = [
        row for row in rows
        if row["dataset"] == dataset and int(row["seed"]) == seed
        and row["method"] == "NeuroSigVIA"
    ]
    if len(matched) != 1:
        raise ValueError(f"Expected one NeuroSigVIA {dataset} seed {seed} reference")
    row = matched[0]
    status_path = Path(row["status"])
    state = dict(
        line.split("=", 1) for line in status_path.read_text().splitlines()
        if "=" in line
    )
    if state.get("state") != "COMPLETED" or state.get("exit_code") != "0":
        raise ValueError(f"Reference run is not successfully completed: {status_path}")
    result_root = Path(row["result"])
    args_path = _single(result_root.rglob("args.json"), "reference args.json")
    result_dir = args_path.parent
    args = json.loads(args_path.read_text())
    if int(args["random_seed"]) != seed or args["dataset_names"] != [dataset_name]:
        raise ValueError("Reference argument seed/dataset mismatch")
    cache_manifest_path = result_dir / f"{dataset_name}_feature_cache_manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    signature = _canonical(cache_manifest["signature"])
    signature_sha256 = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    if signature_sha256[:16] != cache_manifest["cache_key"]:
        raise ValueError("Reference cache key does not match its signature")
    config = cache_manifest["signature"]
    if config["dataset"] != dataset_name or config["split_seed"] != 42:
        raise ValueError("Expected the exact dataset and fixed split seed 42")
    split_path = result_dir / "splits" / f"{dataset_name}_subject_split.csv"
    split_sha256 = _sha256(split_path)
    if split_sha256 != config["split_audit_sha256"]:
        raise ValueError("Reference subject split audit hash mismatch")
    with split_path.open(encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    if not split_rows:
        raise ValueError("Reference subject split audit is empty")

    summary_path = result_dir / "adaptive_graph" / dataset_name / "adaptive_graph_summary.json"
    reference_summary = json.loads(summary_path.read_text())
    checkpoint_path = summary_path.parent / reference_summary["checkpoint"]
    # weights_only=True disables arbitrary pickle globals.  The state dict is
    # immediately discarded; only constructor and provenance metadata survive.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint.pop("model_state_dict", None)
    model_config = dict(checkpoint["model_constructor_configuration"])
    if checkpoint["architecture"] != reference_summary["architecture"]:
        raise ValueError("Reference checkpoint/summary architecture mismatch")
    if checkpoint["feature_cache"]["signature"] != signature:
        raise ValueError("Reference checkpoint/cache manifest signature mismatch")
    if checkpoint["checkpoint_metric_effective"] != args["patch_checkpoint_metric"]:
        raise ValueError("Reference checkpoint metric mismatch")

    from data_loading.experiment import load_data
    from runners.neurosigvia import _split_input_identity
    from src.adaptive_graph_training import load_adaptive_graph_feature_cache
    from src.patch_mindts import _labels_to_indices, _map_labels, _subject_ids_digest

    bundle, data_manifest = load_data(dataset, smoke=False)
    input_identity = _split_input_identity(
        bundle.train_loader, bundle.train_labels,
        bundle.test_loader, bundle.test_labels,
        bundle.vali_loader, bundle.vali_labels,
    )
    if input_identity != config["split_input_identity"]:
        raise ValueError("Real source input/ordered-label identity differs from reference")
    _, classes, class_to_index = _labels_to_indices(bundle.train_labels)
    classes = np.asarray(classes)
    if not np.array_equal(classes, np.asarray(checkpoint["classes"])):
        raise ValueError("Source training classes differ from reference checkpoint")
    if int(model_config["num_classes"]) != len(classes):
        raise ValueError("Reference model class count mismatch")
    if int(model_config["num_channels"]) != int(config["channels"]):
        raise ValueError("Reference model/cache channel count mismatch")
    cache_dir = Path(row["cache"]) / cache_manifest["relative_cache_subdir"]
    splits = {}
    split_provenance = {}
    for split in ("train", "vali", "test"):
        source = getattr(bundle, f"{split}_loader").dataset
        original_labels = np.asarray(getattr(bundle, f"{split}_labels"))
        subject_ids = np.asarray(source.sample_subject_ids).copy()
        if original_labels.ndim != 1 or subject_ids.shape != original_labels.shape:
            raise ValueError(f"{split} per-sample subject metadata shape mismatch")
        _verify_subjects(split_rows, split, original_labels, subject_ids)
        if split == "vali" and _subject_ids_digest(subject_ids) != checkpoint["validation_subject_ids_sha256"]:
            raise ValueError("Validation subject ORDER differs from reference checkpoint")
        cache_path = cache_dir / f"adaptive_graph_{split}.npz"
        cached = load_adaptive_graph_feature_cache(
            cache_path, original_labels, signature,
            window_size=int(args["outer_patch_size"]),
            stride=int(args["outer_patch_stride"]),
            expected_channels=int(model_config["num_channels"]),
        )
        if cached is None:
            raise FileNotFoundError(f"Required reference cache is missing: {cache_path}")
        if int(cached["mantis_channel_tokens"].shape[-1]) != int(model_config["temporal_dim"]):
            raise ValueError(f"{split} Mantis dimension differs from reference model")
        splits[split] = {
            key: cached[key] for key in
            ("mantis_channel_tokens", "patch_mask", "valid_fraction")
        }
        splits[split]["labels"] = np.asarray(_map_labels(original_labels, class_to_index), dtype=np.int64)
        splits[split]["subject_ids"] = subject_ids
        split_provenance[split] = {
            "cache_path": str(cache_path),
            "cache_sha256": _sha256(cache_path),
            "sample_count": len(original_labels),
            "subject_count": len(np.unique(subject_ids)),
            "ordered_subject_ids_sha256": _subject_ids_digest(subject_ids),
            "mantis_shape": list(cached["mantis_channel_tokens"].shape),
            "source_input_identity": input_identity["validation" if split == "vali" else split],
        }
        del cached
    sets = [set(splits[split]["subject_ids"].tolist()) for split in ("train", "vali", "test")]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Subject overlap across train/validation/test")
    provenance = {
        "reference_run": manifest_path.parent.name,
        "reference_manifest": str(manifest_path),
        "reference_manifest_sha256": _sha256(manifest_path),
        "reference_dataset": dataset,
        "reference_seed": seed,
        "reference_args": str(args_path),
        "reference_args_sha256": _sha256(args_path),
        "reference_summary": str(summary_path),
        "reference_summary_sha256": _sha256(summary_path),
        "reference_checkpoint": str(checkpoint_path),
        "reference_checkpoint_sha256": _sha256(checkpoint_path),
        "reference_architecture": checkpoint["architecture"],
        "reference_weights_loaded_into_model": False,
        "model_configuration_sha256": hashlib.sha256(_canonical(model_config).encode()).hexdigest(),
        "feature_cache_manifest": str(cache_manifest_path),
        "feature_cache_manifest_sha256": _sha256(cache_manifest_path),
        "feature_cache_signature_sha256": signature_sha256,
        "subject_split_path": str(split_path),
        "subject_split_sha256": split_sha256,
        "split_seed": 42,
        "subject_overlap": [0, 0, 0],
        "cache_identity_matches_real_source": True,
        "source_loader_manifest": data_manifest,
        "splits": split_provenance,
    }
    provenance["provenance_sha256"] = hashlib.sha256(_canonical(provenance).encode()).hexdigest()
    return {
        "model_config": model_config,
        "args": args,
        "classes": classes,
        "splits": splits,
        "provenance": provenance,
        "reference_summary": reference_summary,
    }
