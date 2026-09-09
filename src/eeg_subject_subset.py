"""Explicit ADFTD subject quotas; keep every window of each selected subject."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


VERSION = "adftd_subject_selection_full_windows_v2"
SPLITS = ("train", "vali", "test")
QUOTAS_ENV = "NEUROSIGVIA_ADFTD_SUBJECT_QUOTAS_JSON"
SEED_ENV = "NEUROSIGVIA_ADFTD_SUBJECT_SELECTION_SEED"
MANIFEST_ENV = "NEUROSIGVIA_ADFTD_SUBJECT_SELECTION_MANIFEST"
MANIFEST_SHA_ENV = "NEUROSIGVIA_ADFTD_SUBJECT_SELECTION_MANIFEST_SHA256"


def canonical_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def integer(value, name):
    result = int(value)
    if isinstance(value, bool) or str(result) != str(value):
        raise ValueError(f"{name} must be an integer")
    return result


def explicit_subject_ids(inventory, selected_ids, expected_window_counts=None):
    """Validate a frozen external cohort, preserving full windows and split order.

    This function never uses an RNG. The caller verifies the external manifest's
    own signature before passing its IDs; source membership, classes, and window
    counts are independently verified here against the strict local inventory.
    """
    if set(selected_ids) != set(SPLITS):
        raise ValueError("Explicit subject IDs must specify train, vali and test")
    records = {record.subject_id: record for record in inventory.records}
    result, assigned = {}, set()
    for split in SPLITS:
        ids = [integer(sid, f"{split} subject ID") for sid in selected_ids[split]]
        if len(ids) != len(set(ids)) or assigned.intersection(ids):
            raise ValueError("Repeated selected subject or cross-split subject overlap")
        if not set(ids).issubset(inventory.split_ids[split]):
            raise ValueError(f"Explicit {split} subject is absent from its original source split")
        if {records[sid].label for sid in ids} != {0, 1, 2}:
            raise ValueError(f"Explicit {split} selection must include HC, FTD and AD")
        result[split] = tuple(sid for sid in inventory.split_ids[split] if sid in set(ids))
        assigned.update(ids)
        if expected_window_counts is not None:
            count = sum(records[sid].window_count for sid in result[split])
            if count != integer(expected_window_counts[split], f"{split} expected window count"):
                raise ValueError(f"Explicit {split} cohort windows differ from the external manifest")
    return result


def subject_subset_configuration(explicit=None):
    """No quota means full data. An explicit disabled config ignores ambient settings."""
    if explicit is not None and explicit.get("enabled") is False:
        return None
    if explicit is None:
        old_fraction = os.environ.get("NEUROSIGVIA_ADFTD_SUBSET_FRACTION")
        if old_fraction is not None and float(old_fraction) != 1:
            raise ValueError("Remove obsolete ADFTD window-subset fraction; this batch selects subjects")
        manifest_path = os.environ.get(MANIFEST_ENV)
        if manifest_path:
            if os.environ.get(QUOTAS_ENV):
                raise ValueError("Choose the fixed selection manifest; remove obsolete random subject quotas")
            raw_bytes = Path(manifest_path).read_bytes()
            digest = hashlib.sha256(raw_bytes).hexdigest()
            expected = os.environ.get(MANIFEST_SHA_ENV)
            if expected and expected != digest:
                raise ValueError("External subject selection manifest file SHA-256 differs")
            explicit = {
                "selection_manifest": json.loads(raw_bytes),
                "selection_manifest_file_sha256": digest,
            }
        else:
            raw = os.environ.get(QUOTAS_ENV)
            if not raw:
                return None
            explicit = {"quotas": json.loads(raw), "selection_seed": os.environ.get(SEED_ENV, "42")}
    if "selection_manifest" in explicit:
        if set(explicit) != {"selection_manifest", "selection_manifest_file_sha256"}:
            raise ValueError("Unexpected fixed subject selection configuration fields")
        fixed = explicit["selection_manifest"]
        if fixed.get("schema_version") != 1 or fixed.get("dataset_name") != "ADFTD" or fixed.get("sampling_unit") != "subject":
            raise ValueError("Unsupported external subject selection manifest schema")
        if fixed.get("window_policy") != "all_windows_from_selected_subjects_no_window_subsampling" or fixed.get("eeg_train_fraction") != 1.0:
            raise ValueError("Subject selection must retain every original window")
        if set(fixed["splits"]) != set(SPLITS):
            raise ValueError("External selection must include train, vali and test")
        ordered = [[split, integer(sid, "fixed subject ID")] for split in SPLITS for sid in fixed["splits"][split]["subject_ids"]]
        digest = hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode("utf-8")).hexdigest()
        if digest != fixed["ordered_split_subject_sha256"]:
            raise ValueError("External ordered split-subject selection hash differs")
        return {
            "selection_manifest": fixed,
            "selection_manifest_file_sha256": str(explicit["selection_manifest_file_sha256"]),
        }
    if set(explicit) != {"quotas", "selection_seed"}:
        raise ValueError("Subject subset needs explicit quotas and selection_seed")
    quotas = explicit["quotas"]
    if set(quotas) != set(SPLITS):
        raise ValueError("Subject quotas must specify train, vali and test")
    normalized = {}
    for split in SPLITS:
        if set(map(str, quotas[split])) != {"0", "1", "2"}:
            raise ValueError(f"{split} must specify quotas for HC=0, FTD=1, AD=2")
        normalized[split] = {str(label): integer(value, f"{split} class {label} quota") for label, value in quotas[split].items()}
        if any(value < 1 for value in normalized[split].values()):
            raise ValueError("Every split must retain at least one subject from each class")
    seed = integer(explicit["selection_seed"], "Subject selection seed")
    if not 0 <= seed < 2**32:
        raise ValueError("Subject selection seed must be in [0, 2**32)")
    return {"quotas": normalized, "selection_seed": seed}


def verify_fixed_manifest(inventory, config):
    """Verify the supplied feature files and local split, then use exact IDs."""
    fixed = config["selection_manifest"]
    if fixed["source_protocol"] != inventory.protocol or fixed["source_subject_count"] != len(inventory.records):
        raise ValueError("External selection source protocol/cohort differs")
    if fixed["source_label_file"]["sha256"] != inventory.label_sha256:
        raise ValueError("External selection source label SHA differs")
    if (inventory.data_root / "Label/label.npy").stat().st_size != fixed["source_label_file"]["byte_size"]:
        raise ValueError("External selection source label byte size differs")
    if fixed["original_split_subject_ids"] != {s:list(ids) for s,ids in inventory.split_ids.items()}:
        raise ValueError("External original subject split differs from this server")
    records = {r.subject_id:r for r in inventory.records}
    if fixed["source_subject_label_mapping"] != {str(sid):r.label for sid,r in records.items()}:
        raise ValueError("External subject-label mapping differs from this server")
    selected = explicit_subject_ids(
        inventory, {s:v["subject_ids"] for s,v in fixed["splits"].items()},
        {s:v["window_count"] for s,v in fixed["splits"].items()},
    )
    total_bytes = 0
    for split in SPLITS:
        entry = fixed["splits"][split]
        if list(selected[split]) != entry["subject_ids"]:
            raise ValueError("External selected subject order differs from original split order")
        if entry["subject_count"] != len(selected[split]):
            raise ValueError("External selected subject count differs")
        if entry["class_subject_counts"] != {str(label):sum(records[sid].label == label for sid in selected[split]) for label in (0,1,2)}:
            raise ValueError("External selected class counts differ")
        if [r["subject_id"] for r in entry["subjects"]] != list(selected[split]):
            raise ValueError("External subject feature records differ from selected IDs")
        for row in entry["subjects"]:
            record = records[row["subject_id"]]
            actual_path = record.feature_path
            expected_path = (inventory.data_root / row["feature_relative_path"]).resolve()
            if expected_path != actual_path.resolve():
                raise ValueError("External feature relative path differs from selected source")
            if row["label_id"] != record.label or row["window_count"] != record.window_count:
                raise ValueError("External selected subject label/window count differs")
            if row["feature_shape"] != [record.window_count, 256, 19] or row["source_dtype"] != "float64":
                raise ValueError("External feature shape/dtype differs")
            if row["window_policy"] != "all_source_windows" or row["source_window_indices"] != {"convention":"zero_based_half_open_range","start":0,"step":1,"stop":record.window_count}:
                raise ValueError("External manifest requests partial subject windows")
            size = actual_path.stat().st_size
            if size != row["feature_byte_size"]:
                raise ValueError("External selected feature byte size differs")
            digest = hashlib.sha256()
            with actual_path.open("rb") as handle:
                for chunk in iter(lambda:handle.read(4*1024*1024),b""):
                    digest.update(chunk)
            if digest.hexdigest() != row["feature_sha256"]:
                raise ValueError(f"Selected subject {record.subject_id} feature SHA differs")
            total_bytes += size
    if total_bytes != fixed["selected_feature_total_byte_size"] or sum(map(len,selected.values())) != fixed["selected_subject_count"]:
        raise ValueError("External selected subject/byte totals differ")
    return selected


def build_subject_subset(inventory, explicit=None):
    if inventory.dataset_name != "ADFTD":
        return None, None
    config = subject_subset_configuration(explicit)
    if config is None:
        return None, None
    records = {record.subject_id: record for record in inventory.records}
    fixed = config.get("selection_manifest")
    fixed_selected = verify_fixed_manifest(inventory, config) if fixed is not None else None
    manifest = {
        "version": VERSION,
        "dataset": "ADFTD",
        "selection_unit": "subject",
        "window_policy": "all_original_windows_of_selected_subjects",
        "training_seed_affects_selection": False,
        "config": config,
        "source_label_sha256": inventory.label_sha256,
        "source_content_sha256": inventory.content_sha256,
        "source_split_protocol": inventory.protocol,
        "selection_method": "external_fixed_subject_ids" if fixed is not None else "stratified_subject_quota_pcg64",
        "selection_seed_role": "provenance_only_no_resampling" if fixed is not None else "random_subject_sampling",
        "selection_seed": fixed["subset_seed"] if fixed is not None else config["selection_seed"],
        "ordered_split_subject_sha256": fixed["ordered_split_subject_sha256"] if fixed is not None else None,
        "external_selection_manifest_file_sha256": config.get("selection_manifest_file_sha256"),
        "external_selected_feature_hashes_verified": fixed is not None,
        "splits": {},
    }
    selected = {}
    for split_index, split in enumerate(SPLITS):
        original_ids = inventory.split_ids[split]
        chosen_set = set(fixed_selected[split]) if fixed_selected is not None else set()
        if fixed_selected is None:
            for label in (0, 1, 2):
                available = sorted(sid for sid in original_ids if records[sid].label == label)
                quota = config["quotas"][split][str(label)]
                if quota > len(available):
                    raise ValueError(f"{split} class {label} quota exceeds its original subject count")
                rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([
                    config["selection_seed"], split_index, label,
                ])))
                chosen_set.update(int(sid) for sid in rng.choice(available, size=quota, replace=False))
        selected[split] = tuple(sid for sid in original_ids if sid in chosen_set)
        source_windows = sum(records[sid].window_count for sid in original_ids)
        selected_windows = sum(records[sid].window_count for sid in selected[split])
        classes = {}
        for label in (0, 1, 2):
            source_class = [sid for sid in original_ids if records[sid].label == label]
            selected_class = [sid for sid in selected[split] if records[sid].label == label]
            source_class_windows = sum(records[sid].window_count for sid in source_class)
            selected_class_windows = sum(records[sid].window_count for sid in selected_class)
            classes[str(label)] = {
                "source_subject_count": len(source_class),
                "selected_subject_count": len(selected_class),
                "selected_subject_ids": selected_class,
                "source_window_count": source_class_windows,
                "selected_window_count": selected_class_windows,
                "subject_selection_fraction": len(selected_class) / len(source_class),
                "window_selection_fraction": selected_class_windows / source_class_windows,
                "source_subject_share": len(source_class) / len(original_ids),
                "selected_subject_share": len(selected_class) / len(selected[split]),
                "source_window_share": source_class_windows / source_windows,
                "selected_window_share": selected_class_windows / selected_windows,
            }
        manifest["splits"][split] = {
            "source_subject_count": len(original_ids),
            "selected_subject_count": len(selected[split]),
            "selected_subject_ids": list(selected[split]),
            "source_window_count": source_windows,
            "selected_window_count": selected_windows,
            "subject_selection_fraction": len(selected[split]) / len(original_ids),
            "window_selection_fraction": selected_windows / source_windows,
            "classes": classes,
            "subjects": [{
                "subject_id": sid, "label": records[sid].label,
                "selected": sid in chosen_set,
                "source_window_count": records[sid].window_count,
                "selected_window_count": records[sid].window_count if sid in chosen_set else 0,
            } for sid in original_ids],
        }
    manifest["totals"] = {
        name: sum(values[name] for values in manifest["splits"].values())
        for name in ("source_subject_count", "selected_subject_count", "source_window_count", "selected_window_count")
    }
    totals = manifest["totals"]
    totals["subject_selection_fraction"] = totals["selected_subject_count"] / totals["source_subject_count"]
    totals["window_selection_fraction"] = totals["selected_window_count"] / totals["source_window_count"]
    for values in manifest["splits"].values():
        values["source_split_subject_share"] = values["source_subject_count"] / totals["source_subject_count"]
        values["selected_split_subject_share"] = values["selected_subject_count"] / totals["selected_subject_count"]
        values["source_split_window_share"] = values["source_window_count"] / totals["source_window_count"]
        values["selected_split_window_share"] = values["selected_window_count"] / totals["selected_window_count"]
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest, selected


def validate_subject_subset_manifest(manifest):
    value = dict(manifest)
    expected = value.pop("manifest_sha256", None)
    if not expected or canonical_sha256(value) != expected:
        raise ValueError("ADFTD subject subset manifest SHA-256 mismatch")
    if (value.get("version"), value.get("dataset"), value.get("selection_unit"), value.get("window_policy")) != (
        VERSION, "ADFTD", "subject", "all_original_windows_of_selected_subjects"
    ):
        raise ValueError("Unsupported ADFTD subject subset manifest")
    return subject_subset_configuration(value["config"])


def subject_subset_metadata(manifest):
    """Compact explicit identity for args, cache signature and run protocols."""
    if manifest is None:
        return None
    return {
        "selection_unit": "subject",
        "window_policy": manifest["window_policy"],
        "subject_subset_manifest_sha256": manifest["manifest_sha256"],
        "external_selection_manifest_file_sha256": manifest["external_selection_manifest_file_sha256"],
        "ordered_split_subject_sha256": manifest["ordered_split_subject_sha256"],
        "selected_subject_ids": {s:v["selected_subject_ids"] for s,v in manifest["splits"].items()},
        "selected_window_counts": {s:v["selected_window_count"] for s,v in manifest["splits"].items()},
        "subject_selection_fraction": manifest["totals"]["subject_selection_fraction"],
        "window_selection_fraction": manifest["totals"]["window_selection_fraction"],
    }
