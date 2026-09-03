#!/usr/bin/env python3
"""Regenerate a test feature cache and verify one archived TimeMosaic checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

import numpy as np
import torch


METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_auroc",
    "macro_auprc",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("adftd", "tdbrain", "apava", "shimmer10", "pads11"),
    )
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--visual-batch-size", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--cache-namespace",
        help="Optional fresh-cache namespace shared by one verification batch.",
    )
    parser.add_argument(
        "--reuse-verification-cache",
        action="store_true",
        help="Reuse a cache created inside the same unique verification namespace.",
    )
    arguments = parser.parse_args()

    project_root = arguments.project_root.expanduser().resolve()
    asset_root = arguments.asset_root.expanduser().resolve()
    runtime_root = arguments.runtime_root.expanduser().resolve()
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    import experiment_common as common

    common.EEG_ROOT = runtime_root / "data" / "eeg"
    common.WEARABLE_ROOT = runtime_root / "data" / "wearable"
    common.VISION_PATH = (
        asset_root
        / "pretrained"
        / "CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    common.MANTIS_PATH = asset_root / "pretrained" / "Mantis-8M"

    from selector_host import SelectorComparisonModel
    from src import patch_mindts
    from src.medformer_graph import TemporalGranularityGraphBank
    from src.neurosigvit import get_neurosigvit

    run_dir = (
        runtime_root
        / "reference_results"
        / arguments.dataset
        / "timemosaic"
        / f"seed_{arguments.seed}"
    )
    expected = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    training = json.loads(
        (run_dir / "training_config.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        run_dir / "artifacts" / "patch_mindts_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    bundle, observed_data_manifest = common.load_data(arguments.dataset, smoke=False)
    if observed_data_manifest != expected["data"]:
        raise SystemExit("loaded data manifest does not match the archived run")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to regenerate CLIP and Mantis features")
    device = torch.device("cuda:0")
    vision = get_neurosigvit(
        model_name=str(common.VISION_PATH),
        model_layer=14,
        aggregation="mean",
        stride=None,
        patch_size=None,
        image_mode="med_activity_graph",
        med_activity_patch_lengths=(4, 8, 16),
        med_activity_adaptive_granularity=True,
        med_activity_granularity_bank=((4,), (8,), (16,)),
    ).to(device)
    if not isinstance(vision.med_activity_granularity_bank, TemporalGranularityGraphBank):
        raise SystemExit("unexpected Activity Graph implementation")
    from mantis.architecture import Mantis8M

    mantis = Mantis8M(device=device).from_pretrained(str(common.MANTIS_PATH)).to(device)
    vision.requires_grad_(False).eval()
    mantis.requires_grad_(False).eval()

    feature_sources = {}
    for folder in (project_root / "src", project_root / "data_loading"):
        for path in sorted(folder.rglob("*.py")):
            feature_sources[str(path.relative_to(project_root))] = common.digest_file(path)
    signature = common.canonical_hash(
        {
            "schema": "selector_common_features_v1",
            "source": feature_sources,
            "data": observed_data_manifest,
            "weights": common.weight_manifest(),
            "window": 64,
            "stride": 64,
            "bank": [[4], [8], [16]],
            "vision_layer": 14,
            "aggregation": "mean",
            "dtype": "float16",
            "smoke": False,
        }
    )
    namespace = arguments.cache_namespace or f"single_{uuid.uuid4().hex}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", namespace):
        raise SystemExit("cache namespace may contain only letters, numbers, . _ and -")
    cache = (
        runtime_root
        / "verification_feature_cache"
        / namespace
        / arguments.dataset
        / signature[:20]
    )
    cache.mkdir(parents=True, exist_ok=arguments.reuse_verification_cache)
    test_features = patch_mindts._get_patch_feature_split(
        "test",
        bundle.test_loader,
        bundle.test_labels,
        vision,
        mantis,
        device,
        64,
        64,
        arguments.visual_batch_size,
        cache,
        signature,
        3,
    )

    model = SelectorComparisonModel(
        **checkpoint["model_constructor_configuration"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    indices = np.arange(len(bundle.test_labels), dtype=np.int64)
    loader = patch_mindts._build_patch_loader(
        test_features,
        bundle.test_labels,
        indices,
        batch_size=int(training["batch_size"]),
        shuffle=False,
    )
    subject_ids = np.asarray(bundle.test_loader.dataset.sample_subject_ids)
    observed, _ = patch_mindts._evaluate_patch_metrics_only(
        model,
        loader,
        checkpoint["classes"],
        device,
        f"Verify {arguments.dataset} seed {arguments.seed}",
        sample_subject_ids=subject_ids,
    )
    deltas = {
        name: float(observed[name]) - float(expected["test_metrics"][name])
        for name in METRICS
    }
    passed = all(math.isfinite(value) and abs(value) <= arguments.tolerance for value in deltas.values())
    result = {
        "status": "verified" if passed else "mismatch",
        "dataset": arguments.dataset,
        "seed": arguments.seed,
        "tolerance": arguments.tolerance,
        "expected": {name: expected["test_metrics"][name] for name in METRICS},
        "observed": {name: observed[name] for name in METRICS},
        "delta": deltas,
        "cache": str(cache / "patch_test.npz"),
        "cache_sha256": common.digest_file(cache / "patch_test.npz"),
        "current_cache_signature": signature,
        "archived_cache_signature": expected["cache_signature"],
    }
    output = runtime_root / "verification" / f"{arguments.dataset}_seed_{arguments.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
