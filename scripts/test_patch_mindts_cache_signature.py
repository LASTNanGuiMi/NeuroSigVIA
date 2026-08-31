#!/usr/bin/env python3
import ast
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PATH = REPO_ROOT / "main.py"


def load_signature_helper():
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    names = {
        "_sampled_file_digest",
        "_full_file_digest",
        "_checkpoint_identity",
        "build_feature_cache_signature",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert {node.name for node in selected} == names

    import hashlib
    from functools import lru_cache

    namespace = {
        "hashlib": hashlib,
        "json": json,
        "lru_cache": lru_cache,
        "Path": Path,
        "anonymize_runtime_value": (
            lambda value: Path(value).name if value else value
        ),
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(MAIN_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["build_feature_cache_signature"]


def make_args(vit_path, mantis_path, **overrides):
    values = {
        "datasets": "eeg",
        "dataset_names": ["TDBRAIN"],
        "aaai27_label_mode": "original",
        "eeg_protocol": "medformer_code_exact",
        "eeg_normalization": "per_window_per_channel_standard_scaler_ddof0",
        "falltl_protocol": "comparison_binary",
        "har_channels": "all",
        "uci_protocol": "official_subject",
        "random_seed": 42,
        "val_ratio": 0.25,
        "custom_test_ratio": 0.2,
        "falltl_target_length": 2048,
        "window_size": 200,
        "window_stride": 100,
        "max_windows_per_file": None,
        "image_mode": "med_activity_graph",
        "med_activity_patch_lengths": (2, 4, 8),
        "med_activity_channel_mix": 0.35,
        "med_activity_router_temperature": 0.2,
        "med_activity_router_mix": 0.5,
        "med_activity_adaptive_granularity": True,
        "med_activity_granularity_bank": (
            (1, 2, 4),
            (2, 4, 8),
            (4, 8, 16),
        ),
        "aggregation": "mean",
        "stride": None,
        "vit_1_name": str(vit_path),
        "vit_1_layer": 14,
        "vit_2_name": None,
        "vit_2_layer": None,
        "mantis": True,
        "mantis_name": str(mantis_path),
        "moment": None,
        "modal_interaction": "patch_mindts",
        "outer_patch_size": 64,
        "outer_patch_stride": 64,
        "patch_granularity_router_mode": "adaptive_v4",
        "patch_checkpoint_metric": "auto",
        # These values configure trainable post-cache modules and therefore
        # must not participate in the frozen-feature signature.
        "patch_alignment_dim": 256,
        "patch_alignment_temperature": 0.1,
        "patch_alignment_weight": 0.1,
        "fusion_dim": 512,
        "fusion_heads": 4,
        "visual_encode_batch_size": 16,
        "classifier_type": "mlp",
        "mlp_hidden_dim": 512,
        "mlp_num_layers": 2,
        "mlp_dropout": 0.1,
        "mlp_lr": 1e-4,
        "mlp_weight_decay": 1e-4,
        "mlp_class_weight": "none",
        "mlp_epochs": 20,
        "mlp_early_stop_patience": 0,
        "batch_size": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def build_signature(build, args, identities, split_audit="split-a"):
    return build(
        args,
        "TDBRAIN",
        33,
        None,
        split_audit_sha256=split_audit,
        **identities,
    )


def main():
    identities = {
        "split_input_identity": {
            "train": {"content_sha256": "train-a"},
            "validation": {"content_sha256": "validation-a"},
            "test": {"content_sha256": "test-a"},
        },
        "feature_code_identity": {
            "manifest_sha256": "patch-code-a",
        },
        "runtime_identity": {
            "python": "3.11.0",
            "packages": {"torch": "2.7.1", "open_clip_torch": "2.32.0"},
        },
    }

    with tempfile.TemporaryDirectory(prefix="patch_cache_signature_") as temp_dir:
        root = Path(temp_dir)
        vit_a = root / "vit-a" / "checkpoint.bin"
        vit_b = root / "vit-b" / "checkpoint.bin"
        mantis_a = root / "mantis-a" / "checkpoint.bin"
        mantis_b = root / "mantis-b" / "checkpoint.bin"
        for path in (vit_a, vit_b, mantis_a, mantis_b):
            path.parent.mkdir(parents=True, exist_ok=True)
        vit_a.write_bytes(b"vit checkpoint a")
        vit_b.write_bytes(b"vit checkpoint b with different weights")
        mantis_a.write_bytes(b"mantis checkpoint a")
        mantis_b.write_bytes(b"mantis checkpoint b with different weights")

        build = load_signature_helper()
        base_args = make_args(vit_a, mantis_a)
        base = build_signature(build, base_args, identities)
        assert build_signature(build, base_args, identities) == base

        configuration = json.loads(base)
        assert configuration["schema"] == 9
        assert configuration["feature_layout"] == (
            "patch_mindts_structured_tokens_v1"
        )
        assert configuration["outer_patch_size"] == 64
        assert configuration["outer_patch_stride"] == 64
        assert configuration["med_activity_granularity_bank"] == [
            [1, 2, 4],
            [2, 4, 8],
            [4, 8, 16],
        ]
        assert configuration["vit_1_identity"]["full_manifest_sha256"]
        assert configuration["mantis_identity"]["full_manifest_sha256"]

        # Every input or preprocessing choice that can change a frozen token
        # must invalidate the cache key.
        invalidating_overrides = (
            {"outer_patch_size": 128},
            {"outer_patch_stride": 32},
            {
                "med_activity_granularity_bank": (
                    (1, 2, 4),
                    (2, 4, 8),
                    (8, 16, 32),
                )
            },
            {"med_activity_granularity_bank": ((4,), (8,), (16,))},
            {
                "med_activity_granularity_bank": (
                    (4, 8, 16),
                    (2, 4, 8),
                    (1, 2, 4),
                )
            },
            {"vit_1_layer": 12},
            {"aggregation": "cls_token"},
            {"vit_1_name": str(vit_b)},
            {"mantis_name": str(mantis_b)},
        )
        for overrides in invalidating_overrides:
            candidate = make_args(vit_a, mantis_a, **overrides)
            assert build_signature(build, candidate, identities) != base, overrides

        changed_identities = (
            {
                **identities,
                "split_input_identity": {
                    "train": {"content_sha256": "train-b"},
                    "test": {"content_sha256": "test-a"},
                },
            },
            {
                **identities,
                "feature_code_identity": {
                    "manifest_sha256": "patch-code-b",
                },
            },
            {
                **identities,
                "runtime_identity": {
                    "python": "3.11.1",
                    "packages": {"torch": "2.7.1"},
                },
            },
        )
        for changed in changed_identities:
            assert build_signature(build, base_args, changed) != base
        assert build_signature(build, base_args, identities, "split-b") != base

        # These parameters are consumed only after raw frozen tokens have been
        # loaded, so changing any of them must reuse the exact same cache.
        non_invalidating_overrides = (
            {"patch_alignment_weight": 0.75},
            {"patch_alignment_temperature": 0.25},
            {"patch_alignment_dim": 128},
            {"fusion_heads": 8},
            {"fusion_dim": 256},
            {"mlp_hidden_dim": 128},
            {"mlp_num_layers": 3},
            {"mlp_dropout": 0.35},
            {"mlp_lr": 5e-4},
            {"mlp_weight_decay": 0.0},
            {"mlp_class_weight": "balanced"},
            {"mlp_epochs": 99},
            {"mlp_early_stop_patience": 5},
            {"batch_size": 2},
            {"visual_encode_batch_size": 1},
            {"patch_granularity_router_mode": "uniform"},
            {"patch_checkpoint_metric": "window_macro_f1"},
        )
        for overrides in non_invalidating_overrides:
            candidate = make_args(vit_a, mantis_a, **overrides)
            assert build_signature(build, candidate, identities) == base, overrides

    print("PATCH MINDTS CACHE SIGNATURE VALIDATION PASSED")


if __name__ == "__main__":
    main()
