#!/usr/bin/env python3
import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


MAIN_PATH = Path(__file__).resolve().parent.parent / "main.py"


def load_signature_helpers():
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    selected = []
    names = {
        "_sampled_file_digest",
        "_full_file_digest",
        "_checkpoint_identity",
        "build_feature_cache_signature",
    }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {}
    code = compile(module, str(MAIN_PATH), "exec")
    import hashlib
    import json
    from functools import lru_cache

    namespace.update(
        {
            "hashlib": hashlib,
            "json": json,
            "lru_cache": lru_cache,
            "Path": Path,
            "anonymize_runtime_value": lambda value: Path(value).name if value else value,
        }
    )
    exec(code, namespace)
    return namespace["build_feature_cache_signature"]


def make_args(model_path, **overrides):
    values = {
        "datasets": "uci",
        "dataset_names": ["UCIHAR"],
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
        "med_activity_adaptive_granularity": False,
        "med_activity_granularity_bank": (
            (1, 2, 4),
            (2, 4, 8),
            (4, 8, 16),
        ),
        "med_activity_granularity_hidden_dim": 64,
        "med_activity_granularity_temperature": 1.0,
        "med_activity_granularity_base_prior": 0.9,
        "med_activity_granularity_balance_weight": 0.01,
        "med_activity_granularity_entropy_weight": 0.001,
        "aggregation": "mean",
        "stride": None,
        "vit_1_name": model_path,
        "vit_1_layer": 14,
        "vit_2_name": None,
        "vit_2_layer": None,
        "mantis": False,
        "mantis_name": None,
        "moment": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def main():
    build = load_signature_helpers()
    with tempfile.TemporaryDirectory(prefix="cache_signature_") as temp_dir:
        model = Path(temp_dir) / "checkpoint.bin"
        model.write_bytes(b"first checkpoint")
        base_args = make_args(str(model))
        base = build(base_args, "UCIHAR", 9, None, "split-a")
        assert hashlib.sha256(base.encode("utf-8")).hexdigest() == (
            "a261fc2985264848f9a0bdce4f8bf479902e70e39497c6c472aff1b8af51efd7"
        )

        assert build(base_args, "UCIHAR", 9, None, "split-a") == base
        assert build(base_args, "UCIHAR", 9, None, "split-b") != base
        assert build(make_args(str(model), val_ratio=0.2), "UCIHAR", 9, None, "split-a") != base
        assert build(make_args(str(model), med_activity_router_mix=0.25), "UCIHAR", 9, None, "split-a") != base
        assert build(make_args(str(model), falltl_target_length=1024), "UCIHAR", 9, None, "split-a") != base
        # Fixed mode preserves the old schema/key and does not depend on an
        # unused candidate bank.
        assert json.loads(base)["schema"] == 5
        assert build(
            make_args(
                str(model),
                med_activity_granularity_bank=((2, 4, 8), (4, 8, 16)),
            ),
            "UCIHAR",
            9,
            None,
            "split-a",
        ) == base

        adaptive_args = make_args(
            str(model),
            med_activity_adaptive_granularity=True,
        )
        try:
            build(adaptive_args, "UCIHAR", 9, None, "split-a")
        except ValueError as exc:
            assert "require split input" in str(exc)
        else:
            raise AssertionError("Adaptive signature accepted missing provenance")
        adaptive_identities = {
            "split_input_identity": {
                "train": {"content_sha256": "data-a"},
                "test": {"content_sha256": "data-b"},
            },
            "feature_code_identity": {"manifest_sha256": "code-a"},
            "runtime_identity": {"python": "3.11.0", "torch": "2.7.1"},
        }
        adaptive = build(
            adaptive_args,
            "UCIHAR",
            9,
            None,
            "split-a",
            **adaptive_identities,
        )
        try:
            build(
                make_args(
                    "laion/remote-model",
                    med_activity_adaptive_granularity=True,
                ),
                "UCIHAR",
                9,
                None,
                "split-a",
                **adaptive_identities,
            )
        except ValueError as exc:
            assert "resolved local checkpoint" in str(exc)
        else:
            raise AssertionError("Adaptive cache accepted an unresolved model id")
        adaptive_config = json.loads(adaptive)
        assert adaptive != base
        assert adaptive_config["schema"] == 7
        assert adaptive_config["feature_layout"] == (
            "granularity_bank_scale_major_flat_v1"
        )
        assert build(
            make_args(
                str(model),
                med_activity_adaptive_granularity=True,
                med_activity_granularity_bank=(
                    (1, 2, 3),
                    (2, 4, 8),
                    (4, 8, 16),
                ),
            ),
            "UCIHAR",
            9,
            None,
            "split-a",
            **adaptive_identities,
        ) != adaptive
        # Selector hyperparameters operate after frozen feature extraction and
        # therefore must not invalidate the candidate-feature cache.
        assert build(
            make_args(
                str(model),
                med_activity_adaptive_granularity=True,
                med_activity_granularity_hidden_dim=128,
                med_activity_granularity_temperature=0.5,
                med_activity_granularity_base_prior=0.8,
                med_activity_granularity_balance_weight=0.0,
                med_activity_granularity_entropy_weight=0.0,
            ),
            "UCIHAR",
            9,
            None,
            "split-a",
            **adaptive_identities,
        ) == adaptive
        for identity_name, changed_identity in (
            (
                "split_input_identity",
                {"train": {"content_sha256": "changed-data"}},
            ),
            (
                "feature_code_identity",
                {"manifest_sha256": "changed-code"},
            ),
            (
                "runtime_identity",
                {"python": "3.11.1", "torch": "2.7.1"},
            ),
        ):
            identities = dict(adaptive_identities)
            identities[identity_name] = changed_identity
            assert build(
                adaptive_args,
                "UCIHAR",
                9,
                None,
                "split-a",
                **identities,
            ) != adaptive
        assert build(
            make_args(str(model), eeg_protocol="medformer_code_exact"),
            "UCIHAR",
            9,
            None,
            "split-a",
        ) == base

        model.write_bytes(b"second checkpoint with different content")
        # The production process caches checkpoint identities because weights do
        # not mutate mid-run. Clear the helper cache to emulate a new process.
        namespace_build = load_signature_helpers()
        changed_model = namespace_build(base_args, "UCIHAR", 9, None, "split-a")
        assert changed_model != base

        # Adaptive mode uses a complete streaming hash, so an equal-size change
        # in the checkpoint middle cannot reuse the cache.
        middle_model = Path(temp_dir) / "middle_checkpoint.bin"
        original_bytes = bytearray(b"a" * (3 * 1024 * 1024))
        middle_model.write_bytes(original_bytes)
        middle_args = make_args(
            str(middle_model), med_activity_adaptive_granularity=True
        )
        middle_before = load_signature_helpers()(
            middle_args,
            "UCIHAR",
            9,
            None,
            "split-a",
            **adaptive_identities,
        )
        original_bytes[len(original_bytes) // 2] = ord("b")
        middle_model.write_bytes(original_bytes)
        middle_after = load_signature_helpers()(
            middle_args,
            "UCIHAR",
            9,
            None,
            "split-a",
            **adaptive_identities,
        )
        assert middle_after != middle_before
    print("FEATURE CACHE SIGNATURE VALIDATION PASSED")


if __name__ == "__main__":
    main()
