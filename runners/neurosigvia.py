import ast
import hashlib
import importlib.metadata
import json
import os
import sys
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Older Transformers releases emit this during import and include the local
# site-packages path in stderr. It is unrelated to this project's behavior and
# can disclose a reviewer's or author's machine path in captured run logs.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.utils\._pytree\._register_pytree_node` is deprecated\..*",
    category=FutureWarning,
    module=r"transformers\.utils\.generic",
)

import numpy as np
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from aeon.datasets.tsc_datasets import multivariate, univariate
from mantis.architecture import Mantis8M
from mantis.trainer import MantisTrainer
from momentfm import MOMENTPipeline
from tqdm import tqdm

from src.analysis import (
    get_intrinsic_dimension,
    get_principal_components,
    measure_alignment,
)
from src.arguments import parse_args
from src.classifier import train_classifier
from src.datautils import (
    WEARABLE_DATASET_NAMES,
    EEG_MEDFORMER_DATASET_NAMES,
    get_falltl_comparison_dataloaders,
    get_wearable_dataloaders,
    get_eeg_medformer_dataloaders,
    get_uci_har_official_dataloaders,
    get_dataloader,
    write_falltl_comparison_split_audit,
    write_wearable_split_audit,
    write_eeg_medformer_split_audit,
    write_uci_har_subject_split_audit,
)
from src.embedding import concat_embeddings, embed
from src.mlp_classifier import train_mlp_classifier
from src.neurosigvia import get_neurosigvia
from src.patch_mindts import (
    PATCH_MINDTS_ARCHITECTURE,
    train_patch_mindts_classifier,
)
from src.adaptive_graph_training import (
    NEUROSIGVIA_ARCHITECTURE,
    NEUROSIGVIA_CACHE_ARCHITECTURE,
    train_neurosigvia_classifier,
)
from src.adaptive_cache_reuse import promote_adaptive_static_caches
from src.adaptive_cache_identity import (
    KNOWN_LEGACY_ADAPTIVE_ARCHITECTURE,
    KNOWN_LEGACY_ADAPTIVE_CACHE_SCHEMA,
    KNOWN_TIMEMOSAIC_MODEL_ARCHITECTURE,
    KNOWN_TIMEMOSAIC_STATIC_CACHE_ARCHITECTURE,
    adaptive_static_extractor_code_identity,
    assert_known_static_extractor_compatibility,
    known_legacy_adaptive_code_identity,
    known_timemosaic_code_identity,
)
from src.privacy import anonymize_runtime_arguments, anonymize_runtime_value
from src.utils import (
    get_patch_size,
    save_activity_lineplot_samples,
    save_activity_graph_samples,
    set_random_seed,
    write_result_table,
    write_split_indices,
)


def model_slug(model_name):
    return os.path.basename(os.path.normpath(model_name)).replace("-", "_")


def embed_loader_splits(
    model,
    model_type,
    channels,
    device,
    train_loader,
    test_loader,
    vali_loader=None,
):
    loaders = [train_loader]
    if vali_loader is not None:
        loaders.append(vali_loader)
    loaders.append(test_loader)
    return tuple(
        embed(model, loader, model_type, channels, device) for loader in loaders
    )


def _sampled_file_digest(path, chunk_size=1024 * 1024):
    """Fingerprint a potentially large checkpoint without exposing its path."""
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        if size > chunk_size:
            handle.seek(max(0, size - chunk_size))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()


def _full_file_digest(path, chunk_size=4 * 1024 * 1024):
    """Stream a complete SHA-256 for cache-critical adaptive checkpoints."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=16)
def _checkpoint_identity(path_value, digest_mode="sampled"):
    """Return a path-private identity for a model file or directory."""
    if digest_mode not in {"sampled", "full"}:
        raise ValueError(f"Unsupported checkpoint digest mode: {digest_mode}")
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    identity = {"basename": path.name, "exists": path.exists()}
    if not path.exists():
        return identity

    if path.is_file():
        files = [path]
    else:
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
        files.sort(key=lambda candidate: candidate.relative_to(path).as_posix())
    manifest = hashlib.sha256()
    total_size = 0
    for file_path in files:
        relative = file_path.name if path.is_file() else file_path.relative_to(path).as_posix()
        size = file_path.stat().st_size
        total_size += size
        manifest.update(relative.encode("utf-8"))
        manifest.update(str(size).encode("ascii"))
        file_digest = (
            _full_file_digest(file_path)
            if digest_mode == "full"
            else _sampled_file_digest(file_path)
        )
        manifest.update(file_digest.encode("ascii"))
    manifest_key = (
        "full_manifest_sha256"
        if digest_mode == "full"
        else "sampled_manifest_sha256"
    )
    identity.update(
        {
            "file_count": len(files),
            "total_size": total_size,
            manifest_key: manifest.hexdigest(),
        }
    )
    return identity


def _update_content_digest(digest, value):
    """Hash tensor-like split content including order, shape, and dtype."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"torch_tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        if tensor.numel() > 0:
            byte_view = tensor.view(torch.uint8).numpy()
            digest.update(memoryview(byte_view).cast("B"))
        return
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        digest.update(b"numpy_array\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        if array.dtype.hasobject:
            for item in array.flat:
                _update_content_digest(digest, item)
        else:
            contiguous = np.ascontiguousarray(array)
            if contiguous.nbytes > 0:
                digest.update(memoryview(contiguous).cast("B"))
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_content_digest(digest, key)
            _update_content_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        digest.update(str(len(value)).encode("ascii"))
        for item in value:
            _update_content_digest(digest, item)
        return
    if isinstance(value, np.generic):
        _update_content_digest(digest, value.item())
        return
    payload = json.dumps(
        {"type": type(value).__name__, "value": value},
        sort_keys=True,
        ensure_ascii=False,
        default=repr,
        separators=(",", ":"),
    )
    digest.update(payload.encode("utf-8"))


def _content_identity(value):
    digest = hashlib.sha256()
    _update_content_digest(digest, value)
    return digest.hexdigest()


def _dataset_content_identity(dataset):
    digest = hashlib.sha256()
    digest.update(type(dataset).__name__.encode("utf-8"))
    digest.update(str(len(dataset)).encode("ascii"))
    tensors = getattr(dataset, "tensors", None)
    if tensors is not None:
        _update_content_digest(digest, tensors)
    else:
        for index in range(len(dataset)):
            _update_content_digest(digest, dataset[index])
    return {
        "dataset_type": type(dataset).__name__,
        "sample_count": len(dataset),
        "content_sha256": digest.hexdigest(),
    }


def _split_input_identity(
    train_loader,
    train_labels,
    test_loader,
    test_labels,
    vali_loader=None,
    vali_labels=None,
):
    splits = {
        "train": {
            "inputs": _dataset_content_identity(train_loader.dataset),
            "labels_sha256": _content_identity(np.asarray(train_labels)),
        },
        "test": {
            "inputs": _dataset_content_identity(test_loader.dataset),
            "labels_sha256": _content_identity(np.asarray(test_labels)),
        },
    }
    if vali_loader is not None:
        splits["validation"] = {
            "inputs": _dataset_content_identity(vali_loader.dataset),
            "labels_sha256": _content_identity(np.asarray(vali_labels)),
        }
    return splits


@lru_cache(maxsize=1)
def _feature_extractor_code_identity():
    """Hash only code that can alter frozen candidate features."""
    project_root = Path(__file__).resolve().parents[1]
    full_source_paths = (
        Path("src/neurosigvia.py"),
        Path("src/utils.py"),
        Path("src/activity_graph.py"),
    )
    manifest = hashlib.sha256()
    file_hashes = {}
    for relative_path in full_source_paths:
        file_path = project_root / relative_path
        file_digest = _full_file_digest(file_path)
        relative = relative_path.as_posix()
        file_hashes[relative] = file_digest
        manifest.update(relative.encode("utf-8"))
        manifest.update(file_digest.encode("ascii"))

    extraction_functions = {
        "_set_trainable",
        "_forward_neurosigvia_batch",
        "_forward_moment_batch",
        "_forward_mantis_batch",
        "forward_feature_batch",
        "_extract_feature_split",
    }
    mlp_path = project_root / "src/mlp_classifier.py"
    tree = ast.parse(mlp_path.read_text(encoding="utf-8"))
    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in extraction_functions
    ]
    found_names = {node.name for node in selected_nodes}
    if found_names != extraction_functions:
        missing = sorted(extraction_functions - found_names)
        raise RuntimeError(
            f"Cannot fingerprint frozen feature extraction functions: {missing}"
        )
    relevant_import_aliases = {
        "torch",
        "nn",
        "tqdm",
        "resize_mantis_input",
        "resize_moment_input",
    }
    import_descriptors = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".")[0]
                if bound_name in relevant_import_aliases:
                    import_descriptors.append(
                        ("import", alias.name, alias.asname)
                    )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if bound_name in relevant_import_aliases:
                    import_descriptors.append(
                        (node.module or "", alias.name, alias.asname)
                    )
    extraction_digest = hashlib.sha256()
    extraction_digest.update(
        json.dumps(sorted(import_descriptors), separators=(",", ":")).encode(
            "utf-8"
        )
    )
    for node in sorted(selected_nodes, key=lambda item: item.name):
        extraction_digest.update(node.name.encode("utf-8"))
        extraction_digest.update(
            ast.dump(node, annotate_fields=True, include_attributes=False).encode(
                "utf-8"
            )
        )
    file_hashes["src/mlp_classifier.py:frozen_feature_functions"] = (
        extraction_digest.hexdigest()
    )
    manifest.update(extraction_digest.hexdigest().encode("ascii"))
    return {
        "manifest_sha256": manifest.hexdigest(),
        "components": file_hashes,
    }


@lru_cache(maxsize=1)
def _patch_feature_extractor_code_identity():
    """Hash the structured patch feature path independently of legacy ATGS."""
    project_root = Path(__file__).resolve().parents[1]
    source_paths = (
        Path("src/neurosigvia.py"),
        Path("src/utils.py"),
        Path("src/activity_graph.py"),
    )
    manifest = hashlib.sha256()
    components = {}
    for relative_path in source_paths:
        digest = _full_file_digest(project_root / relative_path)
        relative = relative_path.as_posix()
        components[relative] = digest
        manifest.update(relative.encode("utf-8"))
        manifest.update(digest.encode("ascii"))

    patch_path = project_root / "src/patch_mindts.py"
    tree = ast.parse(patch_path.read_text(encoding="utf-8"))
    extraction_functions = {
        "make_temporal_patches",
        "_encode_visual_images",
        "_line_images_for_chunk",
        "_extract_line_tokens",
        "_extract_graph_tokens",
        "_extract_mantis_channel_tokens",
        "extract_patch_feature_batch",
        "_validate_patch_feature_bundle",
        "save_patch_feature_cache",
        "load_patch_feature_cache",
        "_extract_patch_feature_split",
        "_get_patch_feature_split",
    }
    extraction_classes = {"TemporalPatchBatch"}
    extraction_constants = {
        "PATCH_FEATURE_CACHE_SCHEMA_VERSION",
        "PATCH_FEATURE_ARCHITECTURE",
        "PATCH_TAIL_POLICY",
        "PATCH_FEATURE_KEYS",
    }
    selected_nodes = []
    found_names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in extraction_functions:
                selected_nodes.append((node.name, node))
                found_names.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name in extraction_classes:
            selected_nodes.append((node.name, node))
            found_names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in extraction_constants:
                    selected_nodes.append((target.id, node))
                    found_names.add(target.id)
    expected_names = extraction_functions | extraction_classes | extraction_constants
    if found_names != expected_names:
        missing = sorted(expected_names - found_names)
        raise RuntimeError(
            f"Cannot fingerprint patch feature extraction components: {missing}"
        )
    extraction_digest = hashlib.sha256()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            extraction_digest.update(
                ast.dump(
                    node,
                    annotate_fields=True,
                    include_attributes=False,
                ).encode("utf-8")
            )
    for name, node in sorted(selected_nodes, key=lambda item: item[0]):
        extraction_digest.update(name.encode("utf-8"))
        extraction_digest.update(
            ast.dump(node, annotate_fields=True, include_attributes=False).encode(
                "utf-8"
            )
        )
    component_name = "src/patch_mindts.py:patch_feature_components"
    component_digest = extraction_digest.hexdigest()
    components[component_name] = component_digest
    manifest.update(component_name.encode("utf-8"))
    manifest.update(component_digest.encode("ascii"))
    return {
        "manifest_sha256": manifest.hexdigest(),
        "components": components,
    }


@lru_cache(maxsize=1)
def _adaptive_graph_feature_extractor_code_identity():
    """Hash only code that materializes cached raw/Line/Mantis tensors."""
    project_root = Path(__file__).resolve().parents[1]
    return adaptive_static_extractor_code_identity(project_root)


def _adaptive_static_encoder_contract(vision_model, mantis_model):
    """Describe the actual frozen encoders used to materialize static tokens."""

    if vision_model is None or mantis_model is None:
        raise ValueError("adaptive static caching requires vision and Mantis models")
    processor = getattr(vision_model, "processor", None)
    processor_transforms = getattr(processor, "transforms", None)
    if processor_transforms is not None:
        processor_transforms = [repr(value) for value in processor_transforms]
    backbone = getattr(vision_model, "vit", None)
    mantis_network = getattr(mantis_model, "network", None)
    return {
        "vision_wrapper_class": (
            f"{type(vision_model).__module__}.{type(vision_model).__qualname__}"
        ),
        "vision_backbone_class": (
            f"{type(backbone).__module__}.{type(backbone).__qualname__}"
            if backbone is not None
            else None
        ),
        "vision_processor_class": (
            f"{type(processor).__module__}.{type(processor).__qualname__}"
            if processor is not None
            else None
        ),
        "vision_processor_transforms": processor_transforms,
        "vision_layer_idx": getattr(vision_model, "layer_idx", None),
        "vision_aggregation": getattr(vision_model, "aggregation", None),
        "vision_image_mode": getattr(vision_model, "image_mode", None),
        "mantis_wrapper_class": (
            f"{type(mantis_model).__module__}.{type(mantis_model).__qualname__}"
        ),
        "mantis_network_class": (
            f"{type(mantis_network).__module__}.{type(mantis_network).__qualname__}"
            if mantis_network is not None
            else None
        ),
    }


@lru_cache(maxsize=1)
def _feature_runtime_identity():
    packages = {}
    for package in (
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "open_clip_torch",
        "mantis-tsfm",
        "momentfm",
        "transformers",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "packages": packages,
    }


def build_feature_cache_signature(
    args,
    dataset,
    channels,
    patch_size,
    split_audit_sha256=None,
    split_input_identity=None,
    feature_code_identity=None,
    runtime_identity=None,
    static_encoder_contract=None,
):
    adaptive_granularity = bool(
        getattr(args, "med_activity_adaptive_granularity", False)
    )
    patch_mindts = getattr(args, "modal_interaction", None) == "patch_mindts"
    adaptive_graph = (
        getattr(args, "modal_interaction", None) == "adaptive_granularity"
    )
    fixed_activity_graph = (
        not adaptive_graph
        and not patch_mindts
        and not adaptive_granularity
        and getattr(args, "image_mode", None)
        in {"activity_graph", "med_activity_graph", "activity_matrix"}
    )
    integrity_hashed_features = adaptive_granularity or adaptive_graph
    if integrity_hashed_features and any(
        identity is None
        for identity in (
            split_input_identity,
            feature_code_identity,
            runtime_identity,
        )
    ):
        raise ValueError(
            "Adaptive feature-cache signatures require split input, feature "
            "code, and runtime identities."
        )
    if integrity_hashed_features:
        checkpoint_paths = {
            "vit_1_name": args.vit_1_name,
            "vit_2_name": args.vit_2_name,
            "mantis_name": args.mantis_name if args.mantis else None,
        }
        unresolved = [
            name
            for name, value in checkpoint_paths.items()
            if value and not Path(value).expanduser().exists()
        ]
        if unresolved:
            raise ValueError(
                "Adaptive feature caching requires resolved local checkpoint "
                "files/directories for complete hashing; unresolved model "
                f"arguments: {unresolved}. Use a local snapshot or disable "
                "--feature_cache_dir."
            )
        if args.moment:
            raise ValueError(
                "Adaptive feature caching with MOMENT requires a resolved "
                "local checkpoint identity, which this CLI does not expose. "
                "Disable --moment or --feature_cache_dir."
            )
    has_fixed_split = args.datasets in {"wearable", "eeg"} or (
        args.datasets == "falltl" and args.falltl_protocol == "comparison_binary"
    ) or (
        args.datasets == "uci" and args.uci_protocol == "official_subject"
    )
    configuration = {
        # Schema 11 scopes the adaptive cache to raw/Line/Mantis tensors.  The
        # fixed Activity Graph schema is also bumped because those caches hold
        # graph-derived embeddings and must not survive renderer replacement.
        "schema": (
            11
            if adaptive_graph
            else 9
            if patch_mindts
            else 7
            if adaptive_granularity
            else 6
            if fixed_activity_graph
            else 5
        ),
        "dataset_group": args.datasets,
        "dataset": dataset,
        "dataset_names": args.dataset_names,
        "channels": channels,
        "label_mode": args.wearable_label_mode,
        "eeg_protocol": args.eeg_protocol,
        "eeg_normalization": args.eeg_normalization,
        "falltl_protocol": args.falltl_protocol,
        "har_channels": args.har_channels,
        "uci_protocol": args.uci_protocol,
        "split_seed": 42 if has_fixed_split else args.random_seed,
        "split_audit_sha256": split_audit_sha256,
        **({"adftd_subject_subset": args.adftd_subject_subset}
           if getattr(args, "adftd_subject_subset", None) is not None else {}),
        "val_ratio": args.val_ratio,
        "custom_test_ratio": args.custom_test_ratio,
        "falltl_target_length": args.falltl_target_length,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "max_windows_per_file": args.max_windows_per_file,
        "image_mode": args.image_mode,
        "med_activity_patch_lengths": (
            None
            if patch_mindts or adaptive_graph
            else args.med_activity_patch_lengths
        ),
        "med_activity_channel_mix": (
            None if adaptive_graph else args.med_activity_channel_mix
        ),
        "med_activity_router_temperature": (
            None
            if patch_mindts or adaptive_graph
            else args.med_activity_router_temperature
        ),
        "med_activity_router_mix": (
            None
            if patch_mindts or adaptive_graph
            else args.med_activity_router_mix
        ),
        "aggregation": args.aggregation,
        "patch_size": patch_size,
        "stride": args.stride,
        "vit_1_name": anonymize_runtime_value(args.vit_1_name),
        "vit_1_identity": _checkpoint_identity(
            args.vit_1_name,
            "full" if integrity_hashed_features else "sampled",
        ),
        "vit_1_layer": args.vit_1_layer,
        "vit_2_name": anonymize_runtime_value(args.vit_2_name),
        "vit_2_identity": _checkpoint_identity(
            args.vit_2_name,
            "full" if integrity_hashed_features else "sampled",
        ),
        "vit_2_layer": args.vit_2_layer,
        "mantis_name": (
            anonymize_runtime_value(args.mantis_name) if args.mantis else None
        ),
        "mantis_identity": (
            _checkpoint_identity(
                args.mantis_name,
                "full" if integrity_hashed_features else "sampled",
            )
            if args.mantis
            else None
        ),
        "moment": args.moment,
    }
    if adaptive_graph:
        configuration.update(
            {
                "feature_layout": "raw_windows_line_mantis_v1",
                "architecture": NEUROSIGVIA_CACHE_ARCHITECTURE,
                "cache_scope": "raw_windows_line_mantis_static_v1",
                "outer_patch_size": args.outer_patch_size,
                "outer_patch_stride": args.outer_patch_stride,
                "tail_policy": "right_zero_pad_then_crop_valid_prefix_v1",
                "line_plot_layout": "stacked_channel_lanes_v1",
                "split_input_identity": split_input_identity,
                "feature_code_identity": feature_code_identity,
                "runtime_identity": runtime_identity,
                "static_encoder_contract": static_encoder_contract,
            }
        )
    if adaptive_graph and static_encoder_contract is None:
        raise ValueError(
            "adaptive static cache signatures require the actual frozen "
            "encoder construction contract"
        )
    elif patch_mindts:
        configuration.update(
            {
                "med_activity_adaptive_granularity": True,
                "med_activity_granularity_bank": getattr(
                    args,
                    "med_activity_granularity_bank",
                ),
                "feature_layout": "patch_mindts_structured_tokens_v1",
                "outer_patch_size": args.outer_patch_size,
                "outer_patch_stride": args.outer_patch_stride,
                "tail_policy": "right_zero_pad_then_crop_valid_prefix_v1",
                "line_plot_layout": "stacked_channel_lanes_v1",
                "mantis_patch_resize": "valid_prefix_linear_to_512_v1",
                "split_input_identity": split_input_identity,
                "feature_code_identity": feature_code_identity,
                "runtime_identity": runtime_identity,
            }
        )
    elif adaptive_granularity:
        configuration.update(
            {
                "med_activity_adaptive_granularity": True,
                "med_activity_granularity_bank": getattr(
                    args,
                    "med_activity_granularity_bank",
                ),
                "feature_layout": "granularity_bank_scale_major_flat_v1",
                "split_input_identity": split_input_identity,
                "feature_code_identity": feature_code_identity,
                "runtime_identity": runtime_identity,
            }
        )
    elif fixed_activity_graph:
        configuration.update(
            {
                "activity_graph_feature_code_identity": (
                    _feature_extractor_code_identity()
                ),
                "activity_graph_cache_policy": (
                    "graph_derived_embeddings_invalidate_on_renderer_code_v1"
                ),
            }
        )
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))


def _known_legacy_adaptive_cache_signature(
    args,
    dataset,
    channels,
    patch_size,
    *,
    family,
    split_audit_sha256,
    split_input_identity,
    runtime_identity,
    static_encoder_contract,
):
    """Reconstruct one exact, audited schema-10 cache signature."""

    if family not in {"neurosigvia_97ade80", "timemosaic_7f38ff7"}:
        raise ValueError(f"unsupported legacy adaptive-cache family: {family}")
    current = json.loads(
        build_feature_cache_signature(
            args,
            dataset,
            channels,
            patch_size,
            split_audit_sha256=split_audit_sha256,
            split_input_identity=split_input_identity,
            feature_code_identity=_adaptive_graph_feature_extractor_code_identity(),
            runtime_identity=runtime_identity,
            static_encoder_contract=static_encoder_contract,
        )
    )
    current.pop("cache_scope", None)
    current.pop("static_encoder_contract", None)
    current["schema"] = KNOWN_LEGACY_ADAPTIVE_CACHE_SCHEMA
    current["split_seed"] = 42
    current["med_activity_channel_mix"] = args.med_activity_channel_mix
    current["activity_graph_selection"] = (
        "raw_region_16_hard_st_4_8_16_before_graph_propagation"
    )

    if family == "neurosigvia_97ade80":
        current.update(
            {
                "architecture": KNOWN_LEGACY_ADAPTIVE_ARCHITECTURE,
                "granularity_gate_temperature": args.granularity_gate_temperature,
                "granularity_balance_weight": args.granularity_balance_weight,
                "granularity_graph_token_grid": args.granularity_graph_token_grid,
                "granularity_gate_checkpoint_identity": _checkpoint_identity(
                    args.granularity_gate_checkpoint, "full"
                ),
                "granularity_freeze_gate": args.granularity_freeze_gate,
                "feature_code_identity": known_legacy_adaptive_code_identity(),
            }
        )
    else:
        current.update(
            {
                "architecture": KNOWN_TIMEMOSAIC_MODEL_ARCHITECTURE,
                "timemosaic_gate_temperature": args.granularity_gate_temperature,
                "timemosaic_selector_balance_weight": (
                    args.granularity_balance_weight
                ),
                "timemosaic_graph_token_grid": args.granularity_graph_token_grid,
                "timemosaic_gate_checkpoint_identity": _checkpoint_identity(
                    args.granularity_gate_checkpoint, "full"
                ),
                "timemosaic_freeze_gate": args.granularity_freeze_gate,
                "feature_code_identity": known_timemosaic_code_identity(),
            }
        )
    return json.dumps(current, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    # TDBRAIN 阅读入口：scripts/NeuroSigVIA.sh -> 本模块 -> src/datautils.py。
    # 当前主配置 mlp + adaptive_granularity 最终调用 train_neurosigvia_classifier。
    args = parse_args()

    set_random_seed(args.random_seed)

    timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"

    available_models = []

    if args.vit_1_name:
        available_models.append(model_slug(args.vit_1_name))
    if args.vit_2_name:
        available_models.append(model_slug(args.vit_2_name))
    if args.mantis:
        available_models.append("mantis")
    if args.moment:
        available_models.append(f"moment_{args.moment}")

    has_vision_modality = bool(args.vit_1_name or args.vit_2_name)
    has_timeseries_modality = bool(args.mantis or args.moment)

    if not has_vision_modality and not has_timeseries_modality:
        raise ValueError(
            "At least one modality must be enabled: use a ViT model for vision, "
            "or enable --mantis/--moment for raw time-series embeddings."
        )

    enabled_modalities = []
    if has_vision_modality:
        enabled_modalities.append(f"vision:{args.image_mode}")
    if has_timeseries_modality:
        ts_models = []
        if args.mantis:
            ts_models.append("mantis")
        if args.moment:
            ts_models.append(f"moment-{args.moment}")
        enabled_modalities.append(f"time_series:{'+'.join(ts_models)}")
    print("Enabled modalities:", ", ".join(enabled_modalities))

    available_models = "_".join(available_models)

    result_dir = f"{args.result_dir}/{timestamp}_{args.datasets}_{available_models}_{args.classifier_type}"
    os.makedirs(result_dir, exist_ok=False)

    patch_mindts_enabled = args.modal_interaction == "patch_mindts"
    adaptive_graph_enabled = (
        args.modal_interaction == "adaptive_granularity"
    )
    patch_router_mode = args.patch_granularity_router_mode
    patch_router_v5 = patch_mindts_enabled and patch_router_mode == "adaptive_v5"
    run_protocol = {
        "schema": (
            7
            if adaptive_graph_enabled
            else 6
            if patch_router_v5
            else 5
            if patch_mindts_enabled
            else 2
            if args.med_activity_adaptive_granularity
            else 1
        ),
        "dataset_group": args.datasets,
        "dataset_names": args.dataset_names,
        "random_seed": args.random_seed,
        "image_mode": args.image_mode,
        "modal_interaction": args.modal_interaction,
        "wearable_label_mode": args.wearable_label_mode,
        "eeg_protocol": args.eeg_protocol,
        "eeg_normalization": args.eeg_normalization,
        "uci_protocol": args.uci_protocol,
        "falltl_protocol": args.falltl_protocol,
        "falltl_target_length": args.falltl_target_length,
        "split_unit": (
            "legacy_subject_file_id"
            if args.datasets == "eeg"
            else "subject"
            if args.datasets in {"wearable", "uci"}
            else "activity_code+trial_no"
            if args.datasets == "falltl"
            and args.falltl_protocol == "comparison_binary"
            else "sample"
        ),
        "metric_unit": (
            "window_and_subject_when_subject_ids_available"
            if patch_mindts_enabled or adaptive_graph_enabled
            else "processed_one_second_window"
            if args.datasets == "eeg"
            else "sample"
        ),
    }
    if adaptive_graph_enabled:
        run_protocol.update(
            {
                "architecture": NEUROSIGVIA_ARCHITECTURE,
                "outer_patch_size": args.outer_patch_size,
                "outer_patch_stride": args.outer_patch_stride,
                "tail_policy": "right_zero_pad_then_crop_valid_prefix_v1",
                "line_plot_layout": "stacked_channel_lanes_v1",
                "activity_graph_count_per_window": 1,
                "activity_graph_generation": (
                    "adaptive_piecewise_mean_waveform_then_yang2022_"
                    "algorithm1_algorithm3_multicolumn"
                ),
                "activity_graph_reference_doi": "10.1109/TII.2022.3142315",
                "activity_graph_channel_propagation": False,
                "activity_graph_reference_canvas_size": (
                    args.activity_graph_canvas_size
                ),
                "activity_graph_output_size": 224,
                "activity_graph_line_width": args.activity_graph_line_width,
                "activity_graph_vertical_margin": (
                    args.activity_graph_vertical_margin
                ),
                "activity_graph_plot_bounds_note": (
                    "paper_underreported_explicit_per_signal_minmax"
                ),
                "granularity_candidates": [4, 8, 16],
                "granularity_region_length": 16,
                "granularity_selection_training": (
                    "hard_straight_through_gumbel_softmax"
                ),
                "granularity_selection_evaluation": "argmax_one_hot",
                "granularity_gate_temperature": args.granularity_gate_temperature,
                "granularity_balance_weight": (
                    args.granularity_balance_weight
                ),
                "granularity_gate_checkpoint": (
                    anonymize_runtime_value(args.granularity_gate_checkpoint)
                    if args.granularity_gate_checkpoint
                    else None
                ),
                "granularity_freeze_gate": args.granularity_freeze_gate,
                "visual_fusion": (
                    "pooled_line_query_activity_graph_spatial_key_value_"
                    "cross_attention"
                ),
                "graph_spatial_token_grid": args.granularity_graph_token_grid,
                "alignment_scope": (
                    "symmetric_within_sample_N_by_N_visual_mantis_no_cross_"
                    "batch_negatives"
                ),
                "alignment_dim": args.patch_alignment_dim,
                "alignment_temperature": args.patch_alignment_temperature,
                "alignment_weight": args.patch_alignment_weight,
                "temporal_visual_fusion": "concat_attn",
                "temporal_visual_fusion_semantics": (
                    "branch_projection_then_two_token_self_attention_then_"
                    "flatten"
                ),
                "temporal_visual_branch_order": [
                    "cross_attention_visual",
                    "mantis_temporal",
                ],
                "patch_pooling": "valid_fraction_weighted_mean_v1",
                "checkpoint_metric": args.patch_checkpoint_metric,
                "historical_checkpoint_compatible": False,
                "previous_concat_mlp_checkpoint_compatible": False,
                "historical_path": "patch_mindts",
            }
        )
    elif patch_mindts_enabled:
        run_protocol.update(
            {
                "med_activity_adaptive_granularity": True,
                "med_activity_patch_lengths": args.med_activity_patch_lengths,
                "med_activity_base_renderer_used": False,
                "med_activity_granularity_bank": (
                    args.med_activity_granularity_bank
                ),
                "feature_layout": "patch_mindts_structured_tokens_v1",
                "architecture": PATCH_MINDTS_ARCHITECTURE,
                "outer_patch_size": args.outer_patch_size,
                "outer_patch_stride": args.outer_patch_stride,
                "tail_policy": "right_zero_pad_then_crop_valid_prefix_v1",
                "line_plot_layout": "stacked_channel_lanes_v1",
                "granularity_selection": (
                    "line_q_graph_kv_relation_sparse_topk_v5"
                    if patch_router_mode == "adaptive_v5"
                    else "bounded_temperature_dot_scale_specific_interaction_mlp_v41"
                    if patch_router_mode == "adaptive_v41"
                    else "fixed_uniform"
                    if patch_router_mode == "uniform"
                    else "bounded_rbf_dataset_sample_patch_hierarchical_v4"
                ),
                "granularity_router_mode": patch_router_mode,
                "checkpoint_metric": args.patch_checkpoint_metric,
                "early_stopping": {
                    "strategy": args.mlp_early_stop_strategy,
                    "patience": args.mlp_early_stop_patience,
                    "min_epochs": args.mlp_early_stop_min_epochs,
                    "ema_decay": args.mlp_early_stop_ema_decay,
                    "min_delta": args.mlp_early_stop_min_delta,
                    "checkpoint_selection_uses_raw_metric": True,
                },
                "granularity_decision_scope": (
                    "sample_global_and_patch_local_relation_blend_v5"
                    if patch_router_mode == "adaptive_v5"
                    else "dataset_prior_margin_gated_sample_global_patch_local_v41"
                    if patch_router_mode == "adaptive_v41"
                    else "none_fixed_uniform"
                    if patch_router_mode == "uniform"
                    else "dataset_prior_sample_global_query_patch_local_query_v4"
                ),
                "granularity_expert_pattern": "parallel_multiscale_experts",
                "granularity_expert_encoding": (
                    "single_scale_neutral_rgb_v3"
                    if all(
                        len(regime) == 1
                        for regime in args.med_activity_granularity_bank
                    )
                    else "legacy_three_scale_rgb_v2"
                ),
                "granularity_reference_boundary": (
                    "per_window_topk_experts_with_project_specific_line_q_graph_kv_"
                    "scoring_see_SOURCE_NOTES.md"
                ),
                "granularity_weight_layout": "one_shared_distribution_per_patch_BNK",
                "granularity_route_density": (
                    "sparse_topk_train_and_inference"
                    if patch_router_mode == "adaptive_v5"
                    else "fixed_uniform"
                    if patch_router_mode == "uniform"
                    else "dense_softmax_train_and_inference"
                ),
                "granularity_visual_token": (
                    "graph_value_context_only_no_query_residual"
                ),
                "granularity_selector_temperature": (
                    args.med_activity_granularity_temperature
                ),
                "granularity_balance_weight": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_balance_weight
                ),
                "granularity_entropy_weight": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_entropy_weight
                ),
                "granularity_mix_shrinkage_weight": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_mix_shrinkage_weight
                ),
                "granularity_prior_kl_weight": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_prior_kl_weight
                ),
                "granularity_usage_floor": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_usage_floor
                ),
                "granularity_usage_ema_decay": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_usage_ema_decay
                ),
                "granularity_entropy_floor": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_entropy_floor
                ),
                "granularity_entropy_ceiling": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_entropy_ceiling
                ),
                "granularity_local_mix_max": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_local_mix_max
                ),
                "granularity_local_mix_init": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_local_mix_init
                ),
                "granularity_global_mix_max": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_global_mix_max
                ),
                "granularity_global_mix_init": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_global_mix_init
                ),
                "granularity_evidence_half_saturation": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_evidence_half_saturation
                ),
                "granularity_minimum_weight": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_minimum_weight
                ),
                "granularity_score_cap": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_score_cap
                ),
                "granularity_scorer_hidden_dim": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_scorer_hidden_dim
                ),
                "granularity_confidence_half_saturation": (
                    None
                    if patch_router_v5
                    else args.med_activity_granularity_confidence_half_saturation
                ),
                "v5_top_k": (
                    args.patch_router_top_k if patch_router_v5 else None
                ),
                "v5_training_noise_std": (
                    args.patch_router_training_noise_std
                    if patch_router_v5
                    else None
                ),
                "v5_train_inference_route_match": (
                    args.patch_router_training_noise_std == 0.0
                    if patch_router_v5
                    else None
                ),
                "v5_local_weight": (
                    args.patch_router_local_weight if patch_router_v5 else None
                ),
                "v5_relation_hidden_dim": (
                    args.patch_router_relation_hidden_dim
                    if patch_router_v5
                    else None
                ),
                "v5_relation_residual_scale": (
                    args.patch_router_relation_residual_scale
                    if patch_router_v5
                    else None
                ),
                "v5_key_adapter_scale": (
                    args.patch_router_key_adapter_scale
                    if patch_router_v5
                    else None
                ),
                "v5_value_adapter_scale": (
                    args.patch_router_value_adapter_scale
                    if patch_router_v5
                    else None
                ),
                "v5_route_budget_weight": (
                    args.patch_router_route_budget_weight
                    if patch_router_v5
                    else None
                ),
                "v5_route_budget_definition": (
                    "mse_sample_equal_post_topk_marginal_to_uniform"
                    if patch_router_v5
                    else None
                ),
                "v5_load_balance_weight": (
                    args.patch_router_load_balance_weight
                    if patch_router_v5
                    else None
                ),
                "v5_load_balance_definition": (
                    "cv_squared_sample_equal_pre_topk_clean_softmax_marginal"
                    if patch_router_v5
                    else None
                ),
                "granularity_regularization_policy": (
                    "post_topk_route_budget_plus_pre_topk_load_balance_v5"
                    if patch_router_mode == "adaptive_v5"
                    else "configurable_usage_entropy_mix_prior_v41"
                    if patch_router_mode == "adaptive_v41"
                    else "none_fixed_uniform"
                    if patch_router_mode == "uniform"
                    else "current_marginal_usage_floor_entropy_band_"
                    "mix_shrinkage_prior_kl_v4"
                ),
                "alignment_scope": "within_sample_N_by_N_no_cross_batch_negatives",
                "alignment_router_gradient": "stopped_at_route_weights_v4",
                "alignment_scale_normalization": "divide_by_log_valid_patch_count_v3",
                "alignment_dim": args.patch_alignment_dim,
                "alignment_temperature": args.patch_alignment_temperature,
                "alignment_weight": args.patch_alignment_weight,
                "patch_pooling": "valid_fraction_weighted_mean_v1",
                "patch_atgs_artifact_layout": "patch_atgs/<dataset>/v5",
                "hard_top1_enabled": False,
            }
        )
    elif args.med_activity_adaptive_granularity:
        run_protocol.update(
            {
                "med_activity_adaptive_granularity": True,
                "med_activity_patch_lengths": args.med_activity_patch_lengths,
                "med_activity_granularity_bank": (
                    args.med_activity_granularity_bank
                ),
                "feature_layout": "granularity_bank_scale_major_flat_v1",
                "granularity_selector_hidden_dim": (
                    args.med_activity_granularity_hidden_dim
                ),
                "granularity_selector_temperature": (
                    args.med_activity_granularity_temperature
                ),
                "granularity_selector_base_prior": (
                    args.med_activity_granularity_base_prior
                ),
                "granularity_selector_balance_weight": (
                    args.med_activity_granularity_balance_weight
                ),
                "granularity_selector_entropy_weight": (
                    args.med_activity_granularity_entropy_weight
                ),
                "atgs_artifact_layout": "atgs/<dataset>/v1",
            }
        )
    with open(f"{result_dir}/protocol.json", "w", encoding="utf-8") as handle:
        json.dump(run_protocol, handle, indent=2, sort_keys=True)

    # Save parsed arguments as json dictionary
    args_dict = anonymize_runtime_arguments(args)
    with open(f"{result_dir}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.datasets == "ucr":
        datasets = univariate
    elif args.datasets == "uea":
        datasets = multivariate
    elif args.datasets == "uci":
        datasets = ["UCIHAR"]
    elif args.datasets == "flaap":
        datasets = ["FLAAP"]
    elif args.datasets == "falltl":
        datasets = ["FallTL"]
    elif args.datasets == "feng":
        datasets = ["Feng"]
    elif args.datasets == "wearable":
        datasets = list(WEARABLE_DATASET_NAMES)
    elif args.datasets == "eeg":
        datasets = list(EEG_MEDFORMER_DATASET_NAMES)
    else:
        raise ValueError(
            "Only UCR, UEA, UCI, FLAAP, FallTL, Feng, wearable, and EEG benchmarks "
            "are available."
        )

    if args.dataset_names:
        unavailable = sorted(set(args.dataset_names) - set(datasets))
        if unavailable:
            raise ValueError(
                f"Dataset(s) not found in {args.datasets.upper()}: {unavailable}"
            )
        datasets = args.dataset_names

    for dataset in tqdm(datasets):
        print(dataset)

        fixed_validation_split = False
        vali_loader = None
        vali_labels = None
        audit_path = None
        if args.datasets == "eeg":
            fixed_validation_split = True
            # TDBRAIN 固定 34/8/8 个 legacy 文件编号，窗口总数为 4320/960/960。
            # loader 每批仅返回 (X,)，X=[B,33,256]；对应的一维 labels 由 bundle 单独提供。
            eeg_bundle = get_eeg_medformer_dataloaders(dataset, args)
            train_loader = eeg_bundle.train_loader
            train_labels = eeg_bundle.train_labels
            vali_loader = eeg_bundle.vali_loader
            vali_labels = eeg_bundle.vali_labels
            test_loader = eeg_bundle.test_loader
            test_labels = eeg_bundle.test_labels
            audit_path = write_eeg_medformer_split_audit(eeg_bundle, result_dir)
            if getattr(args, "adftd_subject_subset", None) is not None:
                with open(f"{result_dir}/args.json", "w") as subset_args_file:
                    json.dump(vars(args), subset_args_file, indent=4)
            print(f"EEG subject split audit: {audit_path}")
        elif args.datasets == "wearable":
            fixed_validation_split = True
            wearable_bundle = get_wearable_dataloaders(dataset, args)
            train_loader = wearable_bundle.train_loader
            train_labels = wearable_bundle.train_labels
            vali_loader = wearable_bundle.vali_loader
            vali_labels = wearable_bundle.vali_labels
            test_loader = wearable_bundle.test_loader
            test_labels = wearable_bundle.test_labels
            audit_path = write_wearable_split_audit(wearable_bundle, result_dir)
            print(f"Subject split audit: {audit_path}")
        elif args.datasets == "uci" and args.uci_protocol == "official_subject":
            fixed_validation_split = True
            uci_bundle = get_uci_har_official_dataloaders(args)
            train_loader = uci_bundle.train_loader
            train_labels = uci_bundle.train_labels
            vali_loader = uci_bundle.vali_loader
            vali_labels = uci_bundle.vali_labels
            test_loader = uci_bundle.test_loader
            test_labels = uci_bundle.test_labels
            audit_path = write_uci_har_subject_split_audit(uci_bundle, result_dir)
            print(f"UCI HAR subject split audit: {audit_path}")
        elif (
            args.datasets == "falltl"
            and args.falltl_protocol == "comparison_binary"
        ):
            fixed_validation_split = True
            falltl_bundle = get_falltl_comparison_dataloaders(args)
            train_loader = falltl_bundle.train_loader
            train_labels = falltl_bundle.train_labels
            vali_loader = falltl_bundle.vali_loader
            vali_labels = falltl_bundle.vali_labels
            test_loader = falltl_bundle.test_loader
            test_labels = falltl_bundle.test_labels
            audit_path = write_falltl_comparison_split_audit(
                falltl_bundle, result_dir
            )
            print(f"FallTL split audit: {audit_path}")
        else:
            train_loader, train_labels, test_loader, test_labels = get_dataloader(
                dataset, args
            )

        split_audit_sha256 = (
            hashlib.sha256(Path(audit_path).read_bytes()).hexdigest()
            if audit_path is not None
            else None
        )

        sample_count = len(train_loader.dataset) + len(test_loader.dataset)
        if vali_loader is not None:
            sample_count += len(vali_loader.dataset)
        print("Samples: ", sample_count)
        if (
            args.image_mode in {"activity_graph", "med_activity_graph"}
            and not adaptive_graph_enabled
        ):
            save_activity_graph_samples(
                result_dir=result_dir,
                dataset=dataset,
                dataloader=train_loader,
                num_samples=args.save_activity_graph_samples,
                image_mode=args.image_mode,
                med_activity_patch_lengths=args.med_activity_patch_lengths,
                med_activity_channel_mix=args.med_activity_channel_mix,
                med_activity_router_temperature=(
                    args.med_activity_router_temperature
                ),
                med_activity_router_mix=args.med_activity_router_mix,
                med_activity_adaptive_granularity=(
                    args.med_activity_adaptive_granularity
                ),
                med_activity_granularity_bank=(
                    args.med_activity_granularity_bank
                ),
            )
            save_activity_lineplot_samples(
                result_dir=result_dir,
                dataset=dataset,
                dataloader=train_loader,
                num_samples=args.save_activity_lineplot_samples,
            )
        elif (
            args.image_mode == "multichannel_line_plot"
            or adaptive_graph_enabled
        ):
            save_activity_lineplot_samples(
                result_dir=result_dir,
                dataset=dataset,
                dataloader=train_loader,
                num_samples=args.save_activity_lineplot_samples,
            )
        channels, T = train_loader.dataset[0][0].shape

        # TDBRAIN 在此得到 channels=33、T=256；后续 outer_patch_size=64 形成 4 个内部块。
        mantis_embedding = None
        moment_embedding = None
        vision_embedding_1 = None
        vision_embedding_2 = None
        mantis_model = None
        moment_model = None

        # Embedding with Mantis TSFM
        if args.mantis:
            network = Mantis8M(device=device)
            network = network.from_pretrained(args.mantis_name)
            network = network.to(device)

            if args.classifier_type == "mlp":
                mantis_model = network
            else:
                mantis_model = MantisTrainer(device=device, network=network)
                mantis_embedding = embed_loader_splits(
                    mantis_model,
                    "mantis",
                    channels,
                    device,
                    train_loader,
                    test_loader,
                    vali_loader=vali_loader,
                )

        # Embedding with MOMENT TSFM
        if args.moment:
            moment = MOMENTPipeline.from_pretrained(
                f"AutonLab/MOMENT-1-{args.moment}",
                model_kwargs={"task_name": "embedding"},
            )
            moment.init()
            moment.to(device).float()
            moment.eval()
            moment_model = moment

            if args.classifier_type != "mlp":
                moment_embedding = embed_loader_splits(
                    moment,
                    "moment",
                    channels,
                    device,
                    train_loader,
                    test_loader,
                    vali_loader=vali_loader,
                )

        patch_sizes = get_patch_size(patch_size=args.patch_size, T=T)
        if args.image_mode in {
            "line_plot",
            "multichannel_line_plot",
            "activity_graph",
            "med_activity_graph",
            "activity_matrix",
        }:
            patch_sizes = [None]

        for p in patch_sizes:
            neurosigvia_1 = None
            neurosigvia_2 = None

            if p:
                print(f"Patch size: {p}")

            # Embedding with the NeuroSigVIA visual branch (1st ViT configuration)
            if args.vit_1_name:
                neurosigvia_1 = get_neurosigvia(
                    model_name=args.vit_1_name,
                    model_layer=args.vit_1_layer,
                    aggregation=args.aggregation,
                    stride=args.stride,
                    patch_size=p,
                    image_mode=args.image_mode,
                    med_activity_patch_lengths=args.med_activity_patch_lengths,
                    med_activity_channel_mix=args.med_activity_channel_mix,
                    med_activity_router_temperature=(
                        args.med_activity_router_temperature
                    ),
                    med_activity_router_mix=args.med_activity_router_mix,
                    med_activity_adaptive_granularity=(
                        args.med_activity_adaptive_granularity
                    ),
                    med_activity_granularity_bank=(
                        args.med_activity_granularity_bank
                    ),
                )
                neurosigvia_1 = neurosigvia_1.to(device=device)
                neurosigvia_1.eval()

                if args.classifier_type != "mlp":
                    vision_embedding_1 = embed_loader_splits(
                        neurosigvia_1,
                        "neurosigvia",
                        channels,
                        device,
                        train_loader,
                        test_loader,
                        vali_loader=vali_loader,
                    )

            # Embedding with the NeuroSigVIA visual branch (2nd ViT configuration)
            if args.vit_2_name:
                neurosigvia_2 = get_neurosigvia(
                    model_name=args.vit_2_name,
                    model_layer=args.vit_2_layer,
                    aggregation=args.aggregation,
                    stride=args.stride,
                    patch_size=p,
                    image_mode=args.image_mode,
                    med_activity_patch_lengths=args.med_activity_patch_lengths,
                    med_activity_channel_mix=args.med_activity_channel_mix,
                    med_activity_router_temperature=(
                        args.med_activity_router_temperature
                    ),
                    med_activity_router_mix=args.med_activity_router_mix,
                    med_activity_adaptive_granularity=(
                        args.med_activity_adaptive_granularity
                    ),
                    med_activity_granularity_bank=(
                        args.med_activity_granularity_bank
                    ),
                )
                neurosigvia_2 = neurosigvia_2.to(device=device)
                neurosigvia_2.eval()

                if args.classifier_type != "mlp":
                    vision_embedding_2 = embed_loader_splits(
                        neurosigvia_2,
                        "neurosigvia",
                        channels,
                        device,
                        train_loader,
                        test_loader,
                        vali_loader=vali_loader,
                    )

            # Linear classification
            if args.classifier_type:
                if args.classifier_type == "mlp":
                    feature_cache_signature = None
                    feature_cache_dir = None
                    if args.feature_cache_dir:
                        adaptive_split_identity = None
                        adaptive_feature_code_identity = None
                        adaptive_runtime_identity = None
                        static_encoder_contract = None
                        if (
                            args.med_activity_adaptive_granularity
                            or adaptive_graph_enabled
                        ):
                            adaptive_split_identity = _split_input_identity(
                                train_loader=train_loader,
                                train_labels=train_labels,
                                test_loader=test_loader,
                                test_labels=test_labels,
                                vali_loader=vali_loader,
                                vali_labels=vali_labels,
                            )
                            if adaptive_graph_enabled:
                                adaptive_feature_code_identity = (
                                    _adaptive_graph_feature_extractor_code_identity()
                                )
                                static_encoder_contract = _adaptive_static_encoder_contract(
                                    neurosigvia_1, mantis_model
                                )
                            elif args.modal_interaction == "patch_mindts":
                                adaptive_feature_code_identity = (
                                    _patch_feature_extractor_code_identity()
                                )
                            else:
                                adaptive_feature_code_identity = (
                                    _feature_extractor_code_identity()
                                )
                            adaptive_runtime_identity = (
                                _feature_runtime_identity()
                            )
                        feature_cache_signature = build_feature_cache_signature(
                            args,
                            dataset,
                            channels,
                            p,
                            split_audit_sha256=split_audit_sha256,
                            split_input_identity=adaptive_split_identity,
                            feature_code_identity=adaptive_feature_code_identity,
                            runtime_identity=adaptive_runtime_identity,
                            static_encoder_contract=static_encoder_contract,
                        )
                        feature_cache_key = hashlib.sha256(
                            feature_cache_signature.encode("utf-8")
                        ).hexdigest()[:16]
                        feature_cache_dir = os.path.join(
                            args.feature_cache_dir,
                            dataset,
                            feature_cache_key,
                        )
                        cache_manifest = {
                            "schema": 1,
                            "dataset": dataset,
                            "cache_key": feature_cache_key,
                            "relative_cache_subdir": os.path.join(
                                dataset, feature_cache_key
                            ),
                            "signature": json.loads(feature_cache_signature),
                        }
                        if adaptive_graph_enabled and args.reuse_static_cache_dir:
                            legacy_signatures = {}
                            legacy_skip_reason = None
                            if cache_manifest["signature"]["split_seed"] == 42:
                                try:
                                    assert_known_static_extractor_compatibility(
                                        adaptive_feature_code_identity
                                    )
                                except ValueError as error:
                                    legacy_skip_reason = str(error)
                                else:
                                    for family, architecture in (
                                        ("neurosigvia_97ade80", NEUROSIGVIA_CACHE_ARCHITECTURE),
                                        ("timemosaic_7f38ff7", KNOWN_TIMEMOSAIC_STATIC_CACHE_ARCHITECTURE),
                                    ):
                                        previous_signature = _known_legacy_adaptive_cache_signature(
                                            args, dataset, channels, p, family=family,
                                            split_audit_sha256=split_audit_sha256,
                                            split_input_identity=adaptive_split_identity,
                                            runtime_identity=adaptive_runtime_identity,
                                            static_encoder_contract=static_encoder_contract,
                                        )
                                        legacy_signatures[previous_signature] = architecture
                            cache_manifest["static_cache_reuse"] = promote_adaptive_static_caches(
                                destination_dir=feature_cache_dir,
                                signature=feature_cache_signature,
                                legacy_signatures=legacy_signatures,
                                search_root=args.reuse_static_cache_dir,
                                split_loaders={
                                    "train": (train_loader, train_labels),
                                    "vali": (vali_loader, vali_labels),
                                    "test": (test_loader, test_labels),
                                },
                                channels=channels,
                                window_size=args.outer_patch_size,
                                stride=args.outer_patch_stride,
                            )
                            if legacy_skip_reason is not None:
                                cache_manifest["static_cache_reuse"]["legacy_skip_reason"] = legacy_skip_reason
                                print(legacy_skip_reason)
                        manifest_path = Path(result_dir) / (
                            f"{dataset}_feature_cache_manifest.json"
                        )
                        manifest_path.write_text(
                            json.dumps(cache_manifest, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                        print(f"Feature cache key: {feature_cache_key}")
                    if args.modal_interaction == "adaptive_granularity":
                        # train_neurosigvia_classifier 依次准备静态 line/Mantis 特征、在线活动图、
                        # 融合与 MLP；4 个 64 点块最终汇聚成一个 256 点窗口的 logits[B,2]。
                        # 这里直接传入固定 val_loader；random_seed=42/43/44 不重划分 TDBRAIN 受试者。
                        val_metrics, test_metrics, train_indices, val_indices = (
                            train_neurosigvia_classifier(
                                train_loader=train_loader,
                                train_labels=train_labels,
                                test_loader=test_loader,
                                test_labels=test_labels,
                                channels=channels,
                                device=device,
                                batch_size=args.batch_size,
                                random_seed=args.random_seed,
                                val_ratio=args.val_ratio,
                                hidden_dim=args.mlp_hidden_dim,
                                num_layers=args.mlp_num_layers,
                                dropout=args.mlp_dropout,
                                lr=args.mlp_lr,
                                weight_decay=args.mlp_weight_decay,
                                class_weight=args.mlp_class_weight,
                                epochs=args.mlp_epochs,
                                early_stop_patience=(
                                    args.mlp_early_stop_patience
                                ),
                                early_stop_strategy=(
                                    args.mlp_early_stop_strategy
                                ),
                                early_stop_min_epochs=(
                                    args.mlp_early_stop_min_epochs
                                ),
                                early_stop_warmup_epochs=(
                                    args.mlp_early_stop_warmup_epochs
                                ),
                                early_stop_ema_decay=(
                                    args.mlp_early_stop_ema_decay
                                ),
                                early_stop_min_delta=(
                                    args.mlp_early_stop_min_delta
                                ),
                                lr_scheduler_type=args.mlp_lr_scheduler,
                                lr_scheduler_patience=(
                                    args.mlp_lr_scheduler_patience
                                ),
                                lr_scheduler_factor=(
                                    args.mlp_lr_scheduler_factor
                                ),
                                lr_scheduler_min_lr=(
                                    args.mlp_lr_scheduler_min_lr
                                ),
                                fusion_dim=args.fusion_dim,
                                fusion_heads=args.fusion_heads,
                                alignment_dim=args.patch_alignment_dim,
                                alignment_temperature=(
                                    args.patch_alignment_temperature
                                ),
                                alignment_weight=args.patch_alignment_weight,
                                outer_patch_size=args.outer_patch_size,
                                outer_patch_stride=args.outer_patch_stride,
                                visual_encode_batch_size=(
                                    args.visual_encode_batch_size
                                ),
                                vision_model=neurosigvia_1,
                                mantis_model=mantis_model,
                                val_loader=vali_loader,
                                val_labels=vali_labels,
                                feature_cache_dir=feature_cache_dir,
                                feature_cache_signature=(
                                    feature_cache_signature
                                ),
                                gate_temperature=(
                                    args.granularity_gate_temperature
                                ),
                                activity_graph_canvas_size=(
                                    args.activity_graph_canvas_size
                                ),
                                activity_graph_line_width=(
                                    args.activity_graph_line_width
                                ),
                                activity_graph_vertical_margin=(
                                    args.activity_graph_vertical_margin
                                ),
                                selector_balance_weight=(
                                    args.granularity_balance_weight
                                ),
                                checkpoint_metric=args.patch_checkpoint_metric,
                                channel_hidden_dim=(
                                    args.med_activity_granularity_hidden_dim
                                ),
                                graph_token_grid=(
                                    args.granularity_graph_token_grid
                                ),
                                gate_checkpoint=(
                                    args.granularity_gate_checkpoint
                                ),
                                freeze_gate=args.granularity_freeze_gate,
                                artifact_dir=(
                                    Path(result_dir)
                                    / "adaptive_graph"
                                    / str(dataset)
                                    .replace("/", "_")
                                    .replace("\\", "_")
                                ),
                            )
                        )
                    elif args.modal_interaction == "patch_mindts":
                        val_metrics, test_metrics, train_indices, val_indices = (
                            train_patch_mindts_classifier(
                                train_loader=train_loader,
                                train_labels=train_labels,
                                test_loader=test_loader,
                                test_labels=test_labels,
                                channels=channels,
                                device=device,
                                batch_size=args.batch_size,
                                random_seed=args.random_seed,
                                val_ratio=args.val_ratio,
                                hidden_dim=args.mlp_hidden_dim,
                                num_layers=args.mlp_num_layers,
                                dropout=args.mlp_dropout,
                                lr=args.mlp_lr,
                                weight_decay=args.mlp_weight_decay,
                                class_weight=args.mlp_class_weight,
                                epochs=args.mlp_epochs,
                                early_stop_patience=(
                                    args.mlp_early_stop_patience
                                ),
                                early_stop_strategy=(
                                    args.mlp_early_stop_strategy
                                ),
                                early_stop_min_epochs=(
                                    args.mlp_early_stop_min_epochs
                                ),
                                early_stop_ema_decay=(
                                    args.mlp_early_stop_ema_decay
                                ),
                                early_stop_min_delta=(
                                    args.mlp_early_stop_min_delta
                                ),
                                fusion_dim=args.fusion_dim,
                                fusion_heads=args.fusion_heads,
                                alignment_dim=args.patch_alignment_dim,
                                alignment_temperature=(
                                    args.patch_alignment_temperature
                                ),
                                alignment_weight=args.patch_alignment_weight,
                                outer_patch_size=args.outer_patch_size,
                                outer_patch_stride=args.outer_patch_stride,
                                visual_encode_batch_size=(
                                    args.visual_encode_batch_size
                                ),
                                vision_model=neurosigvia_1,
                                mantis_model=mantis_model,
                                val_loader=vali_loader,
                                val_labels=vali_labels,
                                feature_cache_dir=feature_cache_dir,
                                feature_cache_signature=(
                                    feature_cache_signature
                                ),
                                granularity_temperature=(
                                    args.med_activity_granularity_temperature
                                ),
                                granularity_balance_weight=(
                                    args.med_activity_granularity_balance_weight
                                ),
                                granularity_entropy_weight=(
                                    args.med_activity_granularity_entropy_weight
                                ),
                                granularity_mix_shrinkage_weight=(
                                    args.med_activity_granularity_mix_shrinkage_weight
                                ),
                                granularity_prior_kl_weight=(
                                    args.med_activity_granularity_prior_kl_weight
                                ),
                                granularity_usage_floor=(
                                    args.med_activity_granularity_usage_floor
                                ),
                                granularity_usage_ema_decay=(
                                    args.med_activity_granularity_usage_ema_decay
                                ),
                                granularity_entropy_floor=(
                                    args.med_activity_granularity_entropy_floor
                                ),
                                granularity_entropy_ceiling=(
                                    args.med_activity_granularity_entropy_ceiling
                                ),
                                granularity_router_mode=(
                                    args.patch_granularity_router_mode
                                ),
                                granularity_local_mix_max=(
                                    args.med_activity_granularity_local_mix_max
                                ),
                                granularity_local_mix_init=(
                                    args.med_activity_granularity_local_mix_init
                                ),
                                granularity_global_mix_max=(
                                    args.med_activity_granularity_global_mix_max
                                ),
                                granularity_global_mix_init=(
                                    args.med_activity_granularity_global_mix_init
                                ),
                                granularity_evidence_half_saturation=(
                                    args.med_activity_granularity_evidence_half_saturation
                                ),
                                granularity_minimum_weight=(
                                    args.med_activity_granularity_minimum_weight
                                ),
                                granularity_score_cap=(
                                    args.med_activity_granularity_score_cap
                                ),
                                granularity_scorer_hidden_dim=(
                                    args.med_activity_granularity_scorer_hidden_dim
                                ),
                                granularity_confidence_half_saturation=(
                                    args.med_activity_granularity_confidence_half_saturation
                                ),
                                router_top_k=args.patch_router_top_k,
                                router_training_noise_std=(
                                    args.patch_router_training_noise_std
                                ),
                                router_local_weight=(
                                    args.patch_router_local_weight
                                ),
                                router_relation_hidden_dim=(
                                    args.patch_router_relation_hidden_dim
                                ),
                                router_relation_residual_scale=(
                                    args.patch_router_relation_residual_scale
                                ),
                                router_key_adapter_scale=(
                                    args.patch_router_key_adapter_scale
                                ),
                                router_value_adapter_scale=(
                                    args.patch_router_value_adapter_scale
                                ),
                                router_route_budget_weight=(
                                    args.patch_router_route_budget_weight
                                    if args.patch_granularity_router_mode
                                    == "adaptive_v5"
                                    else 0.0
                                ),
                                router_load_balance_weight=(
                                    args.patch_router_load_balance_weight
                                    if args.patch_granularity_router_mode
                                    == "adaptive_v5"
                                    else 0.0
                                ),
                                checkpoint_metric=args.patch_checkpoint_metric,
                                channel_hidden_dim=(
                                    args.med_activity_granularity_hidden_dim
                                ),
                                artifact_dir=(
                                    Path(result_dir)
                                    / "patch_atgs"
                                    / str(dataset)
                                    .replace("/", "_")
                                    .replace("\\", "_")
                                ),
                            )
                        )
                    else:
                        val_metrics, test_metrics, train_indices, val_indices = (
                            train_mlp_classifier(
                            train_loader=train_loader,
                            train_labels=train_labels,
                            test_loader=test_loader,
                            test_labels=test_labels,
                            channels=channels,
                            device=device,
                            batch_size=args.batch_size,
                            random_seed=args.random_seed,
                            val_ratio=args.val_ratio,
                            hidden_dim=args.mlp_hidden_dim,
                            num_layers=args.mlp_num_layers,
                            dropout=args.mlp_dropout,
                            lr=args.mlp_lr,
                            weight_decay=args.mlp_weight_decay,
                            class_weight=args.mlp_class_weight,
                            epochs=args.mlp_epochs,
                            early_stop_patience=args.mlp_early_stop_patience,
                            modal_interaction=args.modal_interaction,
                            fusion_dim=args.fusion_dim,
                            fusion_heads=args.fusion_heads,
                            cross_attn_query=args.cross_attn_query,
                            mask_prob=args.mask_prob,
                            pretrain_epochs=args.pretrain_epochs,
                            vision_model_1=(
                                neurosigvia_1 if args.vit_1_name else None
                            ),
                            vision_model_2=(
                                neurosigvia_2 if args.vit_2_name else None
                            ),
                            mantis_model=mantis_model,
                            moment_model=moment_model,
                            val_loader=vali_loader,
                            val_labels=vali_labels,
                            feature_cache_dir=feature_cache_dir,
                            feature_cache_signature=feature_cache_signature,
                            granularity_hidden_dim=(
                                args.med_activity_granularity_hidden_dim
                            ),
                            granularity_temperature=(
                                args.med_activity_granularity_temperature
                            ),
                            granularity_base_prior=(
                                args.med_activity_granularity_base_prior
                            ),
                            granularity_balance_weight=(
                                args.med_activity_granularity_balance_weight
                            ),
                            granularity_entropy_weight=(
                                args.med_activity_granularity_entropy_weight
                            ),
                            artifact_dir=(
                                Path(result_dir)
                                / "atgs"
                                / str(dataset).replace("/", "_").replace("\\", "_")
                                if args.med_activity_adaptive_granularity
                                else None
                            ),
                            )
                        )
                else:
                    combined_embeddings = concat_embeddings(
                        vision_embedding_1,
                        vision_embedding_2,
                        mantis_embedding,
                        moment_embedding,
                    )

                    if fixed_validation_split:
                        train_embeds, vali_embeds, test_embeds = combined_embeddings
                    else:
                        train_embeds, test_embeds = combined_embeddings
                        vali_embeds = None

                    val_metrics, test_metrics, train_indices, val_indices = train_classifier(
                        train_embeds,
                        train_labels,
                        test_embeds,
                        test_labels,
                        args.classifier_type,
                        args.random_seed,
                        args.val_ratio,
                        val_embeds=vali_embeds,
                        val_labels=vali_labels,
                    )
                print(
                    "Val metrics: "
                    + ", ".join(
                        f"{metric}={value:.4f}"
                        for metric, value in val_metrics.items()
                    )
                )
                print(
                    "Test metrics: "
                    + ", ".join(
                        f"{metric}={value:.4f}"
                        for metric, value in test_metrics.items()
                    )
                )

                if not fixed_validation_split:
                    write_split_indices(
                        result_dir=result_dir,
                        dataset=dataset,
                        train_indices=train_indices,
                        val_indices=val_indices,
                        random_seed=args.random_seed,
                        val_ratio=args.val_ratio,
                    )

                write_result_table(
                    result_dir=result_dir,
                    dataset=dataset,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    patch_size=p,
                    image_mode=args.image_mode,
                )

            # Measure alignment of representation spaces using mutual kNN
            elif args.measure_alignment:
                measure_alignment(
                    mantis_embedding,
                    moment_embedding,
                    vision_embedding_1,
                    vision_embedding_2,
                    dataset,
                    result_dir,
                )

            # Compute intrinsic dimension or number of principal components
            elif args.get_intrinsic_dimension or args.get_principal_components:
                embeddings = [
                    e
                    for e in [
                        vision_embedding_1,
                        vision_embedding_2,
                        mantis_embedding,
                        moment_embedding,
                    ]
                    if e is not None
                ]

                assert (
                    len(embeddings) == 1
                ), "Compute intrinsic dimensionality only for one model."

                embedding = np.concatenate(embeddings[0], axis=0).transpose(2, 0, 1)

                if args.get_intrinsic_dimension:
                    get_intrinsic_dimension(embedding, dataset, result_dir)
                if args.get_principal_components:
                    get_principal_components(embedding, dataset, result_dir)

            else:
                raise ValueError(
                    "Please choose: linear probing, intrinsic dimension, principal components, alignment."
                )

            torch.cuda.empty_cache()
