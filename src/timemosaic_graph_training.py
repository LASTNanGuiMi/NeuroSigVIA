"""Independent training path for adaptive TimeMosaic-style Activity Graphs.

This module deliberately does not change the legacy ``patch_mindts`` cache or
trainer.  Its static cache contains raw temporal windows plus frozen line-plot
and Mantis features.  During every train/evaluation forward pass, the raw
windows are routed by ``TimeMosaicAdaptiveActivityGraphRenderer`` to form one
Activity Graph, which is then encoded as spatial OpenCLIP tokens.  Consequently
the granularity gate is part of graph construction and can receive gradients
through the frozen visual encoder.

The visual encoder parameters are frozen, but its forward pass is not wrapped
in ``torch.no_grad`` during training.  This distinction is required to retain
the image-to-gate gradient path.  The visual encoder itself is external to the
trainable model and is therefore intentionally absent from checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torch.utils.data import DataLoader, SequentialSampler, Subset, TensorDataset
from tqdm import tqdm

from src.classifier import compute_metrics_from_predictions
from src.medformer_graph.timemosaic_adaptive import (
    TimeMosaicAdaptiveActivityGraphRenderer,
)
from src.patch_mindts import (
    PATCH_TAIL_POLICY,
    _EarlyStoppingMonitor,
    _aggregate_subject_predictions,
    _balanced_class_weights,
    _build_patch_lr_scheduler,
    _checkpoint_selection_key,
    _extract_line_tokens,
    _extract_mantis_channel_tokens,
    _labels_to_indices,
    _loader_sample_subject_ids,
    _map_labels,
    _merge_subject_metrics,
    _resolve_checkpoint_metric,
    _subject_ids_digest,
    make_temporal_patches,
)
from src.timemosaic_patch_pipeline import (
    OPENCLIP_SPATIAL_GRID_SIZE,
    TimeMosaicPatchFusionModule,
    compress_openclip_spatial_tokens,
)
from src.utils import get_split, set_random_seed


TIMEMOSAIC_GRAPH_CACHE_SCHEMA_VERSION = 1
TIMEMOSAIC_GRAPH_CACHE_ARCHITECTURE = "timemosaic_adaptive_graph_static_v1"
TIMEMOSAIC_GRAPH_CHECKPOINT_SCHEMA_VERSION = 1
TIMEMOSAIC_GRAPH_ARCHITECTURE = "timemosaic_adaptive_graph_crossattn_v1"

TIMEMOSAIC_GRAPH_STATIC_KEYS = (
    "raw_windows",
    "line_tokens",
    "mantis_channel_tokens",
    "patch_mask",
    "valid_fraction",
    "valid_lengths",
)


def _signature_model_identity(signature, key: str):
    if not str(signature or "").strip():
        return None
    try:
        payload = json.loads(str(signature))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload.get(key)


def _sampled_module_state_fingerprint(
    module: nn.Module,
    *,
    values_per_tensor: int = 64,
) -> str:
    """Fingerprint every state entry using metadata and fixed-position values.

    A full ViT-H byte hash would add a multi-gigabyte device-to-host transfer
    to every run.  This fingerprint instead covers every parameter/buffer name,
    dtype, and shape plus evenly spaced values from each tensor.  The cache
    signature may additionally carry the full on-disk checkpoint manifest.
    """
    digest = hashlib.sha256()
    digest.update(
        f"{type(module).__module__}.{type(module).__qualname__}".encode("utf-8")
    )
    named_state = [
        (f"parameter:{name}", tensor)
        for name, tensor in module.named_parameters()
    ]
    named_state.extend(
        (f"buffer:{name}", tensor) for name, tensor in module.named_buffers()
    )
    for name, tensor in sorted(named_state, key=lambda item: item[0]):
        detached = tensor.detach()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(json.dumps(list(detached.shape)).encode("ascii"))
        element_count = detached.numel()
        if element_count == 0:
            continue
        sample_count = min(int(values_per_tensor), int(element_count))
        if sample_count == element_count:
            sample = detached.reshape(-1)
        else:
            indices = torch.linspace(
                0,
                element_count - 1,
                steps=sample_count,
                device=detached.device,
            ).round().to(dtype=torch.long)
            sample = detached.reshape(-1).index_select(0, indices)
        if sample.is_complex():
            sample = torch.view_as_real(sample).float()
        elif sample.is_floating_point():
            sample = sample.float()
        elif sample.dtype == torch.bool:
            sample = sample.to(dtype=torch.uint8)
        else:
            sample = sample.to(dtype=torch.int64)
        digest.update(sample.contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _encoder_contract(
    encoder: nn.Module,
    role: str,
    *,
    source_identity=None,
) -> dict[str, Any]:
    processor = getattr(encoder, "processor", None)
    processor_transforms = getattr(processor, "transforms", None)
    if processor_transforms is not None:
        processor_transforms = [repr(transform) for transform in processor_transforms]
    backbone = getattr(encoder, "vit", None)
    return {
        "role": str(role),
        "wrapper_class": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
        "backbone_class": (
            f"{type(backbone).__module__}.{type(backbone).__qualname__}"
            if backbone is not None
            else None
        ),
        "processor_class": (
            f"{type(processor).__module__}.{type(processor).__qualname__}"
            if processor is not None
            else None
        ),
        "processor_transforms": processor_transforms,
        "layer_idx": getattr(encoder, "layer_idx", None),
        "aggregation": getattr(encoder, "aggregation", None),
        "image_mode": getattr(encoder, "image_mode", None),
        "sampled_state_sha256": _sampled_module_state_fingerprint(encoder),
        "sampled_values_per_tensor": 64,
        "source_checkpoint_identity": source_identity,
    }


def _assert_encoder_contract(
    expected: Mapping[str, Any],
    encoder: nn.Module,
    role: str,
) -> dict[str, Any]:
    current = _encoder_contract(encoder, role)
    compared_keys = (
        "wrapper_class",
        "backbone_class",
        "processor_class",
        "processor_transforms",
        "layer_idx",
        "aggregation",
        "image_mode",
        "sampled_state_sha256",
    )
    mismatches = {
        key: {"expected": expected.get(key), "actual": current.get(key)}
        for key in compared_keys
        if expected.get(key) != current.get(key)
    }
    if mismatches:
        raise ValueError(
            f"{role} encoder does not match the checkpoint contract: {mismatches}"
        )
    return current


def _one_dimensional_labels(labels, name: str) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    return array


def _cache_label_array(labels) -> np.ndarray:
    """Return labels that can be loaded safely with ``allow_pickle=False``."""
    array = _one_dimensional_labels(labels, "labels")
    if array.dtype == object:
        array = array.astype(str)
    return array


def _validate_static_bundle(
    bundle: Mapping[str, torch.Tensor],
    *,
    expected_window_size: int | None = None,
    expected_channels: int | None = None,
) -> dict[str, torch.Tensor]:
    missing = set(TIMEMOSAIC_GRAPH_STATIC_KEYS) - set(bundle)
    if missing:
        raise ValueError(
            "TimeMosaic graph cache is missing " + ", ".join(sorted(missing))
        )
    values = {key: bundle[key] for key in TIMEMOSAIC_GRAPH_STATIC_KEYS}
    if not all(torch.is_tensor(value) for value in values.values()):
        raise TypeError("all TimeMosaic graph cache entries must be tensors")

    raw = values["raw_windows"]
    line = values["line_tokens"]
    mantis = values["mantis_channel_tokens"]
    mask = values["patch_mask"]
    fraction = values["valid_fraction"]
    lengths = values["valid_lengths"]
    if raw.ndim != 4 or line.ndim != 3 or mantis.ndim != 4:
        raise ValueError(
            "expected raw [S,N,C,T], line [S,N,Dv], and Mantis [S,N,C,Dt]"
        )
    sample_patch_shape = tuple(raw.shape[:2])
    if tuple(line.shape[:2]) != sample_patch_shape:
        raise ValueError("line tokens do not share raw sample/patch axes")
    if tuple(mantis.shape[:2]) != sample_patch_shape:
        raise ValueError("Mantis tokens do not share raw sample/patch axes")
    if mantis.shape[2] != raw.shape[2]:
        raise ValueError("raw and Mantis channel counts do not match")
    for name, value in (
        ("patch_mask", mask),
        ("valid_fraction", fraction),
        ("valid_lengths", lengths),
    ):
        if tuple(value.shape) != sample_patch_shape:
            raise ValueError(
                f"{name} must have shape {sample_patch_shape}, got {tuple(value.shape)}"
            )
    if not raw.is_floating_point():
        raise TypeError("raw_windows must use a floating dtype")
    for name in ("line_tokens", "mantis_channel_tokens", "valid_fraction"):
        if not values[name].is_floating_point():
            raise TypeError(f"{name} must use a floating dtype")
        if not torch.isfinite(values[name]).all():
            raise ValueError(f"cache contains non-finite {name}")
    if line.shape[-1] < 1 or mantis.shape[-1] < 1:
        raise ValueError("cached feature dimensions must be non-empty")
    if raw.shape[2] < 1 or raw.shape[3] < 1:
        raise ValueError("cached raw channel/time dimensions must be non-empty")
    if expected_window_size is not None and raw.shape[-1] != expected_window_size:
        raise ValueError(
            f"expected raw window size {expected_window_size}, got {raw.shape[-1]}"
        )
    if expected_channels is not None and raw.shape[2] != expected_channels:
        raise ValueError(
            f"expected {expected_channels} channels, got {raw.shape[2]}"
        )

    mask_bool = mask.to(dtype=torch.bool)
    lengths_long = lengths.to(dtype=torch.long)
    if not torch.equal(mask_bool, lengths_long > 0):
        raise ValueError("patch_mask must equal valid_lengths > 0")
    if bool((lengths_long < 0).any()) or bool(
        (lengths_long > raw.shape[-1]).any()
    ):
        raise ValueError("valid_lengths entries must lie in [0, window_size]")
    fraction_float = fraction.float()
    expected_fraction = lengths_long.float() / float(raw.shape[-1])
    if not torch.allclose(
        fraction_float,
        expected_fraction,
        rtol=1e-3,
        atol=1e-3,
    ):
        raise ValueError("valid_fraction must equal valid_lengths / window_size")
    if bool((mask_bool.sum(dim=1) == 0).any()):
        raise ValueError("every cached sample must contain at least one valid window")
    return values


def save_timemosaic_graph_feature_cache(
    path,
    bundle,
    labels,
    signature,
    *,
    window_size: int,
    stride: int,
) -> None:
    """Atomically save raw windows and the two frozen feature branches."""
    if not str(signature or "").strip():
        raise ValueError(
            "a non-empty cache signature is required to prevent stale-feature reuse"
        )
    path = Path(path)
    validated = _validate_static_bundle(
        bundle,
        expected_window_size=int(window_size),
    )
    labels_array = _cache_label_array(labels)
    if len(labels_array) != len(validated["raw_windows"]):
        raise ValueError("cache labels and feature sample count do not match")
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(
            TIMEMOSAIC_GRAPH_CACHE_SCHEMA_VERSION, dtype=np.int64
        ),
        "architecture": np.asarray(TIMEMOSAIC_GRAPH_CACHE_ARCHITECTURE),
        "cache_signature": np.asarray(str(signature or "")),
        "tail_policy": np.asarray(PATCH_TAIL_POLICY),
        "window_size": np.asarray(int(window_size), dtype=np.int64),
        "stride": np.asarray(int(stride), dtype=np.int64),
        "labels": labels_array,
    }
    arrays.update(
        {
            key: tensor.detach().cpu().numpy()
            for key, tensor in validated.items()
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_timemosaic_graph_feature_cache(
    path,
    labels,
    signature,
    *,
    window_size: int,
    stride: int,
    expected_channels: int | None = None,
) -> dict[str, torch.Tensor] | None:
    """Load a validated independent cache, or return ``None`` on a cold miss."""
    if not str(signature or "").strip():
        raise ValueError(
            "a non-empty cache signature is required to prevent stale-feature reuse"
        )
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cached:
        required = {
            "schema_version",
            "architecture",
            "cache_signature",
            "tail_policy",
            "window_size",
            "stride",
            "labels",
            *TIMEMOSAIC_GRAPH_STATIC_KEYS,
        }
        missing = required - set(cached.files)
        if missing:
            raise ValueError(f"cache is missing {sorted(missing)}: {path}")
        if int(cached["schema_version"].item()) != (
            TIMEMOSAIC_GRAPH_CACHE_SCHEMA_VERSION
        ):
            raise ValueError(f"cache schema mismatch: {path}")
        if str(cached["architecture"].item()) != (
            TIMEMOSAIC_GRAPH_CACHE_ARCHITECTURE
        ):
            raise ValueError(f"cache architecture mismatch: {path}")
        if str(cached["cache_signature"].item()) != str(signature or ""):
            raise ValueError(f"cache signature mismatch: {path}")
        if str(cached["tail_policy"].item()) != PATCH_TAIL_POLICY:
            raise ValueError(f"cache tail policy mismatch: {path}")
        if int(cached["window_size"].item()) != int(window_size):
            raise ValueError(f"cache window size mismatch: {path}")
        if int(cached["stride"].item()) != int(stride):
            raise ValueError(f"cache stride mismatch: {path}")
        if not np.array_equal(cached["labels"], _cache_label_array(labels)):
            raise ValueError(f"cache labels do not match: {path}")
        bundle = {
            key: torch.from_numpy(cached[key].copy())
            for key in TIMEMOSAIC_GRAPH_STATIC_KEYS
        }
    validated = _validate_static_bundle(
        bundle,
        expected_window_size=int(window_size),
        expected_channels=expected_channels,
    )
    if len(validated["raw_windows"]) != len(_cache_label_array(labels)):
        raise ValueError(f"cache sample count mismatch: {path}")
    print(f"Loaded TimeMosaic graph static features: {path}")
    return validated


@torch.no_grad()
def extract_timemosaic_graph_feature_batch(
    batch,
    vision_model,
    mantis_model,
    device,
    *,
    window_size: int = 64,
    stride: int = 64,
    encode_batch_size: int = 16,
    lengths=None,
) -> dict[str, torch.Tensor]:
    """Cache raw windows, pooled line tokens, and channel-wise Mantis tokens."""
    if encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be positive")
    if vision_model is None or mantis_model is None:
        raise ValueError("both vision_model and mantis_model are required")
    vision_model.requires_grad_(False)
    mantis_model.requires_grad_(False)
    vision_model.eval()
    mantis_model.eval()

    temporal = make_temporal_patches(
        batch,
        window_size=window_size,
        stride=stride,
        lengths=lengths,
    )
    batch_size, patch_count, channels, patch_length = temporal.patches.shape
    flat_windows = temporal.patches.reshape(
        batch_size * patch_count,
        channels,
        patch_length,
    ).detach().cpu().float()
    flat_lengths = temporal.valid_lengths.reshape(-1).detach().cpu()
    flat_valid = temporal.patch_mask.reshape(-1).detach().cpu()
    valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("feature batch contains no valid temporal windows")
    valid_windows = flat_windows.index_select(0, valid_indices)
    valid_lengths = flat_lengths.index_select(0, valid_indices)

    line_valid = _extract_line_tokens(
        valid_windows,
        valid_lengths,
        vision_model,
        device,
        encode_batch_size,
    )
    mantis_valid = _extract_mantis_channel_tokens(
        valid_windows,
        valid_lengths,
        mantis_model,
        device,
        encode_batch_size,
    )
    total_patches = batch_size * patch_count
    line = torch.zeros(
        (total_patches, line_valid.shape[-1]), dtype=torch.float32
    )
    mantis = torch.zeros(
        (
            total_patches,
            mantis_valid.shape[1],
            mantis_valid.shape[2],
        ),
        dtype=torch.float32,
    )
    line.index_copy_(0, valid_indices, line_valid.float())
    mantis.index_copy_(0, valid_indices, mantis_valid.float())
    bundle = {
        # Raw windows remain float32 because the trainable renderer consumes
        # them online; the two frozen representations use compact float16.
        "raw_windows": flat_windows.reshape(
            batch_size,
            patch_count,
            channels,
            patch_length,
        ),
        "line_tokens": line.reshape(batch_size, patch_count, -1).half(),
        "mantis_channel_tokens": mantis.reshape(
            batch_size,
            patch_count,
            mantis.shape[1],
            mantis.shape[2],
        ).half(),
        "patch_mask": temporal.patch_mask.detach().cpu(),
        "valid_fraction": temporal.valid_fraction.detach().cpu().half(),
        "valid_lengths": temporal.valid_lengths.detach().cpu(),
    }
    return _validate_static_bundle(
        bundle,
        expected_window_size=window_size,
        expected_channels=channels,
    )


@torch.no_grad()
def _extract_static_split(
    loader,
    vision_model,
    mantis_model,
    device,
    *,
    window_size: int,
    stride: int,
    encode_batch_size: int,
    expected_channels: int,
) -> dict[str, torch.Tensor]:
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError(
            "static feature extraction requires a sequential source loader; "
            "a shuffled sampler would misalign cached samples and external labels"
        )
    batches = {key: [] for key in TIMEMOSAIC_GRAPH_STATIC_KEYS}
    for loader_batch in tqdm(
        loader,
        desc="Extract TimeMosaic graph static features",
        leave=False,
    ):
        if len(loader_batch) == 1:
            signals = loader_batch[0]
            lengths = None
        elif len(loader_batch) == 2:
            signals, lengths = loader_batch
        else:
            raise ValueError(
                "feature loaders must yield (signals,) or (signals, lengths)"
            )
        features = extract_timemosaic_graph_feature_batch(
            signals,
            vision_model,
            mantis_model,
            device,
            window_size=window_size,
            stride=stride,
            encode_batch_size=encode_batch_size,
            lengths=lengths,
        )
        for key in TIMEMOSAIC_GRAPH_STATIC_KEYS:
            batches[key].append(features[key])
    if not batches["raw_windows"]:
        raise ValueError("cannot extract features from an empty split")
    try:
        bundle = {key: torch.cat(value, dim=0) for key, value in batches.items()}
    except RuntimeError as error:
        raise ValueError(
            "loader batches produced different temporal-window counts; use a "
            "split loader with one consistent padded time width"
        ) from error
    return _validate_static_bundle(
        bundle,
        expected_window_size=window_size,
        expected_channels=expected_channels,
    )


def _get_static_split(
    split_name,
    loader,
    labels,
    vision_model,
    mantis_model,
    device,
    *,
    window_size: int,
    stride: int,
    encode_batch_size: int,
    feature_cache_dir,
    feature_cache_signature,
    expected_channels: int,
) -> dict[str, torch.Tensor]:
    cache_path = None
    if feature_cache_dir:
        cache_path = (
            Path(feature_cache_dir) / f"timemosaic_graph_{split_name}.npz"
        )
        cached = load_timemosaic_graph_feature_cache(
            cache_path,
            labels,
            feature_cache_signature,
            window_size=window_size,
            stride=stride,
            expected_channels=expected_channels,
        )
        if cached is not None:
            return cached
    bundle = _extract_static_split(
        loader,
        vision_model,
        mantis_model,
        device,
        window_size=window_size,
        stride=stride,
        encode_batch_size=encode_batch_size,
        expected_channels=expected_channels,
    )
    if len(bundle["raw_windows"]) != len(_one_dimensional_labels(labels, "labels")):
        raise ValueError(
            f"{split_name} loader sample count does not match supplied labels"
        )
    if cache_path is not None:
        save_timemosaic_graph_feature_cache(
            cache_path,
            bundle,
            labels,
            feature_cache_signature,
            window_size=window_size,
            stride=stride,
        )
        print(f"Saved TimeMosaic graph static features: {cache_path}")
    return bundle


class TimeMosaicGraphClassifier(nn.Module):
    """Trainable adaptive renderer plus visual/temporal fusion classifier."""

    def __init__(
        self,
        visual_dim: int,
        temporal_dim: int,
        num_channels: int,
        num_classes: int,
        *,
        fusion_dim: int = 512,
        fusion_heads: int = 4,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 512,
        classifier_num_layers: int = 2,
        channel_hidden_dim: int = 64,
        alignment_dim: int = 256,
        alignment_temperature: float = 0.1,
        graph_image_size: int = 224,
        graph_token_grid: int = OPENCLIP_SPATIAL_GRID_SIZE,
        adaptive_channel_mix: float = 0.35,
        adaptive_temperature: float = 0.5,
        freeze_adaptive_gate: bool = False,
        cross_attention_ffn_hidden_dim: int | None = None,
        cross_attention_bias: bool = True,
        adaptive_gate_checkpoint=None,
        strict_gate_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self._expected_vision_encoder_contract: Mapping[str, Any] | None = None
        self._validated_vision_encoder_object_id: int | None = None
        if (
            isinstance(graph_token_grid, bool)
            or not isinstance(graph_token_grid, int)
            or graph_token_grid < 2
        ):
            raise ValueError("graph_token_grid must be an integer of at least 2")
        self.graph_token_grid = int(graph_token_grid)
        self.renderer = TimeMosaicAdaptiveActivityGraphRenderer(
            img_size=graph_image_size,
            channel_mix=adaptive_channel_mix,
            temperature=adaptive_temperature,
            gate_checkpoint=adaptive_gate_checkpoint,
            freeze_gate=freeze_adaptive_gate,
            strict_gate_checkpoint=strict_gate_checkpoint,
        )
        self.fusion = TimeMosaicPatchFusionModule(
            visual_dim=visual_dim,
            temporal_dim=temporal_dim,
            num_channels=num_channels,
            num_classes=num_classes,
            fusion_dim=fusion_dim,
            fusion_heads=fusion_heads,
            dropout=dropout,
            classifier_hidden_dim=classifier_hidden_dim,
            classifier_num_layers=classifier_num_layers,
            channel_hidden_dim=channel_hidden_dim,
            alignment_dim=alignment_dim,
            alignment_temperature=alignment_temperature,
            cross_attention_ffn_hidden_dim=cross_attention_ffn_hidden_dim,
            cross_attention_bias=cross_attention_bias,
        )
        # Gate checkpoint provenance is recorded separately; reconstruction
        # uses the state dict and must not depend on the original path.
        self.constructor_configuration: dict[str, Any] = {
            "visual_dim": int(visual_dim),
            "temporal_dim": int(temporal_dim),
            "num_channels": int(num_channels),
            "num_classes": int(num_classes),
            "fusion_dim": int(fusion_dim),
            "fusion_heads": int(fusion_heads),
            "dropout": float(dropout),
            "classifier_hidden_dim": int(classifier_hidden_dim),
            "classifier_num_layers": int(classifier_num_layers),
            "channel_hidden_dim": int(channel_hidden_dim),
            "alignment_dim": int(alignment_dim),
            "alignment_temperature": float(alignment_temperature),
            "graph_image_size": int(graph_image_size),
            "graph_token_grid": self.graph_token_grid,
            "adaptive_channel_mix": float(adaptive_channel_mix),
            "adaptive_temperature": float(adaptive_temperature),
            "freeze_adaptive_gate": bool(freeze_adaptive_gate),
            "cross_attention_ffn_hidden_dim": (
                None
                if cross_attention_ffn_hidden_dim is None
                else int(cross_attention_ffn_hidden_dim)
            ),
            "cross_attention_bias": bool(cross_attention_bias),
        }
        fusion_configuration = dict(self.fusion.configuration)
        fusion_configuration["visual_token_count"] = self.graph_token_grid**2
        self.configuration: dict[str, Any] = {
            **self.constructor_configuration,
            "architecture": TIMEMOSAIC_GRAPH_ARCHITECTURE,
            "renderer": self.renderer.provenance(),
            "fusion": fusion_configuration,
            "graph_count_per_valid_window": 1,
            "graph_spatial_grid_size": self.graph_token_grid,
            "graph_spatial_token_count": self.graph_token_grid**2,
            "vision_parameters_in_checkpoint": False,
        }

    def get_config(self) -> dict[str, Any]:
        return dict(self.constructor_configuration)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]):
        return cls(**dict(config))

    def bind_vision_encoder_contract(
        self,
        contract: Mapping[str, Any],
        *,
        validated_encoder=None,
    ) -> None:
        """Bind the frozen OpenCLIP identity stored by a checkpoint."""
        if not isinstance(contract, Mapping):
            raise TypeError("vision encoder contract must be a mapping")
        self._expected_vision_encoder_contract = dict(contract)
        self._validated_vision_encoder_object_id = (
            id(validated_encoder) if validated_encoder is not None else None
        )

    def _validate_bound_vision_encoder(self, vision_model) -> None:
        if self._expected_vision_encoder_contract is None:
            return
        if self._validated_vision_encoder_object_id == id(vision_model):
            return
        _assert_encoder_contract(
            self._expected_vision_encoder_contract,
            vision_model,
            "vision",
        )
        self._validated_vision_encoder_object_id = id(vision_model)

    @staticmethod
    def _vision_hidden(
        vision_model,
        images: torch.Tensor,
        use_gradient_checkpointing: bool,
    ) -> torch.Tensor:
        if (
            use_gradient_checkpointing
            and torch.is_grad_enabled()
            and images.requires_grad
        ):
            hidden = gradient_checkpoint(
                vision_model.forward_vit,
                images,
                use_reentrant=False,
            )
        else:
            hidden = vision_model.forward_vit(images)
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise ValueError(
                "adaptive graph training requires OpenCLIP hidden tokens "
                f"[batch, CLS+patches, dim], got {getattr(hidden, 'shape', None)}. "
                "Use a fixed OpenCLIP layer_idx rather than stacked layers."
            )
        if (
            torch.is_grad_enabled()
            and images.requires_grad
            and not hidden.requires_grad
        ):
            raise RuntimeError(
                "the visual encoder detached its image input; the adaptive "
                "granularity gate cannot be trained through this encoder"
            )
        return hidden

    def _encode_adaptive_graphs(
        self,
        raw_windows: torch.Tensor,
        valid_lengths: torch.Tensor,
        patch_mask: torch.Tensor,
        vision_model,
        *,
        encode_batch_size: int,
        spatial_grid_size: int,
        use_gradient_checkpointing: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if raw_windows.ndim != 4:
            raise ValueError("raw_windows must have shape [B,N,C,T]")
        if valid_lengths.shape != raw_windows.shape[:2]:
            raise ValueError("valid_lengths must share raw [B,N] axes")
        if patch_mask.shape != raw_windows.shape[:2]:
            raise ValueError("patch_mask must share raw [B,N] axes")
        if encode_batch_size <= 0:
            raise ValueError("encode_batch_size must be positive")

        batch_size, patch_count, channels, window_size = raw_windows.shape
        flat_raw = raw_windows.reshape(-1, channels, window_size)
        flat_lengths = valid_lengths.reshape(-1)
        flat_mask = patch_mask.reshape(-1).to(dtype=torch.bool)
        valid_indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError("batch contains no valid temporal windows")

        token_chunks: list[torch.Tensor] = []
        probability_sums: list[torch.Tensor] = []
        hard_sums: list[torch.Tensor] = []
        decision_counts: list[torch.Tensor] = []
        entropy_sums: list[torch.Tensor] = []
        for start in range(0, int(valid_indices.numel()), encode_batch_size):
            chunk_indices = valid_indices[start : start + encode_batch_size]
            chunk_raw = flat_raw.index_select(0, chunk_indices)
            chunk_lengths = flat_lengths.index_select(0, chunk_indices)
            images, diagnostics = self.renderer(
                chunk_raw,
                valid_lengths=chunk_lengths,
                return_diagnostics=True,
            )
            hidden = self._vision_hidden(
                vision_model,
                images,
                use_gradient_checkpointing,
            )
            spatial = compress_openclip_spatial_tokens(
                hidden,
                output_grid_size=spatial_grid_size,
            )
            spatial = vision_model.project_pooled_representation(spatial)
            if not torch.is_tensor(spatial) or spatial.ndim != 3:
                raise ValueError(
                    "visual projection must preserve [batch, spatial, dim]"
                )
            token_chunks.append(F.normalize(spatial.float(), dim=-1, eps=1e-8))

            region_mask = diagnostics["region_valid_mask"]
            clean_probs = diagnostics["clean_probs"].float()
            hard_weights = diagnostics["hard_weights"].float()
            expanded_mask = region_mask.unsqueeze(-1)
            probability_sums.append(
                (clean_probs * expanded_mask).sum(dim=(0, 1, 2))
            )
            hard_sums.append(
                (hard_weights * expanded_mask).sum(dim=(0, 1, 2))
            )
            decision_count = region_mask.sum().to(dtype=torch.float32)
            decision_counts.append(decision_count)
            entropy = -(
                clean_probs.clamp_min(1e-8).log() * clean_probs
            ).sum(dim=-1)
            entropy_sums.append((entropy * region_mask).sum())

        valid_tokens = torch.cat(token_chunks, dim=0)
        if valid_tokens.shape[-1] != self.fusion.visual_dim:
            raise ValueError(
                "graph spatial token dimension does not match cached line "
                f"dimension: graph={valid_tokens.shape[-1]}, "
                f"line={self.fusion.visual_dim}"
            )
        flat_tokens = valid_tokens.new_zeros(
            (
                batch_size * patch_count,
                valid_tokens.shape[1],
                valid_tokens.shape[2],
            )
        )
        flat_tokens = flat_tokens.index_copy(0, valid_indices, valid_tokens)

        probability_sum = torch.stack(probability_sums).sum(dim=0)
        hard_sum = torch.stack(hard_sums).sum(dim=0)
        decision_count = torch.stack(decision_counts).sum().clamp_min(1.0)
        soft_usage = probability_sum / decision_count
        hard_usage = hard_sum / decision_count
        uniform = torch.full_like(soft_usage, 1.0 / soft_usage.numel())
        balance_loss = (soft_usage - uniform).abs().mean()
        mean_entropy = torch.stack(entropy_sums).sum() / decision_count
        graph_tokens = flat_tokens.reshape(
            batch_size,
            patch_count,
            valid_tokens.shape[1],
            valid_tokens.shape[2],
        )
        return graph_tokens, {
            "selector_balance_loss": balance_loss,
            "selector_soft_usage": soft_usage,
            "selector_hard_usage": hard_usage,
            "selector_probability_sum": probability_sum,
            "selector_hard_count": hard_sum,
            "selector_valid_count": decision_count,
            "selector_mean_entropy": mean_entropy,
        }

    def forward(
        self,
        raw_windows: torch.Tensor,
        line_tokens: torch.Tensor,
        mantis_channel_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        valid_fraction: torch.Tensor,
        valid_lengths: torch.Tensor,
        vision_model,
        *,
        visual_encode_batch_size: int = 16,
        graph_spatial_grid_size: int | None = None,
        vision_gradient_checkpointing: bool = True,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        self._validate_bound_vision_encoder(vision_model)
        if graph_spatial_grid_size is None:
            graph_spatial_grid_size = self.graph_token_grid
        elif graph_spatial_grid_size != self.graph_token_grid:
            raise ValueError(
                "graph_spatial_grid_size cannot override the checkpoint/model "
                f"contract {self.graph_token_grid}; got {graph_spatial_grid_size}"
            )
        graph_tokens, selector_details = self._encode_adaptive_graphs(
            raw_windows,
            valid_lengths,
            patch_mask,
            vision_model,
            encode_batch_size=visual_encode_batch_size,
            spatial_grid_size=graph_spatial_grid_size,
            use_gradient_checkpointing=vision_gradient_checkpointing,
        )
        logits, details = self.fusion(
            line_tokens,
            graph_tokens,
            mantis_channel_tokens,
            patch_mask,
            valid_fraction,
            return_attention_weights=return_attention_weights,
        )
        details.update(selector_details)
        return logits, details


def _build_static_loader(bundle, labels, indices, batch_size, shuffle):
    dataset = TensorDataset(
        bundle["raw_windows"],
        bundle["line_tokens"],
        bundle["mantis_channel_tokens"],
        bundle["patch_mask"],
        bundle["valid_fraction"],
        bundle["valid_lengths"],
        torch.as_tensor(labels, dtype=torch.long),
        torch.arange(len(labels), dtype=torch.long),
    )
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _forward_training_batch(
    model,
    batch,
    vision_model,
    device,
    *,
    visual_encode_batch_size: int,
    graph_spatial_grid_size: int,
    vision_gradient_checkpointing: bool,
):
    raw, line, mantis, mask, fraction, lengths, labels, sample_indices = batch
    logits, details = model(
        raw.to(device=device, dtype=torch.float32),
        line.to(device=device, dtype=torch.float32),
        mantis.to(device=device, dtype=torch.float32),
        mask.to(device=device, dtype=torch.bool),
        fraction.to(device=device, dtype=torch.float32),
        lengths.to(device=device, dtype=torch.long),
        vision_model,
        visual_encode_batch_size=visual_encode_batch_size,
        graph_spatial_grid_size=graph_spatial_grid_size,
        vision_gradient_checkpointing=vision_gradient_checkpointing,
    )
    return logits, details, labels.to(device=device, dtype=torch.long), sample_indices


def _run_epoch(
    model,
    loader,
    vision_model,
    optimizer,
    criterion,
    device,
    *,
    alignment_weight: float,
    selector_balance_weight: float,
    visual_encode_batch_size: int,
    graph_spatial_grid_size: int,
    vision_gradient_checkpointing: bool,
):
    model.train()
    totals = {
        "loss": 0.0,
        "task_loss": 0.0,
        "alignment_loss": 0.0,
        "selector_balance_loss": 0.0,
        "selector_mean_entropy": 0.0,
    }
    probability_sum = None
    hard_count = None
    valid_count = 0.0
    sample_count = 0
    for batch in tqdm(loader, desc="Train adaptive Activity Graph", leave=False):
        optimizer.zero_grad(set_to_none=True)
        logits, details, labels, _ = _forward_training_batch(
            model,
            batch,
            vision_model,
            device,
            visual_encode_batch_size=visual_encode_batch_size,
            graph_spatial_grid_size=graph_spatial_grid_size,
            vision_gradient_checkpointing=vision_gradient_checkpointing,
        )
        task_loss = criterion(logits, labels).mean()
        alignment_loss = details["alignment_loss"]
        selector_balance_loss = details["selector_balance_loss"]
        loss = (
            task_loss
            + alignment_weight * alignment_loss
            + selector_balance_weight * selector_balance_loss
        )
        if not torch.isfinite(loss):
            raise ValueError("adaptive Activity Graph training produced non-finite loss")
        loss.backward()
        trainable_gate_parameters = [
            parameter
            for parameter in model.renderer.gate.region_cls.parameters()
            if parameter.requires_grad
        ]
        if trainable_gate_parameters and not all(
            parameter.grad is not None for parameter in trainable_gate_parameters
        ):
            raise RuntimeError(
                "adaptive gate did not receive gradients through the online "
                "Activity Graph path"
            )
        if trainable_gate_parameters and not any(
            bool((parameter.grad.detach().abs() > 0).any())
            for parameter in trainable_gate_parameters
        ):
            raise RuntimeError(
                "adaptive gate gradients are all zero; verify the online "
                "OpenCLIP-to-renderer path with selector_balance_weight=0"
            )
        if any(
            not torch.isfinite(parameter.grad).all()
            for parameter in trainable_gate_parameters
        ):
            raise ValueError("adaptive gate produced non-finite gradients")
        optimizer.step()

        count = int(labels.shape[0])
        sample_count += count
        totals["loss"] += float(loss.detach()) * count
        totals["task_loss"] += float(task_loss.detach()) * count
        totals["alignment_loss"] += float(alignment_loss.detach()) * count
        totals["selector_balance_loss"] += (
            float(selector_balance_loss.detach()) * count
        )
        totals["selector_mean_entropy"] += (
            float(details["selector_mean_entropy"].detach()) * count
        )
        batch_probability_sum = details["selector_probability_sum"].detach()
        batch_hard_count = details["selector_hard_count"].detach()
        probability_sum = (
            batch_probability_sum
            if probability_sum is None
            else probability_sum + batch_probability_sum
        )
        hard_count = (
            batch_hard_count
            if hard_count is None
            else hard_count + batch_hard_count
        )
        valid_count += float(details["selector_valid_count"].detach())
        del logits, details, loss, task_loss, alignment_loss, selector_balance_loss
    if sample_count == 0 or probability_sum is None or hard_count is None:
        raise ValueError("cannot train on an empty feature split")
    statistics = {key: value / sample_count for key, value in totals.items()}
    statistics["selector_soft_usage"] = (
        probability_sum / max(valid_count, 1.0)
    ).cpu().tolist()
    statistics["selector_hard_usage"] = (
        hard_count / max(valid_count, 1.0)
    ).cpu().tolist()
    statistics["selector_valid_decisions"] = int(round(valid_count))
    return statistics


@torch.no_grad()
def _evaluate(
    model,
    loader,
    classes,
    vision_model,
    device,
    description,
    *,
    sample_subject_ids=None,
    visual_encode_batch_size: int,
    graph_spatial_grid_size: int,
):
    model.eval()
    sample_indices_all = []
    y_true_all = []
    y_pred_all = []
    y_score_all = []
    probability_sum = None
    hard_count = None
    valid_count = 0.0
    for batch in tqdm(loader, desc=description, leave=False):
        logits, details, labels, sample_indices = _forward_training_batch(
            model,
            batch,
            vision_model,
            device,
            visual_encode_batch_size=visual_encode_batch_size,
            graph_spatial_grid_size=graph_spatial_grid_size,
            vision_gradient_checkpointing=False,
        )
        probabilities = torch.softmax(logits.float(), dim=-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("adaptive Activity Graph classifier returned non-finite scores")
        sample_indices_all.append(sample_indices.numpy())
        y_true_all.append(labels.cpu().numpy())
        y_pred_all.append(probabilities.argmax(dim=-1).cpu().numpy())
        y_score_all.append(probabilities.cpu().numpy())
        batch_probability_sum = details["selector_probability_sum"].detach()
        batch_hard_count = details["selector_hard_count"].detach()
        probability_sum = (
            batch_probability_sum
            if probability_sum is None
            else probability_sum + batch_probability_sum
        )
        hard_count = (
            batch_hard_count
            if hard_count is None
            else hard_count + batch_hard_count
        )
        valid_count += float(details["selector_valid_count"].detach())
        del logits, details, probabilities
    if not y_true_all or probability_sum is None or hard_count is None:
        raise ValueError("cannot evaluate an empty feature split")
    details = {
        "sample_index": np.concatenate(sample_indices_all),
        "y_true": np.concatenate(y_true_all),
        "y_pred": np.concatenate(y_pred_all),
        "y_score": np.concatenate(y_score_all),
        "selector_soft_usage": (
            probability_sum / max(valid_count, 1.0)
        ).cpu().numpy(),
        "selector_hard_usage": (
            hard_count / max(valid_count, 1.0)
        ).cpu().numpy(),
        "selector_valid_decisions": int(round(valid_count)),
    }
    window_metrics = compute_metrics_from_predictions(
        details["y_true"],
        details["y_pred"],
        details["y_score"],
        np.arange(len(classes)),
    )
    subject_metrics = None
    if sample_subject_ids is not None:
        subject_metrics, subject_details = _aggregate_subject_predictions(
            details["y_true"],
            details["y_score"],
            details["sample_index"],
            sample_subject_ids,
            classes,
        )
        details.update(subject_details)
    return _merge_subject_metrics(window_metrics, subject_metrics), details


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _json_safe(value):
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json_dump(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(
                _json_safe(payload),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_checkpoint(
    artifact_dir,
    model,
    classes,
    training_history,
    best_epoch,
    validation_metrics,
    *,
    checkpoint_metric_requested,
    checkpoint_metric_effective,
    validation_subject_ids,
    feature_cache_signature,
    outer_patch_size,
    outer_patch_stride,
    graph_spatial_grid_size,
    alignment_weight,
    selector_balance_weight,
    early_stop_configuration,
    scheduler_configuration,
    external_encoder_contracts,
) -> Path:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "timemosaic_graph_checkpoint.pt"
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "schema_version": TIMEMOSAIC_GRAPH_CHECKPOINT_SCHEMA_VERSION,
        "architecture": TIMEMOSAIC_GRAPH_ARCHITECTURE,
        "model_state_dict": _cpu_state_dict(model),
        "model_constructor_configuration": model.get_config(),
        "model_configuration": dict(model.configuration),
        "external_encoder_contracts": _json_safe(external_encoder_contracts),
        "classes": _json_safe(np.asarray(classes)),
        "best_epoch": int(best_epoch),
        "validation_metrics": _json_safe(validation_metrics),
        "checkpoint_metric_requested": checkpoint_metric_requested,
        "checkpoint_metric_effective": checkpoint_metric_effective,
        "validation_subject_ids_sha256": _subject_ids_digest(
            validation_subject_ids
        ),
        "feature_cache": {
            "schema_version": TIMEMOSAIC_GRAPH_CACHE_SCHEMA_VERSION,
            "architecture": TIMEMOSAIC_GRAPH_CACHE_ARCHITECTURE,
            "signature": str(feature_cache_signature or ""),
            "keys": list(TIMEMOSAIC_GRAPH_STATIC_KEYS),
        },
        "protocol": {
            "outer_patch_size": int(outer_patch_size),
            "outer_patch_stride": int(outer_patch_stride),
            "tail_policy": PATCH_TAIL_POLICY,
            "graph_spatial_grid_size": int(graph_spatial_grid_size),
            "graphs_per_valid_window": 1,
            "vision_encoder_parameters_frozen": True,
            "vision_encoder_parameters_included": False,
            "mantis_encoder_parameters_included": False,
            "line_and_mantis_features_cached": True,
            "activity_graph_generated_online": True,
            "alignment_weight": float(alignment_weight),
            "selector_balance_weight": float(selector_balance_weight),
        },
        "early_stopping": _json_safe(early_stop_configuration),
        "lr_scheduler": _json_safe(scheduler_configuration),
        "training_history": _json_safe(training_history),
    }
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_timemosaic_graph_checkpoint(
    path,
    *,
    map_location="cpu",
    strict: bool = True,
    vision_model=None,
    mantis_model=None,
):
    """Reconstruct the trainable model and bind its external encoder contract.

    Supplying ``vision_model`` validates it immediately.  If omitted, the
    reconstructed model validates the first visual encoder passed to
    :meth:`TimeMosaicGraphClassifier.forward`.  ``mantis_model`` can be
    supplied to validate the encoder used to create new cached Mantis tokens.
    """
    try:
        payload = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError("TimeMosaic graph checkpoint must contain a mapping")
    if payload.get("schema_version") != TIMEMOSAIC_GRAPH_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("TimeMosaic graph checkpoint schema mismatch")
    if payload.get("architecture") != TIMEMOSAIC_GRAPH_ARCHITECTURE:
        raise ValueError("TimeMosaic graph checkpoint architecture mismatch")
    model = TimeMosaicGraphClassifier.from_config(
        payload["model_constructor_configuration"]
    )
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    contracts = payload.get("external_encoder_contracts")
    if not isinstance(contracts, Mapping) or not isinstance(
        contracts.get("vision"), Mapping
    ):
        raise ValueError("checkpoint is missing its external vision contract")
    if vision_model is not None:
        _assert_encoder_contract(contracts["vision"], vision_model, "vision")
    model.bind_vision_encoder_contract(
        contracts["vision"],
        validated_encoder=vision_model,
    )
    if mantis_model is not None:
        if not isinstance(contracts.get("mantis"), Mapping):
            raise ValueError("checkpoint is missing its external Mantis contract")
        _assert_encoder_contract(contracts["mantis"], mantis_model, "mantis")
    return model, payload


def validate_timemosaic_graph_encoders(
    checkpoint_payload: Mapping[str, Any],
    *,
    vision_model,
    mantis_model=None,
) -> None:
    """Validate encoders before creating inference features from a checkpoint."""
    if not isinstance(checkpoint_payload, Mapping):
        raise TypeError("checkpoint_payload must be a mapping")
    contracts = checkpoint_payload.get("external_encoder_contracts")
    if not isinstance(contracts, Mapping):
        raise ValueError("checkpoint is missing external encoder contracts")
    if not isinstance(contracts.get("vision"), Mapping):
        raise ValueError("checkpoint is missing its external vision contract")
    _assert_encoder_contract(contracts["vision"], vision_model, "vision")
    if mantis_model is not None:
        if not isinstance(contracts.get("mantis"), Mapping):
            raise ValueError("checkpoint is missing its external Mantis contract")
        _assert_encoder_contract(contracts["mantis"], mantis_model, "mantis")


def train_timemosaic_graph_classifier(
    train_loader,
    train_labels,
    test_loader,
    test_labels,
    channels,
    device,
    batch_size,
    random_seed,
    val_ratio,
    hidden_dim,
    num_layers,
    dropout,
    lr,
    weight_decay,
    epochs,
    early_stop_patience,
    fusion_dim,
    fusion_heads,
    alignment_dim,
    alignment_temperature,
    alignment_weight,
    outer_patch_size=64,
    outer_patch_stride=64,
    visual_encode_batch_size=16,
    class_weight="none",
    vision_model=None,
    mantis_model=None,
    val_loader=None,
    val_labels=None,
    feature_cache_dir=None,
    feature_cache_signature=None,
    gate_temperature=0.5,
    channel_mix=0.35,
    selector_balance_weight=0.001,
    checkpoint_metric="auto",
    channel_hidden_dim=64,
    artifact_dir=None,
    cross_attention_ffn_hidden_dim=None,
    cross_attention_bias=True,
    early_stop_strategy="raw_selection_key",
    early_stop_min_epochs=0,
    early_stop_warmup_epochs=0,
    early_stop_ema_decay=0.6,
    early_stop_min_delta=0.0,
    lr_scheduler_type="none",
    lr_scheduler_patience=4,
    lr_scheduler_factor=0.5,
    lr_scheduler_min_lr=1e-6,
    graph_image_size=224,
    graph_token_grid=OPENCLIP_SPATIAL_GRID_SIZE,
    gate_checkpoint=None,
    freeze_gate=False,
    strict_gate_checkpoint=True,
    vision_gradient_checkpointing=True,
):
    """Train the corrected adaptive-graph path and return the legacy four-tuple.

    Returns ``(validation_metrics, test_metrics, train_indices, val_indices)``.
    The source loaders retain the legacy contract: each batch must be either
    ``(signals,)`` or ``(signals, lengths)`` and labels are supplied separately.
    """
    device = torch.device(device)
    positive_integer_arguments = {
        "channels": channels,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "epochs": epochs,
        "fusion_dim": fusion_dim,
        "fusion_heads": fusion_heads,
        "alignment_dim": alignment_dim,
        "outer_patch_size": outer_patch_size,
        "outer_patch_stride": outer_patch_stride,
        "visual_encode_batch_size": visual_encode_batch_size,
        "channel_hidden_dim": channel_hidden_dim,
        "graph_image_size": graph_image_size,
        "graph_token_grid": graph_token_grid,
    }
    for name, value in positive_integer_arguments.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
    if graph_token_grid != OPENCLIP_SPATIAL_GRID_SIZE:
        raise ValueError(
            "graph_token_grid must be 4 so the corrected protocol retains "
            "exactly 16 Activity Graph K/V tokens"
        )
    if outer_patch_stride > outer_patch_size:
        raise ValueError("outer_patch_stride cannot exceed outer_patch_size")
    if fusion_dim % fusion_heads != 0:
        raise ValueError("fusion_dim must be divisible by fusion_heads")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must lie in [0, 1)")
    for name, value in (
        ("lr", lr),
        ("alignment_temperature", alignment_temperature),
        ("gate_temperature", gate_temperature),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in (
        ("weight_decay", weight_decay),
        ("alignment_weight", alignment_weight),
        ("selector_balance_weight", selector_balance_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(channel_mix) or not (
        0.0 <= channel_mix <= 1.0
    ):
        raise ValueError("channel_mix must lie in [0, 1]")
    if vision_model is None or mantis_model is None:
        raise ValueError("vision_model and mantis_model are required")
    if feature_cache_dir and not str(feature_cache_signature or "").strip():
        raise ValueError(
            "feature_cache_dir requires a non-empty feature_cache_signature"
        )
    vision_backbone = getattr(vision_model, "vit", None)
    vision_transformer = getattr(vision_backbone, "transformer", None)
    if not (
        callable(getattr(vision_backbone, "_embeds", None))
        and hasattr(vision_transformer, "resblocks")
    ):
        raise ValueError(
            "patch_timemosaic_graph requires the differentiable OpenCLIP visual "
            "backbone; Hugging Face image processors detach rendered tensors"
        )
    if freeze_gate and gate_checkpoint is None:
        raise ValueError("freeze_gate requires gate_checkpoint")
    if (
        not math.isfinite(lr_scheduler_min_lr)
        or lr_scheduler_min_lr < 0.0
        or lr_scheduler_min_lr > lr
    ):
        raise ValueError("lr_scheduler_min_lr must lie in [0, lr]")

    train_labels = _one_dimensional_labels(train_labels, "train_labels")
    test_labels = _one_dimensional_labels(test_labels, "test_labels")
    if len(train_labels) != len(train_loader.dataset):
        raise ValueError("train_labels must match train_loader.dataset")
    if len(test_labels) != len(test_loader.dataset):
        raise ValueError("test_labels must match test_loader.dataset")
    has_fixed_validation = val_loader is not None or val_labels is not None
    if has_fixed_validation and (val_loader is None or val_labels is None):
        raise ValueError("val_loader and val_labels must be supplied together")
    if has_fixed_validation:
        val_labels = _one_dimensional_labels(val_labels, "val_labels")
        if len(val_labels) != len(val_loader.dataset):
            raise ValueError("val_labels must match val_loader.dataset")

    set_random_seed(random_seed)
    vision_model.requires_grad_(False)
    mantis_model.requires_grad_(False)
    vision_model.eval()
    mantis_model.eval()
    external_encoder_contracts = {
        "vision": _encoder_contract(
            vision_model,
            "vision",
            source_identity=_signature_model_identity(
                feature_cache_signature,
                "vit_1_identity",
            ),
        ),
        "mantis": _encoder_contract(
            mantis_model,
            "mantis",
            source_identity=_signature_model_identity(
                feature_cache_signature,
                "mantis_identity",
            ),
        ),
    }

    if has_fixed_validation:
        train_indices = list(range(len(train_loader.dataset)))
        val_indices = list(range(len(val_loader.dataset)))
    else:
        train_indices, val_indices = get_split(
            train_loader.dataset,
            frac=val_ratio,
            random_seed=random_seed,
        )
    if not train_indices or not val_indices:
        raise ValueError("training and validation splits must both be non-empty")

    train_subject_ids = _loader_sample_subject_ids(train_loader)
    test_subject_ids = _loader_sample_subject_ids(test_loader)
    validation_subject_ids = (
        _loader_sample_subject_ids(val_loader)
        if has_fixed_validation
        else train_subject_ids
    )
    checkpoint_metric_effective = _resolve_checkpoint_metric(
        checkpoint_metric,
        train_subject_ids,
        validation_subject_ids,
        train_indices,
        val_indices,
    )
    print(
        "TimeMosaic graph checkpoint selection: "
        f"requested={checkpoint_metric} effective={checkpoint_metric_effective}"
    )

    train_label_indices, classes, class_to_index = _labels_to_indices(train_labels)
    if len(classes) < 2:
        raise ValueError("classification requires at least two training classes")
    test_label_indices = _map_labels(test_labels, class_to_index)
    train_features = _get_static_split(
        "train",
        train_loader,
        train_labels,
        vision_model,
        mantis_model,
        device,
        window_size=outer_patch_size,
        stride=outer_patch_stride,
        encode_batch_size=visual_encode_batch_size,
        feature_cache_dir=feature_cache_dir,
        feature_cache_signature=feature_cache_signature,
        expected_channels=channels,
    )
    test_features = _get_static_split(
        "test",
        test_loader,
        test_labels,
        vision_model,
        mantis_model,
        device,
        window_size=outer_patch_size,
        stride=outer_patch_stride,
        encode_batch_size=visual_encode_batch_size,
        feature_cache_dir=feature_cache_dir,
        feature_cache_signature=feature_cache_signature,
        expected_channels=channels,
    )
    if has_fixed_validation:
        val_label_indices = _map_labels(val_labels, class_to_index)
        val_features = _get_static_split(
            "vali",
            val_loader,
            val_labels,
            vision_model,
            mantis_model,
            device,
            window_size=outer_patch_size,
            stride=outer_patch_stride,
            encode_batch_size=visual_encode_batch_size,
            feature_cache_dir=feature_cache_dir,
            feature_cache_signature=feature_cache_signature,
            expected_channels=channels,
        )

    # Reset after optional cold-cache extraction so cache hits and misses start
    # model initialization and stochastic routing from the same RNG state.
    set_random_seed(random_seed)
    fit_loader = _build_static_loader(
        train_features,
        train_label_indices,
        train_indices,
        batch_size,
        shuffle=True,
    )
    if has_fixed_validation:
        validation_loader = _build_static_loader(
            val_features,
            val_label_indices,
            val_indices,
            batch_size,
            shuffle=False,
        )
    else:
        validation_loader = _build_static_loader(
            train_features,
            train_label_indices,
            val_indices,
            batch_size,
            shuffle=False,
        )
    test_feature_loader = _build_static_loader(
        test_features,
        test_label_indices,
        range(len(test_label_indices)),
        batch_size,
        shuffle=False,
    )

    visual_dim = int(train_features["line_tokens"].shape[-1])
    temporal_dim = int(train_features["mantis_channel_tokens"].shape[-1])
    if int(train_features["raw_windows"].shape[2]) != channels:
        raise ValueError("cached raw channel count does not match channels")
    comparison_splits = [("test", test_features)]
    if has_fixed_validation:
        comparison_splits.append(("validation", val_features))
    for split_name, features in comparison_splits:
        if int(features["line_tokens"].shape[-1]) != visual_dim:
            raise ValueError(f"{split_name} line feature dimension mismatch")
        if int(features["mantis_channel_tokens"].shape[-1]) != temporal_dim:
            raise ValueError(f"{split_name} Mantis feature dimension mismatch")

    model = TimeMosaicGraphClassifier(
        visual_dim=visual_dim,
        temporal_dim=temporal_dim,
        num_channels=channels,
        num_classes=len(classes),
        fusion_dim=fusion_dim,
        fusion_heads=fusion_heads,
        dropout=dropout,
        classifier_hidden_dim=hidden_dim,
        classifier_num_layers=num_layers,
        channel_hidden_dim=channel_hidden_dim,
        alignment_dim=alignment_dim,
        alignment_temperature=alignment_temperature,
        graph_image_size=graph_image_size,
        graph_token_grid=graph_token_grid,
        adaptive_channel_mix=channel_mix,
        adaptive_temperature=gate_temperature,
        freeze_adaptive_gate=freeze_gate,
        cross_attention_ffn_hidden_dim=cross_attention_ffn_hidden_dim,
        cross_attention_bias=cross_attention_bias,
        adaptive_gate_checkpoint=gate_checkpoint,
        strict_gate_checkpoint=strict_gate_checkpoint,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = _build_patch_lr_scheduler(
        optimizer,
        lr_scheduler_type,
        patience=lr_scheduler_patience,
        factor=lr_scheduler_factor,
        min_lr=lr_scheduler_min_lr,
    )
    if class_weight == "balanced":
        weights = _balanced_class_weights(
            train_label_indices[np.asarray(train_indices, dtype=np.int64)],
            len(classes),
            device,
        )
    elif class_weight == "none":
        weights = None
    else:
        raise ValueError(f"unsupported class weighting: {class_weight}")
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")

    early_stop_monitor = _EarlyStoppingMonitor(
        strategy=early_stop_strategy,
        patience=early_stop_patience,
        min_epochs=early_stop_min_epochs,
        warmup_epochs=early_stop_warmup_epochs,
        ema_decay=early_stop_ema_decay,
        min_delta=early_stop_min_delta,
    )
    best_selection_key = None
    best_state = None
    best_epoch = 0
    early_stopped = False
    stopped_epoch = None
    history = []
    for epoch in range(epochs):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        statistics = _run_epoch(
            model,
            fit_loader,
            vision_model,
            optimizer,
            criterion,
            device,
            alignment_weight=alignment_weight,
            selector_balance_weight=selector_balance_weight,
            visual_encode_batch_size=visual_encode_batch_size,
            graph_spatial_grid_size=graph_token_grid,
            vision_gradient_checkpointing=vision_gradient_checkpointing,
        )
        validation_metrics, validation_details = _evaluate(
            model,
            validation_loader,
            classes,
            vision_model,
            device,
            "Validate adaptive Activity Graph",
            sample_subject_ids=validation_subject_ids,
            visual_encode_batch_size=visual_encode_batch_size,
            graph_spatial_grid_size=graph_token_grid,
        )
        selection_key = _checkpoint_selection_key(
            validation_metrics,
            checkpoint_metric_effective,
        )
        checkpoint_improved = (
            best_selection_key is None or selection_key > best_selection_key
        )
        early_stop_state = early_stop_monitor.update(selection_key, epoch + 1)
        scheduler_stepped = False
        if scheduler is not None and epoch + 1 > early_stop_warmup_epochs:
            scheduler.step(float(selection_key[0]))
            scheduler_stepped = True
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_record = {
            "epoch": epoch + 1,
            **statistics,
            "learning_rate": learning_rate,
            "next_learning_rate": next_learning_rate,
            "lr_scheduler_stepped": scheduler_stepped,
            "validation_macro_f1": float(validation_metrics["macro_f1"]),
            "validation_subject_macro_f1": validation_metrics.get(
                "subject_macro_f1"
            ),
            "validation_selector_soft_usage": validation_details[
                "selector_soft_usage"
            ],
            "validation_selector_hard_usage": validation_details[
                "selector_hard_usage"
            ],
            "checkpoint_selection_key": list(selection_key),
            "checkpoint_improved": checkpoint_improved,
            "early_stop_raw_score": early_stop_state["raw_score"],
            "early_stop_smoothed_score": early_stop_state["smoothed_score"],
            "early_stop_best_score": early_stop_state["best_score"],
            "early_stop_improved": early_stop_state["improved"],
            "early_stop_epochs_without_improvement": early_stop_state[
                "epochs_without_improvement"
            ],
        }
        history.append(epoch_record)
        print(
            f"Adaptive graph epoch {epoch + 1}/{epochs} | "
            f"loss={statistics['loss']:.4f} | "
            f"task={statistics['task_loss']:.4f} | "
            f"alignment={statistics['alignment_loss']:.4f} | "
            f"selector_balance={statistics['selector_balance_loss']:.4f} | "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f} | "
            f"val_subject_macro_f1="
            f"{validation_metrics.get('subject_macro_f1', float('nan')):.4f} | "
            f"lr={learning_rate:.3e}->{next_learning_rate:.3e} | "
            f"hard_usage={statistics['selector_hard_usage']}"
        )
        if checkpoint_improved:
            best_selection_key = selection_key
            best_state = _cpu_state_dict(model)
            best_epoch = epoch + 1
        if early_stop_state["should_stop"]:
            early_stopped = True
            stopped_epoch = epoch + 1
            print(f"Adaptive graph early stopping at epoch {stopped_epoch}")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    val_metrics, val_details = _evaluate(
        model,
        validation_loader,
        classes,
        vision_model,
        device,
        "Evaluate best adaptive graph validation",
        sample_subject_ids=validation_subject_ids,
        visual_encode_batch_size=visual_encode_batch_size,
        graph_spatial_grid_size=graph_token_grid,
    )
    test_metrics, test_details = _evaluate(
        model,
        test_feature_loader,
        classes,
        vision_model,
        device,
        "Evaluate best adaptive graph test",
        sample_subject_ids=test_subject_ids,
        visual_encode_batch_size=visual_encode_batch_size,
        graph_spatial_grid_size=graph_token_grid,
    )

    validation_subject_ids_used = (
        np.asarray(validation_subject_ids)[np.asarray(val_indices)]
        if validation_subject_ids is not None
        else None
    )
    if artifact_dir is not None:
        early_stop_configuration = {
            "strategy": early_stop_strategy,
            "patience": early_stop_patience,
            "min_epochs": early_stop_min_epochs,
            "warmup_epochs": early_stop_warmup_epochs,
            "ema_decay": early_stop_ema_decay,
            "min_delta": early_stop_min_delta,
            "best_score": early_stop_monitor.best_score,
            "early_stopped": early_stopped,
            "stopped_epoch": stopped_epoch,
        }
        scheduler_configuration = {
            "type": lr_scheduler_type,
            "patience": lr_scheduler_patience,
            "factor": lr_scheduler_factor,
            "min_lr": lr_scheduler_min_lr,
            "final_lr": float(optimizer.param_groups[0]["lr"]),
        }
        checkpoint_path = _save_checkpoint(
            artifact_dir,
            model,
            classes,
            history,
            best_epoch,
            val_metrics,
            checkpoint_metric_requested=checkpoint_metric,
            checkpoint_metric_effective=checkpoint_metric_effective,
            validation_subject_ids=validation_subject_ids_used,
            feature_cache_signature=feature_cache_signature,
            outer_patch_size=outer_patch_size,
            outer_patch_stride=outer_patch_stride,
            graph_spatial_grid_size=graph_token_grid,
            alignment_weight=alignment_weight,
            selector_balance_weight=selector_balance_weight,
            early_stop_configuration=early_stop_configuration,
            scheduler_configuration=scheduler_configuration,
            external_encoder_contracts=external_encoder_contracts,
        )
        _atomic_json_dump(
            Path(artifact_dir) / "timemosaic_graph_summary.json",
            {
                "schema_version": TIMEMOSAIC_GRAPH_CHECKPOINT_SCHEMA_VERSION,
                "architecture": TIMEMOSAIC_GRAPH_ARCHITECTURE,
                "checkpoint": checkpoint_path.name,
                "best_epoch": best_epoch,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "validation_selector_soft_usage": val_details[
                    "selector_soft_usage"
                ],
                "validation_selector_hard_usage": val_details[
                    "selector_hard_usage"
                ],
                "test_selector_soft_usage": test_details[
                    "selector_soft_usage"
                ],
                "test_selector_hard_usage": test_details[
                    "selector_hard_usage"
                ],
                "training_history": history,
            },
        )

    return val_metrics, test_metrics, train_indices, val_indices


__all__ = [
    "TIMEMOSAIC_GRAPH_ARCHITECTURE",
    "TIMEMOSAIC_GRAPH_CACHE_ARCHITECTURE",
    "TIMEMOSAIC_GRAPH_CACHE_SCHEMA_VERSION",
    "TIMEMOSAIC_GRAPH_CHECKPOINT_SCHEMA_VERSION",
    "TIMEMOSAIC_GRAPH_STATIC_KEYS",
    "TimeMosaicGraphClassifier",
    "extract_timemosaic_graph_feature_batch",
    "load_timemosaic_graph_checkpoint",
    "load_timemosaic_graph_feature_cache",
    "save_timemosaic_graph_feature_cache",
    "train_timemosaic_graph_classifier",
    "validate_timemosaic_graph_encoders",
]
