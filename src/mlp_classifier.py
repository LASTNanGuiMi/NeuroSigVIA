import json
import math
import os
import re
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from src.classifier import compute_metrics_from_predictions
from src.granularity_selector import AdaptiveTemporalGranularitySelector
from src.utils import (
    get_split,
    resize_mantis_input,
    resize_moment_input,
    set_random_seed,
)


FEATURE_CACHE_SCHEMA_VERSION = 2

# Historical import path retained for external callers.
AdaptiveGranularitySelector = AdaptiveTemporalGranularitySelector


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_classes):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"MLP must contain at least one layer, got {num_layers}.")

        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FusionModule(nn.Module):
    def __init__(
        self,
        branch_dims,
        modal_interaction="concat",
        fusion_dim=512,
        fusion_heads=4,
        cross_attn_query="ts",
        branch_names=None,
        branch_granularity_counts=None,
        branch_granularity_base_indices=None,
        branch_granularity_labels=None,
        granularity_hidden_dim=64,
        granularity_temperature=1.0,
        granularity_base_prior=0.9,
    ):
        super().__init__()

        if not branch_dims:
            raise ValueError("FusionModule requires at least one enabled branch.")
        if modal_interaction not in {
            "concat",
            "concat_attn",
            "cross_attn_gate",
            "masked_pretrain",
        }:
            raise ValueError(f"Unsupported modal_interaction: {modal_interaction}")

        self.modal_interaction = modal_interaction
        self.fusion_dim = fusion_dim
        self.fusion_heads = fusion_heads
        self.cross_attn_query = cross_attn_query
        self.input_branch_dims = [int(dim) for dim in branch_dims]
        branch_count = len(self.input_branch_dims)
        self.branch_names = list(branch_names or [
            f"branch_{index}" for index in range(branch_count)
        ])
        self.branch_granularity_counts = [
            int(count)
            for count in (
                branch_granularity_counts
                if branch_granularity_counts is not None
                else [1] * branch_count
            )
        ]
        self.branch_granularity_base_indices = [
            int(base_index)
            for base_index in (
                branch_granularity_base_indices
                if branch_granularity_base_indices is not None
                else [0] * branch_count
            )
        ]
        if branch_granularity_labels is None:
            branch_granularity_labels = [
                tuple(f"candidate_{index}" for index in range(count))
                for count in self.branch_granularity_counts
            ]
        self.branch_granularity_labels = [
            tuple(labels) for labels in branch_granularity_labels
        ]

        metadata = {
            "branch_names": self.branch_names,
            "branch_granularity_counts": self.branch_granularity_counts,
            "branch_granularity_base_indices": (
                self.branch_granularity_base_indices
            ),
            "branch_granularity_labels": self.branch_granularity_labels,
        }
        for name, values in metadata.items():
            if len(values) != branch_count:
                raise ValueError(
                    f"{name} must contain {branch_count} values, got {len(values)}."
                )
        if len(set(self.branch_names)) != branch_count:
            raise ValueError(f"Branch names must be unique, got {self.branch_names}.")

        self.granularity_selectors = nn.ModuleList()
        self.branch_dims = []
        # Selector initialization uses an isolated RNG stream.  Restoring the
        # caller's RNG state before projections/MLP initialization keeps the
        # shared head bit-comparable to a fixed-granularity run at the same
        # random seed.
        with torch.random.fork_rng(devices=[]):
            for input_dim, count, base_index, labels in zip(
                self.input_branch_dims,
                self.branch_granularity_counts,
                self.branch_granularity_base_indices,
                self.branch_granularity_labels,
            ):
                if count < 1:
                    raise ValueError(
                        f"Granularity candidate count must be positive, got {count}."
                    )
                if input_dim % count != 0:
                    raise ValueError(
                        f"Branch width {input_dim} is not divisible by its {count} "
                        "granularity candidates."
                    )
                if len(labels) != count:
                    raise ValueError(
                        f"Expected {count} granularity labels, got {labels}."
                    )
                feature_dim = input_dim // count
                self.branch_dims.append(feature_dim)
                if count > 1:
                    self.granularity_selectors.append(
                        AdaptiveTemporalGranularitySelector(
                            feature_dim=feature_dim,
                            num_granularities=count,
                            hidden_dim=granularity_hidden_dim,
                            temperature=granularity_temperature,
                            base_index=base_index,
                            base_prior=granularity_base_prior,
                        )
                    )
                else:
                    if base_index != 0:
                        raise ValueError(
                            "A single-candidate branch must use base_index=0."
                        )
                    self.granularity_selectors.append(nn.Identity())
        self.last_granularity_weights = {}
        self.last_granularity_fallback_codes = {}
        self.last_granularity_fallback_indices = {}
        self._granularity_weights_for_loss = {}
        self._granularity_valid_for_loss = {}

        self.projections = nn.ModuleList()
        if self.modal_interaction == "concat":
            self.output_dim = int(sum(self.branch_dims))
        else:
            for dim in self.branch_dims:
                self.projections.append(nn.Linear(dim, fusion_dim))

            if self.modal_interaction == "concat_attn":
                self.output_dim = fusion_dim * len(self.branch_dims)
                if fusion_dim % fusion_heads != 0:
                    raise ValueError(
                        f"fusion_dim ({fusion_dim}) must be divisible by fusion_heads ({fusion_heads})."
                    )
                self.attention = nn.MultiheadAttention(
                    embed_dim=fusion_dim,
                    num_heads=fusion_heads,
                    batch_first=True,
                )
            elif self.modal_interaction == "cross_attn_gate":
                if len(self.branch_dims) < 2:
                    raise ValueError(
                        "cross_attn_gate requires at least two enabled branches."
                    )
                self.output_dim = fusion_dim
                if fusion_dim % fusion_heads != 0:
                    raise ValueError(
                        f"fusion_dim ({fusion_dim}) must be divisible by fusion_heads ({fusion_heads})."
                    )
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=fusion_dim,
                    num_heads=fusion_heads,
                    batch_first=True,
                )
                self.gate = nn.Linear(2 * fusion_dim, fusion_dim)
            elif self.modal_interaction == "masked_pretrain":
                self.output_dim = fusion_dim
                self.fusion_encoder = nn.Sequential(
                    nn.Linear(fusion_dim * len(self.branch_dims), fusion_dim),
                    nn.ReLU(),
                    nn.Linear(fusion_dim, fusion_dim),
                )
                self.reconstruction_heads = nn.ModuleList(
                    [nn.Linear(fusion_dim, fusion_dim) for _ in self.branch_dims]
                )
            else:
                self.attention = None

    def _project_branches(self, branch_embeddings):
        if len(branch_embeddings) != len(self.branch_dims):
            raise ValueError(
                f"Expected {len(self.branch_dims)} branch embeddings, got {len(branch_embeddings)}."
            )

        selected_embeddings = []
        self.last_granularity_weights = {}
        self.last_granularity_fallback_codes = {}
        self.last_granularity_fallback_indices = {}
        self._granularity_weights_for_loss = {}
        self._granularity_valid_for_loss = {}
        for branch_index, (embedding, selector, count) in enumerate(
            zip(
                branch_embeddings,
                self.granularity_selectors,
                self.branch_granularity_counts,
            )
        ):
            expected_dim = self.input_branch_dims[branch_index]
            if embedding.ndim != 2 or embedding.shape[1] != expected_dim:
                raise ValueError(
                    f"Branch {self.branch_names[branch_index]} expects "
                    f"[batch, {expected_dim}], got {tuple(embedding.shape)}."
                )
            if count > 1:
                branch_name = self.branch_names[branch_index]
                try:
                    embedding, weights = selector(embedding)
                except FloatingPointError as exc:
                    raise FloatingPointError(
                        f"Adaptive granularity branch {branch_name!r}: {exc}"
                    ) from exc
                self._granularity_weights_for_loss[branch_name] = weights
                self.last_granularity_weights[branch_name] = weights.detach()
                fallback_codes = getattr(selector, "last_fallback_codes", None)
                if fallback_codes is not None:
                    self._granularity_valid_for_loss[
                        branch_name
                    ] = fallback_codes == 0
                    self.last_granularity_fallback_codes[
                        branch_name
                    ] = fallback_codes.detach()
                    fallback_indices = getattr(
                        selector, "last_fallback_indices", None
                    )
                    if fallback_indices is not None:
                        self.last_granularity_fallback_indices[
                            branch_name
                        ] = fallback_indices.detach()
                else:
                    self._granularity_valid_for_loss[branch_name] = torch.ones(
                        weights.shape[0],
                        dtype=torch.bool,
                        device=weights.device,
                    )
            selected_embeddings.append(embedding)

        if self.modal_interaction == "concat":
            return selected_embeddings

        return [
            projection(embedding)
            for projection, embedding in zip(self.projections, selected_embeddings)
        ]

    def granularity_regularization_loss(
        self,
        balance_weight=0.01,
        entropy_weight=0.001,
        eps=1e-8,
    ):
        """Return anti-collapse loss and its normalized diagnostic terms.

        ``balance`` is KL(batch marginal || uniform) / log(K).  ``entropy`` is
        the mean per-sample entropy / log(K).  Minimizing the first prevents a
        global one-regime collapse; minimizing the much more weakly weighted
        second prevents the solution from becoming a uniform soft average.
        """
        if (
            not math.isfinite(balance_weight)
            or not math.isfinite(entropy_weight)
            or balance_weight < 0.0
            or entropy_weight < 0.0
        ):
            raise ValueError(
                "Granularity regularization weights must be finite and non-negative."
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}.")

        branch_weights = list(self._granularity_weights_for_loss.items())
        if not branch_weights:
            reference = next(self.parameters(), None)
            zero = (
                reference.new_zeros(())
                if reference is not None
                else torch.zeros((), dtype=torch.float32)
            )
            return zero, zero, zero

        balance_terms = []
        entropy_terms = []
        for branch_name, weights in branch_weights:
            valid = self._granularity_valid_for_loss.get(branch_name)
            if valid is not None:
                weights = weights[valid]
            if weights.shape[0] == 0:
                continue
            candidate_count = weights.shape[1]
            log_candidate_count = math.log(candidate_count)
            safe_weights = weights.clamp_min(eps)
            marginal = weights.mean(dim=0).clamp_min(eps)
            balance = (
                marginal * (marginal.log() + log_candidate_count)
            ).sum() / log_candidate_count
            entropy = -(
                safe_weights * safe_weights.log()
            ).sum(dim=1).mean() / log_candidate_count
            balance_terms.append(balance)
            entropy_terms.append(entropy)

        if not balance_terms:
            reference = branch_weights[0][1]
            zero = reference.new_zeros(())
            return zero, zero, zero

        mean_balance = torch.stack(balance_terms).mean()
        mean_entropy = torch.stack(entropy_terms).mean()
        total = balance_weight * mean_balance + entropy_weight * mean_entropy
        return total, mean_balance, mean_entropy

    def forward(self, branch_embeddings):
        if self.modal_interaction == "concat":
            return torch.cat(self._project_branches(branch_embeddings), dim=1)

        projected = self._project_branches(branch_embeddings)

        if self.modal_interaction == "concat_attn":
            tokens = torch.stack(projected, dim=1)
            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            return attended.reshape(attended.shape[0], -1)

        if self.modal_interaction == "cross_attn_gate":
            # For two branches, use them directly as visual and time-series streams.
            # If more than two branches are enabled, we keep the first projected branch
            # as the visual stream and collapse all remaining projected branches into
            # a single auxiliary time-series-side embedding by mean pooling.
            if len(projected) > 2:
                e_img = projected[0]
                e_ts = torch.stack(projected[1:], dim=0).mean(dim=0)
            else:
                e_img, e_ts = projected[0], projected[1]

            if self.cross_attn_query == "ts":
                query, context = e_ts, e_img
                query_is_ts = True
            else:
                query, context = e_img, e_ts
                query_is_ts = False

            e_cross, _ = self.cross_attention(
                query=query.unsqueeze(1),
                key=context.unsqueeze(1),
                value=context.unsqueeze(1),
                need_weights=False,
            )
            e_cross = e_cross.squeeze(1)

            gate_input = torch.cat([e_ts, e_img], dim=1)
            gate = torch.sigmoid(self.gate(gate_input))
            self.last_gate_mean = gate.detach().mean()

            if query_is_ts:
                return gate * e_ts + (1.0 - gate) * e_cross

            return gate * e_img + (1.0 - gate) * e_cross

        if self.modal_interaction == "masked_pretrain":
            return self.fusion_encoder(torch.cat(projected, dim=1))

        # Placeholder for future fusion strategies.
        return torch.cat(projected, dim=1)

    def reconstruct_masked(self, branch_embeddings, mask_prob=0.3, mask_index=None):
        if self.modal_interaction != "masked_pretrain":
            raise ValueError("reconstruct_masked is only available for masked_pretrain.")

        projected = self._project_branches(branch_embeddings)

        if mask_index is None:
            mask_index = torch.randint(len(projected), (1,), device=projected[0].device).item()
        apply_mask = bool((torch.rand((), device=projected[0].device) < mask_prob).item())
        if not apply_mask:
            return None

        masked = []
        for idx, embedding in enumerate(projected):
            if idx == mask_index:
                masked.append(torch.zeros_like(embedding))
            else:
                masked.append(embedding)

        fused = self.fusion_encoder(torch.cat(masked, dim=1))
        recon = self.reconstruction_heads[mask_index](fused)
        target = projected[mask_index]

        return recon, target, mask_index


def _set_trainable(models, is_training):
    for model in models:
        if model is not None:
            model.train(is_training)


def _labels_to_indices(labels):
    classes = np.unique(labels)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    indices = np.asarray([class_to_idx[label] for label in labels], dtype=np.int64)

    return indices, classes, class_to_idx


def _map_labels(labels, class_to_idx):
    unknown = sorted(set(labels) - set(class_to_idx))
    if unknown:
        raise ValueError(f"Test labels contain classes not present in training: {unknown}")

    return np.asarray([class_to_idx[label] for label in labels], dtype=np.int64)


def _balanced_class_weights(label_indices, num_classes, device):
    counts = np.bincount(label_indices, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(
            f"Cannot balance classes with zero training samples: {counts.tolist()}"
        )

    weights = len(label_indices) / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _forward_neurosigvia_batch(model, batch, channels, device):
    granularity_count = int(getattr(model, "feature_granularity_count", 1))
    if granularity_count > 1:
        if model.image_mode != "med_activity_graph":
            raise ValueError(
                "Adaptive vision features require med_activity_graph mode."
            )
        outputs = model.forward_granularities(batch.to(device))
        if outputs.ndim != 3 or outputs.shape[1] != granularity_count:
            raise ValueError(
                "forward_granularities must return [batch, K, dim], got "
                f"{tuple(outputs.shape)} for K={granularity_count}."
            )
        outputs = nn.functional.normalize(outputs, dim=-1)
        return outputs.reshape(outputs.shape[0], -1)

    if model.image_mode in {
        "multichannel_line_plot",
        "activity_graph",
        "med_activity_graph",
        "activity_matrix",
    }:
        outputs = model(batch.to(device))
        return nn.functional.normalize(outputs, dim=-1)

    batch_embeds_dim = []
    for dim in range(channels):
        batch_dim = batch[:, dim, :].unsqueeze(-1).to(device)
        batch_embeds_dim.append(model(batch_dim))

    outputs = torch.cat(batch_embeds_dim, dim=1)

    return nn.functional.normalize(outputs, dim=-1)


def _forward_moment_batch(model, batch, channels, device):
    batch_embeds_dim = []
    for dim in range(channels):
        batch_dim = batch[:, dim, :].unsqueeze(1)
        batch_dim = resize_moment_input(batch_dim).to(device).float()
        outputs = model(x_enc=batch_dim).embeddings
        batch_embeds_dim.append(outputs)

    outputs = torch.cat(batch_embeds_dim, dim=1)

    return nn.functional.normalize(outputs, dim=-1)


def _forward_mantis_batch(model, batch, channels, device):
    batch_embeds_dim = []
    for dim in range(channels):
        batch_dim = batch[:, dim, :].unsqueeze(1)
        batch_dim = resize_mantis_input(batch_dim).to(device).float()
        outputs = model(batch_dim)
        batch_embeds_dim.append(outputs)

    # Mantis emits one 512-D vector per channel.  NeuroSigVIA pools those
    # frozen channel representations into one sample-level temporal token.
    outputs = torch.stack(batch_embeds_dim, dim=1).mean(dim=1)

    return nn.functional.normalize(outputs, dim=-1)


def forward_feature_batch(
    batch,
    channels,
    device,
    vision_model_1=None,
    vision_model_2=None,
    mantis_model=None,
    moment_model=None,
):
    features = []

    if vision_model_1 is not None:
        features.append(
            _forward_neurosigvia_batch(vision_model_1, batch, channels, device)
        )
    if vision_model_2 is not None:
        features.append(
            _forward_neurosigvia_batch(vision_model_2, batch, channels, device)
        )
    if mantis_model is not None:
        features.append(_forward_mantis_batch(mantis_model, batch, channels, device))
    if moment_model is not None:
        features.append(_forward_moment_batch(moment_model, batch, channels, device))

    if not features:
        raise ValueError("At least one differentiable feature branch is required.")

    return features


@torch.no_grad()
def _extract_feature_split(loader, feature_models, channels, device):
    for model in feature_models.values():
        if model is not None:
            model.requires_grad_(False)
    _set_trainable(feature_models.values(), False)
    branch_batches = None

    for (batch,) in tqdm(loader, desc="Extract model features", leave=False):
        features = forward_feature_batch(
            batch=batch,
            channels=channels,
            device=device,
            **feature_models,
        )
        if branch_batches is None:
            branch_batches = [[] for _ in features]
        for batches, feature in zip(branch_batches, features):
            batches.append(feature.detach().cpu())

    if branch_batches is None:
        raise ValueError("Cannot extract features from an empty split.")

    return [torch.cat(batches, dim=0) for batches in branch_batches]


def _feature_branch_names(feature_models):
    return [name for name, model in feature_models.items() if model is not None]


def _feature_branch_granularity_metadata(feature_models):
    counts = []
    base_indices = []
    labels = []
    for model in feature_models.values():
        if model is None:
            continue
        count = int(getattr(model, "feature_granularity_count", 1))
        base_index = int(
            getattr(model, "feature_granularity_base_index", 0)
        )
        branch_labels = tuple(
            getattr(
                model,
                "feature_granularity_labels",
                tuple(f"candidate_{index}" for index in range(count)),
            )
        )
        if count < 1:
            raise ValueError(
                f"Feature branch exposes invalid granularity count {count}."
            )
        if not 0 <= base_index < count:
            raise ValueError(
                f"Feature branch exposes invalid base index {base_index} for K={count}."
            )
        if len(branch_labels) != count:
            raise ValueError(
                f"Feature branch exposes {len(branch_labels)} labels for K={count}."
            )
        counts.append(count)
        base_indices.append(base_index)
        labels.append(branch_labels)
    return counts, base_indices, labels


def _load_feature_cache(
    path,
    expected_labels,
    expected_branch_names,
    expected_signature,
):
    if not path.is_file():
        return None

    with np.load(path, allow_pickle=False) as cached:
        required_keys = {
            "schema_version",
            "cache_signature",
            "branch_names",
            "labels",
        }
        missing_keys = required_keys - set(cached.files)
        if missing_keys:
            raise ValueError(
                f"Feature cache is missing metadata {sorted(missing_keys)}: {path}"
            )
        schema_version = int(cached["schema_version"].item())
        if schema_version != FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Feature cache schema mismatch in {path}: expected "
                f"{FEATURE_CACHE_SCHEMA_VERSION}, got {schema_version}."
            )
        signature = str(cached["cache_signature"].item())
        if signature != (expected_signature or ""):
            raise ValueError(
                f"Feature cache model/configuration mismatch: {path}"
            )
        branch_names = cached["branch_names"].tolist()
        labels = cached["labels"]
        if branch_names != expected_branch_names:
            raise ValueError(
                f"Feature cache branch mismatch in {path}: "
                f"expected {expected_branch_names}, got {branch_names}."
            )
        if not np.array_equal(labels, np.asarray(expected_labels, dtype=np.int64)):
            raise ValueError(f"Feature cache labels do not match current split: {path}")
        branch_keys = [f"branch_{index}" for index in range(len(branch_names))]
        missing_branches = set(branch_keys) - set(cached.files)
        if missing_branches:
            raise ValueError(
                f"Feature cache is missing branches {sorted(missing_branches)}: {path}"
            )
        features = [torch.from_numpy(cached[key].copy()) for key in branch_keys]

    if any(len(feature) != len(labels) for feature in features):
        raise ValueError(f"Feature cache sample count mismatch: {path}")
    if any(feature.ndim != 2 for feature in features):
        raise ValueError(f"Feature cache branches must be two-dimensional: {path}")
    if any(not torch.isfinite(feature).all() for feature in features):
        raise ValueError(f"Feature cache contains non-finite values: {path}")
    print(f"Loaded model features: {path}")
    return features


def _save_feature_cache(path, labels, branch_names, features, cache_signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "schema_version": np.asarray(FEATURE_CACHE_SCHEMA_VERSION, dtype=np.int64),
        "cache_signature": np.asarray(cache_signature or ""),
        "labels": np.asarray(labels, dtype=np.int64),
        "branch_names": np.asarray(branch_names),
    }
    arrays.update(
        {
            f"branch_{index}": feature.numpy()
            for index, feature in enumerate(features)
        }
    )
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved model features: {path}")


def _get_feature_split(
    split,
    loader,
    labels,
    feature_models,
    channels,
    device,
    feature_cache_dir,
    feature_cache_signature,
):
    branch_names = _feature_branch_names(feature_models)
    cache_path = None
    if feature_cache_dir:
        cache_path = Path(feature_cache_dir) / f"{split}.npz"
        cached = _load_feature_cache(
            cache_path,
            labels,
            branch_names,
            feature_cache_signature,
        )
        if cached is not None:
            return cached

    features = _extract_feature_split(loader, feature_models, channels, device)
    if cache_path is not None:
        _save_feature_cache(
            cache_path,
            labels,
            branch_names,
            features,
            feature_cache_signature,
        )
    return features


def _build_feature_loader(features, labels, indices, batch_size, shuffle):
    dataset = TensorDataset(
        *features,
        torch.as_tensor(labels, dtype=torch.long),
    )
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _infer_feature_dim(features):
    if not features:
        raise ValueError("At least one cached feature branch is required.")

    return [feature.shape[1] for feature in features]


def _accumulate_granularity_weights(
    running_sums,
    running_counts,
    fusion_module,
):
    for branch_name, weights in getattr(
        fusion_module,
        "last_granularity_weights",
        {},
    ).items():
        detached = weights.detach().to(dtype=torch.float64, device="cpu")
        if branch_name not in running_sums:
            running_sums[branch_name] = detached.sum(dim=0)
            running_counts[branch_name] = detached.shape[0]
        else:
            running_sums[branch_name] += detached.sum(dim=0)
            running_counts[branch_name] += detached.shape[0]


def _mean_granularity_weights(running_sums, running_counts):
    return {
        branch_name: (weights / max(running_counts[branch_name], 1)).tolist()
        for branch_name, weights in running_sums.items()
    }


def _format_granularity_weights(fusion_module, mean_weights):
    if not mean_weights:
        return ""
    labels_by_name = {
        name: labels
        for name, labels in zip(
            fusion_module.branch_names,
            fusion_module.branch_granularity_labels,
        )
    }
    summaries = []
    for branch_name, weights in mean_weights.items():
        labels = labels_by_name[branch_name]
        assignments = ",".join(
            f"{label}={weight:.4f}"
            for label, weight in zip(labels, weights)
        )
        summaries.append(f"{branch_name}[{assignments}]")
    return "; ".join(summaries)


def _require_finite_rows(values, name):
    finite = torch.isfinite(values).reshape(values.shape[0], -1).all(dim=1)
    if bool(finite.all()):
        return
    bad_indices = torch.nonzero(~finite, as_tuple=False).flatten()
    preview = bad_indices[:8].detach().cpu().tolist()
    suffix = "..." if bad_indices.numel() > len(preview) else ""
    raise FloatingPointError(
        f"Non-finite {name} for {bad_indices.numel()} sample(s); "
        f"batch indices={preview}{suffix}."
    )


def _run_epoch(
    mlp,
    fusion_module,
    loader,
    criterion,
    optimizer,
    device,
    granularity_balance_weight=0.01,
    granularity_entropy_weight=0.001,
):
    _set_trainable([mlp, fusion_module], True)
    total_loss = 0.0
    total_task_loss = 0.0
    total_regularization = 0.0
    total_balance = 0.0
    total_entropy = 0.0
    total_samples = 0
    gate_means = []
    granularity_sums = {}
    granularity_counts = {}

    for *branch_features, labels in tqdm(loader, desc="Train MLP", leave=False):
        branch_features = [feature.to(device) for feature in branch_features]
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        features = fusion_module(branch_features)
        logits = mlp(features)
        _require_finite_rows(logits, "classifier logits")
        task_loss = criterion(logits, labels)
        regularization, balance, entropy = (
            fusion_module.granularity_regularization_loss(
                balance_weight=granularity_balance_weight,
                entropy_weight=granularity_entropy_weight,
            )
        )
        loss = task_loss + regularization
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("MLP training loss became non-finite.")
        loss.backward()
        optimizer.step()

        _accumulate_granularity_weights(
            granularity_sums,
            granularity_counts,
            fusion_module,
        )

        if getattr(fusion_module, "modal_interaction", None) == "cross_attn_gate":
            gate_mean = getattr(fusion_module, "last_gate_mean", None)
            if gate_mean is not None:
                gate_means.append(float(gate_mean.item()))

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_task_loss += task_loss.item() * batch_size
        total_regularization += regularization.item() * batch_size
        total_balance += balance.item() * batch_size
        total_entropy += entropy.item() * batch_size
        total_samples += batch_size

    mean_gate = float(np.mean(gate_means)) if gate_means else None
    mean_granularity = _mean_granularity_weights(
        granularity_sums,
        granularity_counts,
    )

    denominator = max(total_samples, 1)
    statistics = {
        "loss": total_loss / denominator,
        "task_loss": total_task_loss / denominator,
        "granularity_regularization": total_regularization / denominator,
        "granularity_balance": total_balance / denominator,
        "granularity_entropy": total_entropy / denominator,
    }
    return statistics, mean_gate, mean_granularity


def _run_masked_pretrain_epoch(
    fusion_module,
    loader,
    optimizer,
    device,
    mask_prob,
    granularity_balance_weight=0.01,
    granularity_entropy_weight=0.001,
):
    _set_trainable([fusion_module], True)
    total_loss = 0.0
    total_samples = 0

    for *branch_features, labels in tqdm(loader, desc="Pretrain Fusion", leave=False):
        branch_features = [feature.to(device) for feature in branch_features]
        optimizer.zero_grad(set_to_none=True)
        result = fusion_module.reconstruct_masked(
            branch_embeddings=branch_features,
            mask_prob=mask_prob,
        )
        if result is None:
            continue
        recon, target, _ = result
        reconstruction_loss = nn.functional.mse_loss(recon, target)
        regularization, _, _ = fusion_module.granularity_regularization_loss(
            balance_weight=granularity_balance_weight,
            entropy_weight=granularity_entropy_weight,
        )
        loss = reconstruction_loss + regularization
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Fusion pretraining loss became non-finite.")
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def _evaluate(mlp, fusion_module, loader, classes, device):
    _set_trainable([mlp, fusion_module], False)
    y_true = []
    y_pred = []
    y_score = []
    granularity_sums = {}
    granularity_counts = {}
    granularity_batches = {}
    fallback_code_batches = {}
    fallback_index_batches = {}

    for *branch_features, labels in tqdm(loader, desc="Evaluate MLP", leave=False):
        branch_features = [feature.to(device) for feature in branch_features]
        features = fusion_module(branch_features)
        logits = mlp(features)
        _require_finite_rows(logits, "classifier logits")
        probabilities = torch.softmax(logits, dim=1)
        _require_finite_rows(probabilities, "classifier probabilities")

        y_true.append(labels.cpu().numpy())
        y_pred.append(probabilities.argmax(dim=1).cpu().numpy())
        y_score.append(probabilities.cpu().numpy())
        _accumulate_granularity_weights(
            granularity_sums,
            granularity_counts,
            fusion_module,
        )
        for branch_name, weights in getattr(
            fusion_module, "last_granularity_weights", {}
        ).items():
            granularity_batches.setdefault(branch_name, []).append(
                weights.detach().cpu().numpy()
            )
        for branch_name, codes in getattr(
            fusion_module, "last_granularity_fallback_codes", {}
        ).items():
            fallback_code_batches.setdefault(branch_name, []).append(
                codes.detach().cpu().numpy()
            )
        for branch_name, indices in getattr(
            fusion_module, "last_granularity_fallback_indices", {}
        ).items():
            fallback_index_batches.setdefault(branch_name, []).append(
                indices.detach().cpu().numpy()
            )

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_score = np.concatenate(y_score)
    fusion_module.last_evaluation_granularity_means = (
        _mean_granularity_weights(granularity_sums, granularity_counts)
    )
    fusion_module.last_evaluation_details = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "branches": {
            branch_name: {
                "weights": np.concatenate(batches, axis=0),
                "fallback_codes": np.concatenate(
                    fallback_code_batches.get(
                        branch_name,
                        [np.zeros(len(y_true), dtype=np.int64)],
                    ),
                    axis=0,
                ),
                "fallback_indices": np.concatenate(
                    fallback_index_batches.get(
                        branch_name,
                        [np.full(len(y_true), -1, dtype=np.int64)],
                    ),
                    axis=0,
                ),
            }
            for branch_name, batches in granularity_batches.items()
        },
    }

    return compute_metrics_from_predictions(y_true, y_pred, y_score, np.arange(len(classes)))


def _normalized_selector_entropy(weights, eps=1e-12):
    weights = np.asarray(weights, dtype=np.float64)
    candidate_count = weights.shape[1]
    safe = np.clip(weights, eps, 1.0)
    return -(safe * np.log(safe)).sum(axis=1) / np.log(candidate_count)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json_dump(path, payload):
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


def _safe_npz_branch_name(branch_name):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", branch_name).strip("_")
    return safe or "branch"


def _summarize_granularity_details(
    details,
    fusion_module,
    collapse_threshold=0.95,
    minimum_collapse_samples=3,
):
    metadata = {
        name: {
            "labels": list(labels),
            "base_index": int(base_index),
        }
        for name, labels, base_index in zip(
            fusion_module.branch_names,
            fusion_module.branch_granularity_labels,
            fusion_module.branch_granularity_base_indices,
        )
    }
    summary = {
        "sample_count": int(len(details["y_true"])),
        "branches": {},
    }
    fallback_names = {
        0: "none",
        1: "invalid_candidate",
        2: "legacy_nonfinite_logits",
        3: "indistinguishable_candidates",
        4: "degenerate_weighted_mixture",
    }
    for branch_name, branch in details["branches"].items():
        weights = np.asarray(branch["weights"], dtype=np.float64)
        if weights.ndim != 2 or weights.shape[0] != summary["sample_count"]:
            raise ValueError(
                f"Invalid diagnostic weights for {branch_name}: {weights.shape}."
            )
        if not np.isfinite(weights).all() or not np.allclose(
            weights.sum(axis=1), 1.0, atol=1e-6
        ):
            raise ValueError(
                f"Invalid selector probability rows for diagnostic branch {branch_name}."
            )

        candidate_count = weights.shape[1]
        entropy = _normalized_selector_entropy(weights)
        marginal = weights.mean(axis=0)
        marginal_safe = np.clip(marginal, 1e-12, 1.0)
        marginal_entropy = float(
            -(marginal_safe * np.log(marginal_safe)).sum()
            / np.log(candidate_count)
        )
        mutual_information = float(
            np.clip(marginal_entropy - entropy.mean(), 0.0, 1.0)
        )
        top1 = weights.argmax(axis=1)
        top1_counts = np.bincount(top1, minlength=candidate_count)
        top1_fractions = top1_counts / max(len(top1), 1)
        fallback_codes = np.asarray(branch["fallback_codes"], dtype=np.int64)
        unique_codes, code_counts = np.unique(fallback_codes, return_counts=True)
        fallback_counts = {
            fallback_names.get(int(code), f"unknown_{int(code)}"): int(count)
            for code, count in zip(unique_codes, code_counts)
        }
        fallback_count = int(np.count_nonzero(fallback_codes))

        collapse_reasons = []
        mean_max_weight = float(weights.max(axis=1).mean())
        mean_entropy = float(entropy.mean())
        enough_samples = len(top1) >= minimum_collapse_samples
        if enough_samples:
            # Argmax alone is meaningless for an exactly uniform mixture, so a
            # concentrated top-1 histogram is considered collapse only when
            # the selector is also meaningfully confident.
            if (
                float(top1_fractions.max()) >= collapse_threshold
                and mean_max_weight >= 0.5
            ):
                collapse_reasons.append("top1_assignment_concentration")
            if float(marginal.max()) >= collapse_threshold:
                collapse_reasons.append("mean_weight_concentration")
            if (
                mean_entropy >= 0.98
                and mean_max_weight <= (1.0 / candidate_count + 0.02)
            ):
                collapse_reasons.append("uniform_soft_mixture")

        branch_metadata = metadata[branch_name]
        summary["branches"][branch_name] = {
            "labels": branch_metadata["labels"],
            "base_index": branch_metadata["base_index"],
            "candidate_count": candidate_count,
            "mean_weights": weights.mean(axis=0).tolist(),
            "std_weights": weights.std(axis=0).tolist(),
            "top1_counts": top1_counts.tolist(),
            "top1_fractions": top1_fractions.tolist(),
            "mean_normalized_entropy": mean_entropy,
            "std_normalized_entropy": float(entropy.std()),
            "marginal_normalized_entropy": marginal_entropy,
            "normalized_mutual_information": mutual_information,
            "mean_max_weight": mean_max_weight,
            "std_max_weight": float(weights.max(axis=1).std()),
            "fallback_counts": fallback_counts,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / max(len(top1), 1),
            "collapse_threshold": collapse_threshold,
            "collapse_minimum_samples": minimum_collapse_samples,
            "collapse_assessment": (
                "evaluated" if enough_samples else "insufficient_samples"
            ),
            "collapse_detected": (
                bool(collapse_reasons) if enough_samples else None
            ),
            "collapse_reasons": collapse_reasons,
        }
    return summary


def _save_granularity_diagnostics(
    artifact_dir,
    split_name,
    details,
    sample_indices,
    fusion_module,
):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    if len(sample_indices) != len(details["y_true"]):
        raise ValueError(
            f"{split_name} diagnostic sample-index count does not match predictions."
        )

    arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "sample_index": sample_indices,
        "y_true": np.asarray(details["y_true"], dtype=np.int64),
        "y_pred": np.asarray(details["y_pred"], dtype=np.int64),
        "y_score": np.asarray(details["y_score"], dtype=np.float32),
        "branch_names": np.asarray(list(details["branches"])),
    }
    used_names = set()
    for branch_position, (branch_name, branch) in enumerate(
        details["branches"].items()
    ):
        safe_name = _safe_npz_branch_name(branch_name)
        if safe_name in used_names:
            safe_name = f"{safe_name}_{branch_position}"
        used_names.add(safe_name)
        prefix = f"branch_{branch_position}_{safe_name}"
        weights = np.asarray(branch["weights"], dtype=np.float32)
        arrays[f"{prefix}_weights"] = weights
        arrays[f"{prefix}_top1"] = weights.argmax(axis=1).astype(np.int64)
        arrays[f"{prefix}_normalized_entropy"] = (
            _normalized_selector_entropy(weights).astype(np.float32)
        )
        arrays[f"{prefix}_fallback_code"] = np.asarray(
            branch["fallback_codes"], dtype=np.int64
        )
        arrays[f"{prefix}_fallback_index"] = np.asarray(
            branch["fallback_indices"], dtype=np.int64
        )

    path = artifact_dir / f"atgs_diagnostics_{split_name}.npz"
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return _summarize_granularity_details(details, fusion_module)


def _cpu_state_dict(module):
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def _save_atgs_checkpoint(
    artifact_dir,
    mlp,
    fusion_module,
    classes,
    training_history,
    selection,
    selected_epoch,
    validation_macro_f1,
    balance_weight,
    entropy_weight,
):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    selector_states = {
        branch_name: _cpu_state_dict(selector)
        for branch_name, count, selector in zip(
            fusion_module.branch_names,
            fusion_module.branch_granularity_counts,
            fusion_module.granularity_selectors,
        )
        if count > 1
    }
    mlp_linear_layers = [
        layer for layer in mlp.network if isinstance(layer, nn.Linear)
    ]
    mlp_dropouts = [
        layer.p for layer in mlp.network if isinstance(layer, nn.Dropout)
    ]
    selector_configurations = {
        branch_name: {
            "feature_dim": int(selector.feature_dim),
            "num_granularities": int(selector.num_granularities),
            "hidden_dim": int(selector.scorer[0].out_features),
            "temperature": float(selector.temperature),
            "base_index": int(selector.base_index),
            "base_prior": float(selector.base_prior),
        }
        for branch_name, count, selector in zip(
            fusion_module.branch_names,
            fusion_module.branch_granularity_counts,
            fusion_module.granularity_selectors,
        )
        if count > 1
    }
    checkpoint = {
        "schema_version": 1,
        "selection": selection,
        "selected_epoch": int(selected_epoch),
        "validation_macro_f1": float(validation_macro_f1),
        "classes": [_json_safe(label) for label in classes],
        "branch_names": list(fusion_module.branch_names),
        "branch_granularity_counts": list(
            fusion_module.branch_granularity_counts
        ),
        "branch_granularity_base_indices": list(
            fusion_module.branch_granularity_base_indices
        ),
        "branch_granularity_labels": [
            list(labels) for labels in fusion_module.branch_granularity_labels
        ],
        "granularity_balance_weight": float(balance_weight),
        "granularity_entropy_weight": float(entropy_weight),
        "fusion_configuration": {
            "input_branch_dims": list(fusion_module.input_branch_dims),
            "modal_interaction": fusion_module.modal_interaction,
            "fusion_dim": int(fusion_module.fusion_dim),
            "fusion_heads": int(fusion_module.fusion_heads),
            "cross_attn_query": fusion_module.cross_attn_query,
            "selector_configurations": selector_configurations,
        },
        "mlp_configuration": {
            "input_dim": int(mlp_linear_layers[0].in_features),
            "hidden_dims": [
                int(layer.out_features) for layer in mlp_linear_layers[:-1]
            ],
            "num_layers": len(mlp_linear_layers),
            "dropout": mlp_dropouts,
            "num_classes": int(mlp_linear_layers[-1].out_features),
        },
        "selector_state_dicts": selector_states,
        "fusion_state_dict": _cpu_state_dict(fusion_module),
        "mlp_state_dict": _cpu_state_dict(mlp),
    }
    path = artifact_dir / "atgs_checkpoint.pt"
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    _atomic_json_dump(
        artifact_dir / "atgs_training_history.json",
        {
            "schema_version": 1,
            "selection": selection,
            "selected_epoch": int(selected_epoch),
            "validation_macro_f1": float(validation_macro_f1),
            "epochs": training_history,
        },
    )


def train_mlp_classifier(
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
    modal_interaction,
    fusion_dim,
    fusion_heads,
    cross_attn_query,
    mask_prob,
    pretrain_epochs,
    class_weight="none",
    vision_model_1=None,
    vision_model_2=None,
    mantis_model=None,
    moment_model=None,
    val_loader=None,
    val_labels=None,
    feature_cache_dir=None,
    feature_cache_signature=None,
    granularity_hidden_dim=64,
    granularity_temperature=1.0,
    granularity_base_prior=0.9,
    granularity_balance_weight=0.01,
    granularity_entropy_weight=0.001,
    artifact_dir=None,
):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if early_stop_patience < 0:
        raise ValueError("early_stop_patience must be non-negative")
    if pretrain_epochs < 0:
        raise ValueError("pretrain_epochs must be non-negative")
    if (
        not math.isfinite(granularity_balance_weight)
        or granularity_balance_weight < 0.0
    ):
        raise ValueError(
            "granularity_balance_weight must be finite and non-negative"
        )
    if (
        not math.isfinite(granularity_entropy_weight)
        or granularity_entropy_weight < 0.0
    ):
        raise ValueError(
            "granularity_entropy_weight must be finite and non-negative"
        )
    has_fixed_validation = val_loader is not None or val_labels is not None
    if has_fixed_validation and (val_loader is None or val_labels is None):
        raise ValueError("val_loader and val_labels must be provided together")

    if has_fixed_validation:
        train_indices = list(range(len(train_loader.dataset)))
        val_indices = list(range(len(val_loader.dataset)))
    else:
        train_indices, val_indices = get_split(
            train_loader.dataset,
            frac=val_ratio,
            random_seed=random_seed,
        )

    train_label_indices, classes, class_to_idx = _labels_to_indices(train_labels)
    test_label_indices = _map_labels(test_labels, class_to_idx)

    feature_models = {
        "vision_model_1": vision_model_1,
        "vision_model_2": vision_model_2,
        "mantis_model": mantis_model,
        "moment_model": moment_model,
    }
    train_features = _get_feature_split(
        "train",
        train_loader,
        train_labels,
        feature_models,
        channels,
        device,
        feature_cache_dir,
        feature_cache_signature,
    )
    test_features = _get_feature_split(
        "test",
        test_loader,
        test_labels,
        feature_models,
        channels,
        device,
        feature_cache_dir,
        feature_cache_signature,
    )
    if has_fixed_validation:
        val_label_indices = _map_labels(val_labels, class_to_idx)
        val_features = _get_feature_split(
            "vali",
            val_loader,
            val_labels,
            feature_models,
            channels,
            device,
            feature_cache_dir,
            feature_cache_signature,
        )

    # Feature extraction and cache misses may consume RNG state. Reset here so
    # classifier initialization and training depend only on the outer seed.
    set_random_seed(random_seed)

    if has_fixed_validation:
        mlp_train_loader = _build_feature_loader(
            train_features,
            train_label_indices,
            train_indices,
            batch_size,
            shuffle=True,
        )
        mlp_val_loader = _build_feature_loader(
            val_features,
            val_label_indices,
            val_indices,
            batch_size,
            shuffle=False,
        )
    else:
        mlp_train_loader = _build_feature_loader(
            train_features,
            train_label_indices,
            train_indices,
            batch_size,
            shuffle=True,
        )
        mlp_val_loader = _build_feature_loader(
            train_features,
            train_label_indices,
            val_indices,
            batch_size,
            shuffle=False,
        )
    mlp_test_loader = _build_feature_loader(
        test_features,
        test_label_indices,
        list(range(len(test_label_indices))),
        batch_size,
        shuffle=False,
    )

    feature_dims = _infer_feature_dim(train_features)
    branch_names = _feature_branch_names(feature_models)
    (
        branch_granularity_counts,
        branch_granularity_base_indices,
        branch_granularity_labels,
    ) = _feature_branch_granularity_metadata(feature_models)
    fusion_module = FusionModule(
        branch_dims=feature_dims,
        modal_interaction=modal_interaction,
        fusion_dim=fusion_dim,
        fusion_heads=fusion_heads,
        cross_attn_query=cross_attn_query,
        branch_names=branch_names,
        branch_granularity_counts=branch_granularity_counts,
        branch_granularity_base_indices=branch_granularity_base_indices,
        branch_granularity_labels=branch_granularity_labels,
        granularity_hidden_dim=granularity_hidden_dim,
        granularity_temperature=granularity_temperature,
        granularity_base_prior=granularity_base_prior,
    ).to(device)

    mlp = MLPClassifier(
        input_dim=fusion_module.output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=len(classes),
    ).to(device)

    parameters = list(mlp.parameters())
    parameters.extend(fusion_module.parameters())

    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if class_weight == "balanced":
        criterion_weights = _balanced_class_weights(
            train_label_indices[train_indices], len(classes), device
        )
        print(
            "MLP class weights: "
            + ", ".join(f"{weight:.4f}" for weight in criterion_weights.tolist())
        )
    elif class_weight == "none":
        criterion_weights = None
    else:
        raise ValueError(f"Unsupported MLP class weighting: {class_weight}")
    criterion = nn.CrossEntropyLoss(weight=criterion_weights)

    if modal_interaction == "masked_pretrain" and pretrain_epochs > 0:
        pretrain_optimizer = torch.optim.AdamW(
            fusion_module.parameters(), lr=lr, weight_decay=weight_decay
        )
        for epoch in range(pretrain_epochs):
            loss = _run_masked_pretrain_epoch(
                fusion_module=fusion_module,
                loader=mlp_train_loader,
                optimizer=pretrain_optimizer,
                device=device,
                mask_prob=mask_prob,
                granularity_balance_weight=granularity_balance_weight,
                granularity_entropy_weight=granularity_entropy_weight,
            )
            print(
                f"Fusion pretrain epoch {epoch + 1}/{pretrain_epochs} | loss={loss:.4f}"
            )

    best_val_score = -float("inf")
    best_state = None
    epochs_without_improvement = 0
    val_metrics = None
    best_epoch = None
    last_epoch = 0
    training_history = []

    for epoch in range(epochs):
        last_epoch = epoch + 1
        training_statistics, gate_mean, granularity_means = _run_epoch(
            mlp=mlp,
            fusion_module=fusion_module,
            loader=mlp_train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            granularity_balance_weight=granularity_balance_weight,
            granularity_entropy_weight=granularity_entropy_weight,
        )
        epoch_record = {
            "epoch": epoch + 1,
            **training_statistics,
        }
        epoch_summary = (
            f"MLP epoch {epoch + 1}/{epochs} | "
            f"loss={training_statistics['loss']:.4f} | "
            f"task_loss={training_statistics['task_loss']:.4f}"
        )
        if granularity_means:
            epoch_summary += (
                " | atgs_reg="
                f"{training_statistics['granularity_regularization']:.6f}"
                " | atgs_balance="
                f"{training_statistics['granularity_balance']:.4f}"
                " | atgs_entropy="
                f"{training_statistics['granularity_entropy']:.4f}"
            )
        if gate_mean is not None:
            epoch_summary += f" | gate_mean={gate_mean:.4f}"
        granularity_summary = _format_granularity_weights(
            fusion_module,
            granularity_means,
        )
        if granularity_summary:
            epoch_summary += f" | granularity={granularity_summary}"
        print(epoch_summary)

        if early_stop_patience > 0:
            val_metrics = _evaluate(
                mlp, fusion_module, mlp_val_loader, classes, device
            )
            val_score = val_metrics["macro_f1"]
            if not np.isfinite(val_score):
                raise FloatingPointError(
                    f"Validation macro F1 became non-finite at epoch {epoch + 1}."
                )
            epoch_record["validation_metrics"] = deepcopy(val_metrics)
            if val_score > best_val_score:
                best_val_score = val_score
                epochs_without_improvement = 0
                best_epoch = epoch + 1
                best_state = {
                    "mlp": deepcopy(mlp.state_dict()),
                    "fusion_module": deepcopy(fusion_module.state_dict()),
                }
            else:
                epochs_without_improvement += 1

            print(
                f"MLP epoch {epoch + 1}/{epochs} | val_macro_f1={val_score:.4f} | best_val_macro_f1={best_val_score:.4f}"
            )
            if epochs_without_improvement >= early_stop_patience:
                print(
                    f"Early stopping after {epoch + 1} epochs | patience={early_stop_patience}"
                )
                break
        training_history.append(epoch_record)

    # If early stopping breaks, its final epoch record has not passed the line
    # after the stopping check above.
    if last_epoch > len(training_history):
        training_history.append(epoch_record)

    if best_state is not None:
        mlp.load_state_dict(best_state["mlp"])
        fusion_module.load_state_dict(best_state["fusion_module"])

    adaptive_granularity = any(
        count > 1 for count in fusion_module.branch_granularity_counts
    )
    save_atgs_artifacts = adaptive_granularity and artifact_dir is not None
    train_diagnostics = None
    if save_atgs_artifacts:
        mlp_train_evaluation_loader = _build_feature_loader(
            train_features,
            train_label_indices,
            train_indices,
            batch_size,
            shuffle=False,
        )
        _evaluate(
            mlp,
            fusion_module,
            mlp_train_evaluation_loader,
            classes,
            device,
        )
        train_diagnostics = deepcopy(fusion_module.last_evaluation_details)

    # Always recompute final diagnostics after restoring the selected state.
    val_metrics = _evaluate(
        mlp, fusion_module, mlp_val_loader, classes, device
    )
    val_diagnostics = (
        deepcopy(fusion_module.last_evaluation_details)
        if save_atgs_artifacts
        else None
    )
    val_granularity_means = deepcopy(
        getattr(fusion_module, "last_evaluation_granularity_means", {})
    )
    test_metrics = _evaluate(
        mlp, fusion_module, mlp_test_loader, classes, device
    )
    test_diagnostics = (
        deepcopy(fusion_module.last_evaluation_details)
        if save_atgs_artifacts
        else None
    )
    test_granularity_means = deepcopy(
        getattr(fusion_module, "last_evaluation_granularity_means", {})
    )
    for split_name, weights in (
        ("validation", val_granularity_means),
        ("test", test_granularity_means),
    ):
        summary = _format_granularity_weights(fusion_module, weights)
        if summary:
            print(f"Adaptive granularity {split_name} mean weights: {summary}")

    if save_atgs_artifacts:
        artifact_dir = Path(artifact_dir)
        split_payloads = (
            ("train", train_diagnostics, train_indices),
            ("validation", val_diagnostics, val_indices),
            (
                "test",
                test_diagnostics,
                list(range(len(test_label_indices))),
            ),
        )
        diagnostic_summary = {
            "schema_version": 1,
            "splits": {},
        }
        for split_name, details, sample_indices in split_payloads:
            split_summary = _save_granularity_diagnostics(
                artifact_dir=artifact_dir,
                split_name=split_name,
                details=details,
                sample_indices=sample_indices,
                fusion_module=fusion_module,
            )
            diagnostic_summary["splits"][split_name] = split_summary
            for branch_name, branch_summary in split_summary["branches"].items():
                if branch_summary["collapse_detected"]:
                    warnings.warn(
                        "Adaptive granularity collapse warning: "
                        f"split={split_name}, branch={branch_name}, "
                        "reasons="
                        f"{branch_summary['collapse_reasons']}",
                        RuntimeWarning,
                    )
        _atomic_json_dump(
            artifact_dir / "atgs_diagnostics_summary.json",
            diagnostic_summary,
        )

        if best_state is not None:
            selection = "best_validation_epoch"
            selected_epoch = best_epoch
        else:
            selection = "final_epoch"
            selected_epoch = last_epoch
        _save_atgs_checkpoint(
            artifact_dir=artifact_dir,
            mlp=mlp,
            fusion_module=fusion_module,
            classes=classes,
            training_history=training_history,
            selection=selection,
            selected_epoch=selected_epoch,
            validation_macro_f1=val_metrics["macro_f1"],
            balance_weight=granularity_balance_weight,
            entropy_weight=granularity_entropy_weight,
        )
        print(f"Saved adaptive granularity artifacts: {artifact_dir}")

    return val_metrics, test_metrics, train_indices, val_indices
