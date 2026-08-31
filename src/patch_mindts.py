"""Patch-level cross-view fusion for NeuroSigViT.

This module implements the time-patch path without changing the legacy
sample-level MLP/ATGS contracts.  Every branch consumes the same temporal
window, the visual granularity selector uses the line-plot token as query and
the Activity Graph candidates as keys/values, and alignment negatives are
restricted to other valid windows from the same sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from src.classifier import compute_metrics_from_predictions
from src.neurosigvit import preprocess_stacked_multichannel_lineplot
from src.utils import get_split, resize_mantis_input, set_random_seed


PATCH_FEATURE_CACHE_SCHEMA_VERSION = 1
PATCH_FEATURE_ARCHITECTURE = "patch_mindts_v1"
PATCH_MINDTS_ARCHITECTURE = "patch_mindts_v4"
PATCH_TAIL_POLICY = "right_zero_pad_then_crop_valid_prefix_v1"
PATCH_POOLING_POLICY = "valid_fraction_weighted_mean_v1"

PATCH_FEATURE_KEYS = (
    "line_tokens",
    "graph_tokens",
    "mantis_channel_tokens",
    "patch_mask",
    "valid_fraction",
    "valid_lengths",
)


@dataclass(frozen=True)
class TemporalPatchBatch:
    """One shared temporal partition consumed by all three feature branches."""

    patches: torch.Tensor
    time_mask: torch.Tensor
    patch_mask: torch.Tensor
    valid_lengths: torch.Tensor
    valid_fraction: torch.Tensor


def make_temporal_patches(
    inputs,
    window_size=64,
    stride=64,
    lengths=None,
):
    """Split ``[B, C, T]`` into aligned, right-padded temporal windows.

    ``lengths`` optionally describes the true prefix length of each sample
    when the input tensor was padded by a data loader.  Values outside those
    prefixes are zeroed in the returned patches and are never considered
    valid by downstream renderers.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive integers")
    if stride > window_size:
        raise ValueError(
            "stride cannot exceed window_size because that would leave time points uncovered"
        )

    tensor = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs)
    if tensor.ndim != 3:
        raise ValueError(
            f"inputs must have shape [batch, channels, time], got {tuple(tensor.shape)}"
        )
    batch_size, channels, time_steps = tensor.shape
    if batch_size < 1 or channels < 1 or time_steps < 1:
        raise ValueError(f"input dimensions must be non-empty, got {tuple(tensor.shape)}")

    if lengths is None:
        true_lengths = torch.full(
            (batch_size,),
            time_steps,
            dtype=torch.long,
            device=tensor.device,
        )
    else:
        true_lengths = torch.as_tensor(
            lengths,
            dtype=torch.long,
            device=tensor.device,
        )
        if true_lengths.ndim != 1 or len(true_lengths) != batch_size:
            raise ValueError(
                f"lengths must have shape [{batch_size}], got {tuple(true_lengths.shape)}"
            )
        if torch.any(true_lengths <= 0):
            raise ValueError("every sample must contain at least one valid time point")
        if torch.any(true_lengths > time_steps):
            raise ValueError(
                f"sample lengths cannot exceed the input time dimension {time_steps}"
            )

    # The padded tensor width, rather than the largest length in this raw
    # mini-batch, defines N.  This keeps the structured cache shape stable
    # across loader batches while ``true_lengths`` only controls validity.
    if time_steps <= window_size:
        patch_count = 1
    else:
        patch_count = 1 + math.ceil((time_steps - window_size) / stride)

    required_time = (patch_count - 1) * stride + window_size
    if required_time > time_steps:
        padded = F.pad(tensor, (0, required_time - time_steps), value=0)
    else:
        padded = tensor

    patches = padded.unfold(-1, window_size, stride)[..., :patch_count, :]
    patches = patches.permute(0, 2, 1, 3).contiguous()

    starts = torch.arange(
        patch_count,
        dtype=torch.long,
        device=tensor.device,
    ) * stride
    offsets = torch.arange(
        window_size,
        dtype=torch.long,
        device=tensor.device,
    )
    absolute_positions = starts[:, None] + offsets[None, :]
    time_mask = absolute_positions[None, :, :] < true_lengths[:, None, None]
    valid_lengths = time_mask.sum(dim=-1)
    patch_mask = valid_lengths > 0
    valid_fraction = valid_lengths.to(dtype=torch.float32) / float(window_size)

    patches = patches.masked_fill(~time_mask[:, :, None, :], 0)
    return TemporalPatchBatch(
        patches=patches,
        time_mask=time_mask,
        patch_mask=patch_mask,
        valid_lengths=valid_lengths,
        valid_fraction=valid_fraction,
    )


class ChannelAttentionPool(nn.Module):
    """Trainable channel pooling for frozen per-channel Mantis tokens."""

    def __init__(
        self,
        input_dim,
        output_dim,
        num_channels,
        hidden_dim=64,
    ):
        super().__init__()
        if min(input_dim, output_dim, num_channels, hidden_dim) <= 0:
            raise ValueError("channel-attention dimensions must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_channels = int(num_channels)
        self.value_projection = nn.Linear(input_dim, output_dim)
        self.value_norm = nn.LayerNorm(output_dim)
        self.channel_embeddings = nn.Parameter(
            torch.empty(num_channels, output_dim)
        )
        self.scorer = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.channel_embeddings, mean=0.0, std=0.02)

    def forward(self, tokens, patch_mask=None):
        if tokens.ndim != 4:
            raise ValueError(
                "Mantis channel tokens must have shape [B, N, C, D], got "
                f"{tuple(tokens.shape)}"
            )
        if tokens.shape[2] != self.num_channels or tokens.shape[3] != self.input_dim:
            raise ValueError(
                "Mantis channel-token shape does not match the configured "
                f"[C={self.num_channels}, D={self.input_dim}]: {tuple(tokens.shape)}"
            )

        values = self.value_projection(tokens)
        score_inputs = self.value_norm(values) + self.channel_embeddings.view(
            1, 1, self.num_channels, self.output_dim
        )
        scores = self.scorer(score_inputs).squeeze(-1)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=values.dtype)
        pooled = torch.sum(weights.unsqueeze(-1) * values, dim=2)

        if patch_mask is not None:
            if patch_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"patch_mask must have shape {tuple(tokens.shape[:2])}, "
                    f"got {tuple(patch_mask.shape)}"
                )
            valid = patch_mask.to(dtype=torch.bool, device=tokens.device)
            weights = weights * valid.unsqueeze(-1)
            pooled = pooled * valid.unsqueeze(-1)
        return pooled, weights


class PatchGranularityCrossAttention(nn.Module):
    """Line-Q/Activity-Graph-KV routing with versioned policies.

    TimeMosaic motivates making an independent decision for every temporal
    window, while Pathformer motivates keeping multiple scale experts.  The
    actual routing rule here is task-specific.  Legacy v4 uses bounded RBF
    distances and learned hierarchical shrinkage.  V4.1 retains the same
    dataset/sample/patch hierarchy but restores a v3-like dot-product path,
    augments it with a small scale-specific interaction MLP, and admits local
    evidence only in proportion to its detached top1-minus-top2 probability
    margin.  This lets the middle scale learn its own decision surface without
    letting uncertain local routing dominate small datasets.

    V5 keeps the task-specific premise explicit: every route logit is produced
    by an interaction between a line-plot query and an Activity-Graph key.  It
    blends patch-local Q/K scores with sample-global Q/K scores, then applies a
    Pathformer-style Top-K mask and re-normalizes only the selected experts.
    The shared relation scorer and zero-initialized diagonal key adapter avoid
    three unrelated scale-specific classifiers while still allowing the
    middle scale to depart from a strict linear midpoint.  No raw-series gate,
    dataset prior, standalone scale bias, evidence gate, or confidence gate is
    used by the V5 route.

    A single ``[B, N, K]`` distribution is shared by all value groups so there
    is exactly one interpretable granularity decision per temporal window.
    """

    def __init__(
        self,
        visual_dim,
        fusion_dim,
        num_heads,
        num_granularities,
        temperature=1.0,
        dropout=0.0,
        line_query_center=None,
        local_mix_max=0.50,
        local_mix_init=0.10,
        global_mix_max=0.75,
        global_mix_init=0.50,
        evidence_half_saturation=0.05,
        minimum_weight=0.0,
        score_cap=1.0,
        router_mode="adaptive_v4",
        scorer_hidden_dim=32,
        confidence_half_saturation=0.05,
        router_top_k=2,
        router_training_noise_std=0.0,
        router_local_weight=0.5,
        router_relation_hidden_dim=16,
        router_relation_residual_scale=0.25,
        router_key_adapter_scale=0.1,
        router_value_adapter_scale=0.1,
    ):
        super().__init__()
        if min(visual_dim, fusion_dim, num_heads) <= 0:
            raise ValueError("cross-attention dimensions must be positive")
        if num_granularities < 2:
            raise ValueError("adaptive granularity requires at least two candidates")
        if fusion_dim % num_heads != 0:
            raise ValueError(
                f"fusion_dim ({fusion_dim}) must be divisible by num_heads ({num_heads})"
            )
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("granularity temperature must be positive and finite")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("router dropout compatibility value must lie in [0, 1)")
        for name, maximum, initial in (
            ("local_mix", local_mix_max, local_mix_init),
            ("global_mix", global_mix_max, global_mix_init),
        ):
            if not math.isfinite(maximum) or not 0.0 < maximum <= 1.0:
                raise ValueError(f"{name}_max must lie in (0, 1]")
            if not math.isfinite(initial) or not 0.0 < initial < maximum:
                raise ValueError(f"{name}_init must lie strictly inside (0, max)")
        if (
            not math.isfinite(evidence_half_saturation)
            or evidence_half_saturation <= 0.0
        ):
            raise ValueError("evidence_half_saturation must be positive and finite")
        if (
            not math.isfinite(minimum_weight)
            or not 0.0 <= minimum_weight < 1.0 / num_granularities
        ):
            raise ValueError(
                "minimum_weight must lie in [0, uniform expert weight)"
            )
        if not math.isfinite(score_cap) or score_cap <= 0.0:
            raise ValueError("score_cap must be positive and finite")
        if router_mode not in {
            "adaptive_v4",
            "adaptive_v41",
            "adaptive_v5",
            "uniform",
        }:
            raise ValueError(
                "router_mode must be adaptive_v4, adaptive_v41, "
                "adaptive_v5, or uniform"
            )
        if scorer_hidden_dim <= 0:
            raise ValueError("scorer_hidden_dim must be positive")
        if (
            not math.isfinite(confidence_half_saturation)
            or confidence_half_saturation <= 0.0
        ):
            raise ValueError(
                "confidence_half_saturation must be positive and finite"
            )
        if isinstance(router_top_k, bool) or not isinstance(router_top_k, int):
            raise ValueError("router_top_k must be an integer")
        if not 1 <= router_top_k <= num_granularities:
            raise ValueError(
                "router_top_k must lie in [1, num_granularities]"
            )
        if (
            not math.isfinite(router_training_noise_std)
            or router_training_noise_std < 0.0
        ):
            raise ValueError(
                "router_training_noise_std must be finite and non-negative"
            )
        if (
            not math.isfinite(router_local_weight)
            or not 0.0 <= router_local_weight <= 1.0
        ):
            raise ValueError("router_local_weight must lie in [0, 1]")
        if router_relation_hidden_dim <= 0:
            raise ValueError("router_relation_hidden_dim must be positive")
        for name, value in (
            ("router_relation_residual_scale", router_relation_residual_scale),
            ("router_key_adapter_scale", router_key_adapter_scale),
            ("router_value_adapter_scale", router_value_adapter_scale),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.visual_dim = int(visual_dim)
        self.fusion_dim = int(fusion_dim)
        self.num_heads = int(num_heads)
        self.num_granularities = int(num_granularities)
        self.temperature = float(temperature)
        # Kept in the public constructor for checkpoint/API compatibility.
        # Routing itself is deterministic so task/alignment visual forwards
        # remain numerically identical; FusionModule applies dropout later.
        self.dropout = float(dropout)
        self.head_dim = fusion_dim // num_heads
        self.local_mix_max = float(local_mix_max)
        self.global_mix_max = float(global_mix_max)
        self.evidence_half_saturation = float(evidence_half_saturation)
        self.minimum_weight = float(minimum_weight)
        self.score_cap = float(score_cap)
        self.router_mode = str(router_mode)
        self.scorer_hidden_dim = int(scorer_hidden_dim)
        self.confidence_half_saturation = float(confidence_half_saturation)
        self.router_top_k = int(router_top_k)
        self.router_training_noise_std = float(router_training_noise_std)
        self.router_local_weight = float(router_local_weight)
        self.router_relation_hidden_dim = int(router_relation_hidden_dim)
        self.router_relation_residual_scale = float(
            router_relation_residual_scale
        )
        self.router_key_adapter_scale = float(router_key_adapter_scale)
        self.router_value_adapter_scale = float(router_value_adapter_scale)

        # Stateless normalization makes the train-only centre a stable buffer;
        # the following projections retain all learnable affine capacity.
        self.query_input_norm = nn.LayerNorm(
            visual_dim,
            elementwise_affine=False,
        )
        self.graph_input_norm = nn.LayerNorm(
            visual_dim,
            elementwise_affine=False,
        )
        self.query_projection = nn.Linear(visual_dim, fusion_dim, bias=False)
        self.key_projection = nn.Linear(visual_dim, fusion_dim, bias=False)
        self.value_projection = nn.Linear(visual_dim, fusion_dim)
        if self.router_mode == "adaptive_v41":
            self.scale_scorers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(4 * self.head_dim, self.scorer_hidden_dim),
                        nn.GELU(),
                        nn.Linear(self.scorer_hidden_dim, 1, bias=False),
                    )
                    for _ in range(self.num_granularities)
                ]
            )
            for scorer in self.scale_scorers:
                nn.init.normal_(scorer[-1].weight, mean=0.0, std=0.02)
        if self.router_mode == "adaptive_v5":
            # One scorer is shared by every scale and head.  Its inputs are
            # exclusively pairwise Q/K relations, so it cannot learn an
            # independent scale-only gate.
            self.v5_relation_scorer = nn.Sequential(
                nn.Linear(
                    2 * self.head_dim,
                    self.router_relation_hidden_dim,
                    bias=False,
                ),
                nn.GELU(),
                nn.Linear(
                    self.router_relation_hidden_dim,
                    1,
                    bias=False,
                ),
            )
            nn.init.normal_(
                self.v5_relation_scorer[-1].weight,
                mean=0.0,
                std=0.02,
            )
            # Multiplicative, bias-free diagonal adapters start as exact
            # identities.  Candidate identity can therefore modulate how a
            # graph key/value is represented, but it cannot emit a route
            # score without interacting with the line query.
            self.v5_key_adapter_delta = nn.Parameter(
                torch.zeros(num_granularities, fusion_dim)
            )
            self.v5_value_adapter_delta = nn.Parameter(
                torch.zeros(num_granularities, fusion_dim)
            )
        self.dataset_prior_logits = nn.Parameter(torch.zeros(num_granularities))
        self.local_mix_logit = nn.Parameter(
            torch.tensor(
                math.log(local_mix_init / (local_mix_max - local_mix_init)),
                dtype=torch.float32,
            )
        )
        self.global_mix_logit = nn.Parameter(
            torch.tensor(
                math.log(global_mix_init / (global_mix_max - global_mix_init)),
                dtype=torch.float32,
            )
        )
        if self.router_mode == "adaptive_v5":
            # State-schema compatibility only: V5 never consults these legacy
            # dataset-prior or learned mix-gate parameters.
            self.dataset_prior_logits.requires_grad_(False)
            self.local_mix_logit.requires_grad_(False)
            self.global_mix_logit.requires_grad_(False)
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)
        self.norm_1 = nn.LayerNorm(fusion_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(fusion_dim, 4 * fusion_dim),
            nn.GELU(),
            nn.Linear(4 * fusion_dim, fusion_dim),
        )
        self.norm_2 = nn.LayerNorm(fusion_dim)

        if line_query_center is None:
            center = torch.zeros(visual_dim, dtype=torch.float32)
        else:
            center = torch.as_tensor(line_query_center, dtype=torch.float32)
            if center.shape != (visual_dim,):
                raise ValueError(
                    "line_query_center must have shape "
                    f"[{visual_dim}], got {tuple(center.shape)}"
                )
            if not torch.isfinite(center).all():
                raise ValueError("line_query_center must be finite")
        self.register_buffer("line_query_center", center.clone())

    @torch.no_grad()
    def set_line_query_center(self, center):
        center = torch.as_tensor(
            center,
            dtype=self.line_query_center.dtype,
            device=self.line_query_center.device,
        )
        if center.shape != self.line_query_center.shape:
            raise ValueError(
                "line-query centre shape mismatch: expected "
                f"{tuple(self.line_query_center.shape)}, got {tuple(center.shape)}"
            )
        if not torch.isfinite(center).all():
            raise ValueError("line-query centre must be finite")
        self.line_query_center.copy_(center)

    def project_query(self, line_tokens):
        normalized = self.query_input_norm(line_tokens.float())
        centered = normalized - self.line_query_center.view(1, 1, -1)
        return torch.tanh(self.query_projection(centered))

    def _project_graph(self, graph_tokens):
        graph_inputs = self.graph_input_norm(graph_tokens.float())
        keys = torch.tanh(self.key_projection(graph_inputs))
        values = self.value_projection(graph_inputs)
        if self.router_mode == "adaptive_v5":
            key_multiplier = (
                1.0
                + self.router_key_adapter_scale
                * self.v5_key_adapter_delta
            )
            value_multiplier = (
                1.0
                + self.router_value_adapter_scale
                * self.v5_value_adapter_delta
            )
            # V5 adapts uncentred keys.  If scale 8 is exactly the midpoint of
            # scales 4 and 16, candidate-centering would map its key to zero;
            # a multiplicative adapter could then never revive it.  Legacy
            # routes retain their historical candidate-centred keys below.
            routed_keys = keys * key_multiplier.view(
                1,
                1,
                self.num_granularities,
                self.fusion_dim,
            )
            values = values * value_multiplier.view(
                1,
                1,
                self.num_granularities,
                self.fusion_dim,
            )
        else:
            routed_keys = keys - keys.mean(dim=2, keepdim=True)
        return routed_keys, values

    def _candidate_evidence(self, graph_tokens):
        normalized = F.normalize(graph_tokens.float(), dim=-1, eps=1e-8)
        dissimilarities = []
        for left_index in range(self.num_granularities):
            for right_index in range(left_index + 1, self.num_granularities):
                # For nonzero embeddings this is exactly 1-cosine.  The
                # distance form also maps two identical zero vectors to zero
                # evidence instead of the undefined-cosine artefact 1.
                dissimilarities.append(
                    0.5
                    * (
                        normalized[:, :, left_index]
                        - normalized[:, :, right_index]
                    ).square().sum(dim=-1)
                )
        evidence = torch.stack(dissimilarities, dim=-1).mean(dim=-1)
        return (
            evidence / (evidence + self.evidence_half_saturation)
        ).detach()

    def _bounded_distribution(self, logits):
        centered = logits.float() - logits.float().mean(dim=-1, keepdim=True)
        bounded = self.score_cap * torch.tanh(
            centered / (self.temperature * self.score_cap)
        )
        weights = torch.softmax(bounded, dim=-1)
        if self.minimum_weight > 0.0:
            residual_mass = 1.0 - self.num_granularities * self.minimum_weight
            weights = self.minimum_weight + residual_mass * weights
        return weights

    def _v41_distribution(self, logits):
        """Bound scores first, then apply a genuine Softmax temperature."""
        centered = logits.float() - logits.float().mean(dim=-1, keepdim=True)
        bounded = self.score_cap * torch.tanh(centered / self.score_cap)
        weights = torch.softmax(bounded / self.temperature, dim=-1)
        if self.minimum_weight > 0.0:
            residual_mass = 1.0 - self.num_granularities * self.minimum_weight
            weights = self.minimum_weight + residual_mass * weights
        return weights

    def _v41_scores(self, query_heads, key_heads):
        """Return scale-specific dot-plus-interaction scores as ``[B,N,H,K]``."""
        scores = []
        scale = math.sqrt(self.head_dim)
        for candidate_index, scorer in enumerate(self.scale_scorers):
            candidate_keys = key_heads[:, :, candidate_index]
            interactions = torch.cat(
                (
                    query_heads,
                    candidate_keys,
                    query_heads - candidate_keys,
                    query_heads * candidate_keys,
                ),
                dim=-1,
            )
            nonlinear_score = scorer(interactions).squeeze(-1)
            dot_score = torch.sum(
                query_heads * candidate_keys,
                dim=-1,
            ) / scale
            scores.append(dot_score + nonlinear_score)
        return torch.stack(scores, dim=-1)

    def _v5_relation_scores(self, query_heads, key_heads):
        """Return shared dot-plus-relation Q/K scores as ``[B,N,H,K]``.

        ``query_heads`` is ``[B,N,H,Dh]`` and ``key_heads`` is
        ``[B,N,K,H,Dh]``.  Candidate identity is deliberately absent from the
        scorer; the only candidate-specific capacity is the bias-free diagonal
        key adapter applied in :meth:`_project_graph`.
        """
        query_by_candidate = query_heads.unsqueeze(2).expand(
            -1,
            -1,
            self.num_granularities,
            -1,
            -1,
        )
        relation_inputs = torch.cat(
            (
                query_by_candidate * key_heads,
                torch.abs(query_by_candidate - key_heads),
            ),
            dim=-1,
        )
        relation_residual = self.v5_relation_scorer(
            relation_inputs
        ).squeeze(-1)
        dot_scores = torch.sum(
            query_by_candidate * key_heads,
            dim=-1,
        ) / math.sqrt(self.head_dim)
        scores = (
            dot_scores
            + self.router_relation_residual_scale * relation_residual
        )
        return scores.permute(0, 1, 3, 2)

    def _v5_topk_distribution(self, logits, add_training_noise=False):
        """Mask to Top-K experts and re-Softmax the selected logits."""
        route_logits = logits.float()
        if (
            add_training_noise
            and self.training
            and self.router_training_noise_std > 0.0
        ):
            route_logits = route_logits + torch.randn_like(route_logits) * (
                self.router_training_noise_std
            )
        topk_indices = torch.topk(
            route_logits,
            k=self.router_top_k,
            dim=-1,
        ).indices
        topk_mask = torch.zeros_like(route_logits, dtype=torch.bool)
        topk_mask.scatter_(-1, topk_indices, True)
        masked_logits = route_logits.masked_fill(~topk_mask, float("-inf"))
        weights = torch.softmax(masked_logits / self.temperature, dim=-1)
        return weights, topk_mask, topk_indices, route_logits

    def _v5_routing_distribution(
        self,
        projected_query,
        relative_keys,
        patch_mask=None,
        valid_fraction=None,
    ):
        """Route with local and sample-global line-Q/graph-K interactions."""
        local_query_heads = projected_query.float().reshape(
            *projected_query.shape[:2],
            self.num_heads,
            self.head_dim,
        )
        local_key_heads = relative_keys.float().reshape(
            *relative_keys.shape[:3],
            self.num_heads,
            self.head_dim,
        )
        if patch_mask is None:
            valid = torch.ones(
                projected_query.shape[:2],
                dtype=torch.float32,
                device=projected_query.device,
            )
        else:
            valid = patch_mask.to(
                device=projected_query.device,
                dtype=torch.float32,
            )
        if valid_fraction is None:
            importance = valid
        else:
            if valid_fraction.shape != projected_query.shape[:2]:
                raise ValueError(
                    "valid_fraction must share the query batch/patch axes"
                )
            importance = valid * valid_fraction.to(
                device=projected_query.device,
                dtype=torch.float32,
            )
        denominator = importance.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # Both sides of the global interaction are pooled.  In particular,
        # the global query is never compared against a patch-local graph key.
        sample_query_heads = torch.sum(
            local_query_heads * importance[:, :, None, None],
            dim=1,
            keepdim=True,
        ) / denominator[:, :, None, None]
        sample_key_heads = torch.sum(
            local_key_heads * importance[:, :, None, None, None],
            dim=1,
            keepdim=True,
        ) / denominator[:, :, None, None, None]

        local_head_scores = self._v5_relation_scores(
            local_query_heads,
            local_key_heads,
        )
        global_head_scores = self._v5_relation_scores(
            sample_query_heads,
            sample_key_heads,
        )
        patch_local_route_scores = local_head_scores.mean(dim=2)
        sample_global_route_scores = global_head_scores.mean(dim=2).expand(
            -1,
            projected_query.shape[1],
            -1,
        )
        clean_scores = (
            self.router_local_weight * patch_local_route_scores
            + (1.0 - self.router_local_weight) * sample_global_route_scores
        )

        # The clean pre-TopK probabilities provide the differentiable load
        # proxy.  Optional Pathformer-like noise affects expert selection only
        # during training and is disabled by default.
        pre_topk_weights = torch.softmax(
            clean_scores / self.temperature,
            dim=-1,
        )
        (
            shared_weights,
            topk_mask,
            topk_indices,
            route_logits,
        ) = self._v5_topk_distribution(
            clean_scores,
            add_training_noise=True,
        )
        sample_global_weights, _, _, _ = self._v5_topk_distribution(
            sample_global_route_scores,
            add_training_noise=False,
        )
        patch_local_weights, _, _, _ = self._v5_topk_distribution(
            patch_local_route_scores,
            add_training_noise=False,
        )
        per_head_weights = shared_weights.unsqueeze(2).expand(
            -1,
            -1,
            self.num_heads,
            -1,
        )
        shared_scores = shared_weights.clamp_min(1e-12).log()

        sample_probabilities = torch.softmax(
            sample_global_route_scores / self.temperature,
            dim=-1,
        )
        local_probabilities = torch.softmax(
            patch_local_route_scores / self.temperature,
            dim=-1,
        )
        sample_top = torch.topk(
            sample_probabilities,
            k=2,
            dim=-1,
        ).values
        local_top = torch.topk(
            local_probabilities,
            k=2,
            dim=-1,
        ).values
        sample_margin = sample_top[..., 0] - sample_top[..., 1]
        local_margin = local_top[..., 0] - local_top[..., 1]

        batch_size, patch_count = projected_query.shape[:2]
        uniform_weights = torch.full_like(
            shared_weights,
            1.0 / self.num_granularities,
        )
        zero_patch = torch.zeros(
            (batch_size, patch_count),
            dtype=shared_weights.dtype,
            device=shared_weights.device,
        )
        local_mix = shared_weights.new_tensor(self.router_local_weight)
        global_mix = shared_weights.new_tensor(1.0 - self.router_local_weight)
        diagnostics = {
            # Kept only as an explicit no-prior compatibility diagnostic; it
            # does not participate in V5 score construction.
            "dataset_prior_weights": uniform_weights,
            "sample_global_weights": sample_global_weights,
            "patch_local_weights": patch_local_weights,
            "candidate_evidence": zero_patch,
            "global_mix": global_mix,
            "local_mix": local_mix,
            "sample_global_score_margin": sample_margin,
            "patch_local_score_margin": local_margin,
            "sample_global_confidence": zero_patch,
            "patch_local_confidence": zero_patch,
            "effective_global_mix": zero_patch + global_mix,
            "effective_local_mix": zero_patch + local_mix,
            "sample_global_route_scores": sample_global_route_scores,
            "patch_local_route_scores": patch_local_route_scores,
            "router_pre_topk_scores": clean_scores,
            "router_pre_topk_weights": pre_topk_weights,
            "router_topk_mask": topk_mask,
            "router_topk_indices": topk_indices,
            "router_noisy_topk_scores": route_logits,
            "router_noise_std": shared_weights.new_tensor(
                self.router_training_noise_std
            ),
        }
        return shared_scores, shared_weights, per_head_weights, diagnostics

    def _margin_confidence(self, weights):
        """Return probability margin and a detached monotone confidence gate."""
        top_weights = torch.topk(weights.float(), k=2, dim=-1).values
        margin = top_weights[..., 0] - top_weights[..., 1]
        detached_margin = margin.detach()
        confidence = detached_margin / (
            detached_margin + self.confidence_half_saturation
        )
        return margin, confidence

    def _route_distribution(self, logits):
        if self.router_mode == "adaptive_v41":
            return self._v41_distribution(logits)
        return self._bounded_distribution(logits)

    def _routing_distribution(
        self,
        projected_query,
        relative_keys,
        graph_tokens,
        patch_mask=None,
        valid_fraction=None,
    ):
        if self.router_mode == "adaptive_v5":
            return self._v5_routing_distribution(
                projected_query,
                relative_keys,
                patch_mask=patch_mask,
                valid_fraction=valid_fraction,
            )
        local_query_heads = projected_query.float().reshape(
            *projected_query.shape[:2],
            self.num_heads,
            self.head_dim,
        )
        key_heads = relative_keys.float().reshape(
            *relative_keys.shape[:3],
            self.num_heads,
            self.head_dim,
        )
        if patch_mask is None:
            valid = torch.ones(
                projected_query.shape[:2],
                dtype=torch.float32,
                device=projected_query.device,
            )
        else:
            valid = patch_mask.to(device=projected_query.device, dtype=torch.float32)
        if valid_fraction is None:
            importance = valid
        else:
            if valid_fraction.shape != projected_query.shape[:2]:
                raise ValueError("valid_fraction must share the query batch/patch axes")
            importance = valid * valid_fraction.to(
                device=projected_query.device,
                dtype=torch.float32,
            )
        denominator = importance.sum(dim=1, keepdim=True).clamp_min(1e-8)
        sample_query_heads = torch.sum(
            local_query_heads * importance[:, :, None, None],
            dim=1,
            keepdim=True,
        ) / denominator[:, :, None, None]
        sample_query_heads = sample_query_heads.expand_as(local_query_heads)

        if self.router_mode == "adaptive_v41":
            local_route_scores = self._v41_scores(local_query_heads, key_heads)
            sample_route_scores = self._v41_scores(sample_query_heads, key_heads)
            distribution = self._v41_distribution
        else:
            local_route_scores = -0.5 * (
                local_query_heads.unsqueeze(2) - key_heads
            ).square().mean(dim=-1)
            sample_route_scores = -0.5 * (
                sample_query_heads.unsqueeze(2) - key_heads
            ).square().mean(dim=-1)
            # RBF tensors are [B,N,K,H]; expose head before candidate.
            local_route_scores = local_route_scores.permute(0, 1, 3, 2)
            sample_route_scores = sample_route_scores.permute(0, 1, 3, 2)
            distribution = self._bounded_distribution
        prior_logits = self.dataset_prior_logits.float().view(
            1,
            1,
            1,
            self.num_granularities,
        )
        prior_weights = distribution(prior_logits).expand(
            projected_query.shape[0],
            projected_query.shape[1],
            self.num_heads,
            -1,
        )
        sample_weights = distribution(
            prior_logits + sample_route_scores
        )
        local_weights = distribution(
            prior_logits + local_route_scores
        )
        evidence = self._candidate_evidence(graph_tokens)
        global_mix = self.global_mix_max * torch.sigmoid(self.global_mix_logit)
        local_mix = self.local_mix_max * torch.sigmoid(self.local_mix_logit)
        if self.router_mode == "adaptive_v41":
            sample_score_margin, sample_confidence = self._margin_confidence(
                sample_weights
            )
            local_score_margin, local_confidence = self._margin_confidence(
                local_weights
            )
            effective_global_mix = (
                evidence[:, :, None, None]
                * global_mix
                * sample_confidence.unsqueeze(-1)
            )
            effective_local_mix = (
                evidence[:, :, None, None]
                * local_mix
                * local_confidence.unsqueeze(-1)
            )
        else:
            zero_confidence = torch.zeros_like(sample_weights[..., 0])
            sample_score_margin = zero_confidence
            local_score_margin = zero_confidence
            sample_confidence = zero_confidence
            local_confidence = zero_confidence
            effective_global_mix = evidence[:, :, None, None] * global_mix
            effective_local_mix = evidence[:, :, None, None] * local_mix
        dataset_global_weights = (
            (1.0 - effective_global_mix) * prior_weights
            + effective_global_mix * sample_weights
        )
        per_head_weights = (
            (1.0 - effective_local_mix) * dataset_global_weights
            + effective_local_mix * local_weights
        )
        shared_weights = per_head_weights.mean(dim=2)
        # log(probability) is a shared-logit representation.  The clamp keeps
        # diagnostics finite in float underflow/zero-weight override cases;
        # softmax(scores) therefore reconstructs weights to numeric tolerance.
        shared_scores = shared_weights.clamp_min(1e-12).log()
        diagnostics = {
            "dataset_prior_weights": prior_weights.mean(dim=2),
            "sample_global_weights": sample_weights.mean(dim=2),
            "patch_local_weights": local_weights.mean(dim=2),
            "candidate_evidence": evidence,
            "global_mix": global_mix,
            "local_mix": local_mix,
            "sample_global_score_margin": sample_score_margin.mean(dim=2),
            "patch_local_score_margin": local_score_margin.mean(dim=2),
            "sample_global_confidence": sample_confidence.mean(dim=2),
            "patch_local_confidence": local_confidence.mean(dim=2),
            "effective_global_mix": effective_global_mix.mean(dim=2).squeeze(-1),
            "effective_local_mix": effective_local_mix.mean(dim=2).squeeze(-1),
            # Common diagnostic names make result files comparable across
            # router versions.  Legacy modes remain dense and unchanged.
            "sample_global_route_scores": sample_weights.mean(dim=2).clamp_min(
                1e-12
            ).log(),
            "patch_local_route_scores": local_weights.mean(dim=2).clamp_min(
                1e-12
            ).log(),
            "router_pre_topk_scores": shared_scores,
            "router_pre_topk_weights": shared_weights,
            "router_topk_mask": shared_weights > 0.0,
            "router_topk_indices": torch.topk(
                shared_weights,
                k=min(self.router_top_k, self.num_granularities),
                dim=-1,
            ).indices,
            "router_noisy_topk_scores": shared_scores,
            "router_noise_std": shared_weights.new_zeros(()),
        }
        return shared_scores, shared_weights, per_head_weights, diagnostics

    def _postprocess_context(self, context):
        context = self.output_projection(context)
        visual_tokens = self.norm_1(context)
        return self.norm_2(
            visual_tokens + self.feed_forward(visual_tokens)
        )

    def _validate_weight_override(self, weights, line_tokens, patch_mask):
        if weights.shape != (*line_tokens.shape[:2], self.num_granularities):
            raise ValueError(
                "granularity weight override must have shape "
                f"{(*line_tokens.shape[:2], self.num_granularities)}, "
                f"got {tuple(weights.shape)}"
            )
        if not torch.isfinite(weights).all() or torch.any(weights < 0.0):
            raise ValueError("granularity weight override must be finite and non-negative")
        if patch_mask is None:
            valid = torch.ones(
                line_tokens.shape[:2],
                dtype=torch.bool,
                device=line_tokens.device,
            )
        else:
            valid = patch_mask.to(dtype=torch.bool, device=line_tokens.device)
        row_sums = weights.float().sum(dim=-1)
        if not torch.allclose(
            row_sums[valid],
            torch.ones_like(row_sums[valid]),
            rtol=1e-4,
            atol=1e-5,
        ):
            raise ValueError("valid granularity weight override rows must sum to one")
        return weights

    def forward(
        self,
        line_tokens,
        graph_tokens,
        patch_mask=None,
        valid_fraction=None,
        granularity_weights_override=None,
        return_router_diagnostics=False,
    ):
        if line_tokens.ndim != 3:
            raise ValueError(
                f"line tokens must have shape [B, N, D], got {tuple(line_tokens.shape)}"
            )
        if graph_tokens.ndim != 4:
            raise ValueError(
                "graph tokens must have shape [B, N, K, D], got "
                f"{tuple(graph_tokens.shape)}"
            )
        if graph_tokens.shape[:2] != line_tokens.shape[:2]:
            raise ValueError("line and graph tokens must share batch and patch axes")
        if line_tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                f"line token width must be {self.visual_dim}, got {line_tokens.shape[-1]}"
            )
        if (
            graph_tokens.shape[2] != self.num_granularities
            or graph_tokens.shape[3] != self.visual_dim
        ):
            raise ValueError(
                "graph-token shape does not match configured "
                f"[K={self.num_granularities}, D={self.visual_dim}]: "
                f"{tuple(graph_tokens.shape)}"
            )

        projected_query = self.project_query(line_tokens)
        relative_keys, values = self._project_graph(graph_tokens)
        (
            scores,
            routed_weights,
            routed_per_head_weights,
            router_diagnostics,
        ) = (
            self._routing_distribution(
                projected_query,
                relative_keys,
                graph_tokens,
                patch_mask=patch_mask,
                valid_fraction=valid_fraction,
            )
        )
        routed_weights = routed_weights.to(dtype=values.dtype)
        if granularity_weights_override is None:
            mean_weights = routed_weights
            per_head_weights = routed_per_head_weights.to(dtype=values.dtype)
        else:
            mean_weights = self._validate_weight_override(
                granularity_weights_override.to(
                    device=values.device,
                    dtype=values.dtype,
                ),
                line_tokens,
                patch_mask,
            )
            per_head_weights = mean_weights.unsqueeze(2).expand(
                -1, -1, self.num_heads, -1
            )
            scores = mean_weights.float().clamp_min(1e-12).log()

        # Classification receives task gradients through the soft weights.
        task_context = torch.sum(mean_weights.unsqueeze(-1) * values, dim=2)
        visual_tokens = self._postprocess_context(task_context)

        # Alignment sees the numerically identical selected visual token but
        # cannot train Q/K routing.  Values and shared post-processing remain
        # trainable through the alignment objective.
        alignment_context = torch.sum(
            mean_weights.detach().unsqueeze(-1) * values,
            dim=2,
        )
        alignment_visual_tokens = self._postprocess_context(alignment_context)
        if patch_mask is not None:
            if patch_mask.shape != line_tokens.shape[:2]:
                raise ValueError(
                    f"patch_mask must have shape {tuple(line_tokens.shape[:2])}, "
                    f"got {tuple(patch_mask.shape)}"
                )
            valid = patch_mask.to(dtype=torch.bool, device=line_tokens.device)
            visual_tokens = visual_tokens * valid.unsqueeze(-1)
            alignment_visual_tokens = alignment_visual_tokens * valid.unsqueeze(-1)
            mean_weights = mean_weights * valid.unsqueeze(-1)
            per_head_weights = per_head_weights * valid.unsqueeze(-1).unsqueeze(-1)
            scores = scores * valid.unsqueeze(-1)
            for name in (
                "dataset_prior_weights",
                "sample_global_weights",
                "patch_local_weights",
                "sample_global_route_scores",
                "patch_local_route_scores",
                "router_pre_topk_scores",
                "router_pre_topk_weights",
                "router_noisy_topk_scores",
            ):
                router_diagnostics[name] = (
                    router_diagnostics[name] * valid.unsqueeze(-1)
                )
            router_diagnostics["router_topk_mask"] = (
                router_diagnostics["router_topk_mask"]
                & valid.unsqueeze(-1)
            )
            router_diagnostics["router_topk_indices"] = (
                router_diagnostics["router_topk_indices"].masked_fill(
                    ~valid.unsqueeze(-1),
                    -1,
                )
            )
            for name in (
                "candidate_evidence",
                "sample_global_score_margin",
                "patch_local_score_margin",
                "sample_global_confidence",
                "patch_local_confidence",
                "effective_global_mix",
                "effective_local_mix",
            ):
                router_diagnostics[name] = router_diagnostics[name] * valid
        outputs = (
            visual_tokens,
            alignment_visual_tokens,
            mean_weights,
            per_head_weights,
            scores,
        )
        if return_router_diagnostics:
            return (*outputs, router_diagnostics)
        return outputs


def masked_granularity_regularization_terms(
    granularity_weights,
    patch_mask,
    valid_fraction=None,
    usage_reference=None,
    usage_floor=0.05,
    entropy_floor=0.0,
    entropy_ceiling=0.90,
    eps=1e-8,
    return_entropy_components=False,
):
    """Return expert-survival and bounded-entropy penalties.

    The usage term activates when the supplied usage estimate (or the current
    valid-time marginal when omitted) falls below ``usage_floor``; unlike
    per-sample uniform balancing, it does not force every sample to use all
    experts.  The entropy band penalizes both collapse below ``entropy_floor``
    and excessive uncertainty above ``entropy_ceiling``. Partial tail patches
    contribute only their true valid-time fraction.
    """
    if granularity_weights.ndim != 3:
        raise ValueError("granularity_weights must have shape [B, N, K]")
    if patch_mask.shape != granularity_weights.shape[:2]:
        raise ValueError("patch_mask must share granularity batch/patch axes")
    if granularity_weights.shape[-1] < 2:
        raise ValueError("granularity regularization requires at least two candidates")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    if not math.isfinite(usage_floor) or not 0.0 <= usage_floor < 1.0:
        raise ValueError("usage_floor must lie in [0, 1)")
    if usage_floor > 1.0 / granularity_weights.shape[-1]:
        raise ValueError(
            "usage_floor cannot exceed the uniform per-expert usage "
            f"{1.0 / granularity_weights.shape[-1]:.6f}"
        )
    if (
        not math.isfinite(entropy_floor)
        or not 0.0 <= entropy_floor <= 1.0
    ):
        raise ValueError("entropy_floor must lie in [0, 1]")
    if (
        not math.isfinite(entropy_ceiling)
        or not 0.0 <= entropy_ceiling <= 1.0
    ):
        raise ValueError("entropy_ceiling must lie in [0, 1]")
    if entropy_floor > entropy_ceiling:
        raise ValueError("entropy_floor cannot exceed entropy_ceiling")
    if not torch.isfinite(granularity_weights).all():
        raise ValueError("granularity_weights must be finite")

    valid = patch_mask.to(
        device=granularity_weights.device,
        dtype=granularity_weights.dtype,
    )
    if valid_fraction is None:
        importance = valid
    else:
        if valid_fraction.shape != patch_mask.shape:
            raise ValueError("valid_fraction must have the same shape as patch_mask")
        if not torch.isfinite(valid_fraction).all():
            raise ValueError("valid_fraction must be finite")
        fraction = valid_fraction.to(
            device=granularity_weights.device,
            dtype=granularity_weights.dtype,
        )
        if torch.any(fraction < 0.0) or torch.any(fraction > 1.0):
            raise ValueError("valid_fraction must lie in [0, 1]")
        importance = valid * fraction

    denominator = importance.sum()
    if bool(denominator <= 0.0):
        raise ValueError("granularity regularization requires a valid patch")

    valid_rows = patch_mask.to(
        device=granularity_weights.device,
        dtype=torch.bool,
    )
    row_sums = granularity_weights[valid_rows].float().sum(dim=-1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        rtol=1e-4,
        atol=1e-5,
    ):
        raise ValueError("valid granularity-weight rows must sum to one")

    weights = granularity_weights.float().clamp_min(0.0)
    importance_float = importance.float()
    denominator_float = denominator.float()
    candidate_count = weights.shape[-1]
    log_candidate_count = math.log(candidate_count)

    marginal = torch.sum(
        weights * importance_float.unsqueeze(-1),
        dim=(0, 1),
    ) / denominator_float
    if usage_reference is None:
        usage = marginal
    else:
        if usage_reference.shape != marginal.shape:
            raise ValueError(
                "usage_reference must have shape "
                f"{tuple(marginal.shape)}, got {tuple(usage_reference.shape)}"
            )
        usage = usage_reference.float()
        if not torch.isfinite(usage).all() or torch.any(usage < 0.0):
            raise ValueError("usage_reference must be finite and non-negative")
    if usage_floor == 0.0:
        usage_penalty = marginal.sum() * 0.0
    else:
        deficit = torch.relu(usage_floor - usage)
        usage_penalty = torch.mean(deficit.square()) / (usage_floor ** 2)

    safe_weights = weights.clamp_min(eps)
    per_patch_entropy = -torch.sum(weights * safe_weights.log(), dim=-1)
    mean_normalized_entropy = torch.sum(
        per_patch_entropy * importance_float
    ) / (denominator_float * log_candidate_count)
    entropy_floor_penalty = torch.relu(
        entropy_floor - mean_normalized_entropy
    ).square()
    entropy_ceiling_penalty = torch.relu(
        mean_normalized_entropy - entropy_ceiling
    ).square()
    if return_entropy_components:
        return (
            usage_penalty,
            entropy_floor_penalty,
            entropy_ceiling_penalty,
            marginal,
        )
    return (
        usage_penalty,
        entropy_floor_penalty + entropy_ceiling_penalty,
        marginal,
    )


def masked_v5_router_regularization_terms(
    post_topk_weights,
    pre_topk_weights,
    topk_mask,
    patch_mask,
    eps=1e-8,
    valid_fraction=None,
):
    """Return V5 soft-budget and differentiable load-balancing terms.

    TimeMosaic motivates controlling the aggregate granularity budget, while
    Pathformer motivates balanced expert importance/load.  To prevent samples
    with more valid windows from dominating either term, weights are first
    averaged over each sample's valid-time mass and only then averaged equally
    over samples.  ``valid_fraction`` optionally downweights a partial tail
    patch.  The route-budget MSE uses the post-TopK soft route.  CV2 uses the
    *clean pre-TopK* Softmax so it remains differentiable even when
    deterministic Top-K support does not.  Hard-support CV2 is returned only
    as a diagnostic.
    """
    for name, tensor in (
        ("post_topk_weights", post_topk_weights),
        ("pre_topk_weights", pre_topk_weights),
        ("topk_mask", topk_mask),
    ):
        if tensor.ndim != 3:
            raise ValueError(f"{name} must have shape [B, N, K]")
    if pre_topk_weights.shape != post_topk_weights.shape:
        raise ValueError("pre/post TopK weights must have the same shape")
    if topk_mask.shape != post_topk_weights.shape:
        raise ValueError("topk_mask must share the route-weight shape")
    if patch_mask.shape != post_topk_weights.shape[:2]:
        raise ValueError("patch_mask must share the route batch/patch axes")
    if post_topk_weights.shape[-1] < 2:
        raise ValueError("V5 router regularization requires at least two experts")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    if not torch.isfinite(post_topk_weights).all():
        raise ValueError("post_topk_weights must be finite")
    if not torch.isfinite(pre_topk_weights).all():
        raise ValueError("pre_topk_weights must be finite")

    valid_bool = patch_mask.to(
        device=post_topk_weights.device,
        dtype=torch.bool,
    )
    valid = valid_bool.to(dtype=torch.float32)
    if valid_fraction is None:
        importance = valid
    else:
        if valid_fraction.shape != patch_mask.shape:
            raise ValueError(
                "valid_fraction must have the same shape as patch_mask"
            )
        if not torch.isfinite(valid_fraction).all():
            raise ValueError("valid_fraction must be finite")
        fraction = valid_fraction.to(
            device=post_topk_weights.device,
            dtype=torch.float32,
        )
        if torch.any(fraction < 0.0) or torch.any(fraction > 1.0):
            raise ValueError("valid_fraction must lie in [0, 1]")
        importance = valid * fraction
    sample_mass = importance.sum(dim=1, keepdim=True)
    if torch.any(sample_mass <= 0.0):
        raise ValueError("every sample requires at least one valid route patch")
    post = post_topk_weights.float()
    pre = pre_topk_weights.float()
    for name, weights in (("post", post), ("pre", pre)):
        row_sums = weights[valid_bool].sum(dim=-1)
        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=1e-4,
            atol=1e-5,
        ):
            raise ValueError(f"valid {name}-TopK route rows must sum to one")

    sample_post_usage = torch.sum(
        post * importance.unsqueeze(-1),
        dim=1,
    ) / sample_mass
    sample_pre_usage = torch.sum(
        pre * importance.unsqueeze(-1),
        dim=1,
    ) / sample_mass
    support = topk_mask.to(
        device=post_topk_weights.device,
        dtype=torch.float32,
    )
    sample_hard_load = torch.sum(
        support * importance.unsqueeze(-1),
        dim=1,
    ) / sample_mass

    post_marginal = sample_post_usage.mean(dim=0)
    pre_marginal = sample_pre_usage.mean(dim=0)
    hard_load_marginal = sample_hard_load.mean(dim=0)
    target = torch.full_like(
        post_marginal,
        1.0 / post_topk_weights.shape[-1],
    )
    route_budget_loss = torch.mean((post_marginal - target).square())

    def coefficient_of_variation_squared(values):
        mean = values.mean()
        return torch.mean((values - mean).square()) / (
            mean.square() + eps
        )

    load_cv2_loss = coefficient_of_variation_squared(pre_marginal)
    hard_load_cv2 = coefficient_of_variation_squared(
        hard_load_marginal
    ).detach()
    return {
        "route_budget_loss": route_budget_loss,
        "load_cv2_loss": load_cv2_loss,
        "hard_load_cv2": hard_load_cv2,
        "post_topk_marginal": post_marginal,
        "pre_topk_marginal": pre_marginal,
        "hard_load_marginal": hard_load_marginal.detach(),
    }


class MaskedIntraSampleInfoNCE(nn.Module):
    """Symmetric N-by-N patch alignment with no cross-batch negatives.

    Each sample loss is divided by ``log(N_valid)`` so a fixed alignment
    coefficient has a comparable random-logit scale across datasets with
    different numbers of outer windows.
    """

    def __init__(
        self,
        temporal_dim,
        visual_dim,
        projection_dim,
        temperature=0.1,
    ):
        super().__init__()
        if min(temporal_dim, visual_dim, projection_dim) <= 0:
            raise ValueError("alignment dimensions must be positive")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("alignment temperature must be positive and finite")
        self.temperature = float(temperature)
        self.temporal_projection = nn.Linear(temporal_dim, projection_dim)
        self.visual_projection = nn.Linear(visual_dim, projection_dim)

    def forward(
        self,
        temporal_tokens,
        visual_tokens,
        patch_mask,
        valid_fraction=None,
    ):
        if temporal_tokens.ndim != 3 or visual_tokens.ndim != 3:
            raise ValueError("alignment tokens must have shape [B, N, D]")
        if temporal_tokens.shape[:2] != visual_tokens.shape[:2]:
            raise ValueError("temporal and visual tokens must share batch/patch axes")
        if patch_mask.shape != temporal_tokens.shape[:2]:
            raise ValueError("patch_mask must share the alignment batch/patch axes")
        if valid_fraction is not None and valid_fraction.shape != patch_mask.shape:
            raise ValueError("valid_fraction must have the same shape as patch_mask")

        temporal = F.normalize(
            self.temporal_projection(temporal_tokens), dim=-1, eps=1e-8
        )
        visual = F.normalize(
            self.visual_projection(visual_tokens), dim=-1, eps=1e-8
        )
        zero_loss = temporal.sum() * 0.0 + visual.sum() * 0.0
        losses = []
        for sample_index in range(temporal.shape[0]):
            valid_indices = torch.nonzero(
                patch_mask[sample_index].to(dtype=torch.bool),
                as_tuple=False,
            ).flatten()
            if valid_indices.numel() < 2:
                continue
            sample_temporal = temporal[sample_index].index_select(0, valid_indices)
            sample_visual = visual[sample_index].index_select(0, valid_indices)
            logits = (
                sample_temporal.float() @ sample_visual.float().transpose(0, 1)
            ) / self.temperature
            targets = torch.arange(logits.shape[0], device=logits.device)
            row_losses = F.cross_entropy(logits, targets, reduction="none")
            column_losses = F.cross_entropy(
                logits.transpose(0, 1), targets, reduction="none"
            )
            if valid_fraction is None:
                sample_loss = 0.5 * (row_losses.mean() + column_losses.mean())
            else:
                anchor_weights = valid_fraction[sample_index].index_select(
                    0, valid_indices
                ).float().clamp_min(0.0)
                denominator = anchor_weights.sum().clamp_min(1e-8)
                sample_loss = 0.5 * (
                    torch.sum(row_losses * anchor_weights) / denominator
                    + torch.sum(column_losses * anchor_weights) / denominator
                )
            losses.append(sample_loss / math.log(int(valid_indices.numel())))

        if not losses:
            return zero_loss
        return torch.stack(losses).mean()


class _MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super().__init__()
        if num_layers < 1:
            raise ValueError("MLP must contain at least one linear layer")
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
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


def valid_fraction_weighted_pool(
    patch_features,
    patch_mask,
    valid_fraction,
):
    """Pool valid patches while giving a partial tail its true time weight."""
    if patch_features.ndim != 3:
        raise ValueError("patch_features must have shape [B, N, D]")
    if patch_mask.shape != patch_features.shape[:2]:
        raise ValueError("patch_mask must share the feature batch/patch axes")
    if valid_fraction.shape != patch_mask.shape:
        raise ValueError("valid_fraction must have the same shape as patch_mask")
    pooling_weights = (
        valid_fraction.to(dtype=patch_features.dtype, device=patch_features.device)
        * patch_mask.to(dtype=patch_features.dtype, device=patch_features.device)
    )
    denominators = pooling_weights.sum(dim=1, keepdim=True)
    if torch.any(denominators <= 0):
        raise ValueError("every sample must contain at least one valid patch")
    return torch.sum(
        patch_features * pooling_weights.unsqueeze(-1), dim=1
    ) / denominators


class PatchMindTSFusionModule(nn.Module):
    """Patch-internal fusion followed by valid-fraction weighted pooling."""

    def __init__(
        self,
        visual_dim,
        temporal_dim,
        num_channels,
        num_granularities,
        num_classes,
        fusion_dim=512,
        fusion_heads=4,
        dropout=0.1,
        classifier_hidden_dim=512,
        classifier_num_layers=2,
        channel_hidden_dim=64,
        granularity_temperature=1.0,
        alignment_dim=256,
        alignment_temperature=0.1,
        granularity_usage_floor=0.05,
        granularity_usage_ema_decay=0.95,
        granularity_entropy_floor=0.55,
        granularity_entropy_ceiling=1.0,
        line_query_center=None,
        granularity_router_mode="adaptive_v4",
        granularity_local_mix_max=0.50,
        granularity_local_mix_init=0.10,
        granularity_global_mix_max=0.75,
        granularity_global_mix_init=0.50,
        granularity_evidence_half_saturation=0.05,
        granularity_minimum_weight=0.0,
        granularity_score_cap=1.0,
        granularity_scorer_hidden_dim=32,
        granularity_confidence_half_saturation=0.05,
        router_top_k=2,
        router_training_noise_std=0.0,
        router_local_weight=0.5,
        router_relation_hidden_dim=16,
        router_relation_residual_scale=0.25,
        router_key_adapter_scale=0.1,
        router_value_adapter_scale=0.1,
    ):
        super().__init__()
        if num_granularities < 2:
            raise ValueError("adaptive granularity requires at least two experts")
        if (
            not math.isfinite(granularity_usage_floor)
            or not 0.0 <= granularity_usage_floor < 1.0
        ):
            raise ValueError("granularity_usage_floor must lie in [0, 1)")
        if granularity_usage_floor > 1.0 / num_granularities:
            raise ValueError(
                "granularity_usage_floor cannot exceed uniform expert usage"
            )
        if (
            not math.isfinite(granularity_usage_ema_decay)
            or not 0.0 <= granularity_usage_ema_decay < 1.0
        ):
            raise ValueError("granularity_usage_ema_decay must lie in [0, 1)")
        if (
            not math.isfinite(granularity_entropy_floor)
            or not 0.0 <= granularity_entropy_floor <= 1.0
        ):
            raise ValueError("granularity_entropy_floor must lie in [0, 1]")
        if (
            not math.isfinite(granularity_entropy_ceiling)
            or not 0.0 <= granularity_entropy_ceiling <= 1.0
        ):
            raise ValueError("granularity_entropy_ceiling must lie in [0, 1]")
        if granularity_entropy_floor > granularity_entropy_ceiling:
            raise ValueError(
                "granularity_entropy_floor cannot exceed granularity_entropy_ceiling"
            )
        if granularity_router_mode not in {
            "adaptive_v4",
            "adaptive_v41",
            "adaptive_v5",
            "uniform",
        }:
            raise ValueError(
                "granularity_router_mode must be adaptive_v4, adaptive_v41, "
                "adaptive_v5, or uniform"
            )
        self.channel_pool = ChannelAttentionPool(
            input_dim=temporal_dim,
            output_dim=fusion_dim,
            num_channels=num_channels,
            hidden_dim=channel_hidden_dim,
        )
        self.granularity_attention = PatchGranularityCrossAttention(
            visual_dim=visual_dim,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            num_granularities=num_granularities,
            temperature=granularity_temperature,
            dropout=dropout,
            line_query_center=line_query_center,
            local_mix_max=granularity_local_mix_max,
            local_mix_init=granularity_local_mix_init,
            global_mix_max=granularity_global_mix_max,
            global_mix_init=granularity_global_mix_init,
            evidence_half_saturation=granularity_evidence_half_saturation,
            minimum_weight=granularity_minimum_weight,
            score_cap=granularity_score_cap,
            router_mode=granularity_router_mode,
            scorer_hidden_dim=granularity_scorer_hidden_dim,
            confidence_half_saturation=(
                granularity_confidence_half_saturation
            ),
            router_top_k=router_top_k,
            router_training_noise_std=router_training_noise_std,
            router_local_weight=router_local_weight,
            router_relation_hidden_dim=router_relation_hidden_dim,
            router_relation_residual_scale=router_relation_residual_scale,
            router_key_adapter_scale=router_key_adapter_scale,
            router_value_adapter_scale=router_value_adapter_scale,
        )
        self.alignment = MaskedIntraSampleInfoNCE(
            temporal_dim=fusion_dim,
            visual_dim=fusion_dim,
            projection_dim=alignment_dim,
            temperature=alignment_temperature,
        )
        self.patch_fusion = nn.Sequential(
            nn.LayerNorm(2 * fusion_dim),
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.classifier = _MLPHead(
            input_dim=fusion_dim,
            hidden_dim=classifier_hidden_dim,
            num_layers=classifier_num_layers,
            dropout=dropout,
            output_dim=num_classes,
        )
        self.granularity_usage_floor = float(granularity_usage_floor)
        self.granularity_usage_ema_decay = float(granularity_usage_ema_decay)
        self.granularity_entropy_floor = float(granularity_entropy_floor)
        self.granularity_entropy_ceiling = float(granularity_entropy_ceiling)
        self.granularity_router_mode = str(granularity_router_mode)
        self.register_buffer(
            "granularity_usage_ema",
            torch.full(
                (num_granularities,),
                1.0 / num_granularities,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "granularity_usage_ema_initialized",
            torch.tensor(False, dtype=torch.bool),
        )
        self.constructor_configuration = {
            "visual_dim": int(visual_dim),
            "temporal_dim": int(temporal_dim),
            "num_channels": int(num_channels),
            "num_granularities": int(num_granularities),
            "num_classes": int(num_classes),
            "fusion_dim": int(fusion_dim),
            "fusion_heads": int(fusion_heads),
            "dropout": float(dropout),
            "classifier_hidden_dim": int(classifier_hidden_dim),
            "classifier_num_layers": int(classifier_num_layers),
            "channel_hidden_dim": int(channel_hidden_dim),
            "granularity_temperature": float(granularity_temperature),
            "alignment_dim": int(alignment_dim),
            "alignment_temperature": float(alignment_temperature),
            "granularity_usage_floor": float(granularity_usage_floor),
            "granularity_usage_ema_decay": float(granularity_usage_ema_decay),
            "granularity_entropy_floor": float(granularity_entropy_floor),
            "granularity_entropy_ceiling": float(granularity_entropy_ceiling),
            "granularity_router_mode": str(granularity_router_mode),
            "granularity_local_mix_max": float(granularity_local_mix_max),
            "granularity_local_mix_init": float(granularity_local_mix_init),
            "granularity_global_mix_max": float(granularity_global_mix_max),
            "granularity_global_mix_init": float(granularity_global_mix_init),
            "granularity_evidence_half_saturation": float(
                granularity_evidence_half_saturation
            ),
            "granularity_minimum_weight": float(granularity_minimum_weight),
            "granularity_score_cap": float(granularity_score_cap),
            "granularity_scorer_hidden_dim": int(
                granularity_scorer_hidden_dim
            ),
            "granularity_confidence_half_saturation": float(
                granularity_confidence_half_saturation
            ),
            "router_top_k": int(router_top_k),
            "router_training_noise_std": float(
                router_training_noise_std
            ),
            "router_local_weight": float(router_local_weight),
            "router_relation_hidden_dim": int(
                router_relation_hidden_dim
            ),
            "router_relation_residual_scale": float(
                router_relation_residual_scale
            ),
            "router_key_adapter_scale": float(router_key_adapter_scale),
            "router_value_adapter_scale": float(router_value_adapter_scale),
        }
        v41_enabled = granularity_router_mode == "adaptive_v41"
        v5_enabled = granularity_router_mode == "adaptive_v5"
        self.configuration = {
            **self.constructor_configuration,
            "granularity_score": (
                "line_q_graph_k_shared_relation_topk_v5"
                if v5_enabled
                else (
                    "bounded_temperature_dot_scale_specific_interaction_mlp_v41"
                    if v41_enabled
                    else "bounded_rbf_dataset_sample_patch_hierarchical_v4"
                )
            ),
            "granularity_weight_layout": "one_shared_distribution_per_patch_BNK",
            "visual_token_composition": "graph_value_context_only_no_query_residual",
            "granularity_router_inputs": (
                "line_query_activity_graph_keys_only_v5"
                if v5_enabled
                else "legacy_line_query_activity_graph_keys"
            ),
            "granularity_standalone_scale_bias": False if v5_enabled else None,
            "granularity_legacy_prior_gate_parameters": (
                "state_compatibility_only_frozen_unused_v5"
                if v5_enabled
                else "active_legacy_route_parameters"
            ),
            "granularity_global_pooling": (
                "valid_fraction_weighted_line_query_and_graph_keys_v5"
                if v5_enabled
                else "legacy_query_only_pooling"
            ),
            "granularity_decision_scope": (
                "valid_patch_pooled_global_qk_fixed_shrink_patch_local_qk_v5"
                if v5_enabled
                else (
                    "dataset_prior_margin_gated_sample_global_patch_local_v41"
                    if v41_enabled
                    else "dataset_prior_sample_global_query_patch_local_query_v4"
                )
            ),
            "granularity_expert_pattern": "parallel_multiscale_pathformer_inspired",
            "granularity_regularization_policy": (
                "sample_equal_post_topk_soft_budget_pre_topk_load_cv2_v5"
                if v5_enabled
                else (
                    "configurable_usage_entropy_mix_prior_v41"
                    if v41_enabled
                    else "current_marginal_usage_floor_entropy_band_mix_shrinkage_"
                    "prior_kl_ema_diagnostics_only_v4"
                )
            ),
            "granularity_topk_policy": (
                f"selected_logits_resoftmax_top{int(router_top_k)}"
                if v5_enabled
                else "dense_softmax"
            ),
            "granularity_router_noise_policy": (
                "optional_training_only_gaussian_default_zero"
                if v5_enabled
                else "none"
            ),
            "granularity_key_adapter": (
                "zero_initialized_scale_diagonal_bias_free_v5"
                if v5_enabled
                else "none"
            ),
            "granularity_value_adapter": (
                "zero_initialized_scale_diagonal_residual_v5"
                if v5_enabled
                else "none"
            ),
            "alignment_router_gradient": "stopped_at_route_weights_v4",
            "alignment_scale_normalization": "divide_by_log_valid_patch_count_v3",
        }

    def forward(
        self,
        line_tokens,
        graph_tokens,
        mantis_channel_tokens,
        patch_mask,
        valid_fraction,
        granularity_weights_override=None,
    ):
        patch_mask = patch_mask.to(dtype=torch.bool)
        effective_weight_override = granularity_weights_override
        if (
            effective_weight_override is None
            and self.granularity_router_mode == "uniform"
        ):
            effective_weight_override = torch.full(
                (
                    line_tokens.shape[0],
                    line_tokens.shape[1],
                    self.granularity_attention.num_granularities,
                ),
                1.0 / self.granularity_attention.num_granularities,
                dtype=line_tokens.dtype,
                device=line_tokens.device,
            )
            effective_weight_override = (
                effective_weight_override * patch_mask.unsqueeze(-1)
            )
        temporal_tokens, channel_weights = self.channel_pool(
            mantis_channel_tokens,
            patch_mask=patch_mask,
        )
        (
            visual_tokens,
            alignment_visual_tokens,
            granularity_weights,
            per_head_weights,
            granularity_scores,
            router_diagnostics,
        ) = (
            self.granularity_attention(
                line_tokens,
                graph_tokens,
                patch_mask=patch_mask,
                valid_fraction=valid_fraction,
                granularity_weights_override=effective_weight_override,
                return_router_diagnostics=True,
            )
        )
        if self.granularity_router_mode == "uniform":
            # Make diagnostics describe the route that actually reaches the
            # classifier.  The latent adaptive path is intentionally inactive
            # and its frozen random Q/K projections have no experimental
            # meaning in the formal uniform baseline.
            for name in (
                "dataset_prior_weights",
                "sample_global_weights",
                "patch_local_weights",
            ):
                router_diagnostics[name] = granularity_weights
            zero_scalar = granularity_weights.sum() * 0.0
            router_diagnostics["global_mix"] = zero_scalar
            router_diagnostics["local_mix"] = zero_scalar
            router_diagnostics["effective_global_mix"] = torch.zeros_like(
                patch_mask,
                dtype=granularity_weights.dtype,
            )
            router_diagnostics["effective_local_mix"] = torch.zeros_like(
                patch_mask,
                dtype=granularity_weights.dtype,
            )
            for name in (
                "sample_global_score_margin",
                "patch_local_score_margin",
                "sample_global_confidence",
                "patch_local_confidence",
            ):
                router_diagnostics[name] = torch.zeros_like(
                    patch_mask,
                    dtype=granularity_weights.dtype,
                )
        alignment_loss = self.alignment(
            temporal_tokens,
            alignment_visual_tokens,
            patch_mask,
            valid_fraction=valid_fraction,
        )
        patch_features = self.patch_fusion(
            torch.cat([temporal_tokens, visual_tokens], dim=-1)
        )
        patch_features = patch_features * patch_mask.unsqueeze(-1)

        sample_features = valid_fraction_weighted_pool(
            patch_features,
            patch_mask,
            valid_fraction,
        )
        logits = self.classifier(sample_features)

        top1 = granularity_weights.argmax(dim=-1)
        top1 = top1.masked_fill(~patch_mask, -1)
        safe_weights = granularity_weights.float().clamp_min(1e-12)
        entropy = -torch.sum(safe_weights * safe_weights.log(), dim=-1)
        entropy = entropy / math.log(granularity_weights.shape[-1])
        entropy = entropy.masked_fill(~patch_mask, 0.0)
        _, _, granularity_marginal = masked_granularity_regularization_terms(
            granularity_weights,
            patch_mask,
            valid_fraction=valid_fraction,
            usage_floor=0.0,
            entropy_floor=0.0,
            entropy_ceiling=1.0,
        )
        learned_training_route = (
            self.training
            and granularity_weights_override is None
            and self.granularity_router_mode in {"adaptive_v4", "adaptive_v41"}
        )
        if learned_training_route:
            if bool(self.granularity_usage_ema_initialized):
                usage_reference = (
                    self.granularity_usage_ema_decay
                    * self.granularity_usage_ema.detach()
                    + (1.0 - self.granularity_usage_ema_decay)
                    * granularity_marginal
                )
            else:
                # First-batch initialization avoids the long artificial delay
                # caused by decaying from a uniform prior before an unused
                # expert can fall below the survival floor.
                usage_reference = granularity_marginal
            with torch.no_grad():
                self.granularity_usage_ema.copy_(usage_reference.detach())
                self.granularity_usage_ema_initialized.fill_(True)
        else:
            usage_reference = self.granularity_usage_ema.detach()
        (
            granularity_usage_loss,
            granularity_entropy_floor_loss,
            granularity_entropy_ceiling_loss,
            _,
        ) = (
            masked_granularity_regularization_terms(
                granularity_weights,
                patch_mask,
                valid_fraction=valid_fraction,
                # Current marginal keeps the usage gradient at full strength;
                # EMA is retained strictly as a long-run diagnostic.
                usage_reference=None,
                usage_floor=self.granularity_usage_floor,
                entropy_floor=self.granularity_entropy_floor,
                entropy_ceiling=self.granularity_entropy_ceiling,
                return_entropy_components=True,
            )
        )
        granularity_entropy_loss = (
            granularity_entropy_floor_loss + granularity_entropy_ceiling_loss
        )
        zero_router_loss = logits.sum() * 0.0
        if self.granularity_router_mode == "adaptive_v5":
            v5_regularization = masked_v5_router_regularization_terms(
                granularity_weights,
                router_diagnostics["router_pre_topk_weights"],
                router_diagnostics["router_topk_mask"],
                patch_mask,
                valid_fraction=valid_fraction,
            )
            granularity_route_budget_loss = v5_regularization[
                "route_budget_loss"
            ]
            granularity_load_cv2_loss = v5_regularization[
                "load_cv2_loss"
            ]
            granularity_hard_load_cv2 = v5_regularization[
                "hard_load_cv2"
            ]
            granularity_post_topk_sample_equal_marginal = v5_regularization[
                "post_topk_marginal"
            ]
            granularity_pre_topk_sample_equal_marginal = v5_regularization[
                "pre_topk_marginal"
            ]
            granularity_hard_load_marginal = v5_regularization[
                "hard_load_marginal"
            ]
            # V5 replaces the legacy survival/entropy-band penalties rather
            # than stacking them on top of the paper-inspired objectives.
            granularity_usage_loss = zero_router_loss
            granularity_entropy_floor_loss = zero_router_loss
            granularity_entropy_ceiling_loss = zero_router_loss
            granularity_entropy_loss = zero_router_loss
            if self.training and granularity_weights_override is None:
                with torch.no_grad():
                    self.granularity_usage_ema.copy_(
                        granularity_post_topk_sample_equal_marginal.detach()
                    )
                    self.granularity_usage_ema_initialized.fill_(True)
        else:
            granularity_route_budget_loss = zero_router_loss
            granularity_load_cv2_loss = zero_router_loss
            granularity_hard_load_cv2 = zero_router_loss.detach()
            granularity_post_topk_sample_equal_marginal = granularity_marginal
            granularity_pre_topk_sample_equal_marginal = granularity_marginal
            granularity_hard_load_marginal = granularity_marginal.detach()
        if self.granularity_router_mode in {"adaptive_v4", "adaptive_v41"}:
            local_mix_ratio = (
                router_diagnostics["local_mix"]
                / self.granularity_attention.local_mix_max
            )
            global_mix_ratio = (
                router_diagnostics["global_mix"]
                / self.granularity_attention.global_mix_max
            )
            granularity_mix_shrinkage_loss = (
                local_mix_ratio.square()
                + 0.25 * global_mix_ratio.square()
            )
            prior_weights = self.granularity_attention._route_distribution(
                self.granularity_attention.dataset_prior_logits.view(1, 1, 1, -1)
            ).squeeze(0).squeeze(0).squeeze(0)
            granularity_prior_kl_loss = torch.sum(
                prior_weights
                * (
                    prior_weights.clamp_min(1e-12).log()
                    + math.log(len(prior_weights))
                )
            )
        else:
            granularity_mix_shrinkage_loss = zero_router_loss
            granularity_prior_kl_loss = zero_router_loss
        top_scores = torch.topk(granularity_scores, k=2, dim=-1).values
        score_margin = (top_scores[..., 0] - top_scores[..., 1]).masked_fill(
            ~patch_mask,
            0.0,
        )
        return logits, {
            "alignment_loss": alignment_loss,
            "temporal_tokens": temporal_tokens,
            "visual_tokens": visual_tokens,
            "alignment_visual_tokens": alignment_visual_tokens,
            "patch_features": patch_features,
            "sample_features": sample_features,
            "channel_weights": channel_weights,
            "granularity_weights": granularity_weights,
            "granularity_weights_per_head": per_head_weights,
            "granularity_scores": granularity_scores,
            "granularity_score_margin": score_margin,
            "granularity_top1": top1,
            "granularity_max_weight": granularity_weights.max(dim=-1).values,
            "granularity_entropy": entropy,
            "granularity_balance_loss": granularity_usage_loss,
            "granularity_usage_floor_loss": granularity_usage_loss,
            "granularity_entropy_loss": granularity_entropy_loss,
            "granularity_entropy_floor_loss": granularity_entropy_floor_loss,
            "granularity_entropy_ceiling_loss": granularity_entropy_ceiling_loss,
            "granularity_marginal": granularity_marginal,
            "granularity_usage_ema": self.granularity_usage_ema.clone(),
            "granularity_usage_ema_initialized": (
                self.granularity_usage_ema_initialized.clone()
            ),
            "granularity_mix_shrinkage_loss": granularity_mix_shrinkage_loss,
            "granularity_prior_kl_loss": granularity_prior_kl_loss,
            "granularity_route_budget_loss": granularity_route_budget_loss,
            "granularity_load_cv2_loss": granularity_load_cv2_loss,
            "granularity_hard_load_cv2": granularity_hard_load_cv2,
            "granularity_post_topk_sample_equal_marginal": (
                granularity_post_topk_sample_equal_marginal
            ),
            "granularity_pre_topk_sample_equal_marginal": (
                granularity_pre_topk_sample_equal_marginal
            ),
            "granularity_hard_load_marginal": (
                granularity_hard_load_marginal
            ),
            **router_diagnostics,
        }


def _encode_visual_images(model, images, device):
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"visual images must have shape [B, 3, H, W], got {tuple(images.shape)}")
    hidden = model.forward_vit(images.to(device))
    embeddings = model.aggregate_hidden_representations(
        hidden,
        aggregation=model.aggregation,
    )
    if embeddings.ndim != 2:
        raise ValueError(
            f"the visual encoder must return one vector per image, got {tuple(embeddings.shape)}"
        )
    return F.normalize(embeddings.float(), dim=-1, eps=1e-8)


def _line_images_for_chunk(windows, valid_lengths):
    images = [None] * len(windows)
    for length in torch.unique(valid_lengths, sorted=True).tolist():
        length = int(length)
        if length <= 0:
            continue
        indices = torch.nonzero(valid_lengths == length, as_tuple=False).flatten()
        rendered = preprocess_stacked_multichannel_lineplot(
            windows.index_select(0, indices)[..., :length],
        ).cpu()
        for local_index, image in zip(indices.tolist(), rendered):
            images[local_index] = image
    if any(image is None for image in images):
        raise ValueError("line renderer received an empty temporal patch")
    return torch.stack(images, dim=0)


@torch.no_grad()
def _extract_line_tokens(
    windows,
    valid_lengths,
    vision_model,
    device,
    encode_batch_size,
):
    batches = []
    for start in range(0, len(windows), encode_batch_size):
        stop = min(start + encode_batch_size, len(windows))
        images = _line_images_for_chunk(
            windows[start:stop], valid_lengths[start:stop]
        )
        batches.append(_encode_visual_images(vision_model, images, device).cpu())
    return torch.cat(batches, dim=0)


@torch.no_grad()
def _extract_graph_tokens(
    windows,
    valid_lengths,
    vision_model,
    device,
    encode_batch_size,
):
    graph_bank = getattr(vision_model, "med_activity_granularity_bank", None)
    if graph_bank is None or not hasattr(graph_bank, "graphs"):
        raise ValueError(
            "patch_mindts requires a vision model with an adaptive Activity Graph bank"
        )
    candidate_batches = []
    for graph_renderer in graph_bank.graphs:
        batches = []
        for start in range(0, len(windows), encode_batch_size):
            stop = min(start + encode_batch_size, len(windows))
            chunk = windows[start:stop]
            chunk_lengths = valid_lengths[start:stop]
            images = None
            for length in torch.unique(chunk_lengths, sorted=True).tolist():
                length = int(length)
                if length <= 0:
                    continue
                indices = torch.nonzero(
                    chunk_lengths == length, as_tuple=False
                ).flatten()
                rendered = graph_renderer(
                    chunk.index_select(0, indices).to(device)[..., :length]
                )
                if images is None:
                    images = torch.empty(
                        (len(chunk), *rendered.shape[1:]),
                        dtype=rendered.dtype,
                        device=rendered.device,
                    )
                images.index_copy_(0, indices.to(device), rendered)
            if images is None:
                raise ValueError("graph renderer received only empty temporal patches")
            batches.append(_encode_visual_images(vision_model, images, device).cpu())
        candidate_batches.append(torch.cat(batches, dim=0))
    return torch.stack(candidate_batches, dim=1)


@torch.no_grad()
def _extract_mantis_channel_tokens(
    windows,
    valid_lengths,
    mantis_model,
    device,
    encode_batch_size,
):
    channels = windows.shape[1]
    channel_outputs = []
    for channel_index in range(channels):
        outputs_for_channel = None
        for length in torch.unique(valid_lengths, sorted=True).tolist():
            length = int(length)
            if length <= 0:
                continue
            indices = torch.nonzero(valid_lengths == length, as_tuple=False).flatten()
            cropped = windows.index_select(0, indices)[:, channel_index : channel_index + 1, :length]
            resized = resize_mantis_input(cropped).float()
            output_batches = []
            for start in range(0, len(resized), encode_batch_size):
                stop = min(start + encode_batch_size, len(resized))
                output = mantis_model(resized[start:stop].to(device))
                if not torch.is_tensor(output) or output.ndim != 2:
                    raise ValueError(
                        "Mantis must return [batch, feature_dim], got "
                        f"{type(output).__name__} {getattr(output, 'shape', None)}"
                    )
                output_batches.append(F.normalize(output.float(), dim=-1, eps=1e-8).cpu())
            selected_outputs = torch.cat(output_batches, dim=0)
            if outputs_for_channel is None:
                outputs_for_channel = torch.empty(
                    (len(windows), selected_outputs.shape[-1]),
                    dtype=selected_outputs.dtype,
                )
            outputs_for_channel.index_copy_(0, indices, selected_outputs)
        if outputs_for_channel is None:
            raise ValueError("Mantis received only empty temporal patches")
        channel_outputs.append(outputs_for_channel)
    return torch.stack(channel_outputs, dim=1)


@torch.no_grad()
def extract_patch_feature_batch(
    batch,
    vision_model,
    mantis_model,
    device,
    window_size=64,
    stride=64,
    encode_batch_size=16,
    lengths=None,
):
    """Extract structured frozen features for one raw-loader batch."""
    if encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be positive")
    if vision_model is None or mantis_model is None:
        raise ValueError("patch_mindts requires both a vision model and Mantis")

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
        batch_size * patch_count, channels, patch_length
    ).detach().cpu()
    flat_lengths = temporal.valid_lengths.reshape(-1).detach().cpu()
    flat_valid = temporal.patch_mask.reshape(-1).detach().cpu()
    valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
    valid_windows = flat_windows.index_select(0, valid_indices)
    valid_lengths = flat_lengths.index_select(0, valid_indices)

    line_valid = _extract_line_tokens(
        valid_windows,
        valid_lengths,
        vision_model,
        device,
        encode_batch_size,
    )
    graph_valid = _extract_graph_tokens(
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
    line = torch.zeros((total_patches, line_valid.shape[-1]), dtype=torch.float32)
    graph = torch.zeros(
        (total_patches, graph_valid.shape[1], graph_valid.shape[2]),
        dtype=torch.float32,
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
    graph.index_copy_(0, valid_indices, graph_valid.float())
    mantis.index_copy_(0, valid_indices, mantis_valid.float())

    # Frozen features are cached in float16 to keep the structured N/K/C
    # layout practical on the larger EEG splits.  Training always casts them
    # back to float32 before applying trainable heads.
    return {
        "line_tokens": line.reshape(batch_size, patch_count, -1).half(),
        "graph_tokens": graph.reshape(
            batch_size, patch_count, graph.shape[1], graph.shape[2]
        ).half(),
        "mantis_channel_tokens": mantis.reshape(
            batch_size, patch_count, mantis.shape[1], mantis.shape[2]
        ).half(),
        "patch_mask": temporal.patch_mask.detach().cpu(),
        "valid_fraction": temporal.valid_fraction.detach().cpu().half(),
        "valid_lengths": temporal.valid_lengths.detach().cpu(),
    }


def _validate_patch_feature_bundle(bundle, expected_num_granularities=None):
    missing = set(PATCH_FEATURE_KEYS) - set(bundle)
    if missing:
        raise ValueError(f"patch feature bundle is missing {sorted(missing)}")
    line = bundle["line_tokens"]
    graph = bundle["graph_tokens"]
    mantis = bundle["mantis_channel_tokens"]
    mask = bundle["patch_mask"]
    fraction = bundle["valid_fraction"]
    lengths = bundle["valid_lengths"]
    if line.ndim != 3 or graph.ndim != 4 or mantis.ndim != 4:
        raise ValueError(
            "patch features must have layouts [S,N,D], [S,N,K,D], and [S,N,C,D]"
        )
    sample_patch_shape = line.shape[:2]
    if graph.shape[:2] != sample_patch_shape or mantis.shape[:2] != sample_patch_shape:
        raise ValueError("all patch feature branches must share sample and patch axes")
    if mask.shape != sample_patch_shape or fraction.shape != sample_patch_shape or lengths.shape != sample_patch_shape:
        raise ValueError("patch metadata must share the feature sample and patch axes")
    if graph.shape[-1] != line.shape[-1]:
        raise ValueError("line and graph tokens must use the same visual dimension")
    if expected_num_granularities is not None and graph.shape[2] != int(expected_num_granularities):
        raise ValueError(
            f"expected {expected_num_granularities} graph candidates, got {graph.shape[2]}"
        )
    for name in ("line_tokens", "graph_tokens", "mantis_channel_tokens", "valid_fraction"):
        if not torch.isfinite(bundle[name]).all():
            raise ValueError(f"patch feature bundle contains non-finite {name}")
    mask_bool = mask.to(dtype=torch.bool)
    if not torch.equal(mask_bool, lengths > 0):
        raise ValueError("patch_mask must equal valid_lengths > 0")
    if torch.any(lengths < 0):
        raise ValueError("valid_lengths cannot be negative")
    if torch.any(fraction < 0) or torch.any(fraction > 1):
        raise ValueError("valid_fraction must lie in [0, 1]")
    if not torch.equal(fraction > 0, mask_bool):
        raise ValueError("positive valid_fraction entries must match patch_mask")
    positive = fraction > 0
    if positive.any():
        implied_window_sizes = (
            lengths[positive].float() / fraction[positive].float()
        )
        reference_window_size = implied_window_sizes[0]
        if not torch.allclose(
            implied_window_sizes,
            torch.full_like(implied_window_sizes, reference_window_size),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise ValueError(
                "valid_fraction must equal valid_lengths divided by one shared window size"
            )
        if torch.any(lengths[positive].float() > reference_window_size + 1e-3):
            raise ValueError("valid_lengths cannot exceed the inferred window size")
    return {key: bundle[key] for key in PATCH_FEATURE_KEYS}


def save_patch_feature_cache(path, bundle, labels, signature):
    """Atomically save a structured patch-feature split."""
    path = Path(path)
    validated = _validate_patch_feature_bundle(bundle)
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1 or len(labels_array) != len(validated["line_tokens"]):
        raise ValueError("cache labels must be one-dimensional and match sample count")
    arrays = {
        "schema_version": np.asarray(PATCH_FEATURE_CACHE_SCHEMA_VERSION, dtype=np.int64),
        "architecture": np.asarray(PATCH_FEATURE_ARCHITECTURE),
        "cache_signature": np.asarray(signature or ""),
        "labels": labels_array,
    }
    arrays.update({key: value.detach().cpu().numpy() for key, value in validated.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_patch_feature_cache(
    path,
    labels,
    signature,
    expected_num_granularities=None,
):
    """Load and validate a patch cache, returning ``None`` on a cold miss."""
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cached:
        required = {
            "schema_version",
            "architecture",
            "cache_signature",
            "labels",
            *PATCH_FEATURE_KEYS,
        }
        missing = required - set(cached.files)
        if missing:
            raise ValueError(f"patch feature cache is missing {sorted(missing)}: {path}")
        if int(cached["schema_version"].item()) != PATCH_FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError(f"patch feature cache schema mismatch: {path}")
        if str(cached["architecture"].item()) != PATCH_FEATURE_ARCHITECTURE:
            raise ValueError(f"patch feature cache architecture mismatch: {path}")
        if str(cached["cache_signature"].item()) != (signature or ""):
            raise ValueError(f"patch feature cache signature mismatch: {path}")
        if not np.array_equal(cached["labels"], np.asarray(labels)):
            raise ValueError(f"patch feature cache labels do not match: {path}")
        bundle = {
            key: torch.from_numpy(cached[key].copy()) for key in PATCH_FEATURE_KEYS
        }
    validated = _validate_patch_feature_bundle(
        bundle,
        expected_num_granularities=expected_num_granularities,
    )
    if len(validated["line_tokens"]) != len(np.asarray(labels)):
        raise ValueError(f"patch feature cache sample count mismatch: {path}")
    print(f"Loaded patch model features: {path}")
    return validated


@torch.no_grad()
def _extract_patch_feature_split(
    loader,
    vision_model,
    mantis_model,
    device,
    window_size,
    stride,
    encode_batch_size,
):
    batches = {key: [] for key in PATCH_FEATURE_KEYS}
    for loader_batch in tqdm(loader, desc="Extract patch model features", leave=False):
        if len(loader_batch) == 1:
            batch = loader_batch[0]
            lengths = None
        elif len(loader_batch) == 2:
            batch, lengths = loader_batch
        else:
            raise ValueError(
                "patch feature loaders must yield (signals,) or (signals, lengths)"
            )
        features = extract_patch_feature_batch(
            batch=batch,
            vision_model=vision_model,
            mantis_model=mantis_model,
            device=device,
            window_size=window_size,
            stride=stride,
            encode_batch_size=encode_batch_size,
            lengths=lengths,
        )
        for key in PATCH_FEATURE_KEYS:
            batches[key].append(features[key])
    if not batches["line_tokens"]:
        raise ValueError("cannot extract patch features from an empty split")
    bundle = {key: torch.cat(value, dim=0) for key, value in batches.items()}
    return _validate_patch_feature_bundle(bundle)


def _get_patch_feature_split(
    split_name,
    loader,
    labels,
    vision_model,
    mantis_model,
    device,
    window_size,
    stride,
    encode_batch_size,
    feature_cache_dir,
    feature_cache_signature,
    num_granularities,
):
    cache_path = None
    if feature_cache_dir:
        cache_path = Path(feature_cache_dir) / f"patch_{split_name}.npz"
        cached = load_patch_feature_cache(
            cache_path,
            labels,
            feature_cache_signature,
            expected_num_granularities=num_granularities,
        )
        if cached is not None:
            return cached
    bundle = _extract_patch_feature_split(
        loader,
        vision_model,
        mantis_model,
        device,
        window_size,
        stride,
        encode_batch_size,
    )
    _validate_patch_feature_bundle(
        bundle,
        expected_num_granularities=num_granularities,
    )
    if cache_path is not None:
        save_patch_feature_cache(
            cache_path,
            bundle,
            labels,
            feature_cache_signature,
        )
        print(f"Saved patch model features: {cache_path}")
    return bundle


def _labels_to_indices(labels):
    classes = np.unique(labels)
    class_to_index = {label: index for index, label in enumerate(classes)}
    indices = np.asarray(
        [class_to_index[label] for label in labels], dtype=np.int64
    )
    return indices, classes, class_to_index


def _map_labels(labels, class_to_index):
    unknown = sorted(set(labels) - set(class_to_index))
    if unknown:
        raise ValueError(f"labels contain classes absent from training: {unknown}")
    return np.asarray([class_to_index[label] for label in labels], dtype=np.int64)


def _loader_sample_subject_ids(loader):
    """Return validated per-sample subject IDs, or ``None`` when unavailable."""
    dataset = loader.dataset
    subject_ids = getattr(dataset, "sample_subject_ids", None)
    if subject_ids is None and isinstance(dataset, Subset):
        parent_ids = getattr(dataset.dataset, "sample_subject_ids", None)
        if parent_ids is not None:
            subject_ids = np.asarray(parent_ids)[np.asarray(dataset.indices)]
    if subject_ids is None:
        return None
    subject_ids = np.asarray(subject_ids)
    if subject_ids.ndim != 1 or len(subject_ids) != len(dataset):
        raise ValueError(
            "loader.dataset.sample_subject_ids must be one-dimensional and "
            "match the dataset length"
        )
    if subject_ids.dtype == object:
        subject_ids = subject_ids.astype(str)
    return subject_ids.copy()


def _aggregate_subject_predictions(
    y_true,
    y_score,
    sample_indices,
    sample_subject_ids,
    classes,
):
    """Average window probabilities so every subject contributes exactly once."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    sample_subject_ids = np.asarray(sample_subject_ids)
    if y_true.ndim != 1 or sample_indices.shape != y_true.shape:
        raise ValueError("subject aggregation requires aligned one-dimensional labels")
    if y_score.ndim != 2 or y_score.shape[0] != len(y_true):
        raise ValueError("subject aggregation requires [sample, class] probabilities")
    if y_score.shape[1] != len(classes):
        raise ValueError("subject probability class width does not match classes")
    if not np.isfinite(y_score).all():
        raise ValueError("subject aggregation probabilities must be finite")
    if np.any(sample_indices < 0) or np.any(sample_indices >= len(sample_subject_ids)):
        raise ValueError("subject aggregation sample indices are out of range")
    row_sums = y_score.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("subject aggregation rows must be probabilities")

    aligned_subject_ids = sample_subject_ids[sample_indices]
    subject_ids, inverse = np.unique(aligned_subject_ids, return_inverse=True)
    subject_y_true = np.empty(len(subject_ids), dtype=np.int64)
    subject_y_score = np.empty((len(subject_ids), y_score.shape[1]), dtype=np.float64)
    subject_window_count = np.empty(len(subject_ids), dtype=np.int64)
    for subject_index in range(len(subject_ids)):
        member = inverse == subject_index
        labels = np.unique(y_true[member])
        if len(labels) != 1:
            raise ValueError(
                f"subject {subject_ids[subject_index]!r} has inconsistent labels: "
                f"{labels.tolist()}"
            )
        subject_y_true[subject_index] = int(labels[0])
        subject_y_score[subject_index] = y_score[member].mean(axis=0)
        subject_window_count[subject_index] = int(member.sum())
    subject_y_pred = subject_y_score.argmax(axis=-1)
    subject_metrics = compute_metrics_from_predictions(
        subject_y_true,
        subject_y_pred,
        subject_y_score,
        np.arange(len(classes)),
    )
    per_subject_nll = -np.log(
        np.clip(
            subject_y_score[np.arange(len(subject_y_true)), subject_y_true],
            1e-12,
            1.0,
        )
    )
    class_nll = [
        per_subject_nll[subject_y_true == class_index].mean()
        for class_index in np.unique(subject_y_true)
    ]
    subject_metrics["macro_log_loss"] = float(np.mean(class_nll))
    return subject_metrics, {
        "sample_subject_id": aligned_subject_ids,
        "subject_id": subject_ids,
        "subject_y_true": subject_y_true,
        "subject_y_pred": subject_y_pred,
        "subject_y_score": subject_y_score,
        "subject_window_count": subject_window_count,
    }


def _merge_subject_metrics(window_metrics, subject_metrics):
    metrics = dict(window_metrics)
    if subject_metrics is not None:
        metrics.update(
            {
                f"subject_{name}": float(value)
                for name, value in subject_metrics.items()
            }
        )
    return metrics


def _subject_ids_digest(subject_ids):
    if subject_ids is None:
        return None
    payload = "\n".join(str(value) for value in np.asarray(subject_ids).tolist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_checkpoint_metric(
    requested,
    train_subject_ids,
    validation_subject_ids,
    train_indices,
    validation_indices,
):
    if requested not in {"auto", "subject_macro_f1", "window_macro_f1"}:
        raise ValueError(
            "checkpoint_metric must be auto, subject_macro_f1, or window_macro_f1"
        )
    subject_ready = (
        train_subject_ids is not None and validation_subject_ids is not None
    )
    overlap = np.asarray([], dtype=np.int64)
    if subject_ready:
        train_used = np.asarray(train_subject_ids)[np.asarray(train_indices)]
        validation_used = np.asarray(validation_subject_ids)[
            np.asarray(validation_indices)
        ]
        overlap = np.intersect1d(train_used, validation_used)
        subject_ready = overlap.size == 0
    if requested == "subject_macro_f1" and not subject_ready:
        reason = (
            "subject IDs are unavailable"
            if train_subject_ids is None or validation_subject_ids is None
            else f"train/validation subjects overlap: {overlap.tolist()}"
        )
        raise ValueError(f"subject checkpoint selection is unsafe because {reason}")
    if requested == "window_macro_f1":
        return "window_macro_f1"
    if subject_ready:
        return "subject_macro_f1"
    print(
        "Checkpoint selection auto fallback: window_macro_f1 "
        "(validated disjoint subject IDs unavailable)"
    )
    return "window_macro_f1"


def _checkpoint_selection_key(metrics, effective_metric):
    if effective_metric == "subject_macro_f1":
        return (
            float(metrics["subject_macro_f1"]),
            -float(metrics["subject_macro_log_loss"]),
        )
    if effective_metric == "window_macro_f1":
        return (float(metrics["macro_f1"]),)
    raise ValueError(f"unsupported checkpoint metric: {effective_metric}")


class _EarlyStoppingMonitor:
    """Track a stopping signal without changing raw checkpoint selection.

    ``raw_selection_key`` exactly preserves the historical lexicographic
    monitor, including the subject log-loss tie-breaker. ``ema_primary`` is
    intended for small validation cohorts: only the primary F1 component is
    smoothed, while the caller continues to save checkpoints from the raw
    selection key.
    """

    def __init__(
        self,
        strategy,
        patience,
        min_epochs=0,
        ema_decay=0.6,
        min_delta=0.0,
    ):
        if strategy not in {"raw_selection_key", "ema_primary"}:
            raise ValueError(f"unsupported early-stop strategy: {strategy}")
        if patience < 0 or min_epochs < 0:
            raise ValueError("early-stop patience and min_epochs must be non-negative")
        if not math.isfinite(ema_decay) or not 0.0 <= ema_decay < 1.0:
            raise ValueError("early-stop EMA decay must lie in [0, 1)")
        if not math.isfinite(min_delta) or min_delta < 0.0:
            raise ValueError("early-stop min_delta must be finite and non-negative")
        self.strategy = strategy
        self.patience = int(patience)
        self.min_epochs = int(min_epochs)
        self.ema_decay = float(ema_decay)
        self.min_delta = float(min_delta)
        self.ema_score = None
        self.best_score = None
        self.best_raw_key = None
        self.epochs_without_improvement = 0

    def update(self, selection_key, epoch):
        epoch = int(epoch)
        if epoch <= 0:
            raise ValueError("early-stop epoch must be positive")
        raw_key = tuple(float(value) for value in selection_key)
        if not raw_key or not all(math.isfinite(value) for value in raw_key):
            raise ValueError("early-stop selection key must be finite and non-empty")
        raw_score = raw_key[0]

        if self.strategy == "raw_selection_key":
            smoothed_score = raw_score
            improved = self.best_raw_key is None or raw_key > self.best_raw_key
            if improved:
                self.best_raw_key = raw_key
                self.best_score = raw_score
        else:
            if self.ema_score is None:
                self.ema_score = raw_score
            else:
                self.ema_score = (
                    self.ema_decay * self.ema_score
                    + (1.0 - self.ema_decay) * raw_score
                )
            smoothed_score = self.ema_score
            improved = (
                self.best_score is None
                or smoothed_score > self.best_score + self.min_delta
            )
            if improved:
                self.best_score = smoothed_score

        if improved:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        eligible = self.patience > 0 and epoch >= max(1, self.min_epochs)
        should_stop = (
            eligible
            and self.epochs_without_improvement >= self.patience
        )
        return {
            "raw_score": float(raw_score),
            "smoothed_score": float(smoothed_score),
            "best_score": float(self.best_score),
            "improved": bool(improved),
            "epochs_without_improvement": int(
                self.epochs_without_improvement
            ),
            "eligible": bool(eligible),
            "should_stop": bool(should_stop),
        }


def _balanced_class_weights(label_indices, num_classes, device):
    counts = np.bincount(label_indices, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"cannot balance empty classes: {counts.tolist()}")
    weights = len(label_indices) / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _build_patch_loader(bundle, labels, indices, batch_size, shuffle):
    dataset = TensorDataset(
        bundle["line_tokens"],
        bundle["graph_tokens"],
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


def compute_line_query_center(bundle, indices, chunk_size=256):
    """Compute a train-only, valid-time-weighted line-token centre.

    Statistics are computed after stateless LayerNorm, exactly matching the
    selector input.  Validation and test tokens must never be included.
    """
    if chunk_size <= 0:
        raise ValueError("line-query centre chunk_size must be positive")
    required = ("line_tokens", "patch_mask", "valid_fraction")
    missing = [key for key in required if key not in bundle]
    if missing:
        raise ValueError(f"line-query centre bundle is missing keys: {missing}")
    index_tensor = torch.as_tensor(list(indices), dtype=torch.long)
    if index_tensor.ndim != 1 or index_tensor.numel() == 0:
        raise ValueError("line-query centre requires at least one training sample")
    sample_count = int(bundle["line_tokens"].shape[0])
    if torch.any(index_tensor < 0) or torch.any(index_tensor >= sample_count):
        raise ValueError("line-query centre indices are out of range")

    visual_dim = int(bundle["line_tokens"].shape[-1])
    weighted_sum = torch.zeros(visual_dim, dtype=torch.float64)
    weight_sum = torch.zeros((), dtype=torch.float64)
    for start in range(0, len(index_tensor), chunk_size):
        current = index_tensor[start : start + chunk_size]
        line = bundle["line_tokens"].index_select(0, current).float()
        normalized = F.layer_norm(line, (visual_dim,))
        importance = (
            bundle["patch_mask"].index_select(0, current).double()
            * bundle["valid_fraction"].index_select(0, current).double()
        )
        weighted_sum += torch.sum(
            normalized.double() * importance.unsqueeze(-1),
            dim=(0, 1),
        )
        weight_sum += importance.sum()
    if not bool(weight_sum > 0.0):
        raise ValueError("line-query centre requires positive valid-time mass")
    center = (weighted_sum / weight_sum).float()
    if not torch.isfinite(center).all():
        raise ValueError("line-query centre computation produced non-finite values")
    return center


def _forward_patch_training_batch(model, batch, device):
    line, graph, mantis, mask, fraction, lengths, labels, sample_indices = batch
    logits, auxiliary = model(
        line.to(device=device, dtype=torch.float32),
        graph.to(device=device, dtype=torch.float32),
        mantis.to(device=device, dtype=torch.float32),
        mask.to(device=device, dtype=torch.bool),
        fraction.to(device=device, dtype=torch.float32),
    )
    if not torch.isfinite(logits).all():
        raise ValueError("patch classifier produced non-finite logits")
    if not torch.isfinite(auxiliary["alignment_loss"]):
        raise ValueError("patch alignment produced a non-finite loss")
    for name in (
        "granularity_balance_loss",
        "granularity_entropy_loss",
        "granularity_mix_shrinkage_loss",
        "granularity_prior_kl_loss",
        "granularity_route_budget_loss",
        "granularity_load_cv2_loss",
        "granularity_hard_load_cv2",
        "granularity_scores",
    ):
        if not torch.isfinite(auxiliary[name]).all():
            raise ValueError(f"patch granularity produced non-finite {name}")
    return (
        logits,
        auxiliary,
        labels.to(device),
        lengths,
        sample_indices,
    )


def _run_patch_epoch(
    model,
    loader,
    optimizer,
    criterion,
    alignment_weight,
    granularity_balance_weight,
    granularity_entropy_weight,
    granularity_mix_shrinkage_weight,
    granularity_prior_kl_weight,
    device,
    router_route_budget_weight=0.005,
    router_load_balance_weight=0.005,
):
    model.train()
    totals = {
        "loss": 0.0,
        "task_loss": 0.0,
        "alignment_loss": 0.0,
        "granularity_regularization_loss": 0.0,
        "granularity_balance_loss": 0.0,
        "granularity_entropy_loss": 0.0,
        "granularity_entropy_floor_loss": 0.0,
        "granularity_entropy_ceiling_loss": 0.0,
        "granularity_mix_shrinkage_loss": 0.0,
        "granularity_prior_kl_loss": 0.0,
        "granularity_route_budget_loss": 0.0,
        "granularity_load_cv2_loss": 0.0,
        "granularity_hard_load_cv2": 0.0,
        "granularity_local_mix": 0.0,
        "granularity_global_mix": 0.0,
    }
    sample_count = 0
    valid_weight_sum = None
    valid_time_mass = 0.0
    max_weight_sum = 0.0
    entropy_sum = 0.0
    candidate_evidence_sum = 0.0
    sample_global_confidence_sum = 0.0
    patch_local_confidence_sum = 0.0
    effective_local_mix_sum = 0.0
    effective_global_mix_sum = 0.0
    top1_mass = None
    for batch in tqdm(loader, desc="Train patch MLP", leave=False):
        optimizer.zero_grad(set_to_none=True)
        logits, auxiliary, labels, _, _ = _forward_patch_training_batch(
            model, batch, device
        )
        task_losses = criterion(logits, labels)
        if task_losses.ndim != 1 or task_losses.shape[0] != labels.shape[0]:
            raise ValueError(
                "patch task criterion must return one loss per sample; "
                f"got {tuple(task_losses.shape)} for {len(labels)} labels"
            )
        # Keep class weights effective even for batch_size=1.  PyTorch's
        # weighted CrossEntropyLoss(reduction="mean") divides by the sum of
        # weights in the current batch, which cancels the sole class weight.
        task_loss = task_losses.mean()
        alignment_loss = auxiliary["alignment_loss"]
        balance_loss = auxiliary["granularity_balance_loss"]
        entropy_loss = auxiliary["granularity_entropy_loss"]
        entropy_floor_loss = auxiliary["granularity_entropy_floor_loss"]
        entropy_ceiling_loss = auxiliary["granularity_entropy_ceiling_loss"]
        mix_shrinkage_loss = auxiliary["granularity_mix_shrinkage_loss"]
        prior_kl_loss = auxiliary["granularity_prior_kl_loss"]
        route_budget_loss = auxiliary["granularity_route_budget_loss"]
        load_cv2_loss = auxiliary["granularity_load_cv2_loss"]
        hard_load_cv2 = auxiliary["granularity_hard_load_cv2"]
        granularity_regularization = (
            granularity_balance_weight * balance_loss
            + granularity_entropy_weight * entropy_loss
            + granularity_mix_shrinkage_weight * mix_shrinkage_loss
            + granularity_prior_kl_weight * prior_kl_loss
            + router_route_budget_weight * route_budget_loss
            + router_load_balance_weight * load_cv2_loss
        )
        loss = (
            task_loss
            + alignment_weight * alignment_loss
            + granularity_regularization
        )
        if not torch.isfinite(loss):
            raise ValueError("patch training produced a non-finite total loss")
        loss.backward()
        optimizer.step()

        batch_size = len(labels)
        totals["loss"] += float(loss.detach()) * batch_size
        totals["task_loss"] += float(task_loss.detach()) * batch_size
        totals["alignment_loss"] += float(alignment_loss.detach()) * batch_size
        totals["granularity_regularization_loss"] += (
            float(granularity_regularization.detach()) * batch_size
        )
        totals["granularity_balance_loss"] += (
            float(balance_loss.detach()) * batch_size
        )
        totals["granularity_entropy_loss"] += (
            float(entropy_loss.detach()) * batch_size
        )
        totals["granularity_entropy_floor_loss"] += (
            float(entropy_floor_loss.detach()) * batch_size
        )
        totals["granularity_entropy_ceiling_loss"] += (
            float(entropy_ceiling_loss.detach()) * batch_size
        )
        totals["granularity_mix_shrinkage_loss"] += (
            float(mix_shrinkage_loss.detach()) * batch_size
        )
        totals["granularity_prior_kl_loss"] += (
            float(prior_kl_loss.detach()) * batch_size
        )
        totals["granularity_route_budget_loss"] += (
            float(route_budget_loss.detach()) * batch_size
        )
        totals["granularity_load_cv2_loss"] += (
            float(load_cv2_loss.detach()) * batch_size
        )
        totals["granularity_hard_load_cv2"] += (
            float(hard_load_cv2.detach()) * batch_size
        )
        totals["granularity_local_mix"] += (
            float(auxiliary["local_mix"].detach()) * batch_size
        )
        totals["granularity_global_mix"] += (
            float(auxiliary["global_mix"].detach()) * batch_size
        )
        sample_count += batch_size
        valid = batch[3].to(dtype=torch.bool)
        importance = batch[4].float() * valid.float()
        weights = auxiliary["granularity_weights"].detach().cpu()
        current_sum = torch.sum(weights * importance.unsqueeze(-1), dim=(0, 1))
        valid_weight_sum = current_sum if valid_weight_sum is None else valid_weight_sum + current_sum
        current_mass = float(importance.sum())
        valid_time_mass += current_mass
        max_weight_sum += float(
            torch.sum(
                auxiliary["granularity_max_weight"].detach().cpu() * importance
            )
        )
        entropy_sum += float(
            torch.sum(auxiliary["granularity_entropy"].detach().cpu() * importance)
        )
        candidate_evidence_sum += float(
            torch.sum(auxiliary["candidate_evidence"].detach().cpu() * importance)
        )
        sample_global_confidence_sum += float(
            torch.sum(
                auxiliary["sample_global_confidence"].detach().cpu()
                * importance
            )
        )
        patch_local_confidence_sum += float(
            torch.sum(
                auxiliary["patch_local_confidence"].detach().cpu()
                * importance
            )
        )
        effective_local_mix_sum += float(
            torch.sum(
                auxiliary["effective_local_mix"].detach().cpu() * importance
            )
        )
        effective_global_mix_sum += float(
            torch.sum(
                auxiliary["effective_global_mix"].detach().cpu() * importance
            )
        )
        top1 = auxiliary["granularity_top1"].detach().cpu()
        current_top1_mass = torch.stack(
            [
                importance[top1 == candidate_index].sum()
                for candidate_index in range(weights.shape[-1])
            ]
        )
        top1_mass = (
            current_top1_mass
            if top1_mass is None
            else top1_mass + current_top1_mass
        )
    statistics = {key: value / max(sample_count, 1) for key, value in totals.items()}
    statistics["weighted_alignment_loss"] = alignment_weight * statistics["alignment_loss"]
    statistics["weighted_granularity_balance_loss"] = (
        granularity_balance_weight * statistics["granularity_balance_loss"]
    )
    statistics["weighted_granularity_entropy_loss"] = (
        granularity_entropy_weight * statistics["granularity_entropy_loss"]
    )
    statistics["granularity_usage_floor_loss"] = statistics[
        "granularity_balance_loss"
    ]
    statistics["weighted_granularity_usage_floor_loss"] = statistics[
        "weighted_granularity_balance_loss"
    ]
    statistics["weighted_granularity_entropy_floor_loss"] = (
        granularity_entropy_weight
        * statistics["granularity_entropy_floor_loss"]
    )
    statistics["weighted_granularity_entropy_ceiling_loss"] = (
        granularity_entropy_weight
        * statistics["granularity_entropy_ceiling_loss"]
    )
    statistics["weighted_granularity_mix_shrinkage_loss"] = (
        granularity_mix_shrinkage_weight
        * statistics["granularity_mix_shrinkage_loss"]
    )
    statistics["weighted_granularity_prior_kl_loss"] = (
        granularity_prior_kl_weight
        * statistics["granularity_prior_kl_loss"]
    )
    statistics["weighted_granularity_route_budget_loss"] = (
        router_route_budget_weight
        * statistics["granularity_route_budget_loss"]
    )
    statistics["weighted_granularity_load_cv2_loss"] = (
        router_load_balance_weight
        * statistics["granularity_load_cv2_loss"]
    )
    statistics["mean_granularity_max_weight"] = (
        max_weight_sum / valid_time_mass if valid_time_mass > 0.0 else 0.0
    )
    statistics["mean_granularity_entropy"] = (
        entropy_sum / valid_time_mass if valid_time_mass > 0.0 else 0.0
    )
    statistics["mean_candidate_evidence"] = (
        candidate_evidence_sum / valid_time_mass
        if valid_time_mass > 0.0
        else 0.0
    )
    statistics["mean_sample_global_confidence"] = (
        sample_global_confidence_sum / valid_time_mass
        if valid_time_mass > 0.0
        else 0.0
    )
    statistics["mean_patch_local_confidence"] = (
        patch_local_confidence_sum / valid_time_mass
        if valid_time_mass > 0.0
        else 0.0
    )
    statistics["mean_effective_local_mix"] = (
        effective_local_mix_sum / valid_time_mass
        if valid_time_mass > 0.0
        else 0.0
    )
    statistics["mean_effective_global_mix"] = (
        effective_global_mix_sum / valid_time_mass
        if valid_time_mass > 0.0
        else 0.0
    )
    statistics["granularity_top1_fraction"] = (
        (top1_mass / valid_time_mass).tolist()
        if top1_mass is not None and valid_time_mass > 0.0
        else []
    )
    mean_weights = (
        (valid_weight_sum / valid_time_mass).tolist()
        if valid_weight_sum is not None and valid_time_mass > 0.0
        else []
    )
    return statistics, mean_weights


def _make_query_counterfactuals(line_tokens, valid_lengths, patch_mask, valid_fraction):
    """Return deterministic within-sample shuffle and constant-query controls."""
    batch_size, patch_count, visual_dim = line_tokens.shape
    permutation = torch.arange(patch_count).repeat(batch_size, 1)
    valid = patch_mask.to(dtype=torch.bool)
    for sample_index in range(batch_size):
        sample_lengths = valid_lengths[sample_index]
        for length in torch.unique(sample_lengths[valid[sample_index]]):
            indices = torch.nonzero(
                valid[sample_index] & (sample_lengths == length),
                as_tuple=False,
            ).flatten()
            if indices.numel() > 1:
                permutation[sample_index, indices] = indices.roll(1)
    shuffled = torch.gather(
        line_tokens,
        dim=1,
        index=permutation.unsqueeze(-1).expand(-1, -1, visual_dim),
    )
    importance = valid_fraction.float() * valid.float()
    denominator = importance.sum(dim=1, keepdim=True).clamp_min(1e-8)
    sample_mean = torch.sum(
        line_tokens.float() * importance.unsqueeze(-1),
        dim=1,
        keepdim=True,
    ) / denominator.unsqueeze(-1)
    constant = sample_mean.expand(-1, patch_count, -1).to(line_tokens.dtype)
    moved = permutation != torch.arange(patch_count).view(1, -1)
    return shuffled, constant, permutation, moved


@torch.no_grad()
def _evaluate_patch(
    model,
    loader,
    classes,
    device,
    description,
    sample_subject_ids=None,
):
    model.eval()
    details = {
        "sample_index": [],
        "y_true": [],
        "y_pred": [],
        "y_score": [],
        "weights": [],
        "scores": [],
        "score_margin": [],
        "top1": [],
        "max_weight": [],
        "entropy": [],
        "dataset_prior_weights": [],
        "sample_global_weights": [],
        "patch_local_weights": [],
        "candidate_evidence": [],
        "sample_global_score_margin": [],
        "patch_local_score_margin": [],
        "sample_global_confidence": [],
        "patch_local_confidence": [],
        "global_mix": [],
        "local_mix": [],
        "effective_global_mix": [],
        "effective_local_mix": [],
        "sample_global_route_scores": [],
        "patch_local_route_scores": [],
        "router_pre_topk_scores": [],
        "router_pre_topk_weights": [],
        "router_topk_mask": [],
        "router_topk_indices": [],
        "router_noisy_topk_scores": [],
        "channel_weights": [],
        "patch_mask": [],
        "valid_fraction": [],
        "valid_lengths": [],
        "graph_pairwise_cosine": [],
        "projected_query_rms": [],
        "projected_relative_key_rms": [],
        "projected_key_pairwise_l2": [],
        "query_shuffle_weights": [],
        "query_shuffle_top1": [],
        "query_shuffle_y_score": [],
        "query_constant_weights": [],
        "query_constant_top1": [],
        "query_constant_y_score": [],
        "query_shuffle_permutation": [],
        "query_shuffle_moved": [],
        "uniform_y_score": [],
        "dataset_prior_y_score": [],
        "sample_global_y_score": [],
        "patch_local_y_score": [],
        "soft_uniform_visual_effect": [],
    }
    for batch in tqdm(loader, desc=description, leave=False):
        logits, auxiliary, labels, lengths, sample_indices = (
            _forward_patch_training_batch(model, batch, device)
        )
        probabilities = torch.softmax(logits.float(), dim=-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("patch classifier produced non-finite probabilities")
        details["sample_index"].append(sample_indices.numpy())
        details["y_true"].append(labels.cpu().numpy())
        details["y_pred"].append(probabilities.argmax(dim=-1).cpu().numpy())
        details["y_score"].append(probabilities.cpu().numpy())
        details["weights"].append(auxiliary["granularity_weights"].cpu().numpy())
        details["scores"].append(auxiliary["granularity_scores"].cpu().numpy())
        details["score_margin"].append(
            auxiliary["granularity_score_margin"].cpu().numpy()
        )
        details["top1"].append(auxiliary["granularity_top1"].cpu().numpy())
        details["max_weight"].append(
            auxiliary["granularity_max_weight"].cpu().numpy()
        )
        details["entropy"].append(
            auxiliary["granularity_entropy"].cpu().numpy()
        )
        for name in (
            "dataset_prior_weights",
            "sample_global_weights",
            "patch_local_weights",
            "candidate_evidence",
            "sample_global_score_margin",
            "patch_local_score_margin",
            "sample_global_confidence",
            "patch_local_confidence",
            "effective_global_mix",
            "effective_local_mix",
            "sample_global_route_scores",
            "patch_local_route_scores",
            "router_pre_topk_scores",
            "router_pre_topk_weights",
            "router_topk_mask",
            "router_topk_indices",
            "router_noisy_topk_scores",
        ):
            details[name].append(auxiliary[name].cpu().numpy())
        for name in ("global_mix", "local_mix"):
            expanded = auxiliary[name].expand_as(
                auxiliary["candidate_evidence"]
            ) * batch[3].to(device=device, dtype=torch.float32)
            details[name].append(expanded.cpu().numpy())
        details["channel_weights"].append(
            auxiliary["channel_weights"].cpu().numpy()
        )
        details["patch_mask"].append(batch[3].numpy())
        details["valid_fraction"].append(batch[4].float().numpy())
        details["valid_lengths"].append(lengths.numpy())
        normalized_graph = F.normalize(batch[1].float(), dim=-1, eps=1e-8)
        pairwise_cosines = []
        for left_index in range(normalized_graph.shape[2]):
            for right_index in range(left_index + 1, normalized_graph.shape[2]):
                pairwise_cosines.append(
                    torch.sum(
                        normalized_graph[:, :, left_index]
                        * normalized_graph[:, :, right_index],
                        dim=-1,
                    )
                )
        details["graph_pairwise_cosine"].append(
            torch.stack(pairwise_cosines, dim=-1).numpy()
        )

        line, graph, mantis, mask, fraction, valid_lengths, _, _ = batch
        shuffled_line, constant_line, permutation, moved = (
            _make_query_counterfactuals(
                line,
                valid_lengths,
                mask,
                fraction,
            )
        )
        graph_device = graph.to(device=device, dtype=torch.float32)
        line_device = line.to(device=device, dtype=torch.float32)
        mantis_device = mantis.to(device=device, dtype=torch.float32)
        mask_device = mask.to(device=device, dtype=torch.bool)
        fraction_device = fraction.to(device=device, dtype=torch.float32)
        projected_query = model.granularity_attention.project_query(line_device)
        relative_keys, _ = model.granularity_attention._project_graph(graph_device)
        details["projected_query_rms"].append(
            projected_query.float().square().mean(dim=-1).sqrt().cpu().numpy()
        )
        details["projected_relative_key_rms"].append(
            relative_keys.float().square().mean(dim=(-1, -2)).sqrt().cpu().numpy()
        )
        projected_pairwise_l2 = []
        for left_index in range(relative_keys.shape[2]):
            for right_index in range(left_index + 1, relative_keys.shape[2]):
                projected_pairwise_l2.append(
                    torch.linalg.vector_norm(
                        relative_keys[:, :, left_index]
                        - relative_keys[:, :, right_index],
                        dim=-1,
                    ) / math.sqrt(relative_keys.shape[-1])
                )
        details["projected_key_pairwise_l2"].append(
            torch.stack(projected_pairwise_l2, dim=-1).cpu().numpy()
        )
        shuffled_logits, shuffled_auxiliary = model(
            shuffled_line.to(device=device, dtype=torch.float32),
            graph_device,
            mantis_device,
            mask_device,
            fraction_device,
        )
        constant_logits, constant_auxiliary = model(
            constant_line.to(device=device, dtype=torch.float32),
            graph_device,
            mantis_device,
            mask_device,
            fraction_device,
        )
        uniform_weights = torch.full(
            (
                line.shape[0],
                line.shape[1],
                graph.shape[2],
            ),
            1.0 / graph.shape[2],
            dtype=torch.float32,
            device=device,
        ) * mask_device.unsqueeze(-1)
        uniform_logits, uniform_auxiliary = model(
            line_device,
            graph_device,
            mantis_device,
            mask_device,
            fraction_device,
            granularity_weights_override=uniform_weights,
        )
        hierarchy_logits = {}
        for route_name in (
            "dataset_prior_weights",
            "sample_global_weights",
            "patch_local_weights",
        ):
            route_logits, _ = model(
                line_device,
                graph_device,
                mantis_device,
                mask_device,
                fraction_device,
                granularity_weights_override=auxiliary[route_name],
            )
            hierarchy_logits[route_name] = route_logits
        details["query_shuffle_weights"].append(
            shuffled_auxiliary["granularity_weights"].cpu().numpy()
        )
        details["query_shuffle_top1"].append(
            shuffled_auxiliary["granularity_top1"].cpu().numpy()
        )
        details["query_shuffle_y_score"].append(
            torch.softmax(shuffled_logits.float(), dim=-1).cpu().numpy()
        )
        details["query_constant_weights"].append(
            constant_auxiliary["granularity_weights"].cpu().numpy()
        )
        details["query_constant_top1"].append(
            constant_auxiliary["granularity_top1"].cpu().numpy()
        )
        details["query_constant_y_score"].append(
            torch.softmax(constant_logits.float(), dim=-1).cpu().numpy()
        )
        details["query_shuffle_permutation"].append(permutation.numpy())
        details["query_shuffle_moved"].append(moved.numpy())
        details["uniform_y_score"].append(
            torch.softmax(uniform_logits.float(), dim=-1).cpu().numpy()
        )
        for route_name, output_name in (
            ("dataset_prior_weights", "dataset_prior_y_score"),
            ("sample_global_weights", "sample_global_y_score"),
            ("patch_local_weights", "patch_local_y_score"),
        ):
            details[output_name].append(
                torch.softmax(
                    hierarchy_logits[route_name].float(), dim=-1
                ).cpu().numpy()
            )
        visual_effect = torch.linalg.vector_norm(
            auxiliary["visual_tokens"] - uniform_auxiliary["visual_tokens"],
            dim=-1,
        ) / torch.linalg.vector_norm(
            uniform_auxiliary["visual_tokens"],
            dim=-1,
        ).clamp_min(1e-8)
        details["soft_uniform_visual_effect"].append(
            visual_effect.cpu().numpy()
        )
    if not details["y_true"]:
        raise ValueError("cannot evaluate an empty patch-feature split")
    details = {key: np.concatenate(value, axis=0) for key, value in details.items()}
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


@torch.no_grad()
def _evaluate_patch_metrics_only(
    model,
    loader,
    classes,
    device,
    description,
    sample_subject_ids=None,
):
    """Fast validation path without counterfactual forwards or large diagnostics."""
    model.eval()
    details = {
        "sample_index": [],
        "y_true": [],
        "y_pred": [],
        "y_score": [],
    }
    for batch in tqdm(loader, desc=description, leave=False):
        logits, _, labels, _, sample_indices = _forward_patch_training_batch(
            model,
            batch,
            device,
        )
        probabilities = torch.softmax(logits.float(), dim=-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("patch classifier produced non-finite probabilities")
        details["sample_index"].append(sample_indices.numpy())
        details["y_true"].append(labels.cpu().numpy())
        details["y_pred"].append(probabilities.argmax(dim=-1).cpu().numpy())
        details["y_score"].append(probabilities.cpu().numpy())
    if not details["y_true"]:
        raise ValueError("cannot evaluate an empty patch-feature split")
    details = {key: np.concatenate(value, axis=0) for key, value in details.items()}
    window_metrics = compute_metrics_from_predictions(
        details["y_true"],
        details["y_pred"],
        details["y_score"],
        np.arange(len(classes)),
    )
    subject_metrics = None
    if sample_subject_ids is not None:
        subject_metrics, _ = _aggregate_subject_predictions(
            details["y_true"],
            details["y_score"],
            details["sample_index"],
            sample_subject_ids,
            classes,
        )
    return _merge_subject_metrics(window_metrics, subject_metrics)


def _json_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _atomic_json_dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_patch_diagnostics(
    artifact_dir,
    split_name,
    details,
    granularity_labels,
):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "schema_version": np.asarray(5, dtype=np.int64),
        "architecture": np.asarray(PATCH_MINDTS_ARCHITECTURE),
        "granularity_labels": np.asarray(granularity_labels),
        **details,
    }
    path = artifact_dir / f"patch_atgs_diagnostics_{split_name}.npz"
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)

    valid = details["patch_mask"].astype(bool)
    valid_weights = details["weights"][valid]
    valid_top1 = details["top1"][valid]
    valid_importance = details["valid_fraction"][valid].astype(np.float64)
    importance_sum = float(valid_importance.sum())
    candidate_count = len(granularity_labels)
    if importance_sum > 0.0:
        weighted_mean = np.sum(
            valid_weights * valid_importance[:, None],
            axis=0,
        ) / importance_sum
        weighted_variance = np.sum(
            (valid_weights - weighted_mean[None, :]) ** 2
            * valid_importance[:, None],
            axis=0,
        ) / importance_sum
        weighted_top1 = np.asarray(
            [
                valid_importance[valid_top1 == candidate_index].sum()
                for candidate_index in range(candidate_count)
            ],
            dtype=np.float64,
        ) / importance_sum
        weighted_max = float(
            np.sum(details["max_weight"][valid] * valid_importance)
            / importance_sum
        )
        weighted_entropy = float(
            np.sum(details["entropy"][valid] * valid_importance)
            / importance_sum
        )
    else:
        weighted_mean = np.zeros(candidate_count, dtype=np.float64)
        weighted_variance = np.zeros(candidate_count, dtype=np.float64)
        weighted_top1 = np.zeros(candidate_count, dtype=np.float64)
        weighted_max = 0.0
        weighted_entropy = 0.0
    uniform = np.full(candidate_count, 1.0 / candidate_count)
    valid_pairwise_cosine = details["graph_pairwise_cosine"][valid]
    valid_projected_query_rms = details["projected_query_rms"][valid]
    valid_projected_key_rms = details["projected_relative_key_rms"][valid]
    valid_projected_pairwise_l2 = details["projected_key_pairwise_l2"][valid]
    router_distributions = {
        name: details[name][valid]
        for name in (
            "dataset_prior_weights",
            "sample_global_weights",
            "patch_local_weights",
        )
    }
    summary = {
        "sample_count": int(len(details["y_true"])),
        "valid_patch_count": int(valid.sum()),
        "granularity_labels": list(granularity_labels),
        "mean_weights": weighted_mean.tolist(),
        "std_weights": np.sqrt(weighted_variance).tolist(),
        "weight_quantiles": {
            "q05": np.quantile(valid_weights, 0.05, axis=0).tolist(),
            "q50": np.quantile(valid_weights, 0.50, axis=0).tolist(),
            "q95": np.quantile(valid_weights, 0.95, axis=0).tolist(),
        } if len(valid_weights) else {
            "q05": [0.0] * candidate_count,
            "q50": [0.0] * candidate_count,
            "q95": [0.0] * candidate_count,
        },
        "mean_l1_from_uniform": (
            float(
                np.sum(
                    np.abs(valid_weights - uniform[None, :]).sum(axis=-1)
                    * valid_importance
                ) / importance_sum
            )
            if importance_sum > 0.0
            else 0.0
        ),
        "top1_counts": np.bincount(
            valid_top1,
            minlength=candidate_count,
        ).tolist(),
        "top1_time_weighted_fraction": weighted_top1.tolist(),
        "mean_max_weight": weighted_max,
        "mean_normalized_entropy": weighted_entropy,
        "mean_per_patch_effective_granularity_count": float(
            np.clip(
                math.exp(
                    float(np.clip(weighted_entropy, 0.0, 1.0))
                    * math.log(candidate_count)
                ),
                1.0,
                float(candidate_count),
            )
        ),
        "global_effective_granularity_count": float(
            np.clip(
                math.exp(
                    -np.sum(
                        weighted_mean
                        * np.log(np.clip(weighted_mean, 1e-12, None))
                    )
                ),
                1.0,
                float(candidate_count),
            )
        ),
        "mean_score_margin": (
            float(
                np.sum(details["score_margin"][valid] * valid_importance)
                / importance_sum
            )
            if importance_sum > 0.0
            else 0.0
        ),
        "graph_candidate_pairs": [
            [left_index, right_index]
            for left_index in range(candidate_count)
            for right_index in range(left_index + 1, candidate_count)
        ],
        "graph_pairwise_cosine_mean": (
            (
                np.sum(
                    valid_pairwise_cosine * valid_importance[:, None],
                    axis=0,
                ) / importance_sum
            ).tolist()
            if importance_sum > 0.0
            else []
        ),
        "graph_pairwise_cosine_quantiles": {
            "q05": np.quantile(valid_pairwise_cosine, 0.05, axis=0).tolist(),
            "q50": np.quantile(valid_pairwise_cosine, 0.50, axis=0).tolist(),
            "q95": np.quantile(valid_pairwise_cosine, 0.95, axis=0).tolist(),
        } if len(valid_pairwise_cosine) else {
            "q05": [],
            "q50": [],
            "q95": [],
        },
        "router_projected_space": {
            "query_rms_mean": (
                float(
                    np.sum(valid_projected_query_rms * valid_importance)
                    / importance_sum
                )
                if importance_sum > 0.0
                else 0.0
            ),
            "relative_key_rms_mean": (
                float(
                    np.sum(valid_projected_key_rms * valid_importance)
                    / importance_sum
                )
                if importance_sum > 0.0
                else 0.0
            ),
            "relative_key_rms_quantiles": {
                "q05": float(np.quantile(valid_projected_key_rms, 0.05)),
                "q50": float(np.quantile(valid_projected_key_rms, 0.50)),
                "q95": float(np.quantile(valid_projected_key_rms, 0.95)),
            } if len(valid_projected_key_rms) else {
                "q05": 0.0,
                "q50": 0.0,
                "q95": 0.0,
            },
            "key_pairwise_l2_mean": (
                (
                    np.sum(
                        valid_projected_pairwise_l2
                        * valid_importance[:, None],
                        axis=0,
                    ) / importance_sum
                ).tolist()
                if importance_sum > 0.0
                else []
            ),
        },
        "router_hierarchy": {
            f"mean_{name}": (
                (
                    np.sum(values * valid_importance[:, None], axis=0)
                    / importance_sum
                ).tolist()
                if importance_sum > 0.0
                else [0.0] * candidate_count
            )
            for name, values in router_distributions.items()
        },
    }
    summary["router_hierarchy"].update(
        {
            name: (
                float(
                    np.sum(details[name][valid] * valid_importance)
                    / importance_sum
                )
                if importance_sum > 0.0
                else 0.0
            )
            for name in (
                "candidate_evidence",
                "sample_global_score_margin",
                "patch_local_score_margin",
                "sample_global_confidence",
                "patch_local_confidence",
                "global_mix",
                "local_mix",
                "effective_global_mix",
                "effective_local_mix",
            )
        }
    )
    valid_topk_mask = details["router_topk_mask"][valid].astype(np.float64)
    valid_pre_topk_weights = details["router_pre_topk_weights"][valid]
    if importance_sum > 0.0:
        weighted_pre_topk = np.sum(
            valid_pre_topk_weights * valid_importance[:, None],
            axis=0,
        ) / importance_sum
        weighted_hard_load = np.sum(
            valid_topk_mask * valid_importance[:, None],
            axis=0,
        ) / importance_sum
        hard_load_mean = float(weighted_hard_load.mean())
        hard_load_cv2 = float(
            np.mean((weighted_hard_load - hard_load_mean) ** 2)
            / max(hard_load_mean ** 2, 1e-12)
        )
        active_count = valid_topk_mask.sum(axis=-1)
        mean_active_count = float(
            np.sum(active_count * valid_importance) / importance_sum
        )
    else:
        weighted_pre_topk = np.zeros(candidate_count, dtype=np.float64)
        weighted_hard_load = np.zeros(candidate_count, dtype=np.float64)
        hard_load_cv2 = 0.0
        mean_active_count = 0.0
    summary["router_topk"] = {
        "mean_active_experts_per_patch": mean_active_count,
        "pre_topk_mean_weights": weighted_pre_topk.tolist(),
        "hard_support_fraction": weighted_hard_load.tolist(),
        "hard_support_cv2": hard_load_cv2,
    }
    summary["router_hierarchy"]["prediction_probability_total_variation"] = {
        route_name: float(
            np.mean(
                0.5
                * np.abs(details["y_score"] - details[score_name]).sum(axis=-1)
            )
        )
        for route_name, score_name in (
            ("dataset_prior", "dataset_prior_y_score"),
            ("sample_global", "sample_global_y_score"),
            ("patch_local", "patch_local_y_score"),
        )
    }
    summary["window_metrics"] = compute_metrics_from_predictions(
        details["y_true"],
        details["y_pred"],
        details["y_score"],
        np.arange(details["y_score"].shape[1]),
    )
    if "subject_y_true" in details:
        subject_metrics = compute_metrics_from_predictions(
            details["subject_y_true"],
            details["subject_y_pred"],
            details["subject_y_score"],
            np.arange(details["subject_y_score"].shape[1]),
        )
        subject_nll = -np.log(
            np.clip(
                details["subject_y_score"]
                [
                    np.arange(len(details["subject_y_true"])),
                    details["subject_y_true"],
                ],
                1e-12,
                1.0,
            )
        )
        subject_metrics["macro_log_loss"] = float(
            np.mean(
                [
                    subject_nll[
                        details["subject_y_true"] == class_index
                    ].mean()
                    for class_index in np.unique(details["subject_y_true"])
                ]
            )
        )
        summary["subject_count"] = int(len(details["subject_y_true"]))
        summary["subject_aggregation"] = "mean_window_softmax_probability"
        summary["subject_metrics"] = subject_metrics
    shuffled_weights = details["query_shuffle_weights"][valid]
    constant_weights = details["query_constant_weights"][valid]
    shuffled_top1 = details["query_shuffle_top1"][valid]
    constant_top1 = details["query_constant_top1"][valid]
    moved = details["query_shuffle_moved"][valid].astype(bool)
    moved_importance = valid_importance * moved.astype(np.float64)
    moved_mass = float(moved_importance.sum())
    shuffled_tv = 0.5 * np.abs(valid_weights - shuffled_weights).sum(axis=-1)
    constant_tv = 0.5 * np.abs(valid_weights - constant_weights).sum(axis=-1)
    confident = details["score_margin"][valid] >= 0.05
    confident_moved_importance = (
        valid_importance * moved.astype(np.float64) * confident.astype(np.float64)
    )
    confident_moved_mass = float(confident_moved_importance.sum())
    sample_moved = details["query_shuffle_moved"].any(axis=1)
    shuffle_prediction_tv = 0.5 * np.abs(
        details["y_score"] - details["query_shuffle_y_score"]
    ).sum(axis=-1)
    summary["query_counterfactuals"] = {
        "shuffle_coverage_time_fraction": (
            moved_mass / importance_sum if importance_sum > 0.0 else 0.0
        ),
        "shuffle_weighted_total_variation": (
            float(np.sum(shuffled_tv * moved_importance) / moved_mass)
            if moved_mass > 0.0
            else 0.0
        ),
        "shuffle_top1_flip_time_fraction": (
            float(
                np.sum((valid_top1 != shuffled_top1) * moved_importance)
                / moved_mass
            )
            if moved_mass > 0.0
            else 0.0
        ),
        "shuffle_confident_top1_flip_time_fraction": (
            float(
                np.sum(
                    (valid_top1 != shuffled_top1)
                    * confident_moved_importance
                ) / confident_moved_mass
            )
            if confident_moved_mass > 0.0
            else 0.0
        ),
        "constant_query_weighted_total_variation": (
            float(np.sum(constant_tv * valid_importance) / importance_sum)
            if importance_sum > 0.0
            else 0.0
        ),
        "constant_query_top1_flip_time_fraction": (
            float(
                np.sum((valid_top1 != constant_top1) * valid_importance)
                / importance_sum
            )
            if importance_sum > 0.0
            else 0.0
        ),
        "shuffle_prediction_probability_total_variation": float(
            np.mean(shuffle_prediction_tv[sample_moved])
            if sample_moved.any()
            else 0.0
        ),
        "shuffle_prediction_probability_total_variation_all_samples": float(
            np.mean(shuffle_prediction_tv)
        ),
        "constant_prediction_probability_total_variation": float(
            np.mean(
                0.5
                * np.abs(
                    details["y_score"] - details["query_constant_y_score"]
                ).sum(axis=-1)
            )
        ),
    }
    summary["soft_route_effect"] = {
        "visual_relative_l2_mean": (
            float(
                np.sum(
                    details["soft_uniform_visual_effect"][valid]
                    * valid_importance
                ) / importance_sum
            )
            if importance_sum > 0.0
            else 0.0
        ),
        "prediction_probability_total_variation": float(
            np.mean(
                0.5
                * np.abs(
                    details["y_score"] - details["uniform_y_score"]
                ).sum(axis=-1)
            )
        ),
    }
    return summary


def _save_patch_checkpoint(
    artifact_dir,
    model,
    classes,
    training_history,
    selected_epoch,
    validation_metrics,
    checkpoint_selection_requested,
    checkpoint_selection_effective,
    validation_subject_ids,
    outer_patch_size,
    outer_patch_stride,
    granularity_bank,
    alignment_weight,
    granularity_balance_weight,
    granularity_entropy_weight,
    granularity_mix_shrinkage_weight,
    granularity_prior_kl_weight,
    router_route_budget_weight=0.0,
    router_load_balance_weight=0.0,
    early_stop_strategy="raw_selection_key",
    early_stop_patience=0,
    early_stop_min_epochs=0,
    early_stop_ema_decay=0.6,
    early_stop_min_delta=0.0,
    early_stop_best_score=None,
    early_stopped=False,
    stopped_epoch=None,
):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 5,
        "architecture": PATCH_MINDTS_ARCHITECTURE,
        "tail_policy": PATCH_TAIL_POLICY,
        "pooling_policy": PATCH_POOLING_POLICY,
        "outer_patch_size": int(outer_patch_size),
        "outer_patch_stride": int(outer_patch_stride),
        "granularity_bank": [list(regime) for regime in granularity_bank],
        "alignment_weight": float(alignment_weight),
        "granularity_balance_weight": float(granularity_balance_weight),
        "granularity_entropy_weight": float(granularity_entropy_weight),
        "granularity_mix_shrinkage_weight": float(
            granularity_mix_shrinkage_weight
        ),
        "granularity_prior_kl_weight": float(granularity_prior_kl_weight),
        "router_route_budget_weight": float(router_route_budget_weight),
        "router_load_balance_weight": float(router_load_balance_weight),
        "granularity_regularization_policy": model.configuration[
            "granularity_regularization_policy"
        ],
        "selected_epoch": int(selected_epoch),
        # Retain the historical key with explicit window semantics.
        "validation_macro_f1": float(validation_metrics["macro_f1"]),
        "validation_subject_macro_f1": (
            float(validation_metrics["subject_macro_f1"])
            if "subject_macro_f1" in validation_metrics
            else None
        ),
        "checkpoint_selection_requested": checkpoint_selection_requested,
        "checkpoint_selection_effective": checkpoint_selection_effective,
        "checkpoint_selection_unit": (
            "subject"
            if checkpoint_selection_effective == "subject_macro_f1"
            else "window"
        ),
        "checkpoint_selection_tie_breaker": (
            "subject_macro_log_loss:min,epoch:min"
            if checkpoint_selection_effective == "subject_macro_f1"
            else "epoch:min"
        ),
        "early_stopping": {
            "strategy": early_stop_strategy,
            "patience": int(early_stop_patience),
            "min_epochs": int(early_stop_min_epochs),
            "ema_decay": float(early_stop_ema_decay),
            "min_delta": float(early_stop_min_delta),
            "best_monitor_score": (
                float(early_stop_best_score)
                if early_stop_best_score is not None
                else None
            ),
            "early_stopped": bool(early_stopped),
            "stopped_epoch": (
                int(stopped_epoch) if stopped_epoch is not None else None
            ),
            "checkpoint_selection_uses_raw_metric": True,
        },
        "selected_validation_window_metrics": {
            key: float(value)
            for key, value in validation_metrics.items()
            if not key.startswith("subject_")
        },
        "selected_validation_subject_metrics": {
            key.removeprefix("subject_"): float(value)
            for key, value in validation_metrics.items()
            if key.startswith("subject_")
        },
        "subject_aggregation": "mean_window_softmax_probability",
        "validation_subject_id_sha256": _subject_ids_digest(
            validation_subject_ids
        ),
        "validation_subject_count": (
            int(len(np.unique(validation_subject_ids)))
            if validation_subject_ids is not None
            else None
        ),
        "classes": [_json_safe(label) for label in classes],
        "model_constructor_configuration": dict(model.constructor_configuration),
        "model_configuration": dict(model.configuration),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "training_history": training_history,
    }
    path = artifact_dir / "patch_mindts_checkpoint.pt"
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def train_patch_mindts_classifier(
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
    granularity_temperature=1.0,
    granularity_balance_weight=0.01,
    granularity_entropy_weight=0.01,
    granularity_mix_shrinkage_weight=0.005,
    granularity_prior_kl_weight=0.001,
    granularity_usage_floor=0.05,
    granularity_usage_ema_decay=0.95,
    granularity_entropy_floor=0.55,
    granularity_entropy_ceiling=1.0,
    granularity_router_mode="adaptive_v4",
    granularity_local_mix_max=0.50,
    granularity_local_mix_init=0.10,
    granularity_global_mix_max=0.75,
    granularity_global_mix_init=0.50,
    granularity_evidence_half_saturation=0.05,
    granularity_minimum_weight=0.0,
    granularity_score_cap=1.0,
    granularity_scorer_hidden_dim=32,
    granularity_confidence_half_saturation=0.05,
    checkpoint_metric="auto",
    channel_hidden_dim=64,
    artifact_dir=None,
    router_top_k=2,
    router_training_noise_std=0.0,
    router_local_weight=0.5,
    router_relation_hidden_dim=16,
    router_relation_residual_scale=0.25,
    router_key_adapter_scale=0.1,
    router_value_adapter_scale=0.1,
    router_route_budget_weight=0.005,
    router_load_balance_weight=0.005,
    early_stop_strategy="raw_selection_key",
    early_stop_min_epochs=0,
    early_stop_ema_decay=0.6,
    early_stop_min_delta=0.0,
):
    """Train the dedicated patch-level classifier and return legacy metrics API."""
    if epochs <= 0 or early_stop_patience < 0:
        raise ValueError("epochs must be positive and patience non-negative")
    if early_stop_strategy not in {"raw_selection_key", "ema_primary"}:
        raise ValueError(f"unsupported early-stop strategy: {early_stop_strategy}")
    if early_stop_min_epochs < 0 or early_stop_min_epochs > epochs:
        raise ValueError("early-stop min_epochs must lie in [0, epochs]")
    if (
        not math.isfinite(early_stop_ema_decay)
        or not 0.0 <= early_stop_ema_decay < 1.0
    ):
        raise ValueError("early-stop EMA decay must lie in [0, 1)")
    if not math.isfinite(early_stop_min_delta) or early_stop_min_delta < 0.0:
        raise ValueError("early-stop min_delta must be finite and non-negative")
    if not math.isfinite(alignment_weight) or alignment_weight < 0.0:
        raise ValueError("alignment_weight must be finite and non-negative")
    for name, value in (
        ("granularity_balance_weight", granularity_balance_weight),
        ("granularity_entropy_weight", granularity_entropy_weight),
        ("granularity_mix_shrinkage_weight", granularity_mix_shrinkage_weight),
        ("granularity_prior_kl_weight", granularity_prior_kl_weight),
        ("router_route_budget_weight", router_route_budget_weight),
        ("router_load_balance_weight", router_load_balance_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if (
        not math.isfinite(granularity_usage_floor)
        or not 0.0 <= granularity_usage_floor < 1.0
    ):
        raise ValueError("granularity_usage_floor must lie in [0, 1)")
    if (
        not math.isfinite(granularity_usage_ema_decay)
        or not 0.0 <= granularity_usage_ema_decay < 1.0
    ):
        raise ValueError("granularity_usage_ema_decay must lie in [0, 1)")
    if (
        not math.isfinite(granularity_entropy_floor)
        or not 0.0 <= granularity_entropy_floor <= 1.0
    ):
        raise ValueError("granularity_entropy_floor must lie in [0, 1]")
    if (
        not math.isfinite(granularity_entropy_ceiling)
        or not 0.0 <= granularity_entropy_ceiling <= 1.0
    ):
        raise ValueError("granularity_entropy_ceiling must lie in [0, 1]")
    if granularity_entropy_floor > granularity_entropy_ceiling:
        raise ValueError("granularity_entropy_floor cannot exceed entropy_ceiling")
    if granularity_router_mode not in {
        "adaptive_v4",
        "adaptive_v41",
        "adaptive_v5",
        "uniform",
    }:
        raise ValueError(
            "granularity_router_mode must be adaptive_v4, adaptive_v41, "
            "adaptive_v5, or uniform"
        )
    if granularity_scorer_hidden_dim <= 0:
        raise ValueError("granularity_scorer_hidden_dim must be positive")
    if (
        not math.isfinite(granularity_confidence_half_saturation)
        or granularity_confidence_half_saturation <= 0.0
    ):
        raise ValueError(
            "granularity_confidence_half_saturation must be positive and finite"
        )
    if vision_model is None or mantis_model is None:
        raise ValueError("patch_mindts requires vision_model and mantis_model")
    graph_bank = getattr(vision_model, "med_activity_granularity_bank", None)
    if graph_bank is None or not hasattr(graph_bank, "patch_length_bank"):
        raise ValueError("vision_model does not expose an adaptive granularity bank")
    granularity_bank = tuple(tuple(regime) for regime in graph_bank.patch_length_bank)
    num_granularities = len(granularity_bank)
    if num_granularities != 3:
        raise ValueError("patch_mindts requires exactly three graph experts")
    regime_widths = {len(regime) for regime in granularity_bank}
    if len(regime_widths) != 1 or not regime_widths.issubset({1, 3}):
        raise ValueError(
            "patch_mindts graph experts must be either three single-scale "
            "neutral-RGB regimes or three legacy three-scale RGB regimes"
        )
    for regime in granularity_bank:
        if tuple(sorted(set(int(value) for value in regime))) != tuple(
            int(value) for value in regime
        ):
            raise ValueError("scales inside each graph expert must be strictly increasing")
    if len(set(granularity_bank)) != num_granularities:
        raise ValueError("patch_mindts graph experts must be distinct")
    internal_scales = tuple(
        sorted({int(scale) for regime in granularity_bank for scale in regime})
    )
    if outer_patch_size <= internal_scales[-1]:
        raise ValueError(
            "outer_patch_size must exceed the largest internal graph scale"
        )
    if granularity_usage_floor > 1.0 / num_granularities:
        raise ValueError(
            "granularity_usage_floor cannot exceed uniform expert usage"
        )

    has_fixed_validation = val_loader is not None or val_labels is not None
    if has_fixed_validation and (val_loader is None or val_labels is None):
        raise ValueError("val_loader and val_labels must be supplied together")
    if has_fixed_validation:
        train_indices = list(range(len(train_loader.dataset)))
        val_indices = list(range(len(val_loader.dataset)))
    else:
        train_indices, val_indices = get_split(
            train_loader.dataset,
            frac=val_ratio,
            random_seed=random_seed,
        )

    train_subject_ids = _loader_sample_subject_ids(train_loader)
    test_subject_ids = _loader_sample_subject_ids(test_loader)
    if has_fixed_validation:
        validation_subject_ids = _loader_sample_subject_ids(val_loader)
    else:
        validation_subject_ids = train_subject_ids
    checkpoint_metric_effective = _resolve_checkpoint_metric(
        checkpoint_metric,
        train_subject_ids,
        validation_subject_ids,
        train_indices,
        val_indices,
    )
    print(
        f"Patch checkpoint selection: requested={checkpoint_metric} "
        f"effective={checkpoint_metric_effective}"
    )

    train_label_indices, classes, class_to_index = _labels_to_indices(train_labels)
    test_label_indices = _map_labels(test_labels, class_to_index)
    train_features = _get_patch_feature_split(
        "train",
        train_loader,
        train_labels,
        vision_model,
        mantis_model,
        device,
        outer_patch_size,
        outer_patch_stride,
        visual_encode_batch_size,
        feature_cache_dir,
        feature_cache_signature,
        num_granularities,
    )
    test_features = _get_patch_feature_split(
        "test",
        test_loader,
        test_labels,
        vision_model,
        mantis_model,
        device,
        outer_patch_size,
        outer_patch_stride,
        visual_encode_batch_size,
        feature_cache_dir,
        feature_cache_signature,
        num_granularities,
    )
    if has_fixed_validation:
        val_label_indices = _map_labels(val_labels, class_to_index)
        val_features = _get_patch_feature_split(
            "vali",
            val_loader,
            val_labels,
            vision_model,
            mantis_model,
            device,
            outer_patch_size,
            outer_patch_stride,
            visual_encode_batch_size,
            feature_cache_dir,
            feature_cache_signature,
            num_granularities,
        )

    set_random_seed(random_seed)
    if has_fixed_validation:
        fit_loader = _build_patch_loader(
            train_features,
            train_label_indices,
            train_indices,
            batch_size,
            shuffle=True,
        )
        validation_loader = _build_patch_loader(
            val_features,
            val_label_indices,
            val_indices,
            batch_size,
            shuffle=False,
        )
    else:
        fit_loader = _build_patch_loader(
            train_features,
            train_label_indices,
            train_indices,
            batch_size,
            shuffle=True,
        )
        validation_loader = _build_patch_loader(
            train_features,
            train_label_indices,
            val_indices,
            batch_size,
            shuffle=False,
        )
    test_feature_loader = _build_patch_loader(
        test_features,
        test_label_indices,
        range(len(test_label_indices)),
        batch_size,
        shuffle=False,
    )

    visual_dim = int(train_features["line_tokens"].shape[-1])
    temporal_dim = int(train_features["mantis_channel_tokens"].shape[-1])
    feature_channels = int(train_features["mantis_channel_tokens"].shape[2])
    if feature_channels != channels:
        raise ValueError(
            f"cached Mantis channel count {feature_channels} does not match data {channels}"
        )
    line_query_center = compute_line_query_center(
        train_features,
        train_indices,
    )
    model = PatchMindTSFusionModule(
        visual_dim=visual_dim,
        temporal_dim=temporal_dim,
        num_channels=channels,
        num_granularities=num_granularities,
        num_classes=len(classes),
        fusion_dim=fusion_dim,
        fusion_heads=fusion_heads,
        dropout=dropout,
        classifier_hidden_dim=hidden_dim,
        classifier_num_layers=num_layers,
        channel_hidden_dim=channel_hidden_dim,
        granularity_temperature=granularity_temperature,
        alignment_dim=alignment_dim,
        alignment_temperature=alignment_temperature,
        granularity_usage_floor=granularity_usage_floor,
        granularity_usage_ema_decay=granularity_usage_ema_decay,
        granularity_entropy_floor=granularity_entropy_floor,
        granularity_entropy_ceiling=granularity_entropy_ceiling,
        line_query_center=line_query_center,
        granularity_router_mode=granularity_router_mode,
        granularity_local_mix_max=granularity_local_mix_max,
        granularity_local_mix_init=granularity_local_mix_init,
        granularity_global_mix_max=granularity_global_mix_max,
        granularity_global_mix_init=granularity_global_mix_init,
        granularity_evidence_half_saturation=(
            granularity_evidence_half_saturation
        ),
        granularity_minimum_weight=granularity_minimum_weight,
        granularity_score_cap=granularity_score_cap,
        granularity_scorer_hidden_dim=granularity_scorer_hidden_dim,
        granularity_confidence_half_saturation=(
            granularity_confidence_half_saturation
        ),
        router_top_k=router_top_k,
        router_training_noise_std=router_training_noise_std,
        router_local_weight=router_local_weight,
        router_relation_hidden_dim=router_relation_hidden_dim,
        router_relation_residual_scale=router_relation_residual_scale,
        router_key_adapter_scale=router_key_adapter_scale,
        router_value_adapter_scale=router_value_adapter_scale,
    ).to(device)
    if granularity_router_mode == "uniform":
        for parameter in (
            *model.granularity_attention.query_projection.parameters(),
            *model.granularity_attention.key_projection.parameters(),
            model.granularity_attention.dataset_prior_logits,
            model.granularity_attention.local_mix_logit,
            model.granularity_attention.global_mix_logit,
        ):
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    if class_weight == "balanced":
        weights = _balanced_class_weights(
            train_label_indices[train_indices], len(classes), device
        )
    elif class_weight == "none":
        weights = None
    else:
        raise ValueError(f"unsupported class weighting: {class_weight}")
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")

    best_selection_key = None
    best_state = None
    best_epoch = 0
    early_stop_monitor = _EarlyStoppingMonitor(
        strategy=early_stop_strategy,
        patience=early_stop_patience,
        min_epochs=early_stop_min_epochs,
        ema_decay=early_stop_ema_decay,
        min_delta=early_stop_min_delta,
    )
    early_stopped = False
    stopped_epoch = None
    training_history = []
    for epoch in range(epochs):
        statistics, mean_weights = _run_patch_epoch(
            model,
            fit_loader,
            optimizer,
            criterion,
            alignment_weight,
            granularity_balance_weight,
            granularity_entropy_weight,
            granularity_mix_shrinkage_weight,
            granularity_prior_kl_weight,
            device,
            router_route_budget_weight=router_route_budget_weight,
            router_load_balance_weight=router_load_balance_weight,
        )
        validation_metrics = _evaluate_patch_metrics_only(
            model,
            validation_loader,
            classes,
            device,
            "Validate patch MLP",
            sample_subject_ids=validation_subject_ids,
        )
        selection_key = _checkpoint_selection_key(
            validation_metrics,
            checkpoint_metric_effective,
        )
        checkpoint_improved = (
            best_selection_key is None or selection_key > best_selection_key
        )
        early_stop_state = early_stop_monitor.update(selection_key, epoch + 1)
        epoch_record = {
            "epoch": epoch + 1,
            **statistics,
            "validation_macro_f1": float(validation_metrics["macro_f1"]),
            "validation_subject_macro_f1": (
                float(validation_metrics["subject_macro_f1"])
                if "subject_macro_f1" in validation_metrics
                else None
            ),
            "validation_subject_macro_log_loss": (
                float(validation_metrics["subject_macro_log_loss"])
                if "subject_macro_log_loss" in validation_metrics
                else None
            ),
            "checkpoint_selection_key": list(selection_key),
            "checkpoint_improved": bool(checkpoint_improved),
            "early_stop_raw_score": early_stop_state["raw_score"],
            "early_stop_smoothed_score": early_stop_state["smoothed_score"],
            "early_stop_best_score": early_stop_state["best_score"],
            "early_stop_improved": early_stop_state["improved"],
            "early_stop_epochs_without_improvement": early_stop_state[
                "epochs_without_improvement"
            ],
            "early_stop_eligible": early_stop_state["eligible"],
            "mean_granularity_weights": mean_weights,
        }
        training_history.append(epoch_record)
        print(
            f"Patch MLP epoch {epoch + 1}/{epochs} | "
            f"loss={statistics['loss']:.4f} | task={statistics['task_loss']:.4f} | "
            f"alignment={statistics['alignment_loss']:.4f} | "
            f"route_usage_floor={statistics['granularity_usage_floor_loss']:.4f} | "
            f"route_entropy_floor="
            f"{statistics['granularity_entropy_floor_loss']:.4f} | "
            f"route_budget="
            f"{statistics['granularity_route_budget_loss']:.4f} | "
            f"route_load_cv2="
            f"{statistics['granularity_load_cv2_loss']:.4f} | "
            f"mix(global/local)="
            f"{statistics['granularity_global_mix']:.3f}/"
            f"{statistics['granularity_local_mix']:.3f} | "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f} | "
            f"val_subject_macro_f1="
            f"{validation_metrics.get('subject_macro_f1', float('nan')):.4f} | "
            f"early_stop(raw/smoothed)="
            f"{early_stop_state['raw_score']:.4f}/"
            f"{early_stop_state['smoothed_score']:.4f} | "
            f"early_stop_wait="
            f"{early_stop_state['epochs_without_improvement']}/"
            f"{early_stop_patience} | "
            f"granularity={mean_weights}"
        )
        if checkpoint_improved:
            best_selection_key = selection_key
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
        if early_stop_state["should_stop"]:
            early_stopped = True
            stopped_epoch = epoch + 1
            print(
                f"Patch MLP early stopping at epoch {epoch + 1} "
                f"(strategy={early_stop_strategy}, min_epochs="
                f"{early_stop_min_epochs}, patience={early_stop_patience})"
            )
            break

    if best_state is None:
        raise RuntimeError("patch training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    val_metrics, val_details = _evaluate_patch(
        model,
        validation_loader,
        classes,
        device,
        "Evaluate best patch validation",
        sample_subject_ids=validation_subject_ids,
    )

    validation_subject_ids_used = (
        np.asarray(validation_subject_ids)[np.asarray(val_indices)]
        if validation_subject_ids is not None
        else None
    )
    if artifact_dir is not None:
        _save_patch_checkpoint(
            artifact_dir,
            model,
            classes,
            training_history,
            best_epoch,
            val_metrics,
            checkpoint_metric,
            checkpoint_metric_effective,
            validation_subject_ids_used,
            outer_patch_size,
            outer_patch_stride,
            granularity_bank,
            alignment_weight,
            granularity_balance_weight,
            granularity_entropy_weight,
            granularity_mix_shrinkage_weight,
            granularity_prior_kl_weight,
            router_route_budget_weight,
            router_load_balance_weight,
            early_stop_strategy=early_stop_strategy,
            early_stop_patience=early_stop_patience,
            early_stop_min_epochs=early_stop_min_epochs,
            early_stop_ema_decay=early_stop_ema_decay,
            early_stop_min_delta=early_stop_min_delta,
            early_stop_best_score=early_stop_monitor.best_score,
            early_stopped=early_stopped,
            stopped_epoch=stopped_epoch,
        )

    test_metrics, test_details = _evaluate_patch(
        model,
        test_feature_loader,
        classes,
        device,
        "Evaluate best patch test",
        sample_subject_ids=test_subject_ids,
    )

    if artifact_dir is not None:
        granularity_labels = tuple(
            "/".join(str(length) for length in regime)
            for regime in granularity_bank
        )
        summaries = {
            "validation": _save_patch_diagnostics(
                artifact_dir,
                "validation",
                val_details,
                granularity_labels,
            ),
            "test": _save_patch_diagnostics(
                artifact_dir,
                "test",
                test_details,
                granularity_labels,
            ),
        }
        _atomic_json_dump(
            Path(artifact_dir) / "patch_atgs_summary.json",
            {
                "schema_version": 5,
                "architecture": PATCH_MINDTS_ARCHITECTURE,
                "splits": summaries,
            },
        )
        _atomic_json_dump(
            Path(artifact_dir) / "patch_mindts_training_history.json",
            {
                "schema_version": 5,
                "architecture": PATCH_MINDTS_ARCHITECTURE,
                "selected_epoch": best_epoch,
                "validation_macro_f1": float(val_metrics["macro_f1"]),
                "validation_subject_macro_f1": (
                    float(val_metrics["subject_macro_f1"])
                    if "subject_macro_f1" in val_metrics
                    else None
                ),
                "checkpoint_selection_requested": checkpoint_metric,
                "checkpoint_selection_effective": checkpoint_metric_effective,
                "checkpoint_selection_key": list(best_selection_key),
                "early_stopping": {
                    "strategy": early_stop_strategy,
                    "patience": int(early_stop_patience),
                    "min_epochs": int(early_stop_min_epochs),
                    "ema_decay": float(early_stop_ema_decay),
                    "min_delta": float(early_stop_min_delta),
                    "best_monitor_score": float(
                        early_stop_monitor.best_score
                    ),
                    "early_stopped": bool(early_stopped),
                    "stopped_epoch": (
                        int(stopped_epoch)
                        if stopped_epoch is not None
                        else None
                    ),
                    "completed_epochs": int(len(training_history)),
                    "checkpoint_selection_uses_raw_metric": True,
                },
                "epochs": training_history,
            },
        )

    return val_metrics, test_metrics, train_indices, val_indices


__all__ = [
    "PATCH_FEATURE_CACHE_SCHEMA_VERSION",
    "PATCH_FEATURE_ARCHITECTURE",
    "PATCH_MINDTS_ARCHITECTURE",
    "PATCH_TAIL_POLICY",
    "PATCH_POOLING_POLICY",
    "TemporalPatchBatch",
    "make_temporal_patches",
    "ChannelAttentionPool",
    "PatchGranularityCrossAttention",
    "masked_granularity_regularization_terms",
    "masked_v5_router_regularization_terms",
    "MaskedIntraSampleInfoNCE",
    "compute_line_query_center",
    "valid_fraction_weighted_pool",
    "PatchMindTSFusionModule",
    "extract_patch_feature_batch",
    "save_patch_feature_cache",
    "load_patch_feature_cache",
    "train_patch_mindts_classifier",
]
