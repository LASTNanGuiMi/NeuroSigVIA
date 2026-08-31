"""Trainable Adaptive Temporal Granularity Selection (ATGS).

The renderer and frozen vision backbone produce ``K`` candidate embeddings per
sample.  ATGS scores those candidates with a shared MLP and returns a weighted
sample representation.  Keeping this selector after frozen feature extraction
allows it to train with the classifier even when candidate embeddings are read
from the feature cache.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AdaptiveTemporalGranularitySelector(nn.Module):
    """Select a temporal-granularity candidate for each cached sample."""

    def __init__(
        self,
        feature_dim,
        num_granularities,
        hidden_dim=64,
        temperature=1.0,
        base_index=0,
        base_prior=0.9,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}.")
        if num_granularities < 2:
            raise ValueError(
                "AdaptiveTemporalGranularitySelector requires at least two "
                "candidates."
            )
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if not 0 <= base_index < num_granularities:
            raise ValueError(
                f"base_index {base_index} is invalid for "
                f"{num_granularities} candidates."
            )
        if not math.isfinite(base_prior) or not 0.0 < base_prior < 1.0:
            raise ValueError(f"base_prior must be in (0, 1), got {base_prior}.")

        self.feature_dim = int(feature_dim)
        self.num_granularities = int(num_granularities)
        self.temperature = float(temperature)
        self.base_index = int(base_index)
        self.base_prior = float(base_prior)

        self.normalization = nn.LayerNorm(self.feature_dim)
        self.scorer = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Begin near the legacy regime while retaining weak sample dependence
        # from the first optimization step.
        nn.init.normal_(self.scorer[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.scorer[-1].bias)

        other_prior = (1.0 - base_prior) / (self.num_granularities - 1)
        initial_prior = torch.full(
            (self.num_granularities,),
            other_prior,
            dtype=torch.float32,
        )
        initial_prior[self.base_index] = base_prior
        self.granularity_bias = nn.Parameter(
            self.temperature * initial_prior.log()
        )
        self.last_weights = None
        self.last_fallback_codes = None
        self.last_fallback_indices = None

    @staticmethod
    def _bad_sample_message(kind, valid_mask):
        bad_indices = torch.nonzero(~valid_mask, as_tuple=False).flatten()
        preview = bad_indices[:8].detach().cpu().tolist()
        suffix = "..." if bad_indices.numel() > len(preview) else ""
        return (
            f"Adaptive granularity received invalid {kind} for "
            f"{bad_indices.numel()} sample(s); batch indices={preview}{suffix}."
        )

    def forward(self, flat_embeddings):
        self.last_weights = None
        self.last_fallback_codes = None
        self.last_fallback_indices = None
        expected_dim = self.num_granularities * self.feature_dim
        if flat_embeddings.ndim != 2 or flat_embeddings.shape[1] != expected_dim:
            raise ValueError(
                "Adaptive granularity expects flattened [batch, K * dim] "
                f"features with width {expected_dim}, got "
                f"{tuple(flat_embeddings.shape)}."
            )

        candidates = flat_embeddings.reshape(
            flat_embeddings.shape[0],
            self.num_granularities,
            self.feature_dim,
        )
        finite_candidates = torch.isfinite(candidates).all(dim=2)
        safe_candidates = torch.nan_to_num(
            candidates,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        candidate_norms = safe_candidates.norm(dim=2)
        valid_candidates = (
            finite_candidates
            & torch.isfinite(candidate_norms)
            & (candidate_norms > 1e-12)
        )
        valid_inputs = valid_candidates.all(dim=1)
        if self.training and not bool(valid_inputs.all()):
            raise FloatingPointError(
                self._bad_sample_message("candidate embeddings", valid_inputs)
            )
        has_valid_candidate = valid_candidates.any(dim=1)
        if not bool(has_valid_candidate.all()):
            raise FloatingPointError(
                self._bad_sample_message(
                    "complete nonzero candidate embeddings", has_valid_candidate
                )
            )
        normalized_candidates = nn.functional.normalize(
            safe_candidates,
            dim=-1,
        )
        logits = self.scorer(
            self.normalization(normalized_candidates)
        ).squeeze(-1)
        logits = logits + self.granularity_bias
        finite_logits = torch.isfinite(logits).all(dim=1)
        if not bool(finite_logits.all()):
            raise FloatingPointError(
                self._bad_sample_message("selector logits", finite_logits)
            )
        safe_logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        weights = torch.softmax(safe_logits / self.temperature, dim=1)

        base_candidate = normalized_candidates[
            :, self.base_index : self.base_index + 1
        ]
        candidate_spread = (
            normalized_candidates - base_candidate
        ).abs().amax(dim=(1, 2))
        identical_candidates = candidate_spread <= 1e-10
        # Codes are persisted with diagnostics: 0=none, 1=invalid candidate
        # input, 2=reserved for legacy non-finite-logit fallback records, and
        # 3=indistinguishable candidates, 4=degenerate weighted mixture.
        # Non-finite logits now fail fast.
        fallback_codes = torch.zeros(
            candidates.shape[0],
            dtype=torch.int64,
            device=candidates.device,
        )
        fallback_codes = torch.where(
            ~valid_inputs,
            torch.ones_like(fallback_codes),
            fallback_codes,
        )
        fallback_codes = torch.where(
            valid_inputs & identical_candidates,
            torch.full_like(fallback_codes, 3),
            fallback_codes,
        )
        fallback = fallback_codes != 0
        first_valid_index = valid_candidates.to(torch.int64).argmax(dim=1)
        fallback_indices = torch.full_like(first_valid_index, self.base_index)
        fallback_indices = torch.where(
            (~valid_candidates[:, self.base_index]),
            first_valid_index,
            fallback_indices,
        )
        fallback_weights = torch.zeros_like(weights)
        fallback_weights.scatter_(1, fallback_indices.unsqueeze(1), 1.0)
        weights = torch.where(fallback.unsqueeze(1), fallback_weights, weights)

        selected_sum = torch.sum(
            weights.unsqueeze(-1) * normalized_candidates,
            dim=1,
        )
        valid_mixture = selected_sum.norm(dim=1) > 1e-12
        if self.training and not bool(valid_mixture.all()):
            raise FloatingPointError(
                self._bad_sample_message("weighted candidate mixtures", valid_mixture)
            )
        fallback_codes = torch.where(
            (~valid_mixture) & (fallback_codes == 0),
            torch.full_like(fallback_codes, 4),
            fallback_codes,
        )
        fallback = fallback_codes != 0
        fallback_weights = torch.zeros_like(weights)
        fallback_weights.scatter_(1, fallback_indices.unsqueeze(1), 1.0)
        weights = torch.where(fallback.unsqueeze(1), fallback_weights, weights)
        selected_sum = torch.sum(
            weights.unsqueeze(-1) * normalized_candidates,
            dim=1,
        )
        final_mixture_norms = selected_sum.norm(dim=1)
        if not bool(
            (torch.isfinite(final_mixture_norms) & (final_mixture_norms > 1e-12)).all()
        ):
            raise FloatingPointError(
                "Adaptive granularity fallback produced an invalid weighted "
                "candidate mixture."
            )
        selected = nn.functional.normalize(selected_sum, dim=-1)
        if not bool(torch.isfinite(weights).all()) or not bool(
            torch.isfinite(selected).all()
        ):
            raise FloatingPointError(
                "Adaptive granularity produced non-finite weights or output."
            )
        self.last_weights = weights.detach()
        self.last_fallback_codes = fallback_codes.detach()
        self.last_fallback_indices = torch.where(
            fallback,
            fallback_indices,
            torch.full_like(fallback_indices, -1),
        ).detach()
        return selected, weights


# Backward-compatible public name used by existing scripts and checkpoints.
AdaptiveGranularitySelector = AdaptiveTemporalGranularitySelector


__all__ = [
    "AdaptiveTemporalGranularitySelector",
    "AdaptiveGranularitySelector",
]
