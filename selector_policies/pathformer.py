"""Pathformer noisy Top-2 selector adapted to a shared projected query.

Official reference: decisionintelligence/pathformer, commit
ea85d82932215e171357da47b3bc82d502344758, layers/AMS.py lines 29-32 and 75-102:
https://github.com/decisionintelligence/pathformer/blob/ea85d82932215e171357da47b3bc82d502344758/layers/AMS.py

The actual upstream code constructs ``w_noise = nn.Linear(input_size,E)``
BEFORE ``w_gate = nn.Linear(input_size,E)``. Both have biases and use PyTorch's
default Linear initialization (weight and bias bounded by 1/sqrt(input_size));
the zero-initialized matrices above them in the source are commented out.
We preserve these layer definitions and construction order, not the full AMS
RNG history, which also constructed other modules before reaching those lines.

Scope: substitute only the selection mechanism over existing graph scales.
The shared projected line query [B,N,F] replaces the upstream raw temporal
vector produced by its learned node reduction. No season/trend decomposition,
start_linear node mapping, expert backbone, forecasting head, pretraining,
graph renderer, or new auxiliary loss is introduced. ``keys`` is accepted
for the common interface but does not influence these query-only gate scores.

The controlled comparison retains its shared v5 CE + 0.1 alignment + 0.005
route-budget + 0.005 load objective. The upstream normal-CDF load estimate
and additional 0.01 CV-squared importance/load auxiliary loss are deliberately
not added, so this is a Pathformer-style selector adaptation, not a full
Pathformer reproduction. ``pre_topk_weights`` is clean softmax for the common
v5 differentiable load proxy, an explicit interface adaptation.

No license declaration was present in the pinned upstream repository. This
file records provenance and makes no claim to grant an upstream license.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PathformerSelector(nn.Module):
    """Per-patch learned noisy Top-2 gates with deterministic evaluation.

    Invalid ``patch_mask`` entries return zero scores/probabilities, false
    supports and -1 indices. ``valid_fraction`` never changes route choices;
    it only weights the scalar training-noise diagnostic over valid patches.
    All-false masks return diagnostic zero. The caller owns masked pooling
    and the common v5 regularizers.
    """

    UPSTREAM_COMMIT = "ea85d82932215e171357da47b3bc82d502344758"
    ADAPTATION_VERSION = "pathformer_query_selector_v1"

    def __init__(self, feature_dim: int, num_candidates: int = 3):
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if isinstance(num_candidates, bool) or not isinstance(num_candidates, int) or num_candidates < 2:
            raise ValueError("num_candidates must be an integer >= 2 for Top-2")
        self.feature_dim = feature_dim
        self.num_candidates = num_candidates
        self.top_k = 2
        self.noise_epsilon = 1e-2
        # Preserve the active upstream layer definitions and their order.
        # Do not replace these default-initialized Linear layers with zeros.
        self.w_noise = nn.Linear(feature_dim, num_candidates)
        self.w_gate = nn.Linear(feature_dim, num_candidates)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        valid_fraction: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(query) or query.ndim != 3:
            raise ValueError("query must have shape [B,N,F]")
        if min(query.shape[:2]) < 1 or query.shape[-1] != self.feature_dim:
            raise ValueError("query must have nonempty B/N axes and configured feature_dim")
        expected_keys = (*query.shape[:2], self.num_candidates, self.feature_dim)
        if not torch.is_tensor(keys) or tuple(keys.shape) != expected_keys:
            raise ValueError(f"keys must have shape {expected_keys}")
        if not query.is_floating_point() or not torch.isfinite(query).all():
            raise ValueError("query must be finite floating point")
        if query.device != self.w_gate.weight.device or query.dtype != self.w_gate.weight.dtype:
            raise ValueError("query device/dtype must match selector parameters")

        valid = torch.ones(query.shape[:2], dtype=torch.bool, device=query.device)
        if patch_mask is not None:
            if patch_mask.shape != query.shape[:2]:
                raise ValueError("patch_mask must share query B/N axes")
            valid = patch_mask.to(device=query.device, dtype=torch.bool)
        diagnostic_importance = valid.to(dtype=query.dtype)
        if valid_fraction is not None:
            if valid_fraction.shape != query.shape[:2]:
                raise ValueError("valid_fraction must share query B/N axes")
            fraction = valid_fraction.to(device=query.device, dtype=query.dtype)
            if not torch.isfinite(fraction).all() or torch.any((fraction < 0) | (fraction > 1)):
                raise ValueError("valid_fraction must be finite and in [0,1]")
            diagnostic_importance = diagnostic_importance * fraction

        clean_scores = self.w_gate(query)
        pre_topk_weights = F.softmax(clean_scores, dim=-1)
        if self.training:
            noise = F.softplus(self.w_noise(query)) + self.noise_epsilon
            # The comparison host supplies an independent per-run generator
            # so gate noise does not advance shared classifier/dropout RNG.
            sampled_noise = torch.randn(
                clean_scores.shape, device=clean_scores.device,
                dtype=clean_scores.dtype, generator=generator,
            )
            route_logits = clean_scores + sampled_noise * noise
            noise_std = (noise.mean(dim=-1) * diagnostic_importance).sum()
            noise_std = noise_std / diagnostic_importance.sum().clamp_min(1e-8)
        else:
            route_logits = clean_scores
            noise_std = clean_scores.new_zeros(())

        # Match upstream top-(k+1), then retain the first k. Upstream needed
        # the extra threshold for its CDF load estimator; keeping this exact
        # order also preserves its tie handling while omitting that estimator.
        top_logits, top_indices = route_logits.topk(
            min(self.top_k + 1, self.num_candidates), dim=-1
        )
        topk_indices = top_indices[..., :self.top_k]
        selected_weights = F.softmax(top_logits[..., :self.top_k], dim=-1)
        weights = torch.zeros_like(route_logits).scatter(-1, topk_indices, selected_weights)
        topk_mask = torch.zeros_like(route_logits, dtype=torch.bool).scatter(-1, topk_indices, True)

        invalid = ~valid.unsqueeze(-1)
        return {
            "clean_scores": clean_scores.masked_fill(invalid, 0),
            "route_logits": route_logits.masked_fill(invalid, 0),
            "weights": weights.masked_fill(invalid, 0),
            "pre_topk_weights": pre_topk_weights.masked_fill(invalid, 0),
            "topk_mask": topk_mask & valid.unsqueeze(-1),
            "topk_indices": topk_indices.masked_fill(invalid, -1),
            "noise_std": noise_std.reshape(()),
        }

    def configuration(self) -> dict:
        """Serializable provenance for this controlled selector adaptation."""
        return {
            "method": self.ADAPTATION_VERSION,
            "upstream_repository": "https://github.com/decisionintelligence/pathformer",
            "upstream_commit": self.UPSTREAM_COMMIT,
            "upstream_component": "layers/AMS.py:noisy_top_k_gating",
            "input": "common_projected_query_only",
            "feature_dim": self.feature_dim,
            "num_candidates": self.num_candidates,
            "top_k": self.top_k,
            "temperature": 1.0,
            "gate": "Linear(F,K,bias=True)_default_initialization",
            "noise": "training_Gaussian_std_softplus(Linear(F,K))_plus_0.01",
            "noise_rng": "caller_supplied_independent_torch_Generator",
            "initialization_order": ["w_noise", "w_gate"],
            "noise_std_diagnostic": "valid_fraction_weighted_mean_learned_std_training_else_zero",
            "pre_topk_weights": "clean_softmax_common_v5_load_proxy",
            "loss": "shared_v5_CE_plus_0.1alignment_plus_0.005budget_plus_0.005load",
            "upstream_cdf_load_and_auxiliary_cv2_added": False,
            "renderer_experts_forecast_pretraining_added": False,
        }
