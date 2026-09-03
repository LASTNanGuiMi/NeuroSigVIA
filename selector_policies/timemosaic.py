"""TimeMosaic-style categorical selector for existing ActivityGraph tokens.

Official source: https://github.com/BenchCouncil/TimeMosaic
Locked commit: 214423b7f0b4653d04620814380a9301580285cc.
``models/TimeMosaic.py:59-144``, class ``AdaptivePatchEmbedding``:
  - lines 73-77: Linear(max_patch_len, 64), ReLU, Linear(64, K);
  - lines 102-108: logits, hard Gumbel-softmax tau=0.5 during training,
    argmax/one-hot during evaluation;
  - lines 127-133: select candidates by multiplying by straight-through weights.

Controlled selector-mechanism adaptation: input is the common projected line
query [B,N,F], not a raw per-channel max-patch-length region. One decision selects
among the existing 4/8/16 ActivityGraph candidates for an outer time window.
Images, cached CLIP/Mantis features, value representations, and subsequent fusion
are unchanged. No encoder, forecasting head, latent-energy image, or pretraining
is included. Keys are accepted solely to validate the shared policy interface.

Upstream ``exp/exp_TimeMosaic.py:149-168`` adds a separate 0.001 usage L1 loss.
This controlled comparison does NOT add it: the parent applies the common v5
classification/alignment and 0.005 budget + 0.005 load objectives to every policy.

Training expands the same straight-through construction as
F.gumbel_softmax(logits, tau=0.5, hard=True), using ONE -log(Exp(1)) noise draw.
This exposes the actual perturbed logits for diagnostics without a second draw.
``route_logits`` are clean logits plus Gumbel noise, before division by tau.
``noise_std`` is pi/sqrt(6) for the standard Gumbel distribution in training,
zero in evaluation or an all-masked batch; it is not a Gaussian noise parameter.
The parent may pass a dedicated torch.Generator so selector noise does not
advance the shared classifier/dropout or data-loader random-number stream.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeMosaicSelector(nn.Module):
    """Trainable MLP plus single-scale straight-through categorical selection."""

    temperature = 0.5
    upstream_commit = "214423b7f0b4653d04620814380a9301580285cc"
    adaptation_version = "timemosaic_existing_graph_selector_v1"

    def __init__(self, feature_dim: int, num_candidates: int = 3):
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer")
        if isinstance(num_candidates, bool) or not isinstance(num_candidates, int) or num_candidates < 2:
            raise ValueError("num_candidates must be an integer >= 2")
        self.feature_dim = feature_dim
        self.num_candidates = num_candidates
        self.region_cls = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_candidates),
        )

    def forward(self, query, keys, patch_mask=None, valid_fraction=None, generator=None):
        if not torch.is_tensor(query) or query.ndim != 3:
            raise ValueError("query must have shape [B,N,F]")
        if min(query.shape[:2]) < 1 or query.shape[-1] != self.feature_dim:
            raise ValueError("query requires nonempty batch/time axes and configured feature_dim")
        expected_keys = (*query.shape[:2], self.num_candidates, self.feature_dim)
        if not torch.is_tensor(keys) or tuple(keys.shape) != expected_keys:
            raise ValueError(f"keys must have shape {expected_keys}")
        if not query.is_floating_point():
            raise ValueError("query must be floating point")
        if patch_mask is None:
            valid = torch.ones(query.shape[:2], dtype=torch.bool, device=query.device)
        else:
            if not torch.is_tensor(patch_mask) or tuple(patch_mask.shape) != tuple(query.shape[:2]):
                raise ValueError("patch_mask must have shape [B,N]")
            valid = patch_mask.to(device=query.device, dtype=torch.bool)
        if valid_fraction is not None:
            if not torch.is_tensor(valid_fraction) or tuple(valid_fraction.shape) != tuple(query.shape[:2]):
                raise ValueError("valid_fraction must have shape [B,N]")
            # This local selector has no temporal pooling. Fractional coverage
            # is consumed by the parent's common losses, not multiplied into
            # normalized per-window categorical weights here.

        clean_scores = self.region_cls(query.to(dtype=self.region_cls[0].weight.dtype))
        pre_topk_weights = F.softmax(clean_scores, dim=-1)
        if self.training:
            noise = -torch.empty_like(clean_scores).exponential_(generator=generator).log()
            route_logits = clean_scores + noise
            soft = F.softmax(route_logits / self.temperature, dim=-1)
            indices = soft.argmax(dim=-1, keepdim=True)
            hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0)
            weights = hard - soft.detach() + soft
            noise_std = clean_scores.new_tensor(math.pi / math.sqrt(6.0)) * valid.any().to(clean_scores.dtype)
        else:
            route_logits = clean_scores
            indices = clean_scores.argmax(dim=-1, keepdim=True)
            hard = torch.zeros_like(clean_scores).scatter_(-1, indices, 1.0)
            weights = hard
            noise_std = clean_scores.new_zeros(())
        topk_mask = hard.to(torch.bool) & valid.unsqueeze(-1)
        return {
            "clean_scores": clean_scores.masked_fill(~valid.unsqueeze(-1), 0.0),
            "route_logits": route_logits.masked_fill(~valid.unsqueeze(-1), 0.0),
            "weights": weights.masked_fill(~valid.unsqueeze(-1), 0.0),
            "pre_topk_weights": pre_topk_weights.masked_fill(~valid.unsqueeze(-1), 0.0),
            "topk_mask": topk_mask,
            "topk_indices": indices.masked_fill(~valid.unsqueeze(-1), -1),
            "noise_std": noise_std,
        }
