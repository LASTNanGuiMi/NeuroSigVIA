"""Adaptive Activity Graph generation for NeuroSigVIA.

Select a temporal granularity for each raw signal region, combine candidate
activity maps, propagate activity across channels, and render one RGB graph.
The independent granularity gate lives in src/temporal_granularity.py. Source
attribution is recorded in src/provenance.py and SOURCE_NOTES.md.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.activity_graph import _inverse_occurrence_row_weights, _pair_covering_order
from src.temporal_granularity import (
    ADAPTATION_VERSION,
    UPSTREAM_COMMIT,
    CheckpointLike,
    AdaptiveGranularityGate,
)


class AdaptiveActivityGraphRenderer(nn.Module):
    """Render one adaptively generated Activity Graph from raw ``[B,C,T]``.

    Each channel is divided into 16-sample regions.  The gate selects patch
    length 4, 8, or 16 for every region.  Candidate RMS-plus-variation maps are
    aligned to the four positions of the 4-sample grid and combined with hard
    straight-through weights.  Channel affinity propagation happens only
    after this selection, followed by pair-covering row order, robust 1--99%
    normalization, bilinear image resizing, and neutral RGB replication.
    """

    region_length = 16
    granularities = (4, 8, 16)
    fine_granularity = 4
    upstream_commit = UPSTREAM_COMMIT
    adaptation_version = ADAPTATION_VERSION

    def __init__(
        self,
        img_size: int = 224,
        channel_mix: float = 0.35,
        temperature: float = 0.5,
        gate_checkpoint: Optional[CheckpointLike] = None,
        freeze_gate: bool = False,
        strict_gate_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(img_size, bool) or not isinstance(img_size, int) or img_size < 1:
            raise ValueError(f"img_size must be a positive integer, got {img_size}")
        if not math.isfinite(channel_mix) or not 0.0 <= channel_mix <= 1.0:
            raise ValueError(f"channel_mix must be in [0, 1], got {channel_mix}")
        self.img_size = int(img_size)
        self.channel_mix = float(channel_mix)
        self.gate = AdaptiveGranularityGate(
            temperature=temperature,
            checkpoint=gate_checkpoint,
            freeze=freeze_gate,
            strict_checkpoint=strict_gate_checkpoint,
        )

    @staticmethod
    def provenance() -> dict[str, Any]:
        return AdaptiveGranularityGate.provenance()

    def load_gate_checkpoint(
        self,
        checkpoint: CheckpointLike,
        strict: bool = True,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        return self.gate.load_gate_checkpoint(checkpoint, strict=strict)

    def freeze_gate(
        self,
        frozen: bool = True,
    ) -> "AdaptiveActivityGraphRenderer":
        self.gate.freeze_gate(frozen)
        return self

    @staticmethod
    def _coerce_valid_lengths(
        valid_lengths: Optional[torch.Tensor],
        batch_size: int,
        time_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        if valid_lengths is None:
            return torch.full(
                (batch_size,),
                time_steps,
                dtype=torch.long,
                device=device,
            )
        lengths = torch.as_tensor(valid_lengths, device=device)
        if lengths.ndim != 1 or lengths.shape[0] != batch_size:
            raise ValueError("valid_lengths must have shape [B]")
        if lengths.dtype == torch.bool or lengths.is_floating_point():
            raise ValueError("valid_lengths must use an integer dtype")
        lengths = lengths.to(dtype=torch.long)
        if bool(((lengths < 1) | (lengths > time_steps)).any()):
            raise ValueError(f"valid_lengths entries must lie in [1, {time_steps}]")
        return lengths

    @staticmethod
    def _normalize_channels(
        signals: torch.Tensor,
        sample_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Robustly normalize valid samples without using padded tail values."""
        valid = sample_valid_mask[:, None, :]
        finite_signals = torch.nan_to_num(
            signals,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        masked = finite_signals.masked_fill(~valid, float("nan"))
        center = torch.nanquantile(masked, 0.5, dim=-1, keepdim=True)
        deviations = finite_signals - center
        absolute_deviations = deviations.abs().masked_fill(~valid, float("nan"))
        mad = torch.nanquantile(
            absolute_deviations,
            0.5,
            dim=-1,
            keepdim=True,
        ) * 1.4826

        counts = valid.sum(dim=-1, keepdim=True).to(dtype=signals.dtype)
        mean = (finite_signals * valid).sum(dim=-1, keepdim=True) / counts
        mean_deviations = finite_signals - mean
        variance = (
            mean_deviations.square() * valid
        ).sum(dim=-1, keepdim=True) / counts
        std = variance.sqrt()
        scale = torch.where(mad > 1e-6, mad, std.clamp_min(1e-6))
        normalized = torch.nan_to_num(
            deviations / scale,
            nan=0.0,
            posinf=8.0,
            neginf=-8.0,
        ).clamp(-8.0, 8.0)
        return normalized.masked_fill(~valid, 0.0)

    @staticmethod
    def _candidate_activity(
        regions: torch.Tensor,
        region_sample_mask: torch.Tensor,
        granularity: int,
    ) -> torch.Tensor:
        """Compute one candidate and align it to the four-cell fine grid."""
        subpatch_count = regions.shape[-1] // granularity
        patches = regions.reshape(*regions.shape[:-1], subpatch_count, granularity)
        mask = region_sample_mask[:, None, :, :].expand(
            regions.shape[0],
            regions.shape[1],
            regions.shape[2],
            regions.shape[3],
        )
        patch_mask = mask.reshape(*mask.shape[:-1], subpatch_count, granularity)

        sample_counts = patch_mask.sum(dim=-1)
        amplitude = torch.sqrt(
            (patches.square() * patch_mask).sum(dim=-1)
            / sample_counts.clamp_min(1).to(dtype=patches.dtype)
            + 1e-8
        )
        if granularity > 1:
            adjacent_valid = patch_mask[..., 1:] & patch_mask[..., :-1]
            variation_counts = adjacent_valid.sum(dim=-1)
            variation = (
                patches.diff(dim=-1).abs() * adjacent_valid
            ).sum(dim=-1) / variation_counts.clamp_min(1).to(dtype=patches.dtype)
            variation = variation.masked_fill(variation_counts == 0, 0.0)
        else:  # Kept explicit if the candidate set is extended later.
            variation = torch.zeros_like(amplitude)
        activity = (amplitude + variation).masked_fill(sample_counts == 0, 0.0)

        fine_width = AdaptiveActivityGraphRenderer.region_length // 4
        if activity.shape[-1] == fine_width:
            return activity
        if activity.shape[-1] == 1:
            return activity.expand(*activity.shape[:-1], fine_width)
        flattened = activity.reshape(-1, 1, activity.shape[-1])
        aligned = F.interpolate(
            flattened,
            size=fine_width,
            mode="linear",
            align_corners=True,
        )
        return aligned.reshape(*activity.shape[:-1], fine_width)

    def _propagate_channels(
        self,
        selected_activity: torch.Tensor,
        fine_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = fine_valid_mask[:, None, :]
        counts = valid.sum(dim=-1, keepdim=True).to(selected_activity.dtype)
        means = (selected_activity * valid).sum(dim=-1, keepdim=True) / counts
        centered = (selected_activity - means) * valid
        norms = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        normalized = centered / norms.clamp_min(1e-8)
        affinity = torch.bmm(normalized, normalized.transpose(1, 2)).abs()
        affinity = torch.nan_to_num(affinity, nan=0.0, posinf=0.0, neginf=0.0)
        affinity = affinity + torch.eye(
            selected_activity.shape[1],
            dtype=selected_activity.dtype,
            device=selected_activity.device,
        ).unsqueeze(0)
        affinity = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        propagated = torch.bmm(affinity, selected_activity)
        mixed = (
            (1.0 - self.channel_mix) * selected_activity
            + self.channel_mix * propagated
        )
        return mixed.masked_fill(~valid, 0.0)

    def _render(self, activity: torch.Tensor) -> torch.Tensor:
        num_channels = activity.shape[1]
        order = torch.tensor(
            _pair_covering_order(num_channels),
            dtype=torch.long,
            device=activity.device,
        )
        row_weights = _inverse_occurrence_row_weights(
            order,
            num_channels,
            activity.dtype,
        )
        ordered = activity.index_select(1, order)
        source_span = ordered.amax(dim=(1, 2), keepdim=True) - ordered.amin(
            dim=(1, 2), keepdim=True
        )
        weighted = ordered * row_weights.view(1, -1, 1)
        # Repetition compensation must not manufacture structure for a
        # constant signal merely because some rows occur more often.
        ordered = torch.where(source_span <= 1e-8, ordered, weighted)

        flattened = ordered.flatten(start_dim=1)
        lower = torch.quantile(flattened, 0.01, dim=1).view(-1, 1, 1)
        upper = torch.quantile(flattened, 0.99, dim=1).view(-1, 1, 1)
        span = upper - lower
        normalized = ((ordered - lower) / span.clamp_min(1e-8)).clamp(0.0, 1.0)
        normalized = torch.where(
            span <= 1e-8,
            torch.full_like(normalized, 0.5),
            normalized,
        )

        image = F.interpolate(
            normalized.unsqueeze(1),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return image.expand(-1, 3, -1, -1).clamp(0.0, 1.0)

    def forward(
        self,
        signals: torch.Tensor,
        valid_lengths: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """Render one RGB graph, optionally returning gate diagnostics.

        Args:
            signals: Floating tensor with shape ``[B,C,T]``.
            valid_lengths: Optional integer tensor ``[B]`` for padded batches.
            return_diagnostics: When true, return ``(image, diagnostics)``.
            generator: Optional device-compatible generator for Gumbel noise.
        """
        if not torch.is_tensor(signals) or signals.ndim != 3:
            raise ValueError("signals must be a tensor with shape [B,C,T]")
        if min(signals.shape) < 1:
            raise ValueError("signals must have non-empty batch/channel/time axes")
        if not signals.is_floating_point():
            raise ValueError("signals must be floating point")

        output_dtype = signals.dtype
        working_dtype = torch.float64 if signals.dtype == torch.float64 else torch.float32
        working = signals.to(dtype=working_dtype)
        lengths = self._coerce_valid_lengths(
            valid_lengths,
            batch_size=working.shape[0],
            time_steps=working.shape[2],
            device=working.device,
        )
        time_axis = torch.arange(working.shape[2], device=working.device)
        sample_valid_mask = time_axis.unsqueeze(0) < lengths.unsqueeze(1)
        working = self._normalize_channels(working, sample_valid_mask)

        original_time_steps = working.shape[2]
        region_count = (original_time_steps + self.region_length - 1) // self.region_length
        padded_time = region_count * self.region_length
        if padded_time != working.shape[2]:
            working = F.pad(working, (0, padded_time - working.shape[2]))
            sample_valid_mask = F.pad(
                sample_valid_mask,
                (0, padded_time - sample_valid_mask.shape[1]),
                value=False,
            )

        regions = working.reshape(
            working.shape[0],
            working.shape[1],
            region_count,
            self.region_length,
        )
        region_sample_mask = sample_valid_mask.reshape(
            working.shape[0],
            region_count,
            self.region_length,
        )
        region_valid = region_sample_mask.any(dim=-1)
        gate_diagnostics = self.gate(
            regions,
            region_valid_mask=region_valid[:, None, :].expand(
                -1, working.shape[1], -1
            ),
            generator=generator,
        )

        candidates = torch.stack(
            [
                self._candidate_activity(regions, region_sample_mask, granularity)
                for granularity in self.granularities
            ],
            dim=3,
        )
        selection_weights = gate_diagnostics["weights"].to(dtype=candidates.dtype)
        selected = torch.einsum("bcrk,bcrkf->bcrf", selection_weights, candidates)

        fine_sample_mask = region_sample_mask.reshape(
            working.shape[0],
            region_count,
            self.region_length // self.fine_granularity,
            self.fine_granularity,
        ).any(dim=-1)
        fine_valid_mask = fine_sample_mask.reshape(working.shape[0], -1)
        selected = selected.reshape(working.shape[0], working.shape[1], -1)
        fine_width = (
            original_time_steps + self.fine_granularity - 1
        ) // self.fine_granularity
        fine_valid_mask = fine_valid_mask[:, :fine_width]
        selected = selected[:, :, :fine_width]
        selected = selected.masked_fill(~fine_valid_mask[:, None, :], 0.0)
        propagated = self._propagate_channels(selected, fine_valid_mask)
        image = self._render(propagated).to(dtype=output_dtype)

        if not return_diagnostics:
            return image
        diagnostics = dict(gate_diagnostics)
        diagnostics.update(
            {
                "granularities": torch.tensor(
                    self.granularities,
                    dtype=torch.long,
                    device=signals.device,
                ),
                "fine_valid_mask": fine_valid_mask,
            }
        )
        return image, diagnostics


__all__ = ["AdaptiveActivityGraphRenderer"]
