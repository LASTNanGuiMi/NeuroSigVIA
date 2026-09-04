"""TimeMosaic-style adaptive Activity Graph generation.

This module adapts only the categorical region gate from TimeMosaic:

* upstream repository: ``https://github.com/BenchCouncil/TimeMosaic``;
* locked upstream commit: ``214423b7f0b4653d04620814380a9301580285cc``;
* reference component: ``models/TimeMosaic.py:AdaptivePatchEmbedding``.

The adaptation boundary is deliberate.  The ``16 -> 64 -> 3`` gate and its
hard straight-through Gumbel selection follow the upstream mechanism.  The
three choices here are NeuroSigViT Activity Graph granularities ``4/8/16``;
RMS/variation activity, channel propagation, pair-covering row order, image
normalization, and RGB rendering are project-specific.  This is therefore not
the TimeMosaic forecasting model and does not include its embedding, encoder,
head, or auxiliary objective.

Unlike the older feature-level selector, the decision is made from raw-signal
regions before graph construction.  During training, the forward value is a
true one-hot choice while gradients use the corresponding soft probabilities.
The selected numeric activity map is formed before cross-channel propagation,
so only one adaptive Activity Graph is rasterized per input sample.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .renderer import _inverse_occurrence_row_weights, _pair_covering_order


UPSTREAM_REPOSITORY = "https://github.com/BenchCouncil/TimeMosaic"
UPSTREAM_COMMIT = "214423b7f0b4653d04620814380a9301580285cc"
UPSTREAM_COMPONENT = "models/TimeMosaic.py:AdaptivePatchEmbedding"
ADAPTATION_VERSION = "timemosaic_adaptive_activity_graph_v1"
ADAPTATION_BOUNDARY = (
    "Only the 16-64-3 categorical gate and hard straight-through Gumbel "
    "selection are adapted from TimeMosaic. Activity statistics, graph "
    "propagation, pair-cover ordering, normalization, and rendering are "
    "NeuroSigViT-specific."
)

PathLike = Union[str, os.PathLike]
CheckpointLike = Union[PathLike, Mapping[str, Any]]


def _load_checkpoint_object(checkpoint: CheckpointLike) -> Mapping[str, Any]:
    """Load a trusted gate checkpoint or return an in-memory mapping."""
    if isinstance(checkpoint, Mapping):
        return checkpoint
    if not isinstance(checkpoint, (str, os.PathLike)):
        raise TypeError("gate checkpoint must be a path or a mapping")

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Gate checkpoint does not exist: {checkpoint_path}")
    try:
        loaded = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:  # PyTorch releases before ``weights_only`` was added.
        loaded = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise TypeError("gate checkpoint must contain a mapping/state dictionary")
    return loaded


def _find_state_mapping(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Find the state dictionary inside common checkpoint containers."""
    state: Mapping[str, Any] = checkpoint
    container_keys = (
        "gate_state_dict",
        "selector_state_dict",
        "model_state_dict",
        "state_dict",
        "model",
    )
    for _ in range(3):
        nested = next(
            (
                state[key]
                for key in container_keys
                if key in state and isinstance(state[key], Mapping)
            ),
            None,
        )
        if nested is None:
            break
        state = nested
    return state


class TimeMosaicRegionGate(nn.Module):
    """Choose one of ``4/8/16`` for every channel-wise 16-sample region."""

    region_length = 16
    granularities = (4, 8, 16)
    upstream_repository = UPSTREAM_REPOSITORY
    upstream_commit = UPSTREAM_COMMIT
    upstream_component = UPSTREAM_COMPONENT
    adaptation_version = ADAPTATION_VERSION

    def __init__(
        self,
        temperature: float = 0.5,
        checkpoint: Optional[CheckpointLike] = None,
        freeze: bool = False,
        strict_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        self.region_cls = nn.Sequential(
            nn.Linear(self.region_length, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.granularities)),
        )
        self.gate_frozen = False
        self.loaded_checkpoint: Optional[str] = None

        if checkpoint is not None:
            self.load_gate_checkpoint(checkpoint, strict=strict_checkpoint)
        self.freeze_gate(freeze)

    @staticmethod
    def provenance() -> dict[str, Any]:
        """Return serializable source and adaptation metadata."""
        return {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_component": UPSTREAM_COMPONENT,
            "adaptation_version": ADAPTATION_VERSION,
            "adaptation_boundary": ADAPTATION_BOUNDARY,
            "region_length": 16,
            "granularities": [4, 8, 16],
            "training_selection": "hard_straight_through_gumbel_softmax",
            "evaluation_selection": "argmax_one_hot",
        }

    def load_gate_checkpoint(
        self,
        checkpoint: CheckpointLike,
        strict: bool = True,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load only ``region_cls`` weights from a gate or full checkpoint.

        Accepted keys may be local (``region_cls.0.weight``), bare sequential
        keys (``0.weight``), or prefixed by wrappers such as ``module.gate.``.
        Unrelated full-model keys are ignored; ``strict`` still requires all
        four gate tensors and validates their shapes.
        """
        checkpoint_object = _load_checkpoint_object(checkpoint)
        source_state = _find_state_mapping(checkpoint_object)
        target_keys = tuple(self.state_dict().keys())
        selected: dict[str, torch.Tensor] = {}

        for source_key, value in source_state.items():
            if not isinstance(source_key, str) or not torch.is_tensor(value):
                continue
            normalized_key = source_key
            while normalized_key.startswith("module."):
                normalized_key = normalized_key[len("module.") :]

            matches = [
                target_key
                for target_key in target_keys
                if normalized_key == target_key
                or normalized_key.endswith("." + target_key)
                or normalized_key == target_key.removeprefix("region_cls.")
            ]
            if not matches:
                continue
            target_key = matches[0]
            if target_key in selected:
                raise ValueError(
                    f"Gate checkpoint maps more than one tensor to {target_key!r}."
                )
            selected[target_key] = value

        if not selected:
            raise KeyError(
                "No TimeMosaic region_cls tensors were found in the gate checkpoint."
            )
        incompatible = self.load_state_dict(selected, strict=strict)
        self.loaded_checkpoint = (
            str(Path(checkpoint).expanduser())
            if isinstance(checkpoint, (str, os.PathLike))
            else "<in-memory mapping>"
        )
        return incompatible

    def freeze_gate(self, frozen: bool = True) -> "TimeMosaicRegionGate":
        """Enable or disable gradient updates of the gate parameters.

        A frozen gate uses deterministic argmax routing even while the enclosing
        classifier is training.  This makes a supplied pretrained gate a stable
        feature generator rather than a source of untrainable Gumbel noise.
        """
        self.gate_frozen = bool(frozen)
        for parameter in self.region_cls.parameters():
            parameter.requires_grad_(not self.gate_frozen)
        return self

    def forward(
        self,
        regions: torch.Tensor,
        region_valid_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        """Route ``[B,C,R,16]`` regions and return live diagnostic tensors."""
        if not torch.is_tensor(regions) or regions.ndim != 4:
            raise ValueError("regions must have shape [B,C,R,16]")
        if regions.shape[-1] != self.region_length or min(regions.shape[:3]) < 1:
            raise ValueError("regions must have non-empty [B,C,R] axes and width 16")
        if not regions.is_floating_point():
            raise ValueError("regions must be floating point")

        expected_mask_shape = regions.shape[:3]
        if region_valid_mask is None:
            valid = torch.ones(
                expected_mask_shape,
                dtype=torch.bool,
                device=regions.device,
            )
        else:
            if (
                not torch.is_tensor(region_valid_mask)
                or tuple(region_valid_mask.shape) != tuple(expected_mask_shape)
            ):
                raise ValueError("region_valid_mask must have shape [B,C,R]")
            valid = region_valid_mask.to(device=regions.device, dtype=torch.bool)

        gate_parameter = next(self.region_cls.parameters())
        if gate_parameter.device != regions.device:
            raise RuntimeError(
                "regions and TimeMosaicRegionGate must be on the same device"
            )
        clean_logits = self.region_cls(regions.to(dtype=gate_parameter.dtype))
        clean_probs = F.softmax(clean_logits, dim=-1)

        if self.training and not self.gate_frozen:
            gumbel_noise = -torch.empty_like(clean_logits).exponential_(
                generator=generator
            ).log()
            route_logits = clean_logits + gumbel_noise
            soft_weights = F.softmax(route_logits / self.temperature, dim=-1)
            indices = soft_weights.argmax(dim=-1)
            hard_weights = F.one_hot(
                indices,
                num_classes=len(self.granularities),
            ).to(dtype=soft_weights.dtype)
            weights = hard_weights - soft_weights.detach() + soft_weights
        else:
            route_logits = clean_logits
            indices = clean_logits.argmax(dim=-1)
            hard_weights = F.one_hot(
                indices,
                num_classes=len(self.granularities),
            ).to(dtype=clean_logits.dtype)
            soft_weights = clean_probs
            weights = hard_weights

        valid_expanded = valid.unsqueeze(-1)
        return {
            "clean_logits": clean_logits.masked_fill(~valid_expanded, 0.0),
            "clean_probs": clean_probs.masked_fill(~valid_expanded, 0.0),
            "route_logits": route_logits.masked_fill(~valid_expanded, 0.0),
            "soft_weights": soft_weights.masked_fill(~valid_expanded, 0.0),
            "hard_weights": hard_weights.masked_fill(~valid_expanded, 0.0),
            "weights": weights.masked_fill(~valid_expanded, 0.0),
            "indices": indices.masked_fill(~valid, -1),
            "region_valid_mask": valid,
        }


class TimeMosaicAdaptiveActivityGraphRenderer(nn.Module):
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
        self.gate = TimeMosaicRegionGate(
            temperature=temperature,
            checkpoint=gate_checkpoint,
            freeze=freeze_gate,
            strict_checkpoint=strict_gate_checkpoint,
        )

    @staticmethod
    def provenance() -> dict[str, Any]:
        return TimeMosaicRegionGate.provenance()

    def load_gate_checkpoint(
        self,
        checkpoint: CheckpointLike,
        strict: bool = True,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        return self.gate.load_gate_checkpoint(checkpoint, strict=strict)

    def freeze_gate(
        self,
        frozen: bool = True,
    ) -> "TimeMosaicAdaptiveActivityGraphRenderer":
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

        fine_width = TimeMosaicAdaptiveActivityGraphRenderer.region_length // 4
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


# Concise aliases for downstream configuration files.
AdaptiveActivityGraphRenderer = TimeMosaicAdaptiveActivityGraphRenderer
TimeMosaicAdaptiveRenderer = TimeMosaicAdaptiveActivityGraphRenderer


__all__ = [
    "ADAPTATION_BOUNDARY",
    "ADAPTATION_VERSION",
    "UPSTREAM_COMMIT",
    "UPSTREAM_COMPONENT",
    "UPSTREAM_REPOSITORY",
    "AdaptiveActivityGraphRenderer",
    "TimeMosaicAdaptiveActivityGraphRenderer",
    "TimeMosaicAdaptiveRenderer",
    "TimeMosaicRegionGate",
]
