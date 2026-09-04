"""Patch-level fusion for adaptively generated Activity Graphs.

This module is the feature-fusion half of the TimeMosaic-style pipeline.  Raw
signals are routed and rendered by
``TimeMosaicAdaptiveActivityGraphRenderer`` before they reach this module.
The resulting Activity Graph keeps a small spatial token grid so a pooled
line-plot token can act as a genuine query over multiple graph keys/values.

The final temporal/visual classifier intentionally remains the compact
``concat -> MLP`` path used by the controlled NeuroSigViT protocol.  The
cross-attention in this file only fuses the two visual views.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.line_graph_cross_attention import LineGraphCrossAttention
from src.medformer_graph.timemosaic_adaptive import (
    ADAPTATION_VERSION,
    TimeMosaicRegionGate,
)
from src.patch_mindts import (
    ChannelAttentionPool,
    MaskedIntraSampleInfoNCE,
    _MLPHead,
    valid_fraction_weighted_pool,
)


OPENCLIP_SPATIAL_GRID_SIZE = 4
OPENCLIP_SPATIAL_TOKEN_COUNT = OPENCLIP_SPATIAL_GRID_SIZE**2
TIMEMOSAIC_PATCH_PIPELINE_VERSION = "timemosaic_patch_pipeline_v1"


def compress_openclip_spatial_tokens(
    hidden_tokens: torch.Tensor,
    output_grid_size: int = OPENCLIP_SPATIAL_GRID_SIZE,
) -> torch.Tensor:
    """Compress OpenCLIP patch tokens to a fixed spatial grid.

    ``hidden_tokens`` must use the OpenCLIP layout
    ``[..., CLS + H*W, D]``.  The leading CLS token is discarded, the square
    patch grid is reconstructed, and adaptive average pooling produces
    ``output_grid_size**2`` spatial tokens.  With the protocol default this is
    a 4 x 4 grid containing 16 graph K/V tokens.

    The function accepts arbitrary leading dimensions.  For example,
    ``[B*N, 1+H*W, D]`` becomes ``[B*N, 16, D]`` and
    ``[B, N, 1+H*W, D]`` becomes ``[B, N, 16, D]``.
    """

    if not torch.is_tensor(hidden_tokens):
        raise TypeError("hidden_tokens must be a torch.Tensor")
    if hidden_tokens.ndim < 3:
        raise ValueError(
            "hidden_tokens must have shape [..., CLS+patches, D], got "
            f"{tuple(hidden_tokens.shape)}"
        )
    if not hidden_tokens.is_floating_point():
        raise TypeError(
            f"hidden_tokens must use a floating dtype, got {hidden_tokens.dtype}"
        )
    if (
        isinstance(output_grid_size, bool)
        or not isinstance(output_grid_size, int)
        or output_grid_size < 1
    ):
        raise ValueError(
            "output_grid_size must be a positive integer, got "
            f"{output_grid_size}"
        )

    sequence_length = hidden_tokens.shape[-2]
    embedding_dim = hidden_tokens.shape[-1]
    spatial_token_count = sequence_length - 1
    source_grid_size = math.isqrt(spatial_token_count)
    if spatial_token_count < 1 or source_grid_size**2 != spatial_token_count:
        raise ValueError(
            "after removing CLS, OpenCLIP tokens must form a square grid; "
            f"got sequence_length={sequence_length}, which leaves "
            f"{spatial_token_count} spatial tokens"
        )
    if source_grid_size < output_grid_size:
        raise ValueError(
            "output_grid_size cannot exceed the OpenCLIP source grid when "
            f"compressing tokens, got source={source_grid_size} and "
            f"output={output_grid_size}"
        )

    leading_shape = hidden_tokens.shape[:-2]
    spatial = hidden_tokens[..., 1:, :].reshape(
        -1,
        source_grid_size,
        source_grid_size,
        embedding_dim,
    )
    spatial = spatial.permute(0, 3, 1, 2)
    compressed = F.adaptive_avg_pool2d(
        spatial,
        output_size=(output_grid_size, output_grid_size),
    )
    compressed = compressed.flatten(start_dim=2).transpose(1, 2)
    return compressed.reshape(
        *leading_shape,
        output_grid_size**2,
        embedding_dim,
    )


class TimeMosaicPatchFusionModule(nn.Module):
    """Fuse one adaptive Activity Graph with line and Mantis features.

    The expected inputs are already encoded patch-level features:

    * pooled line-plot tokens ``[B, N, Dv]``;
    * Activity Graph spatial tokens ``[B, N, P, Dv]``, with ``P >= 2``;
    * per-channel Mantis tokens ``[B, N, C, Dt]``.

    The line token is the query and graph spatial tokens are keys/values.  The
    resulting visual token is aligned with the channel-pooled Mantis token by
    symmetric within-sample InfoNCE.  Classification uses a separate
    ``concat -> MLP`` representation followed by valid-duration pooling.
    """

    def __init__(
        self,
        visual_dim: int,
        temporal_dim: int,
        num_channels: int,
        num_classes: int,
        fusion_dim: int = 512,
        fusion_heads: int = 4,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 512,
        classifier_num_layers: int = 2,
        channel_hidden_dim: int = 64,
        alignment_dim: int = 256,
        alignment_temperature: float = 0.1,
        cross_attention_ffn_hidden_dim: int | None = None,
        cross_attention_bias: bool = True,
    ) -> None:
        super().__init__()

        visual_dim = self._positive_int("visual_dim", visual_dim)
        temporal_dim = self._positive_int("temporal_dim", temporal_dim)
        num_channels = self._positive_int("num_channels", num_channels)
        num_classes = self._positive_int("num_classes", num_classes)
        fusion_dim = self._positive_int("fusion_dim", fusion_dim)
        fusion_heads = self._positive_int("fusion_heads", fusion_heads)
        classifier_hidden_dim = self._positive_int(
            "classifier_hidden_dim", classifier_hidden_dim
        )
        classifier_num_layers = self._positive_int(
            "classifier_num_layers", classifier_num_layers
        )
        channel_hidden_dim = self._positive_int(
            "channel_hidden_dim", channel_hidden_dim
        )
        alignment_dim = self._positive_int("alignment_dim", alignment_dim)
        if cross_attention_ffn_hidden_dim is None:
            cross_attention_ffn_hidden_dim = 4 * fusion_dim
        cross_attention_ffn_hidden_dim = self._positive_int(
            "cross_attention_ffn_hidden_dim",
            cross_attention_ffn_hidden_dim,
        )
        dropout = float(dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be finite and in [0, 1), got {dropout}")
        alignment_temperature = float(alignment_temperature)
        if (
            not math.isfinite(alignment_temperature)
            or alignment_temperature <= 0.0
        ):
            raise ValueError(
                "alignment_temperature must be positive and finite, got "
                f"{alignment_temperature}"
            )
        if not isinstance(cross_attention_bias, bool):
            raise TypeError(
                "cross_attention_bias must be bool, got "
                f"{type(cross_attention_bias).__name__}"
            )

        self.visual_dim = visual_dim
        self.temporal_dim = temporal_dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.fusion_dim = fusion_dim

        self.channel_pool = ChannelAttentionPool(
            input_dim=temporal_dim,
            output_dim=fusion_dim,
            num_channels=num_channels,
            hidden_dim=channel_hidden_dim,
        )
        self.visual_cross_attention = LineGraphCrossAttention(
            line_dim=visual_dim,
            graph_dim=visual_dim,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads,
            ffn_hidden_dim=cross_attention_ffn_hidden_dim,
            dropout=dropout,
            bias=cross_attention_bias,
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

        self.constructor_configuration: dict[str, Any] = {
            "visual_dim": visual_dim,
            "temporal_dim": temporal_dim,
            "num_channels": num_channels,
            "num_classes": num_classes,
            "fusion_dim": fusion_dim,
            "fusion_heads": fusion_heads,
            "dropout": dropout,
            "classifier_hidden_dim": classifier_hidden_dim,
            "classifier_num_layers": classifier_num_layers,
            "channel_hidden_dim": channel_hidden_dim,
            "alignment_dim": alignment_dim,
            "alignment_temperature": alignment_temperature,
            "cross_attention_ffn_hidden_dim": (
                cross_attention_ffn_hidden_dim
            ),
            "cross_attention_bias": cross_attention_bias,
        }
        self.configuration: dict[str, Any] = {
            **self.constructor_configuration,
            "architecture": TIMEMOSAIC_PATCH_PIPELINE_VERSION,
            "activity_graph_generation": ADAPTATION_VERSION,
            "activity_graph_provenance": TimeMosaicRegionGate.provenance(),
            "visual_fusion": (
                "pooled_line_query_activity_graph_spatial_key_value_"
                "cross_attention_v1"
            ),
            "visual_token_count": OPENCLIP_SPATIAL_TOKEN_COUNT,
            "temporal_pooling": "trainable_mantis_channel_attention",
            "alignment": "symmetric_intra_sample_patch_infonce",
            "temporal_visual_fusion": "concat_mlp",
            "sample_pooling": "valid_fraction_weighted_mean",
        }

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer, got {type(value).__name__}"
            )
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return int(value)

    def get_config(self) -> dict[str, Any]:
        """Return an independent JSON-serializable constructor dictionary."""

        return dict(self.constructor_configuration)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
    ) -> "TimeMosaicPatchFusionModule":
        """Reconstruct this module from :meth:`get_config` output."""

        return cls(**dict(config))

    def _validate_inputs(
        self,
        line_tokens: torch.Tensor,
        graph_spatial_tokens: torch.Tensor,
        mantis_channel_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        valid_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        named_tokens = {
            "line_tokens": (line_tokens, 3),
            "graph_spatial_tokens": (graph_spatial_tokens, 4),
            "mantis_channel_tokens": (mantis_channel_tokens, 4),
        }
        for name, (tokens, expected_ndim) in named_tokens.items():
            if not torch.is_tensor(tokens):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tokens.ndim != expected_ndim:
                raise ValueError(
                    f"{name} must have {expected_ndim} dimensions, got "
                    f"shape {tuple(tokens.shape)}"
                )
            if not tokens.is_floating_point():
                raise TypeError(f"{name} must use a floating dtype")

        batch_patches = tuple(line_tokens.shape[:2])
        if tuple(graph_spatial_tokens.shape[:2]) != batch_patches:
            raise ValueError(
                "graph_spatial_tokens must share line batch/patch axes"
            )
        if tuple(mantis_channel_tokens.shape[:2]) != batch_patches:
            raise ValueError(
                "mantis_channel_tokens must share line batch/patch axes"
            )
        if line_tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                f"expected line visual_dim={self.visual_dim}, got "
                f"{line_tokens.shape[-1]}"
            )
        if graph_spatial_tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                f"expected graph visual_dim={self.visual_dim}, got "
                f"{graph_spatial_tokens.shape[-1]}"
            )
        if mantis_channel_tokens.shape[2:] != (
            self.num_channels,
            self.temporal_dim,
        ):
            raise ValueError(
                "expected Mantis [C,Dt]="
                f"[{self.num_channels},{self.temporal_dim}], got "
                f"{tuple(mantis_channel_tokens.shape[2:])}"
            )

        devices = {
            line_tokens.device,
            graph_spatial_tokens.device,
            mantis_channel_tokens.device,
        }
        if len(devices) != 1:
            raise ValueError("line, graph, and Mantis tokens must share a device")
        dtypes = {
            line_tokens.dtype,
            graph_spatial_tokens.dtype,
            mantis_channel_tokens.dtype,
        }
        if len(dtypes) != 1:
            raise ValueError("line, graph, and Mantis tokens must share a dtype")

        if not torch.is_tensor(patch_mask):
            raise TypeError("patch_mask must be a torch.Tensor")
        if not torch.is_tensor(valid_fraction):
            raise TypeError("valid_fraction must be a torch.Tensor")
        if tuple(patch_mask.shape) != batch_patches:
            raise ValueError(
                f"patch_mask must have shape {batch_patches}, got "
                f"{tuple(patch_mask.shape)}"
            )
        if tuple(valid_fraction.shape) != batch_patches:
            raise ValueError(
                f"valid_fraction must have shape {batch_patches}, got "
                f"{tuple(valid_fraction.shape)}"
            )

        device = line_tokens.device
        mask = patch_mask.to(device=device, dtype=torch.bool)
        fractions = valid_fraction.to(device=device, dtype=line_tokens.dtype)
        if not torch.isfinite(fractions).all():
            raise ValueError("valid_fraction must be finite")
        if bool(((fractions < 0.0) | (fractions > 1.0)).any()):
            raise ValueError("valid_fraction entries must lie in [0, 1]")
        if not torch.equal(fractions > 0.0, mask):
            raise ValueError(
                "positive valid_fraction entries must match patch_mask"
            )
        if bool((mask.sum(dim=1) == 0).any()):
            raise ValueError("every sample must contain at least one valid patch")
        return mask, fractions

    def forward(
        self,
        line_tokens: torch.Tensor,
        graph_spatial_tokens: torch.Tensor,
        mantis_channel_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        valid_fraction: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        """Return sample logits and patch-level fusion diagnostics."""

        mask, fractions = self._validate_inputs(
            line_tokens,
            graph_spatial_tokens,
            mantis_channel_tokens,
            patch_mask,
            valid_fraction,
        )
        temporal_tokens, channel_weights = self.channel_pool(
            mantis_channel_tokens,
            patch_mask=mask,
        )

        attention_weights = None
        if return_attention_weights:
            visual_tokens, attention_weights = self.visual_cross_attention(
                line_tokens,
                graph_spatial_tokens,
                patch_mask=mask,
                return_attention_weights=True,
            )
        else:
            visual_tokens = self.visual_cross_attention(
                line_tokens,
                graph_spatial_tokens,
                patch_mask=mask,
            )

        alignment_loss = self.alignment(
            temporal_tokens,
            visual_tokens,
            mask,
            valid_fraction=fractions,
        )
        patch_features = self.patch_fusion(
            torch.cat([temporal_tokens, visual_tokens], dim=-1)
        )
        patch_features = torch.where(
            mask.unsqueeze(-1),
            patch_features,
            torch.zeros_like(patch_features),
        )
        sample_features = valid_fraction_weighted_pool(
            patch_features,
            mask,
            fractions,
        )
        logits = self.classifier(sample_features)

        return logits, {
            "alignment_loss": alignment_loss,
            "temporal_tokens": temporal_tokens,
            "visual_tokens": visual_tokens,
            "patch_features": patch_features,
            "sample_features": sample_features,
            "channel_weights": channel_weights,
            "cross_attention_weights": attention_weights,
        }


__all__ = [
    "OPENCLIP_SPATIAL_GRID_SIZE",
    "OPENCLIP_SPATIAL_TOKEN_COUNT",
    "TIMEMOSAIC_PATCH_PIPELINE_VERSION",
    "TimeMosaicPatchFusionModule",
    "compress_openclip_spatial_tokens",
]
