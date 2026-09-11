"""Patch-level fusion for adaptively generated Activity Graphs.

This module is the feature-fusion half of the adaptive-granularity pipeline.  Raw
signals are routed and rendered by
``AdaptiveActivityGraphRenderer`` before they reach this module.
The resulting Activity Graph keeps a small spatial token grid so a pooled
line-plot token can act as a genuine query over multiple graph keys/values.

The final temporal/visual classifier reuses NeuroSigVIA's existing
``concat_attn`` interaction: project the visual and temporal branches, apply
self-attention over the two branch tokens, flatten the attended tokens, and
pass that representation to the classifier MLP.  The Line-Q/Graph-KV
cross-attention in this file remains responsible only for the two visual views.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.line_graph_cross_attention import LineGraphCrossAttention
from src.temporal_granularity import (
    ADAPTATION_VERSION,
    AdaptiveGranularityGate,
)
from src.mlp_classifier import FusionModule
from src.patch_mindts import (
    ChannelAttentionPool,
    MaskedIntraSampleInfoNCE,
    _MLPHead,
    valid_fraction_weighted_pool,
)


OPENCLIP_SPATIAL_GRID_SIZE = 4
OPENCLIP_SPATIAL_TOKEN_COUNT = OPENCLIP_SPATIAL_GRID_SIZE**2
MULTIMODAL_FUSION_VERSION = "multimodal_fusion_concat_attn_v2"


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

    # 此处 D 是视觉 Transformer 的隐藏宽度，尚未应用 OpenCLIP 的 ln_post/proj 输出投影。
    # 例如 ViT-H/14 在 224 像素输入下为 [M,257,1280]；保留 16 个空间 token 后为 [M,16,1280]。
    # 具体网格和隐藏宽度取决于骨干，本函数通过 CLS 后 token 数的平方性进行核验。
    leading_shape = hidden_tokens.shape[:-2]
    spatial = hidden_tokens[..., 1:, :].reshape(
        -1,
        source_grid_size,
        source_grid_size,
        embedding_dim,
    )
    spatial = spatial.permute(0, 3, 1, 2)
    # [M,D,H,W] -> [M,D,4,4] -> [M,16,D]；任意前导批次轴会在返回前恢复。
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


class AdaptiveGranularityFusionModule(nn.Module):
    """Fuse one adaptive Activity Graph with line and Mantis features.

    The expected inputs are already encoded patch-level features:

    * pooled line-plot tokens ``[B, N, Dv]``;
    * Activity Graph spatial tokens ``[B, N, P, Dv]``, with ``P >= 2``;
    * per-channel Mantis tokens ``[B, N, C, Dt]``.

    The line token is the query and graph spatial tokens are keys/values.  The
    resulting visual token is aligned with the channel-pooled Mantis token by
    symmetric within-sample InfoNCE.  Classification uses the repository's
    existing ``concat_attn`` interaction followed by valid-duration pooling
    and an MLP head.
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

        # 将各外层 patch 的 C 个 Mantis 通道向量 [B,N,C,Dt] 汇聚成 [B,N,F]。
        # C 来自数据/缓存；TDBRAIN 为 33。Dt 是实际 Mantis 输出宽度，F=fusion_dim。
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
        # Reuse the exact concat_attn semantics of the legacy MLP path:
        # branch-specific projections -> two-token self-attention -> flatten.
        # The branch order matches the historical feature extractor, where
        # visual branches precede the Mantis branch.
        self.temporal_visual_fusion = FusionModule(
            branch_dims=[fusion_dim, fusion_dim],
            modal_interaction="concat_attn",
            fusion_dim=fusion_dim,
            fusion_heads=fusion_heads,
            branch_names=["cross_attention_visual", "mantis_temporal"],
        )
        # 两个分支 token 经 concat_attn 后展平为 2F；构造函数默认 F=512 时输入宽度为 1024。
        # 当前 scripts/NeuroSigVIA.sh 显式传 F=128，因此该启动脚本下的分类器输入宽度为 256。
        self.classifier = _MLPHead(
            input_dim=self.temporal_visual_fusion.output_dim,
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
            "architecture": MULTIMODAL_FUSION_VERSION,
            "activity_graph_generation": ADAPTATION_VERSION,
            "activity_graph_provenance": AdaptiveGranularityGate.provenance(),
            "visual_fusion": (
                "pooled_line_query_activity_graph_spatial_key_value_"
                "cross_attention_v1"
            ),
            "visual_token_count": OPENCLIP_SPATIAL_TOKEN_COUNT,
            "temporal_pooling": "trainable_mantis_channel_attention",
            "alignment": "symmetric_intra_sample_patch_infonce",
            "temporal_visual_fusion": "concat_attn",
            "temporal_visual_fusion_semantics": (
                "branch_projection_then_two_token_self_attention_then_flatten"
            ),
            "temporal_visual_branch_order": [
                "cross_attention_visual",
                "mantis_temporal",
            ],
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
    ) -> "AdaptiveGranularityFusionModule":
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

        # 输入轴含义：B=数据窗口数，N=每窗口的外层 patch 数，P=图像空间 token 数。
        # TDBRAIN 使用 256 点窗口、64/64 外层切分时 N=4，当前图像网格 P=16：
        # line [B,4,Dv]，graph [B,4,16,Dv]，Mantis [B,4,33,Dt]，mask/fractions [B,4]。
        # Dv、Dt 由特征缓存及编码器实际输出确定；不能把 OpenCLIP 隐藏宽度直接当作 Dv。
        # 当前启动脚本显式设置 F=128、fusion_heads=2；下文 shape 中 F 表示传入值。
        mask, fractions = self._validate_inputs(
            line_tokens,
            graph_spatial_tokens,
            mantis_channel_tokens,
            patch_mask,
            valid_fraction,
        )
        # 输出 temporal_tokens [B,N,F]、channel_weights [B,N,C]，权重在通道维归一化。
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

        # visual_tokens 同为 [B,N,F]；InfoNCE 在同一数据窗口内部构造 [N_valid,N_valid] 相似度，
        # 对齐相同 patch 的视觉/时序表示。返回标量损失，不在 batch 样本间构造负样本。
        alignment_loss = self.alignment(
            temporal_tokens,
            visual_tokens,
            mask,
            valid_fraction=fractions,
        )
        batch_size, patch_count, _ = visual_tokens.shape
        flat_mask = mask.reshape(-1)
        valid_indices = flat_mask.nonzero(as_tuple=False).squeeze(1)
        flat_visual = visual_tokens.reshape(batch_size * patch_count, -1)
        flat_temporal = temporal_tokens.reshape(batch_size * patch_count, -1)
        # 只取 M_valid 个有效 patch：两路 [M_valid,F] -> 双 token [M_valid,2,F] -> [M_valid,2F]。
        # 这里的自注意力沿两种分支交互；时间 patch 间的最终合并在 valid_fraction_weighted_pool 完成。
        valid_patch_features = self.temporal_visual_fusion(
            [
                flat_visual.index_select(0, valid_indices),
                flat_temporal.index_select(0, valid_indices),
            ]
        )
        flat_patch_features = valid_patch_features.new_zeros(
            batch_size * patch_count,
            self.temporal_visual_fusion.output_dim,
        )
        flat_patch_features = flat_patch_features.index_copy(
            0,
            valid_indices,
            valid_patch_features,
        )
        patch_features = flat_patch_features.reshape(
            batch_size,
            patch_count,
            self.temporal_visual_fusion.output_dim,
        )
        # 回填的 patch_features [B,N,2F] 按有效时长加权为 [B,2F]。
        # TDBRAIN 的四个完整 64 点 patch 的 fraction 均为 1，此时等价于四个 patch 的算术平均。
        sample_features = valid_fraction_weighted_pool(
            patch_features,
            mask,
            fractions,
        )
        # 每个原始 256 点数据窗口输出一个类别向量 [B,num_classes]，不是每个内部 patch 单独输出标签。
        # logits 尚未 softmax；TDBRAIN 当前二分类协议对应 [B,2]。
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
    "MULTIMODAL_FUSION_VERSION",
    "AdaptiveGranularityFusionModule",
    "compress_openclip_spatial_tokens",
]
