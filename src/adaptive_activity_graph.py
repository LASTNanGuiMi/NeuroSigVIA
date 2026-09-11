"""Adaptive granularity followed by the paper's multi-column Activity Graph.

The TimeMosaic-derived gate remains an upstream temporal-resolution selector.
After selection, graph construction follows Yang et al. (IEEE TII 2022): the
selected waveforms are ordered by Algorithm 1 and drawn in the cyclic
previous/current/next layout of Algorithm 3. No RMS activity map, correlation
adjacency, channel propagation, or occurrence weighting is used.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from src.activity_graph import (
    PAPER_ACTIVITY_GRAPH_CANVAS_SIZE,
    PAPER_ACTIVITY_GRAPH_DOI,
    PAPER_ACTIVITY_GRAPH_LAYOUT,
    _coerce_valid_lengths,
    paper_signal_order,
    render_paper_activity_graph,
)
from src.temporal_granularity import (
    ADAPTATION_VERSION,
    UPSTREAM_COMMIT,
    AdaptiveGranularityGate,
    CheckpointLike,
)


class AdaptiveActivityGraphRenderer(nn.Module):
    """Render one adaptive paper-layout Activity Graph from raw ``[B,C,T]``.

    Each channel is divided into 16-sample regions. The gate selects a temporal
    averaging block of 4, 8, or 16 for every region. Those selected waveforms,
    rather than custom activity statistics, enter the cited paper's fixed
    Algorithm 1 + Algorithm 3 graph renderer.
    """

    region_length = 16
    granularities = (4, 8, 16)
    upstream_commit = UPSTREAM_COMMIT
    adaptation_version = ADAPTATION_VERSION

    def __init__(
        self,
        img_size: int = 224,
        channel_mix: float | None = None,
        temperature: float = 0.5,
        gate_checkpoint: Optional[CheckpointLike] = None,
        freeze_gate: bool = False,
        strict_gate_checkpoint: bool = True,
        *,
        canvas_size: int = PAPER_ACTIVITY_GRAPH_CANVAS_SIZE,
        line_width: float = 1.0,
        vertical_margin: float = 0.05,
    ) -> None:
        super().__init__()
        if isinstance(img_size, bool) or not isinstance(img_size, int) or img_size < 1:
            raise ValueError(f"img_size must be a positive integer, got {img_size}")
        if (
            isinstance(canvas_size, bool)
            or not isinstance(canvas_size, int)
            or canvas_size < 3
            or canvas_size % 3 != 0
        ):
            raise ValueError("canvas_size must be a positive multiple of three")
        if not math.isfinite(line_width) or line_width <= 0.0:
            raise ValueError("line_width must be positive and finite")
        if (
            not math.isfinite(vertical_margin)
            or vertical_margin < 0.0
            or vertical_margin >= 0.5
        ):
            raise ValueError("vertical_margin must lie in [0, 0.5)")
        if channel_mix is not None and (
            not math.isfinite(channel_mix) or not 0.0 <= channel_mix <= 1.0
        ):
            raise ValueError(f"channel_mix must be in [0, 1], got {channel_mix}")

        self.img_size = int(img_size)
        self.canvas_size = int(canvas_size)
        self.line_width = float(line_width)
        self.vertical_margin = float(vertical_margin)
        # Kept only so old constructor configurations can be read. It has no
        # effect because the cited Activity Graph does not mix channel values.
        self.channel_mix = None if channel_mix is None else float(channel_mix)
        self.gate = AdaptiveGranularityGate(
            temperature=temperature,
            checkpoint=gate_checkpoint,
            freeze=freeze_gate,
            strict_checkpoint=strict_gate_checkpoint,
        )

    @staticmethod
    def provenance() -> dict[str, Any]:
        metadata = AdaptiveGranularityGate.provenance()
        metadata.update(
            {
                "activity_graph_reference_doi": PAPER_ACTIVITY_GRAPH_DOI,
                "activity_graph_layout": PAPER_ACTIVITY_GRAPH_LAYOUT,
                "activity_graph_signal": "selected_piecewise_mean_waveform",
                "activity_graph_channel_propagation": False,
                "activity_graph_rms_or_difference_statistics": False,
                "paper_reported_reference_canvas": [360, 360],
                "paper_underreported_plot_bounds": True,
            }
        )
        return metadata

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
    def _normalize_gate_input(
        signals: torch.Tensor,
        sample_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize only the gate input without changing graph layout rules."""

        # signals [M,C,T]、sample_valid_mask [M,T] -> [M,C,T]；M 表示本次渲染的外层 patch 数。
        valid = sample_valid_mask[:, None, :]
        finite = torch.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        masked = finite.masked_fill(~valid, float("nan"))
        center = torch.nanquantile(masked, 0.5, dim=-1, keepdim=True)
        deviations = finite - center
        absolute_deviations = deviations.abs().masked_fill(~valid, float("nan"))
        mad = torch.nanquantile(
            absolute_deviations, 0.5, dim=-1, keepdim=True
        ) * 1.4826
        counts = valid.sum(dim=-1, keepdim=True).to(dtype=signals.dtype)
        mean = (finite * valid).sum(dim=-1, keepdim=True) / counts
        variance = (((finite - mean).square()) * valid).sum(
            dim=-1, keepdim=True
        ) / counts
        std = variance.clamp_min(1e-12).sqrt()
        scale = torch.where(mad > 1e-6, mad, std.clamp_min(1e-6))
        normalized = torch.nan_to_num(
            deviations / scale,
            nan=0.0,
            posinf=8.0,
            neginf=-8.0,
        ).clamp(-8.0, 8.0)
        return normalized.masked_fill(~valid, 0.0)

    @staticmethod
    def _candidate_waveform(
        regions: torch.Tensor,
        region_sample_mask: torch.Tensor,
        granularity: int,
    ) -> torch.Tensor:
        """Average each temporal block and restore a 16-point waveform lane."""

        # [M,C,R,16] -> [M,C,R,16/g,g] -> 均值 [M,C,R,16/g] -> [M,C,R,16]。
        # g=4/8/16 时每区分别有 4/2/1 个均值；选择后仍保留 16 点长度，未减少外层 patch 数。
        subpatch_count = regions.shape[-1] // granularity
        patches = regions.reshape(
            *regions.shape[:-1], subpatch_count, granularity
        )
        mask = region_sample_mask[:, None, :, :].expand_as(regions)
        patch_mask = mask.reshape(
            *mask.shape[:-1], subpatch_count, granularity
        )
        counts = patch_mask.sum(dim=-1)
        means = (patches * patch_mask).sum(dim=-1) / counts.clamp_min(1).to(
            dtype=patches.dtype
        )
        means = means.masked_fill(counts == 0, 0.0)
        reconstructed = means.repeat_interleave(granularity, dim=-1)
        return reconstructed.masked_fill(~mask, 0.0)

    def forward(
        self,
        signals: torch.Tensor,
        valid_lengths: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """Render an RGB graph while preserving the gate's gradient path."""

        # 上游 NeuroSigVIAClassifier._encode_adaptive_graphs 传入外层 patch [M,C,T]。
        # TDBRAIN 在 outer_patch_size/stride=64/64 时：[B,33,256] -> [B,4,33,64]，
        # 合并前两轴并按 encode_batch_size 分块后，本函数看到 [M,33,64]（M 不固定为 B）。
        if not torch.is_tensor(signals) or signals.ndim != 3:
            raise ValueError("signals must be a tensor with shape [B,C,T]")
        if min(signals.shape) < 1:
            raise ValueError("signals must have non-empty batch/channel/time axes")
        if not signals.is_floating_point():
            raise ValueError("signals must be floating point")

        output_dtype = signals.dtype
        working_dtype = torch.float64 if signals.dtype == torch.float64 else torch.float32
        working = torch.nan_to_num(
            signals.to(dtype=working_dtype), nan=0.0, posinf=0.0, neginf=0.0
        )
        lengths = _coerce_valid_lengths(
            valid_lengths,
            batch_size=working.shape[0],
            time_steps=working.shape[2],
            device=working.device,
        )
        original_time_steps = working.shape[2]
        time_axis = torch.arange(original_time_steps, device=working.device)
        sample_valid_mask = time_axis.unsqueeze(0) < lengths.unsqueeze(1)
        gate_input = self._normalize_gate_input(working, sample_valid_mask)

        region_count = (original_time_steps + self.region_length - 1) // self.region_length
        # 64 点外层 patch 对应 R=4，无需补齐；若直接调用本类处理其他长度，则 R=ceil(T/16)。
        padded_time = region_count * self.region_length
        if padded_time != original_time_steps:
            pad_width = padded_time - original_time_steps
            working = torch.nn.functional.pad(working, (0, pad_width))
            gate_input = torch.nn.functional.pad(gate_input, (0, pad_width))
            sample_valid_mask = torch.nn.functional.pad(
                sample_valid_mask, (0, pad_width), value=False
            )

        regions = working.reshape(
            working.shape[0], working.shape[1], region_count, self.region_length
        )
        gate_regions = gate_input.reshape(
            working.shape[0], working.shape[1], region_count, self.region_length
        )
        region_sample_mask = sample_valid_mask.reshape(
            working.shape[0], region_count, self.region_length
        )
        region_valid = region_sample_mask.any(dim=-1)
        # 样本掩码 [M,R,16] 先得到区域掩码 [M,R]，再沿通道广播为 Gate 要求的 [M,C,R]。
        gate_diagnostics = self.gate(
            gate_regions,
            region_valid_mask=region_valid[:, None, :].expand(
                -1, working.shape[1], -1
            ),
            generator=generator,
        )

        # TDBRAIN 64 点外层 patch 的候选形状为 [M,33,4,3,16]，不是三张已经编码的图像。
        candidates = torch.stack(
            [
                self._candidate_waveform(
                    regions, region_sample_mask, granularity
                )
                for granularity in self.granularities
            ],
            dim=3,
        )
        selection_weights = gate_diagnostics["weights"].to(dtype=candidates.dtype)
        selected_regions = torch.einsum(
            "bcrk,bcrkt->bcrt", selection_weights, candidates
        )
        # 选中的区域 [M,33,4,16] 拼回 [M,33,64]；输出仍保留所有通道及有效时间点。
        selected = selected_regions.reshape(
            working.shape[0], working.shape[1], padded_time
        )[..., :original_time_steps]
        selected = selected.masked_fill(
            ~sample_valid_mask[:, None, :original_time_steps], 0.0
        )

        # img_size 默认 224，因而通常得到 [M,3,224,224]；RGB 的 3 与 EEG 的 33 个通道不同。
        # 下游在此图像上保留梯度通过冻结 OpenCLIP，以便训练预渲染门控。
        image = render_paper_activity_graph(
            selected,
            valid_lengths=lengths,
            output_size=self.img_size,
            canvas_size=self.canvas_size,
            line_width=self.line_width,
            vertical_margin=self.vertical_margin,
        ).to(dtype=output_dtype)

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
                "sample_valid_mask": sample_valid_mask[:, :original_time_steps],
                "activity_graph_order_length": torch.tensor(
                    len(paper_signal_order(signals.shape[1])),
                    dtype=torch.long,
                    device=signals.device,
                ),
            }
        )
        return image, diagnostics


__all__ = ["AdaptiveActivityGraphRenderer"]
