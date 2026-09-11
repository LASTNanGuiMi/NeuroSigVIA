"""Adaptive temporal granularity selection for NeuroSigVIA.

Route each channel's 16-sample region to a granularity of 4, 8, or 16. The
renderer uses that choice to select a piecewise-mean waveform before applying
the paper's pair-coverage ordering and cyclic three-column Activity Graph
layout. This pre-render integration is specific to NeuroSigVIA. Training
uses hard straight-through selection; evaluation and frozen gates use
deterministic argmax routing. Checkpoint loading and gate freezing belong to
this module. Source attribution is recorded in src/provenance.py and
SOURCE_NOTES.md.
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


from src.provenance import (
    ADAPTATION_BOUNDARY,
    UPSTREAM_COMMIT,
    UPSTREAM_COMPONENT,
    UPSTREAM_REPOSITORY,
)


ADAPTATION_VERSION = "adaptive_granularity_paper_waveform_graph_v2"

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


class AdaptiveGranularityGate(nn.Module):
    """Choose one of ``4/8/16`` for every channel-wise 16-sample region."""

    # 这里的 16 点是外层 patch 内部的区域长度，4/8/16 是区域内求均值的块长。
    # TDBRAIN 的 256 点数据窗口先由上游按 64/64 切成 4 个外层 patch；本门控不负责该切分。
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
        # 共享 MLP 沿最后一维执行 16 -> 64 -> 3；通道数和区域数不进入参数矩阵的大小。
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
            "downstream_representation": "selected_piecewise_mean_waveform",
            "graph_integration": "project_gate_before_yang2022_algorithms_1_and_3",
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
                "No adaptive granularity region_cls tensors were found in the gate checkpoint."
            )
        incompatible = self.load_state_dict(selected, strict=strict)
        self.loaded_checkpoint = (
            str(Path(checkpoint).expanduser())
            if isinstance(checkpoint, (str, os.PathLike))
            else "<in-memory mapping>"
        )
        return incompatible

    def freeze_gate(self, frozen: bool = True) -> "AdaptiveGranularityGate":
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
        # 此处 B 是本次渲染的外层 patch 数（可分块处理），不是原始数据窗口的 batch 大小。
        # 对 TDBRAIN 的单个 64 点 patch：C=33、R=4，输入 [B,33,4,16]。
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
                "regions and AdaptiveGranularityGate must be on the same device"
            )
        clean_logits = self.region_cls(regions.to(dtype=gate_parameter.dtype))
        # 最后一维依次对应块长 4、8、16；clean_probs 不含 Gumbel 噪声或温度缩放，供诊断/平衡项使用。
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

        # logits/probs/weights 均为 [B,C,R,3]；indices 和 region_valid_mask 为 [B,C,R]。
        # indices 的 0/1/2 是候选下标，须通过 granularities 映射为实际块长 4/8/16。
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


__all__ = [
    "ADAPTATION_BOUNDARY",
    "ADAPTATION_VERSION",
    "UPSTREAM_COMMIT",
    "UPSTREAM_COMPONENT",
    "UPSTREAM_REPOSITORY",
    "CheckpointLike",
    "AdaptiveGranularityGate",
]
