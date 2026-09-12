"""Numeric-only NeuroSigVIA without multimodal fusion (v2).

The historical zero-visual-slot variant is kept solely for explicit v1
checkpoint reconstruction. New training uses only channel pooling and an MLP.
"""

import torch
from torch import nn

from src.adaptive_graph_training import NeuroSigVIAClassifier
from src.patch_mindts import valid_fraction_weighted_pool


LEGACY_ARCHITECTURE = "neurosigvia_numeric_only_zero_visual_slot_v1"
ARCHITECTURE = "neurosigvia_numeric_only_no_fusion_v2"


class ZeroVisualSlotClassifier(nn.Module):
    """Historical v1 model; not the default numeric training architecture."""

    def __init__(self, full_model):
        super().__init__()
        self.model_config = dict(full_model.constructor_configuration)
        self.channel_pool = full_model.fusion.channel_pool
        self.temporal_visual_fusion = full_model.fusion.temporal_visual_fusion
        self.classifier = full_model.fusion.classifier
        self.classifier_input_dim = self.temporal_visual_fusion.output_dim

    @classmethod
    def from_config(cls, config):
        # Do not replace this with direct construction of the three modules:
        # renderer/vision initialization consumes RNG before shared weights.
        return cls(NeuroSigVIAClassifier.from_config(config))

    def forward(self, mantis_channel_tokens, patch_mask, valid_fraction):
        if mantis_channel_tokens.ndim != 4:
            raise ValueError("Mantis tokens must be [batch, patch, channel, dim]")
        mask = patch_mask.bool()
        fractions = valid_fraction.float()
        if mask.shape != mantis_channel_tokens.shape[:2] or fractions.shape != mask.shape:
            raise ValueError("patch metadata shape mismatch")
        if not torch.equal(mask, fractions > 0) or not mask.any(dim=1).all():
            raise ValueError("invalid patch mask/fractions")
        if not torch.isfinite(fractions).all() or (fractions < 0).any() or (fractions > 1).any():
            raise ValueError("fractions must be finite in [0, 1]")
        temporal, _ = self.channel_pool(mantis_channel_tokens, patch_mask=mask)
        batch, patches, _ = temporal.shape
        indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        valid_temporal = temporal.reshape(batch * patches, -1).index_select(0, indices)
        features = self.temporal_visual_fusion(
            [torch.zeros_like(valid_temporal), valid_temporal]
        )
        padded = features.new_zeros(batch * patches, self.temporal_visual_fusion.output_dim)
        padded = padded.index_copy(0, indices, features)
        pooled = valid_fraction_weighted_pool(
            padded.reshape(batch, patches, -1), mask, fractions
        )
        return self.classifier(pooled)


class NumericOnlyClassifier(nn.Module):
    """Mantis channel pooling -> duration pooling -> MLP; no fusion slot."""

    def __init__(self, full_model):
        super().__init__()
        self.model_config = dict(full_model.constructor_configuration)
        self.channel_pool = full_model.fusion.channel_pool
        self.classifier_input_dim = int(self.model_config["fusion_dim"])
        self.classifier = full_model.fusion.classifier
        # Removing fusion halves the head input (256 -> 128 in the main run).
        # Preserve the original hidden/output layers, ReLU and dropout. Only
        # this incompatible input projection must be freshly initialized.
        first = self.classifier.network[0]
        self.classifier.network[0] = nn.Linear(
            self.classifier_input_dim, first.out_features,
            bias=first.bias is not None,
            device=first.weight.device, dtype=first.weight.dtype,
        )

    @classmethod
    def from_config(cls, config):
        # A temporary untrained full constructor preserves initialization of
        # the retained channel pool and compatible head layers. No discarded
        # fusion/vision module belongs to the returned model or its forward.
        return cls(NeuroSigVIAClassifier.from_config(config))

    def forward(self, mantis_channel_tokens, patch_mask, valid_fraction):
        if mantis_channel_tokens.ndim != 4:
            raise ValueError("Mantis tokens must be [batch, patch, channel, dim]")
        mask = patch_mask.bool()
        fractions = valid_fraction.float()
        if mask.shape != mantis_channel_tokens.shape[:2] or fractions.shape != mask.shape:
            raise ValueError("patch metadata shape mismatch")
        if not torch.equal(mask, fractions > 0) or not mask.any(dim=1).all():
            raise ValueError("invalid patch mask/fractions")
        if not torch.isfinite(fractions).all() or (fractions < 0).any() or (fractions > 1).any():
            raise ValueError("fractions must be finite in [0, 1]")
        temporal, _ = self.channel_pool(mantis_channel_tokens, patch_mask=mask)
        pooled = valid_fraction_weighted_pool(temporal, mask, fractions)
        return self.classifier(pooled)


def numeric_model_from_checkpoint(checkpoint):
    """Reconstruct an already-loaded v1/v2 payload without silently converting it."""
    architectures = {
        LEGACY_ARCHITECTURE: ZeroVisualSlotClassifier,
        ARCHITECTURE: NumericOnlyClassifier,
    }
    architecture = checkpoint.get("architecture")
    if architecture not in architectures:
        raise ValueError(f"unsupported numeric checkpoint architecture: {architecture!r}")
    model = architectures[architecture].from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model
