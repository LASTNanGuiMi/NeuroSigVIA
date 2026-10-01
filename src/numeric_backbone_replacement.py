"""Trainable conventional numeric encoders for the full NeuroSigVIA model.

The adapters consume the existing outer temporal patches ``[B,N,C,L]`` and
produce one numeric token per outer patch, ``[B,N,F]``.  They deliberately use
the upstream encoder representations rather than the upstream classification
logits.  The visual branch, within-window InfoNCE, ``concat_attn`` fusion and
window-level classifier stay unchanged.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from runners.baselines import import_model
from src.adaptive_graph_training import NeuroSigVIAClassifier
from src.multimodal_fusion import AdaptiveGranularityFusionModule


ARCHITECTURE = "neurosigvia_numeric_backbone_replacement_v1"
BACKBONES = ("Medformer", "TimesNet", "PatchTST")


def backbone_config(
    name: str,
    *,
    sequence_length: int,
    channels: int,
    d_model: int = 128,
    d_ff: int = 256,
    e_layers: int = 2,
    n_heads: int = 8,
    dropout: float = 0.1,
    patch_len_list: str = "2,4,8",
    top_k: int = 3,
    num_kernels: int = 6,
) -> SimpleNamespace:
    if name not in BACKBONES:
        raise ValueError(f"unsupported numeric backbone: {name!r}")
    if sequence_length <= 0 or channels <= 0 or d_model <= 0:
        raise ValueError("sequence_length, channels and d_model must be positive")
    if d_model % n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    return SimpleNamespace(
        task_name="classification",
        seq_len=int(sequence_length),
        pred_len=0,
        enc_in=int(channels),
        dec_in=int(channels),
        c_out=int(channels),
        num_class=2,  # The projection head is removed and never used.
        d_model=int(d_model),
        d_ff=int(d_ff),
        e_layers=int(e_layers),
        d_layers=1,
        n_heads=int(n_heads),
        dropout=float(dropout),
        factor=1,
        activation="gelu",
        embed="timeF",
        freq="h",
        output_attention=False,
        label_len=0,
        moving_avg=25,
        single_channel=False,
        no_inter_attn=False,
        patch_len_list=str(patch_len_list),
        augmentations="none",
        patch_len=16,
        stride=8,
        top_k=int(top_k),
        num_kernels=int(num_kernels),
    )


class ConventionalNumericEncoder(nn.Module):
    """Expose encoder embeddings from a vendored time-series backbone."""

    def __init__(
        self,
        name: str,
        *,
        vendor_root: Path,
        sequence_length: int,
        channels: int,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        patch_len_list: str = "2,4,8",
        top_k: int = 3,
        num_kernels: int = 6,
    ) -> None:
        super().__init__()
        self.name = name
        self.sequence_length = int(sequence_length)
        self.channels = int(channels)
        self.output_dim = int(d_model)
        self.vendor_root = Path(vendor_root).resolve()
        self.config = backbone_config(
            name,
            sequence_length=sequence_length,
            channels=channels,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            n_heads=n_heads,
            dropout=dropout,
            patch_len_list=patch_len_list,
            top_k=top_k,
            num_kernels=num_kernels,
        )
        upstream_type = import_model(name, self.vendor_root)
        self.upstream = upstream_type(self.config)
        # Classification heads/decoders are outside this ablation's interface.
        # Removing them prevents unused parameters from entering the optimizer.
        for attribute in ("projection", "decoder", "dec_pos_embedding", "flatten"):
            if hasattr(self.upstream, attribute):
                delattr(self.upstream, attribute)

    def _encode_valid(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x is [V,C,L]; upstream encoders use [V,L,C].
        x = x.transpose(1, 2).contiguous()
        time = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        time_mask = time < lengths.unsqueeze(1)
        x = x * time_mask.unsqueeze(-1).to(dtype=x.dtype)

        if self.name == "Medformer":
            hidden = self.upstream.enc_embedding(x)
            hidden, _ = self.upstream.encoder(hidden, attn_mask=None)
            hidden = self.upstream.dropout(self.upstream.act(hidden))
            token = hidden.mean(dim=1)
        elif self.name == "PatchTST":
            # PatchTST's instance normalization, computed over the valid prefix only.
            weights = time_mask.unsqueeze(-1).to(dtype=x.dtype)
            count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            means = ((x * weights).sum(dim=1, keepdim=True) / count).detach()
            centered = (x - means) * weights
            stdev = torch.sqrt((centered ** 2).sum(dim=1, keepdim=True) / count + 1e-5)
            hidden, variables = self.upstream.patch_embedding((centered / stdev).permute(0, 2, 1))
            hidden, _ = self.upstream.encoder(hidden)
            hidden = hidden.reshape(-1, variables, hidden.shape[-2], hidden.shape[-1])
            token = self.upstream.dropout(hidden).mean(dim=(1, 2))
        else:
            hidden = self.upstream.enc_embedding(x, None)
            for index in range(self.upstream.layer):
                hidden = self.upstream.layer_norm(self.upstream.model[index](hidden))
            hidden = self.upstream.dropout(self.upstream.act(hidden))
            weights = time_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            token = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        if token.shape != (len(x), self.output_dim):
            raise ValueError(
                f"{self.name} encoder returned {tuple(token.shape)}, expected "
                f"({len(x)},{self.output_dim})"
            )
        if not torch.isfinite(token).all():
            raise ValueError(f"{self.name} encoder produced non-finite tokens")
        return token

    def forward(
        self,
        raw_windows: torch.Tensor,
        valid_lengths: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        if raw_windows.ndim != 4:
            raise ValueError("raw_windows must be [B,N,C,L]")
        if raw_windows.shape[2:] != (self.channels, self.sequence_length):
            raise ValueError(
                f"expected raw [C,L]=[{self.channels},{self.sequence_length}], got "
                f"{tuple(raw_windows.shape[2:])}"
            )
        if valid_lengths.shape != raw_windows.shape[:2] or patch_mask.shape != raw_windows.shape[:2]:
            raise ValueError("valid_lengths and patch_mask must share [B,N]")
        mask = patch_mask.to(dtype=torch.bool)
        lengths = valid_lengths.to(dtype=torch.long)
        if not torch.equal(mask, lengths > 0):
            raise ValueError("patch_mask must equal valid_lengths > 0")
        if bool((lengths > self.sequence_length).any()):
            raise ValueError("valid_lengths exceed the outer patch length")

        batch, patches, channels, length = raw_windows.shape
        flat_mask = mask.reshape(-1)
        indices = flat_mask.nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("batch contains no valid patches")
        flat = raw_windows.reshape(-1, channels, length)
        tokens = self._encode_valid(
            flat.index_select(0, indices),
            lengths.reshape(-1).index_select(0, indices),
        )
        padded = tokens.new_zeros(batch * patches, self.output_dim)
        padded = padded.index_copy(0, indices, tokens)
        return padded.reshape(batch, patches, self.output_dim)

    def provenance(self) -> dict:
        return {
            "name": self.name,
            "vendor_root": str(self.vendor_root),
            "representation": "upstream_encoder_hidden_mean_not_classification_logits",
            "input_shape": ["B", "N", self.channels, self.sequence_length],
            "output_shape": ["B", "N", self.output_dim],
            "config": vars(self.config),
        }


class _DirectTemporalIdentity(nn.Module):
    def forward(self, tokens: torch.Tensor, patch_mask=None):
        return tokens, None


class DirectTemporalFusionModule(AdaptiveGranularityFusionModule):
    """Use already pooled ``[B,N,F]`` numeric tokens in the original fusion."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.channel_pool = _DirectTemporalIdentity()
        self.configuration = dict(self.configuration)
        self.configuration.update(
            temporal_pooling="inside_conventional_numeric_encoder",
            temporal_input="direct_patch_tokens_B_N_F",
            numeric_branch_order="visual_then_conventional_numeric",
        )

    def _validate_inputs(
        self,
        line_tokens: torch.Tensor,
        graph_spatial_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        valid_fraction: torch.Tensor,
    ):
        if line_tokens.ndim != 3 or graph_spatial_tokens.ndim != 4 or temporal_tokens.ndim != 3:
            raise ValueError("expected line [B,N,Dv], graph [B,N,P,Dv], temporal [B,N,F]")
        axes = tuple(line_tokens.shape[:2])
        if tuple(graph_spatial_tokens.shape[:2]) != axes or tuple(temporal_tokens.shape[:2]) != axes:
            raise ValueError("all branches must share [B,N]")
        if line_tokens.shape[-1] != self.visual_dim or graph_spatial_tokens.shape[-1] != self.visual_dim:
            raise ValueError("visual feature dimension mismatch")
        if temporal_tokens.shape[-1] != self.fusion_dim:
            raise ValueError("numeric token dimension must equal fusion_dim")
        if len({line_tokens.device, graph_spatial_tokens.device, temporal_tokens.device}) != 1:
            raise ValueError("all branches must share a device")
        if len({line_tokens.dtype, graph_spatial_tokens.dtype, temporal_tokens.dtype}) != 1:
            raise ValueError("all branches must share a dtype")
        if patch_mask.shape != axes or valid_fraction.shape != axes:
            raise ValueError("patch metadata must have shape [B,N]")
        mask = patch_mask.to(device=line_tokens.device, dtype=torch.bool)
        fractions = valid_fraction.to(device=line_tokens.device, dtype=line_tokens.dtype)
        if not torch.isfinite(fractions).all() or bool(((fractions < 0) | (fractions > 1)).any()):
            raise ValueError("valid_fraction must be finite in [0,1]")
        if not torch.equal(fractions > 0, mask) or bool((mask.sum(dim=1) == 0).any()):
            raise ValueError("invalid patch mask/fractions")
        return mask, fractions


class NumericBackboneReplacementClassifier(NeuroSigVIAClassifier):
    """Full model whose numeric branch is Medformer/PatchTST/TimesNet."""

    def __init__(self, *, backbone_name: str, vendor_root: Path, backbone_config_values=None, **kwargs):
        values = dict(backbone_config_values or {})
        fusion_dim = int(kwargs.get("fusion_dim", 512))
        channels = int(kwargs["num_channels"])
        super().__init__(**kwargs)
        self.numeric_backbone = ConventionalNumericEncoder(
            backbone_name,
            vendor_root=vendor_root,
            sequence_length=64,
            channels=channels,
            d_model=fusion_dim,
            **values,
        )
        fusion_args = {
            key: self.constructor_configuration[key]
            for key in (
                "visual_dim", "temporal_dim", "num_channels", "num_classes",
                "fusion_dim", "fusion_heads", "dropout", "classifier_hidden_dim",
                "classifier_num_layers", "channel_hidden_dim", "alignment_dim",
                "alignment_temperature", "cross_attention_ffn_hidden_dim",
                "cross_attention_bias",
            )
            if key in self.constructor_configuration
        }
        fusion_args["temporal_dim"] = fusion_dim
        self.fusion = DirectTemporalFusionModule(**fusion_args)
        self.backbone_name = backbone_name
        self.constructor_configuration.update(
            architecture=ARCHITECTURE,
            backbone_name=backbone_name,
            backbone=self.numeric_backbone.provenance(),
        )
        self.configuration.update(
            architecture=ARCHITECTURE,
            numeric_backbone=self.numeric_backbone.provenance(),
            fusion=self.fusion.configuration,
            mantis_features_used=False,
            numeric_backbone_trainable=True,
        )

    def forward(
        self,
        raw_windows,
        line_tokens,
        unused_mantis_tokens,
        patch_mask,
        valid_fraction,
        valid_lengths,
        vision_model,
        *,
        visual_encode_batch_size=16,
        graph_spatial_grid_size=None,
        vision_gradient_checkpointing=True,
        return_attention_weights=False,
    ):
        self._validate_bound_vision_encoder(vision_model)
        grid = self.graph_token_grid if graph_spatial_grid_size is None else graph_spatial_grid_size
        if grid != self.graph_token_grid:
            raise ValueError("graph_spatial_grid_size conflicts with the model contract")
        numeric_tokens = self.numeric_backbone(raw_windows, valid_lengths, patch_mask)
        graph_tokens, selector_details = self._encode_adaptive_graphs(
            raw_windows,
            valid_lengths,
            patch_mask,
            vision_model,
            encode_batch_size=visual_encode_batch_size,
            spatial_grid_size=grid,
            use_gradient_checkpointing=vision_gradient_checkpointing,
        )
        logits, details = self.fusion(
            line_tokens,
            graph_tokens,
            numeric_tokens,
            patch_mask,
            valid_fraction,
            return_attention_weights=return_attention_weights,
        )
        details.update(selector_details)
        details["numeric_tokens"] = numeric_tokens
        return logits, details
