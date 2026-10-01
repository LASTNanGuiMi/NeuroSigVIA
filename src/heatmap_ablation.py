"""Fixed heatmaps -> jointly aligned visual/temporal tokens -> concat_attn.

The legacy static-cache key ``line_tokens`` contains HEATMAP embeddings here.
There is no line rendering, activity graph, adaptive gate, or visual cross-attention.
Both encoders are frozen; one shared pair of normalized alignment projections
feeds both symmetric within-sample InfoNCE and the supervised fusion classifier.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.mlp_classifier import FusionModule
from src.patch_fusion import (
    ChannelAttentionPool, MaskedIntraSampleInfoNCE, _MLPHead,
    _encode_visual_images, _extract_mantis_channel_tokens,
    make_temporal_patches, valid_fraction_weighted_pool,
)

HEATMAP_ARCHITECTURE = "neurosigvia_heatmap_aligned_concat_attn_v1"
HEATMAP_REPRESENTATIONS = ("multivariate_heatmap", "patch_heatmap", "ordinary_line", "adaptive_heatmap", "gaf", "tivit_grayscale")
HEATMAP_IMAGE_SIZE = 224
HEATMAP_RENDERING_POLICY = {
    "normalization": "shared_channel_valid_window_minmax_constant_to_0.5",
    "colormap": "matplotlib_viridis_256",
    "resize": "nearest_to_224x224",
    "axes_colorbar_text": False,
    "tail_policy": "crop_valid_prefix_then_gray_patch_cells",
    "patch_layout": "channel_major_then_time_patch_rows",
}


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def heatmap_matrix(window, *, valid_length=None,
                   image_representation="multivariate_heatmap",
                   heatmap_patch_size=8):
    """Return the unnormalized [rows, columns] matrix and valid-cell mask.

    Crop loader/window padding BEFORE rearrangement and normalization. A short
    final patch is padded only for rectangular layout, with separately masked
    cells. Each patch row belongs to one channel; channels never interleave.
    """
    if image_representation not in HEATMAP_REPRESENTATIONS:
        raise ValueError(f"unsupported image_representation: {image_representation}")
    _positive_int("heatmap_patch_size", heatmap_patch_size)
    x = torch.as_tensor(window).detach().cpu().float()
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("window must have nonempty [channels, time] shape")
    if valid_length is None:
        valid_length = x.shape[-1]
    valid_length = int(valid_length)
    if not 1 <= valid_length <= x.shape[-1]:
        raise ValueError("valid_length must lie in [1, window width]")
    x = x[:, :valid_length].contiguous()
    if not torch.isfinite(x).all():
        raise ValueError("valid heatmap values must be finite")
    mask = torch.ones_like(x, dtype=torch.bool)
    if image_representation == "patch_heatmap":
        tail = (-valid_length) % heatmap_patch_size
        x = F.pad(x, (0, tail), value=0)
        mask = F.pad(mask, (0, tail), value=False)
        x = x.reshape(-1, heatmap_patch_size)
        mask = mask.reshape(-1, heatmap_patch_size)
    return x, mask


def render_heatmap(window, *, valid_length=None,
                   image_representation="multivariate_heatmap",
                   heatmap_patch_size=8, image_size=HEATMAP_IMAGE_SIZE):
    """Render an axes-free RGB tensor [3,H,W] with a fixed viridis mapping."""
    from src.imaging_ablation import REPRESENTATIONS, render_image
    if image_representation in REPRESENTATIONS:
        return render_image(window, valid_length=valid_length, image_representation=image_representation, image_size=image_size)
    from matplotlib import colormaps
    _positive_int("image_size", image_size)
    matrix, valid = heatmap_matrix(
        window, valid_length=valid_length,
        image_representation=image_representation,
        heatmap_patch_size=heatmap_patch_size,
    )
    values = matrix[valid]
    low, high = values.min(), values.max()
    if bool(high > low):
        normalized = ((matrix - low) / (high - low)).clamp(0, 1)
    else:
        normalized = torch.full_like(matrix, 0.5)
    lookup = torch.as_tensor(colormaps["viridis"](
        torch.linspace(0, 1, 256).numpy())[:, :3], dtype=torch.float32)
    colors = lookup[(normalized * 255).round().long()]
    colors[~valid] = 0.5
    rgb = colors.permute(2, 0, 1).contiguous()
    return F.interpolate(rgb[None], size=(image_size, image_size),
                         mode="nearest")[0].clamp(0, 1)


@torch.no_grad()
def extract_heatmap_tokens(windows, valid_lengths, vision_model, device,
                          encode_batch_size, *, image_representation,
                          heatmap_patch_size=8):
    _positive_int("encode_batch_size", encode_batch_size)
    outputs = []
    for start in range(0, len(windows), encode_batch_size):
        stop = min(start + encode_batch_size, len(windows))
        images = torch.stack([
            render_heatmap(window, valid_length=int(length),
                           image_representation=image_representation,
                           heatmap_patch_size=heatmap_patch_size)
            for window, length in zip(windows[start:stop], valid_lengths[start:stop])
        ])
        # Exactly the same encoder pooling and feature normalization as line
        # extraction; only the image generator is replaced.
        outputs.append(_encode_visual_images(vision_model, images, device).cpu())
    if not outputs:
        raise ValueError("heatmap extraction requires at least one valid window")
    return torch.cat(outputs)


@torch.no_grad()
def extract_heatmap_feature_batch(
    batch, vision_model, mantis_model, device, *, window_size=64, stride=64,
    encode_batch_size=16, lengths=None,
    image_representation="multivariate_heatmap", heatmap_patch_size=8,
):
    """Return a six-key static bundle, with heatmaps in legacy line_tokens."""
    _positive_int("encode_batch_size", encode_batch_size)
    if vision_model is None or mantis_model is None:
        raise ValueError("both frozen encoders are required")
    for encoder in (vision_model, mantis_model):
        encoder.requires_grad_(False)
        encoder.eval()
    partition = make_temporal_patches(
        batch, window_size=window_size, stride=stride, lengths=lengths)
    b, n, c, length = partition.patches.shape
    raw = partition.patches.reshape(b * n, c, length).detach().cpu().float()
    valid_lengths = partition.valid_lengths.reshape(-1).detach().cpu()
    indices = partition.patch_mask.reshape(-1).detach().cpu().nonzero().flatten()
    windows = raw.index_select(0, indices)
    selected_lengths = valid_lengths.index_select(0, indices)
    visual_valid = extract_heatmap_tokens(
        windows, selected_lengths, vision_model, device, encode_batch_size,
        image_representation=image_representation,
        heatmap_patch_size=heatmap_patch_size,
    )
    temporal_valid = _extract_mantis_channel_tokens(
        windows, selected_lengths, mantis_model, device, encode_batch_size)
    visual = torch.zeros(b * n, visual_valid.shape[-1], dtype=torch.float32)
    temporal = torch.zeros(b * n, c, temporal_valid.shape[-1], dtype=torch.float32)
    visual.index_copy_(0, indices, visual_valid.float())
    temporal.index_copy_(0, indices, temporal_valid.float())
    return {
        "raw_windows": raw.reshape(b, n, c, length),
        "line_tokens": visual.reshape(b, n, -1).half(),
        "mantis_channel_tokens": temporal.reshape(b, n, c, -1).half(),
        "patch_mask": partition.patch_mask.detach().cpu(),
        "valid_fraction": partition.valid_fraction.detach().cpu().half(),
        "valid_lengths": partition.valid_lengths.detach().cpu(),
    }


class JointProjectedInfoNCE(MaskedIntraSampleInfoNCE):
    """Expose the SAME projected tensors to the joint loss and classifier.

    The inherited projection layers and loss algebra preserve the reference
    MaskedIntraSampleInfoNCE objective: symmetric within-sample negatives,
    valid-duration anchor weights, log(N_valid) normalization, and a zero
    contrastive term when fewer than two windows are valid.
    """
    def forward(self, temporal_tokens, visual_tokens, patch_mask,
                valid_fraction=None, *, return_tokens=False):
        if temporal_tokens.ndim != 3 or visual_tokens.ndim != 3:
            raise ValueError("alignment tokens must have shape [B,N,D]")
        if temporal_tokens.shape[:2] != visual_tokens.shape[:2]:
            raise ValueError("temporal and visual tokens must share batch/patch axes")
        if patch_mask.shape != temporal_tokens.shape[:2]:
            raise ValueError("patch_mask must share alignment batch/patch axes")
        if valid_fraction is not None and valid_fraction.shape != patch_mask.shape:
            raise ValueError("valid_fraction must have the same shape as patch_mask")
        temporal = F.normalize(self.temporal_projection(temporal_tokens),
                               dim=-1, eps=1e-8)
        visual = F.normalize(self.visual_projection(visual_tokens), dim=-1, eps=1e-8)
        zero_loss = temporal.sum() * 0.0 + visual.sum() * 0.0
        losses = []
        for sample_index in range(temporal.shape[0]):
            indices = torch.nonzero(patch_mask[sample_index].bool(),
                                    as_tuple=False).flatten()
            if indices.numel() < 2:
                continue
            t = temporal[sample_index].index_select(0, indices)
            v = visual[sample_index].index_select(0, indices)
            logits = (t.float() @ v.float().transpose(0, 1)) / self.temperature
            targets = torch.arange(logits.shape[0], device=logits.device)
            row_losses = F.cross_entropy(logits, targets, reduction="none")
            column_losses = F.cross_entropy(logits.transpose(0, 1), targets,
                                           reduction="none")
            if valid_fraction is None:
                sample_loss = 0.5 * (row_losses.mean() + column_losses.mean())
            else:
                weights = valid_fraction[sample_index].index_select(
                    0, indices).float().clamp_min(0.0)
                denominator = weights.sum().clamp_min(1e-8)
                sample_loss = 0.5 * (
                    torch.sum(row_losses * weights) / denominator
                    + torch.sum(column_losses * weights) / denominator)
            losses.append(sample_loss / math.log(int(indices.numel())))
        loss = torch.stack(losses).mean() if losses else zero_loss
        if return_tokens:
            return loss, temporal, visual
        return loss


class HeatmapFusionModule(nn.Module):
    def __init__(self, visual_dim, temporal_dim, num_channels, num_classes,
                 fusion_dim=512, fusion_heads=4, dropout=0.1,
                 classifier_hidden_dim=512, classifier_num_layers=2,
                 channel_hidden_dim=64, alignment_dim=256,
                 alignment_temperature=0.1):
        super().__init__()
        params = dict(visual_dim=visual_dim, temporal_dim=temporal_dim,
                      num_channels=num_channels, num_classes=num_classes,
                      fusion_dim=fusion_dim, fusion_heads=fusion_heads,
                      classifier_hidden_dim=classifier_hidden_dim,
                      classifier_num_layers=classifier_num_layers,
                      channel_hidden_dim=channel_hidden_dim,
                      alignment_dim=alignment_dim)
        for key, value in params.items():
            _positive_int(key, value)
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.visual_dim, self.temporal_dim = visual_dim, temporal_dim
        self.num_channels, self.num_classes = num_channels, num_classes
        self.fusion_dim = fusion_dim
        self.alignment_enabled = True
        self.temporal_visual_fusion_strategy = "concat_attn"
        self.channel_pool = ChannelAttentionPool(
            temporal_dim, fusion_dim, num_channels, channel_hidden_dim)
        self.alignment = JointProjectedInfoNCE(
            temporal_dim=fusion_dim, visual_dim=visual_dim,
            projection_dim=alignment_dim, temperature=alignment_temperature)
        self.temporal_visual_fusion = FusionModule(
            branch_dims=[alignment_dim, alignment_dim],
            modal_interaction="concat_attn", fusion_dim=fusion_dim,
            fusion_heads=fusion_heads,
            branch_names=["aligned_heatmap_visual", "aligned_mantis_temporal"])
        self.classifier = _MLPHead(
            self.temporal_visual_fusion.output_dim, classifier_hidden_dim,
            classifier_num_layers, dropout, num_classes)
        self.constructor_configuration = {
            **params, "dropout": float(dropout),
            "alignment_temperature": float(alignment_temperature)}
        self.configuration = {
            **self.constructor_configuration,
            "architecture": HEATMAP_ARCHITECTURE,
            "visual_fusion": "none_single_heatmap_embedding",
            "visual_token_count": 1,
            "temporal_pooling": "trainable_mantis_channel_attention",
            "alignment": "symmetric_intra_sample_patch_infonce",
            "alignment_to_fusion": "same_normalized_projected_tensors_no_detach",
            "alignment_projection_dim": alignment_dim,
            "temporal_visual_fusion": "concat_attn",
            "temporal_visual_fusion_semantics":
                "joint_alignment_projection_normalize_then_branch_projection_"
                "then_two_token_self_attention_then_flatten",
            "temporal_visual_branch_order": [
                "aligned_heatmap_visual", "aligned_mantis_temporal"],
            "sample_pooling": "valid_fraction_weighted_mean",
        }

    def get_config(self):
        return dict(self.constructor_configuration)

    @classmethod
    def from_config(cls, config):
        return cls(**dict(config))

    def forward(self, heatmap_tokens, mantis_channel_tokens, patch_mask,
                valid_fraction, *, return_attention_weights=False, pretrain=False):
        if pretrain:
            raise ValueError("Heatmap ablations train alignment and classification jointly")
        if heatmap_tokens.ndim != 3 or mantis_channel_tokens.ndim != 4:
            raise ValueError("expected heatmap [B,N,Dv] and Mantis [B,N,C,Dt]")
        shape = heatmap_tokens.shape[:2]
        if mantis_channel_tokens.shape[:2] != shape:
            raise ValueError("visual and temporal batch/window axes differ")
        if heatmap_tokens.shape[-1] != self.visual_dim:
            raise ValueError("heatmap feature dimension mismatch")
        if mantis_channel_tokens.shape[2:] != (self.num_channels, self.temporal_dim):
            raise ValueError("Mantis channel/feature dimensions mismatch")
        if heatmap_tokens.device != mantis_channel_tokens.device:
            raise ValueError("visual and temporal features must share a device")
        if heatmap_tokens.dtype != mantis_channel_tokens.dtype:
            raise ValueError("visual and temporal features must share a dtype")
        if patch_mask.shape != shape or valid_fraction.shape != shape:
            raise ValueError("mask and valid_fraction must share [B,N] axes")
        mask = patch_mask.to(device=heatmap_tokens.device, dtype=torch.bool)
        fraction = valid_fraction.to(device=heatmap_tokens.device,
                                     dtype=heatmap_tokens.dtype)
        if not torch.isfinite(fraction).all() or bool(((fraction < 0) | (fraction > 1)).any()):
            raise ValueError("valid_fraction must be finite and lie in [0,1]")
        if not torch.equal(mask, fraction > 0) or bool((mask.sum(1) == 0).any()):
            raise ValueError("each sample needs valid windows matching positive fractions")
        # Mask BEFORE linear/attention operations: invalid padded cache cells
        # cannot influence gradients or poison them if supplied as NaN.
        visual_raw = heatmap_tokens.masked_fill(~mask[..., None], 0)
        temporal_raw = mantis_channel_tokens.masked_fill(~mask[..., None, None], 0)
        temporal_tokens, channel_weights = self.channel_pool(temporal_raw, mask)
        alignment_loss, aligned_temporal, aligned_visual = self.alignment(
            temporal_tokens, visual_raw, mask, valid_fraction=fraction,
            return_tokens=True)
        b, n = shape
        indices = mask.reshape(-1).nonzero().flatten()
        valid_branches = [
            aligned_visual.reshape(b * n, -1).index_select(0, indices),
            aligned_temporal.reshape(b * n, -1).index_select(0, indices)]
        fused_valid = self.temporal_visual_fusion(valid_branches)
        fused = fused_valid.new_zeros(b * n, self.temporal_visual_fusion.output_dim)
        fused = fused.index_copy(0, indices, fused_valid).reshape(b, n, -1)
        pooled = valid_fraction_weighted_pool(fused, mask, fraction)
        logits = self.classifier(pooled)
        return logits, {
            "alignment_loss": alignment_loss,
            "reconstruction_loss": logits.new_zeros(()),
            "reconstruction_applied": False, "masked_branch_index": None,
            "temporal_tokens": temporal_tokens, "visual_tokens": visual_raw,
            "aligned_temporal_tokens": aligned_temporal,
            "aligned_visual_tokens": aligned_visual,
            "patch_features": fused, "sample_features": pooled,
            "channel_weights": channel_weights, "cross_attention_weights": None,
        }


class HeatmapClassifier(nn.Module):
    """Classifier adapter for the existing static-loader/checkpoint interface.

    Graph-only constructor flags remain serializable compatibility metadata;
    they instantiate no module and have no effect on the heatmap path.
    """
    def __init__(
        self, visual_dim, temporal_dim, num_channels, num_classes, *,
        fusion_dim=512, fusion_heads=4, dropout=0.1,
        classifier_hidden_dim=512, classifier_num_layers=2,
        channel_hidden_dim=64, alignment_dim=256, alignment_temperature=0.1,
        graph_image_size=224, graph_token_grid=4,
        activity_graph_canvas_size=360, activity_graph_line_width=1.0,
        activity_graph_vertical_margin=0.05, adaptive_temperature=0.5,
        freeze_adaptive_gate=False, cross_attention_ffn_hidden_dim=None,
        cross_attention_bias=True, adaptive_gate_checkpoint=None,
        strict_gate_checkpoint=True, temporal_visual_fusion="concat_attn",
        alignment_enabled=True, mask_prob=0.3,
        image_representation="multivariate_heatmap", heatmap_patch_size=8,
    ):
        super().__init__()
        if image_representation not in HEATMAP_REPRESENTATIONS:
            raise ValueError(f"unsupported image_representation: {image_representation}")
        _positive_int("heatmap_patch_size", heatmap_patch_size)
        if temporal_visual_fusion != "concat_attn" or alignment_enabled is not True:
            raise ValueError("Heatmap ablations require joint alignment followed by concat_attn")
        if adaptive_gate_checkpoint is not None:
            raise ValueError("Heatmap ablations have no adaptive gate checkpoint")
        if graph_image_size != HEATMAP_IMAGE_SIZE:
            raise ValueError("Heatmap image_size is fixed to 224 for the encoder contract")
        self.image_representation = image_representation
        self.heatmap_patch_size = heatmap_patch_size
        self.graph_token_grid = graph_token_grid  # accepted by legacy call sites
        self._expected_vision_encoder_contract = None
        self._validated_vision_encoder_object_id = None
        self.fusion = HeatmapFusionModule(
            visual_dim, temporal_dim, num_channels, num_classes,
            fusion_dim=fusion_dim, fusion_heads=fusion_heads, dropout=dropout,
            classifier_hidden_dim=classifier_hidden_dim,
            classifier_num_layers=classifier_num_layers,
            channel_hidden_dim=channel_hidden_dim, alignment_dim=alignment_dim,
            alignment_temperature=alignment_temperature)
        self.constructor_configuration = {
            **self.fusion.get_config(),
            "graph_image_size": graph_image_size, "graph_token_grid": graph_token_grid,
            "activity_graph_canvas_size": activity_graph_canvas_size,
            "activity_graph_line_width": activity_graph_line_width,
            "activity_graph_vertical_margin": activity_graph_vertical_margin,
            "adaptive_temperature": adaptive_temperature,
            "freeze_adaptive_gate": freeze_adaptive_gate,
            "cross_attention_ffn_hidden_dim": cross_attention_ffn_hidden_dim,
            "cross_attention_bias": cross_attention_bias,
            "strict_gate_checkpoint": strict_gate_checkpoint,
            "temporal_visual_fusion": "concat_attn", "alignment_enabled": True,
            "mask_prob": mask_prob, "image_representation": image_representation,
            "heatmap_patch_size": heatmap_patch_size,
        }
        from src.imaging_ablation import REPRESENTATIONS, rendering_policy
        policy = rendering_policy(image_representation) if image_representation in REPRESENTATIONS else dict(HEATMAP_RENDERING_POLICY)
        self.configuration = {
            **self.constructor_configuration, "architecture": HEATMAP_ARCHITECTURE,
            "fusion": dict(self.fusion.configuration),
            "fusion_input": "aligned_projected_features",
            "visual_cross_attention_enabled": False,
            "granularity_selection_enabled": False,
            "rendering": policy,
            "images_per_valid_window": 1,
            "activity_graph_generated_online": False,
            "heatmap_and_mantis_features_cached": True,
            "legacy_line_tokens_semantics": "frozen_embedding_of_selected_image_representation",
            "removed_modules": ["activity_graph", "adaptive_granularity_gate",
                                "visual_cross_complementary_attention"],
            "ignored_graph_compatibility_parameters": [
                "graph_token_grid", "activity_graph_canvas_size",
                "activity_graph_line_width", "activity_graph_vertical_margin",
                "adaptive_temperature", "freeze_adaptive_gate",
                "cross_attention_ffn_hidden_dim", "cross_attention_bias",
                "strict_gate_checkpoint"],
            "vision_parameters_in_checkpoint": False,
        }

    def get_config(self):
        return dict(self.constructor_configuration)

    @classmethod
    def from_config(cls, config):
        return cls(**dict(config))

    def bind_vision_encoder_contract(self, contract, *, validated_encoder=None):
        if not isinstance(contract, Mapping):
            raise TypeError("vision encoder contract must be a mapping")
        self._expected_vision_encoder_contract = dict(contract)
        self._validated_vision_encoder_object_id = (
            id(validated_encoder) if validated_encoder is not None else None)

    def forward(
        self, raw_windows, line_tokens, mantis_channel_tokens, patch_mask,
        valid_fraction, valid_lengths, vision_model, *,
        visual_encode_batch_size=16, graph_spatial_grid_size=None,
        vision_gradient_checkpointing=True, return_attention_weights=False,
        pretrain=False,
    ):
        if self._expected_vision_encoder_contract is not None and vision_model is not None:
            if self._validated_vision_encoder_object_id != id(vision_model):
                # Lazy import avoids circular dependency with the training adapter.
                from src.adaptive_graph_training import _assert_encoder_contract
                _assert_encoder_contract(self._expected_vision_encoder_contract,
                                         vision_model, "vision")
                self._validated_vision_encoder_object_id = id(vision_model)
        # raw_windows, valid_lengths and online-vision flags are deliberately
        # unused: images and frozen features were validated and cached up front.
        logits, details = self.fusion(
            line_tokens, mantis_channel_tokens, patch_mask, valid_fraction,
            return_attention_weights=return_attention_weights, pretrain=pretrain)
        zero, zero_vector = logits.new_zeros(()), logits.new_zeros(3)
        details.update({
            "selector_balance_loss": zero,
            "selector_soft_usage": zero_vector, "selector_hard_usage": zero_vector,
            "selector_probability_sum": zero_vector, "selector_hard_count": zero_vector,
            "selector_valid_count": zero, "selector_mean_entropy": zero,
        })
        return logits, details


__all__ = ["HEATMAP_ARCHITECTURE", "HEATMAP_RENDERING_POLICY",
           "HEATMAP_REPRESENTATIONS", "HeatmapClassifier", "HeatmapFusionModule",
           "JointProjectedInfoNCE", "heatmap_matrix", "render_heatmap",
           "extract_heatmap_tokens", "extract_heatmap_feature_batch"]
