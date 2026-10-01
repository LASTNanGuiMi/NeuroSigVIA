"""当前 NeuroSigVIA 的仅视觉消融：保留活动图、折线图及其交叉注意力。

此模块不构造 Mantis、时序通道池化、跨模态 InfoNCE 或双模态融合。
活动图仍在线生成，冻结视觉编码器的权重，但保留图像到门控的梯度。
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import SequentialSampler

from src.adaptive_activity_graph import AdaptiveActivityGraphRenderer
from src.adaptive_graph_training import NeuroSigVIAClassifier
from src.line_graph_cross_attention import LineGraphCrossAttention
from src.patch_fusion import (
    _MLPHead,
    _extract_line_tokens,
    make_temporal_patches,
    valid_fraction_weighted_pool,
)


VISION_ONLY_ARCHITECTURE = "neurosigvia_adaptive_graph_visual_only_v1"
VISION_ONLY_CACHE_ARCHITECTURE = "neurosigvia_visual_static_v1"
VISION_ONLY_CACHE_SCHEMA_VERSION = 1
VISUAL_STATIC_KEYS = (
    "raw_windows", "line_tokens", "patch_mask", "valid_fraction", "valid_lengths",
)


def validate_visual_bundle(bundle, *, expected_channels=None, window_size=64):
    """检查纯视觉缓存，拒绝多余模态和错位的样本/patch 轴。"""
    if not isinstance(bundle, Mapping) or set(bundle) != set(VISUAL_STATIC_KEYS):
        raise ValueError(f"visual cache must contain exactly {VISUAL_STATIC_KEYS}")
    if any(not torch.is_tensor(v) for v in bundle.values()):
        raise TypeError("visual cache values must be tensors")
    if any(v.device.type != "cpu" for v in bundle.values()):
        raise ValueError("visual cache tensors must reside on CPU")
    raw, line, mask, fraction, lengths = (bundle[k] for k in VISUAL_STATIC_KEYS)
    if raw.ndim != 4 or line.ndim != 3:
        raise ValueError("expected raw [S,N,C,T] and line [S,N,Dv]")
    if any(d <= 0 for d in raw.shape) or line.shape[-1] <= 0:
        raise ValueError("visual cache dimensions must be non-empty")
    axes = raw.shape[:2]
    if line.shape[:2] != axes or any(v.shape != axes for v in (mask, fraction, lengths)):
        raise ValueError("all visual cache sample/patch axes must match")
    if raw.dtype != torch.float32:
        raise TypeError("raw_windows must use float32")
    if line.dtype not in (torch.float16, torch.float32):
        raise TypeError("line_tokens must use float16 or float32")
    if fraction.dtype not in (torch.float16, torch.float32):
        raise TypeError("valid_fraction must use float16 or float32")
    if mask.dtype != torch.bool or lengths.dtype != torch.int64:
        raise TypeError("patch_mask must be bool and valid_lengths must be int64")
    if any(not bool(torch.isfinite(v).all()) for v in (raw, line, fraction)):
        raise ValueError("visual cache contains non-finite values")
    if raw.shape[-1] != window_size:
        raise ValueError("visual cache window size mismatch")
    if expected_channels is not None and raw.shape[2] != expected_channels:
        raise ValueError("visual cache channel count mismatch")
    if bool(((lengths < 0) | (lengths > window_size)).any()):
        raise ValueError("valid_lengths must lie within the window")
    if not torch.equal(mask, lengths > 0):
        raise ValueError("patch_mask must equal valid_lengths > 0")
    if not torch.allclose(fraction.float(), lengths.float() / window_size, rtol=1e-3, atol=1e-3):
        raise ValueError("valid_fraction must equal valid_lengths / window_size")
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("every sample must contain a valid patch")
    return dict(bundle)


@torch.no_grad()
def extract_visual_split(loader, vision_model, device, *, window_size=64,
                         stride=64, encode_batch_size=4):
    """冷缓存：仅切分原始 EEG 并提取折线图特征；完全不读取时序表征。

    ADFTD 输入 [B,19,256] -> raw [B,4,19,64]、line [B,4,1024]。
    其余三项均为 [B,4]；沿首轴拼接后得到整个 split 的窗口顺序。
    """
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("visual extraction requires a sequential source loader")
    if vision_model is None:
        raise ValueError("vision_model is required")
    if encode_batch_size <= 0 or window_size <= 0 or not 0 < stride <= window_size:
        raise ValueError("invalid encoding batch size or temporal patch geometry")
    vision_model.requires_grad_(False)
    vision_model.eval()
    chunks = {key: [] for key in VISUAL_STATIC_KEYS}
    expected_channels = None
    for item in loader:
        if len(item) == 1:
            signals, lengths = item[0], None
        elif len(item) == 2:
            signals, lengths = item
        else:
            raise ValueError("loader must yield (signals,) or (signals,lengths)")
        temporal = make_temporal_patches(
            signals, window_size=window_size, stride=stride, lengths=lengths,
        )
        batch, patches, channels, width = temporal.patches.shape
        if expected_channels is None:
            expected_channels = channels
        flat = temporal.patches.detach().cpu().float().reshape(-1, channels, width)
        valid_lengths = temporal.valid_lengths.detach().cpu().long()
        patch_mask = temporal.patch_mask.detach().cpu().bool()
        valid = patch_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            raise ValueError("cannot extract visual features from empty patches")
        line_valid = _extract_line_tokens(
            flat.index_select(0, valid), valid_lengths.reshape(-1).index_select(0, valid),
            vision_model, device, encode_batch_size,
        )
        line = torch.zeros((batch * patches, line_valid.shape[-1]), dtype=torch.float32)
        line.index_copy_(0, valid, line_valid.detach().cpu().float())
        values = validate_visual_bundle({
            "raw_windows": flat.reshape(batch, patches, channels, width),
            "line_tokens": line.reshape(batch, patches, -1).half(),
            "patch_mask": patch_mask,
            "valid_fraction": temporal.valid_fraction.detach().cpu().half(),
            "valid_lengths": valid_lengths,
        }, expected_channels=expected_channels, window_size=window_size)
        for key in VISUAL_STATIC_KEYS:
            chunks[key].append(values[key])
    if not chunks["raw_windows"]:
        raise ValueError("cannot extract an empty visual split")
    return validate_visual_bundle(
        {key: torch.cat(parts, dim=0) for key, parts in chunks.items()},
        expected_channels=expected_channels, window_size=window_size,
    )


class _VisualOnlyHead(nn.Module):
    def __init__(self, visual_dim, num_classes, *, fusion_dim, fusion_heads,
                 dropout, classifier_hidden_dim, classifier_num_layers,
                 cross_attention_ffn_hidden_dim, cross_attention_bias):
        super().__init__()
        self.visual_dim = visual_dim
        self.visual_cross_attention = LineGraphCrossAttention(
            line_dim=visual_dim, graph_dim=visual_dim, fusion_dim=fusion_dim,
            num_heads=fusion_heads, dropout=dropout,
            ffn_hidden_dim=cross_attention_ffn_hidden_dim, bias=cross_attention_bias,
        )
        self.classifier = _MLPHead(
            input_dim=fusion_dim, hidden_dim=classifier_hidden_dim,
            num_layers=classifier_num_layers, dropout=dropout, output_dim=num_classes,
        )

    def forward(self, line, graph, mask, fraction, *, return_attention_weights=False):
        result = self.visual_cross_attention(
            line, graph, patch_mask=mask,
            return_attention_weights=return_attention_weights,
        )
        visual, attention = result if return_attention_weights else (result, None)
        # 仅视觉：[B,4,F] -> [B,F] -> [B,3]，不复制特征来伪造双分支。
        sample = valid_fraction_weighted_pool(visual, mask, fraction)
        return self.classifier(sample), {
            "visual_tokens": visual, "patch_features": visual,
            "sample_features": sample, "cross_attention_weights": attention,
        }


class VisionOnlyClassifier(NeuroSigVIAClassifier):
    """复用主方法的在线活动图路径，只删除数值及跨模态部分。"""
    ARCHITECTURE = VISION_ONLY_ARCHITECTURE

    def __init__(self, visual_dim, num_channels, num_classes, *, fusion_dim=128,
                 fusion_heads=2, classifier_hidden_dim=128, classifier_num_layers=2,
                 dropout=0.1, graph_image_size=224, graph_token_grid=4,
                 activity_graph_canvas_size=360, activity_graph_line_width=1.0,
                 activity_graph_vertical_margin=0.05, adaptive_temperature=0.5,
                 freeze_adaptive_gate=False, cross_attention_ffn_hidden_dim=None,
                 cross_attention_bias=True, adaptive_gate_checkpoint=None,
                 strict_gate_checkpoint=True):
        # 不调用完整分类器构造函数，避免先实例化再删除数值/对齐模块。
        nn.Module.__init__(self)
        for key, value in {
            "visual_dim": visual_dim, "num_channels": num_channels,
            "num_classes": num_classes, "fusion_dim": fusion_dim,
            "classifier_hidden_dim": classifier_hidden_dim,
            "classifier_num_layers": classifier_num_layers,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if graph_token_grid != 4:
            raise ValueError("current visual ablation retains the 4x4 graph token grid")
        self._expected_vision_encoder_contract = None
        self._validated_vision_encoder_object_id = None
        self.graph_token_grid = graph_token_grid
        self.num_channels = num_channels
        self.renderer = AdaptiveActivityGraphRenderer(
            img_size=graph_image_size, temperature=adaptive_temperature,
            gate_checkpoint=adaptive_gate_checkpoint, freeze_gate=freeze_adaptive_gate,
            strict_gate_checkpoint=strict_gate_checkpoint,
            canvas_size=activity_graph_canvas_size, line_width=activity_graph_line_width,
            vertical_margin=activity_graph_vertical_margin,
        )
        self.fusion = _VisualOnlyHead(
            visual_dim, num_classes, fusion_dim=fusion_dim, fusion_heads=fusion_heads,
            dropout=dropout, classifier_hidden_dim=classifier_hidden_dim,
            classifier_num_layers=classifier_num_layers,
            cross_attention_ffn_hidden_dim=cross_attention_ffn_hidden_dim,
            cross_attention_bias=cross_attention_bias,
        )
        self.constructor_configuration = {
            "visual_dim": visual_dim, "num_channels": num_channels,
            "num_classes": num_classes, "fusion_dim": fusion_dim,
            "fusion_heads": fusion_heads, "classifier_hidden_dim": classifier_hidden_dim,
            "classifier_num_layers": classifier_num_layers, "dropout": dropout,
            "graph_image_size": graph_image_size, "graph_token_grid": graph_token_grid,
            "activity_graph_canvas_size": activity_graph_canvas_size,
            "activity_graph_line_width": activity_graph_line_width,
            "activity_graph_vertical_margin": activity_graph_vertical_margin,
            "adaptive_temperature": adaptive_temperature,
            "freeze_adaptive_gate": freeze_adaptive_gate,
            "cross_attention_ffn_hidden_dim": cross_attention_ffn_hidden_dim,
            "cross_attention_bias": cross_attention_bias,
        }
        self.configuration = {
            **self.constructor_configuration, "architecture": self.ARCHITECTURE,
            "renderer": self.renderer.provenance(), "visual_only": True,
            "modalities": ["vision"], "visual_views": ["lineplot", "adaptive_activity_graph"],
            "alignment_enabled": False, "temporal_visual_fusion_enabled": False,
            "classifier_input_dim": fusion_dim, "graph_spatial_token_count": 16,
            "vision_parameters_in_checkpoint": False,
        }

    def forward(self, raw_windows, line_tokens, patch_mask, valid_fraction,
                valid_lengths, vision_model, *, encode_batch_size=4,
                vision_gradient_checkpointing=True, return_attention_weights=False):
        self._validate_bound_vision_encoder(vision_model)
        if raw_windows.ndim != 4 or raw_windows.shape[2] != self.num_channels:
            raise ValueError("raw_windows must have shape [B,N,num_channels,T]")
        if raw_windows.shape[-1] != 64:
            raise ValueError("the current visual ablation uses 64-sample inner patches")
        if line_tokens.ndim != 3 or line_tokens.shape[:2] != raw_windows.shape[:2]:
            raise ValueError("line tokens must share raw sample/patch axes")
        if any(v.shape != raw_windows.shape[:2] for v in (patch_mask, valid_fraction, valid_lengths)):
            raise ValueError("mask, fractions and lengths must share raw sample/patch axes")
        if any(p.requires_grad for p in vision_model.parameters()):
            raise ValueError("freeze visual encoder weights before visual ablation training")
        graph, selector = self._encode_adaptive_graphs(
            raw_windows, valid_lengths, patch_mask, vision_model,
            encode_batch_size=encode_batch_size, spatial_grid_size=self.graph_token_grid,
            use_gradient_checkpointing=vision_gradient_checkpointing,
        )
        logits, details = self.fusion(
            line_tokens, graph, patch_mask, valid_fraction,
            return_attention_weights=return_attention_weights,
        )
        details.update(selector)
        return logits, details
