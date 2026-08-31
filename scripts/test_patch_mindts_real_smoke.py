#!/usr/bin/env python3
"""Single-batch CUDA witness for the real CLIP + Mantis patch path."""

import argparse
import sys
from pathlib import Path

import torch
from mantis.architecture import Mantis8M

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.neurosigvit import get_neurosigvit  # noqa: E402
from src.patch_mindts import (  # noqa: E402
    PatchMindTSFusionModule,
    extract_patch_feature_batch,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--mantis-model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--channels", type=int, default=19)
    parser.add_argument("--visual-batch-size", type=int, default=4)
    args = parser.parse_args()

    if args.samples < 1 or args.channels < 1 or args.visual_batch_size < 1:
        raise ValueError("samples, channels and visual-batch-size must be positive")
    device = torch.device(args.device)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    vision = get_neurosigvit(
        model_name=args.vision_model,
        model_layer=14,
        aggregation="mean",
        stride=None,
        patch_size=None,
        image_mode="med_activity_graph",
        med_activity_patch_lengths=(2, 4, 8),
        med_activity_channel_mix=0.35,
        med_activity_router_temperature=0.2,
        med_activity_router_mix=0.5,
        med_activity_adaptive_granularity=True,
        med_activity_granularity_bank=(
            (1, 2, 4),
            (2, 4, 8),
            (4, 8, 16),
        ),
    ).to(device)
    mantis = Mantis8M(device=device)
    mantis = mantis.from_pretrained(args.mantis_model).to(device)

    signals = torch.randn(args.samples, args.channels, 256)
    bundle = extract_patch_feature_batch(
        batch=signals,
        vision_model=vision,
        mantis_model=mantis,
        device=device,
        window_size=64,
        stride=64,
        encode_batch_size=args.visual_batch_size,
    )
    assert bundle["line_tokens"].shape[:2] == (args.samples, 4)
    assert bundle["graph_tokens"].shape[:3] == (args.samples, 4, 3)
    assert bundle["mantis_channel_tokens"].shape[:3] == (
        args.samples,
        4,
        args.channels,
    )
    assert bundle["patch_mask"].all()
    assert torch.equal(
        bundle["valid_lengths"],
        torch.full((args.samples, 4), 64, dtype=torch.long),
    )

    head = PatchMindTSFusionModule(
        visual_dim=bundle["line_tokens"].shape[-1],
        temporal_dim=bundle["mantis_channel_tokens"].shape[-1],
        num_channels=args.channels,
        num_granularities=3,
        num_classes=2,
        fusion_dim=128,
        fusion_heads=2,
        dropout=0.0,
        classifier_hidden_dim=64,
        classifier_num_layers=2,
        alignment_dim=64,
        alignment_temperature=0.1,
    ).to(device)
    logits, auxiliary = head(
        bundle["line_tokens"].to(device=device, dtype=torch.float32),
        bundle["graph_tokens"].to(device=device, dtype=torch.float32),
        bundle["mantis_channel_tokens"].to(device=device, dtype=torch.float32),
        bundle["patch_mask"].to(device),
        bundle["valid_fraction"].to(device=device, dtype=torch.float32),
    )
    loss = logits.square().mean() + 0.1 * auxiliary["alignment_loss"]
    loss.backward()

    weights = auxiliary["granularity_weights"]
    assert weights.shape == (args.samples, 4, 3)
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones((args.samples, 4), device=device),
    )
    assert all(parameter.grad is None for parameter in vision.parameters())
    assert all(parameter.grad is None for parameter in mantis.parameters())
    trainable_gradients = [
        parameter.grad for parameter in head.parameters() if parameter.requires_grad
    ]
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().sum() > 0
        for gradient in trainable_gradients
    )

    peak_mib = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    print(
        "PATCH MINDTS REAL WITNESS",
        f"line={tuple(bundle['line_tokens'].shape)}",
        f"graph={tuple(bundle['graph_tokens'].shape)}",
        f"mantis={tuple(bundle['mantis_channel_tokens'].shape)}",
        f"weights={tuple(weights.shape)}",
        f"logits={tuple(logits.shape)}",
        f"alignment={float(auxiliary['alignment_loss']):.6f}",
        f"peak_mib={peak_mib:.1f}",
        f"device={device}",
    )


if __name__ == "__main__":
    main()
