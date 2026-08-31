#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.medformer_graph import (  # noqa: E402
    AdaptiveTemporalGranularitySelector,
)
from src.neurosigvit import get_neurosigvit  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channels", type=int, default=33)
    args = parser.parse_args()

    if args.channels < 2:
        raise ValueError("CUDA smoke test expects a multichannel input.")

    device = torch.device(args.device)
    torch.manual_seed(42)
    model = get_neurosigvit(
        model_name=args.model,
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
    model.eval().requires_grad_(False)
    signals = torch.randn(1, args.channels, 256, device=device)

    with torch.no_grad():
        candidates = model.forward_granularities(signals)
        legacy = model(signals)
    assert candidates.ndim == 3
    assert candidates.shape[:2] == (1, 3)
    assert candidates.shape[2] == legacy.shape[1]
    assert torch.isfinite(candidates).all()
    torch.testing.assert_close(
        candidates[:, 1],
        legacy,
        rtol=0.0,
        atol=0.0,
    )

    normalized = F.normalize(candidates, dim=-1)
    selector = AdaptiveTemporalGranularitySelector(
        feature_dim=normalized.shape[-1],
        num_granularities=3,
        hidden_dim=64,
        temperature=1.0,
        base_index=1,
        base_prior=0.9,
    ).to(device)
    selected, weights = selector(normalized.reshape(1, -1))
    probe = torch.linspace(
        -1.0,
        1.0,
        selected.shape[-1],
        device=device,
    )
    (selected * probe).sum().backward()
    assert selector.scorer[-1].weight.grad is not None
    assert torch.isfinite(selector.scorer[-1].weight.grad).all()
    assert selector.scorer[-1].weight.grad.abs().sum().item() > 0.0
    assert all(parameter.grad is None for parameter in model.parameters())

    peak_mib = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    print(
        "ADAPTIVE CLIP CUDA WITNESS",
        f"signals={tuple(signals.shape)}",
        f"candidates={tuple(candidates.shape)}",
        f"selected={tuple(selected.shape)}",
        "weights=" + ",".join(f"{value:.6f}" for value in weights[0].tolist()),
        f"peak_mib={peak_mib:.1f}",
        f"device={device}",
    )


if __name__ == "__main__":
    main()
