#!/usr/bin/env python3
"""CPU integration checks for the frozen patch-MindTS feature pipeline."""

import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.medformer_graph import TemporalGranularityGraphBank  # noqa: E402
from src.neurosigvit import (  # noqa: E402
    preprocess_stacked_multichannel_lineplot,
)
from src.patch_mindts import (  # noqa: E402
    _validate_patch_feature_bundle,
    extract_patch_feature_batch,
)


class DummyVision(nn.Module):
    """Small image-statistic encoder backed by the real graph renderers."""

    def __init__(self, output_dim=8):
        super().__init__()
        if output_dim != 8:
            raise ValueError("DummyVision uses eight deterministic image statistics")
        self.output_dim = output_dim
        self.aggregation = "mean"
        self.scale = nn.Parameter(torch.linspace(0.9, 1.1, output_dim))
        self.encoded_image_count = 0
        self.encoded_image_shapes = []
        self.med_activity_granularity_bank = TemporalGranularityGraphBank(
            patch_length_bank=((1, 2, 4), (2, 4, 8), (4, 8, 16)),
            img_size=32,
        )

    def forward_vit(self, images):
        assert images.ndim == 4 and images.shape[1] == 3
        self.encoded_image_count += int(images.shape[0])
        self.encoded_image_shapes.extend(
            [tuple(image.shape) for image in images]
        )
        flattened = images.float().flatten(start_dim=2)
        channel_means = flattened.mean(dim=-1)
        channel_stds = flattened.std(dim=-1, unbiased=False)
        global_minimum = flattened.amin(dim=(1, 2), keepdim=False).unsqueeze(1)
        global_maximum = flattened.amax(dim=(1, 2), keepdim=False).unsqueeze(1)
        return torch.cat(
            (
                channel_means,
                channel_stds,
                global_minimum,
                global_maximum,
            ),
            dim=1,
        )

    def aggregate_hidden_representations(self, hidden, aggregation):
        assert aggregation == self.aggregation
        assert hidden.shape[-1] == self.output_dim
        return hidden * self.scale


class DummyMantis(nn.Module):
    """Return deterministic per-channel statistics in the Mantis layout."""

    def __init__(self, output_dim=6):
        super().__init__()
        if output_dim != 6:
            raise ValueError("DummyMantis uses six deterministic statistics")
        self.output_dim = output_dim
        self.scale = nn.Parameter(torch.linspace(0.8, 1.2, output_dim))

    def forward(self, inputs):
        assert inputs.ndim == 3 and inputs.shape[1] == 1
        values = inputs[:, 0].float()
        features = torch.stack(
            (
                values.mean(dim=1),
                values.std(dim=1, unbiased=False),
                values.amin(dim=1),
                values.amax(dim=1),
                values[:, 0],
                values[:, -1],
            ),
            dim=1,
        )
        return features * self.scale


def make_signals(batch_size=2, channels=3, time_steps=160):
    generator = torch.Generator().manual_seed(117)
    noise = torch.randn(
        batch_size,
        channels,
        time_steps,
        generator=generator,
    )
    time = torch.linspace(0.0, 4.0 * torch.pi, time_steps).view(1, 1, -1)
    frequencies = torch.arange(1, channels + 1).view(1, channels, 1)
    return noise * 0.05 + torch.sin(frequencies * time)


def assert_feature_shapes(bundle, batch_size, patch_count, channels, visual_dim, mantis_dim):
    assert bundle["line_tokens"].shape == (
        batch_size,
        patch_count,
        visual_dim,
    )
    assert bundle["graph_tokens"].shape == (
        batch_size,
        patch_count,
        3,
        visual_dim,
    )
    assert bundle["mantis_channel_tokens"].shape == (
        batch_size,
        patch_count,
        channels,
        mantis_dim,
    )
    assert bundle["patch_mask"].shape == (batch_size, patch_count)
    assert bundle["valid_lengths"].shape == (batch_size, patch_count)
    assert bundle["valid_fraction"].shape == (batch_size, patch_count)


def assert_tail_invariance(reference, changed):
    for key in (
        "line_tokens",
        "graph_tokens",
        "mantis_channel_tokens",
        "valid_fraction",
    ):
        torch.testing.assert_close(
            reference[key],
            changed[key],
            rtol=0.0,
            atol=0.0,
        )
    for key in ("patch_mask", "valid_lengths"):
        assert torch.equal(reference[key], changed[key])


def main():
    device = torch.device("cpu")
    batch_size, channels, time_steps = 2, 3, 160
    visual_dim, mantis_dim = 8, 6
    lengths = torch.tensor([130, 65], dtype=torch.long)
    signals = make_signals(batch_size, channels, time_steps)

    # A two-dimensional window is one multichannel RGB image, not C images.
    lane_image = preprocess_stacked_multichannel_lineplot(signals[0, :, :64])
    assert lane_image.shape == (1, 3, 224, 224)
    assert torch.isfinite(lane_image).all()
    assert lane_image.amin().item() >= 0.0
    assert lane_image.amax().item() <= 1.0

    vision = DummyVision(output_dim=visual_dim)
    mantis = DummyMantis(output_dim=mantis_dim)
    assert all(parameter.requires_grad for parameter in vision.parameters())
    assert all(parameter.requires_grad for parameter in mantis.parameters())

    features = extract_patch_feature_batch(
        batch=signals,
        vision_model=vision,
        mantis_model=mantis,
        device=device,
        window_size=64,
        stride=64,
        encode_batch_size=2,
        lengths=lengths,
    )
    _validate_patch_feature_bundle(features, expected_num_granularities=3)
    assert_feature_shapes(
        features,
        batch_size=batch_size,
        patch_count=3,
        channels=channels,
        visual_dim=visual_dim,
        mantis_dim=mantis_dim,
    )

    expected_lengths = torch.tensor(
        ((64, 64, 2), (64, 1, 0)),
        dtype=torch.long,
    )
    expected_mask = expected_lengths > 0
    assert torch.equal(features["valid_lengths"], expected_lengths)
    assert torch.equal(features["patch_mask"], expected_mask)
    torch.testing.assert_close(
        features["valid_fraction"].float(),
        expected_lengths.float() / 64.0,
        rtol=0.0,
        atol=5e-4,
    )

    effective_patch_count = int(expected_mask.sum().item())
    candidate_count = 3
    expected_visual_images = effective_patch_count * (candidate_count + 1)
    assert vision.encoded_image_count == expected_visual_images
    assert len(vision.encoded_image_shapes) == expected_visual_images
    assert all(shape[0] == 3 for shape in vision.encoded_image_shapes)

    invalid_index = (1, 2)
    assert torch.count_nonzero(features["line_tokens"][invalid_index]) == 0
    assert torch.count_nonzero(features["graph_tokens"][invalid_index]) == 0
    assert (
        torch.count_nonzero(features["mantis_channel_tokens"][invalid_index])
        == 0
    )

    changed_signals = signals.clone()
    for sample_index, true_length in enumerate(lengths.tolist()):
        replacement = torch.linspace(
            1000.0 + sample_index,
            2000.0 + sample_index,
            time_steps - true_length,
        )
        changed_signals[sample_index, :, true_length:] = replacement.unsqueeze(0)

    images_before_second_extraction = vision.encoded_image_count
    changed_features = extract_patch_feature_batch(
        batch=changed_signals,
        vision_model=vision,
        mantis_model=mantis,
        device=device,
        window_size=64,
        stride=64,
        encode_batch_size=2,
        lengths=lengths,
    )
    assert_tail_invariance(features, changed_features)
    assert (
        vision.encoded_image_count - images_before_second_extraction
        == expected_visual_images
    )

    for model in (vision, mantis):
        for parameter in model.parameters():
            assert not parameter.requires_grad
            assert parameter.grad is None

    print("PATCH MINDTS FEATURE PIPELINE VALIDATION PASSED")


if __name__ == "__main__":
    main()
