#!/usr/bin/env python3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.medformer_graph import (  # noqa: E402
    MedformerGraphRenderer,
    TemporalGranularityGraphBank,
)
from src.med_activity_graph import (  # noqa: E402
    MedActitivy_graph,
    MedActivityGraph,
    MedActivityGranularityBank,
    _inverse_occurrence_row_weights,
    _pair_covering_order,
)
from src.neurosigvit import _adaptive_granularity_metadata  # noqa: E402


def test_public_api_compatibility():
    assert MedformerGraphRenderer.__name__ == "MedformerGraphRenderer"
    assert MedformerGraphRenderer.__module__ == "src.medformer_graph.renderer"
    assert TemporalGranularityGraphBank.__name__ == (
        "TemporalGranularityGraphBank"
    )
    assert TemporalGranularityGraphBank.__module__ == (
        "src.medformer_graph.renderer"
    )
    assert MedActitivy_graph is MedformerGraphRenderer
    assert MedActivityGraph is MedformerGraphRenderer
    assert MedActivityGranularityBank is TemporalGranularityGraphBank
    assert TemporalGranularityGraphBank().patch_length_bank == (
        (1, 2, 4),
        (2, 4, 8),
        (4, 8, 16),
    )


def test_neurosigvit_single_scale_bank_metadata():
    base_index, layout = _adaptive_granularity_metadata(
        (4, 8, 16),
        ((4,), (8,), (16,)),
    )
    assert base_index == -1
    assert layout == "single_scale_granularity_bank_scale_major_flat_v3"

    base_index, layout = _adaptive_granularity_metadata(
        (2, 4, 8),
        ((1, 2, 4), (2, 4, 8), (4, 8, 16)),
    )
    assert base_index == 1
    assert layout == "granularity_bank_scale_major_flat_v1"


def make_signals(
    batch_size=2,
    channels=6,
    time_steps=976,
    *,
    dtype=torch.float32,
    device="cpu",
):
    time = torch.linspace(
        0.0,
        8.0 * torch.pi,
        time_steps,
        dtype=dtype,
        device=device,
    ).view(1, 1, -1)
    frequencies = (
        1.0
        + torch.arange(channels, dtype=dtype, device=device).view(1, -1, 1)
        * 0.25
    )
    batch_offsets = (
        torch.arange(batch_size, dtype=dtype, device=device).view(-1, 1, 1)
        * 0.1
    )
    return torch.sin(frequencies * time + batch_offsets) + 0.2 * torch.cos(
        (frequencies + 0.5) * time
    )


def assert_valid_image(image, shape, dtype, device):
    assert image.shape == shape
    assert image.dtype == dtype
    assert image.device == torch.device(device)
    assert torch.isfinite(image).all()
    assert image.amin().item() >= 0.0
    assert image.amax().item() <= 1.0


def test_pair_covering_orders():
    try:
        _pair_covering_order(0)
    except ValueError:
        pass
    else:
        raise AssertionError("A zero-channel graph was accepted")

    for num_channels in range(1, 11):
        first = _pair_covering_order(num_channels)
        assert first == _pair_covering_order(num_channels)
        assert all(0 <= channel < num_channels for channel in first)

        pair_count = num_channels * (num_channels - 1) // 2
        duplicate_count = (
            (num_channels - 2) // 2 if num_channels % 2 == 0 else 0
        )
        assert len(first) == pair_count + duplicate_count + 1

        transitions = Counter(
            tuple(sorted((left, right)))
            for left, right in zip(first, first[1:])
        )
        expected_pairs = {
            (left, right)
            for left in range(num_channels)
            for right in range(left + 1, num_channels)
        }
        assert set(transitions) == expected_pairs

        if num_channels == 1:
            assert first == [0]
        elif num_channels % 2 == 1:
            assert first[0] == first[-1] == 0
            assert set(transitions.values()) == {1}
        else:
            expected_duplicates = {
                (left, left + 1)
                for left in range(1, num_channels - 1, 2)
            }
            actual_duplicates = {
                pair for pair, count in transitions.items() if count == 2
            }
            assert first[0] == 0 and first[-1] == num_channels - 1
            assert actual_duplicates == expected_duplicates
            assert all(count in (1, 2) for count in transitions.values())

        order = torch.tensor(first, dtype=torch.long)
        weights = _inverse_occurrence_row_weights(
            order,
            num_channels,
            torch.float64,
        )
        torch.testing.assert_close(
            weights.mean(),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=0.0,
            atol=1e-12,
        )
        channel_totals = torch.zeros(num_channels, dtype=torch.float64)
        channel_totals.scatter_add_(0, order, weights)
        torch.testing.assert_close(
            channel_totals,
            torch.full_like(channel_totals, len(first) / num_channels),
            rtol=0.0,
            atol=1e-12,
        )


def test_forward_and_edge_cases():
    transform = MedformerGraphRenderer(
        patch_lengths=(2, 4, 8),
        img_size=224,
        channel_mix=0.35,
        router_temperature=0.2,
    )
    signals = make_signals()
    first = transform(signals)
    second = transform(signals)

    assert_valid_image(first, (2, 3, 224, 224), signals.dtype, "cpu")
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    batched_transform = MedActivityGraph(img_size=40)
    batched_signals = make_signals(batch_size=2, channels=6, time_steps=257)
    batched = batched_transform(batched_signals)
    separate = torch.cat(
        [batched_transform(sample) for sample in batched_signals],
        dim=0,
    )
    torch.testing.assert_close(batched, separate, rtol=1e-6, atol=1e-6)

    real_shape_transform = MedActivityGraph(img_size=48)
    for channels, time_steps in (
        (6, 4096),
        (6, 976),
        (9, 128),
        (9, 2048),
    ):
        real_signals = make_signals(
            batch_size=1,
            channels=channels,
            time_steps=time_steps,
        ).squeeze(0)
        real_image = real_shape_transform(real_signals)
        assert_valid_image(real_image, (1, 3, 48, 48), torch.float32, "cpu")

    constant = MedActivityGraph(img_size=32)(torch.ones(1, 6, 17))
    assert_valid_image(constant, (1, 3, 32, 32), torch.float32, "cpu")
    torch.testing.assert_close(
        constant,
        torch.full_like(constant, 0.5),
        rtol=0.0,
        atol=0.0,
    )

    non_finite = make_signals(batch_size=1, time_steps=31)
    non_finite[0, 0, 0] = float("nan")
    non_finite[0, 1, 1] = float("inf")
    non_finite[0, 2, 2] = float("-inf")
    non_finite_image = MedActivityGraph(img_size=32)(non_finite)
    assert_valid_image(
        non_finite_image,
        (1, 3, 32, 32),
        torch.float32,
        "cpu",
    )

    numpy_image = MedActivityGraph(img_size=32)(signals[0].numpy())
    assert_valid_image(numpy_image, (1, 3, 32, 32), torch.float32, "cpu")

    double_signals = make_signals(
        batch_size=1,
        channels=9,
        time_steps=128,
        dtype=torch.float64,
    )
    double_image = MedActivityGraph(img_size=32)(double_signals)
    assert_valid_image(double_image, (1, 3, 32, 32), torch.float64, "cpu")


def test_gradients(device="cpu"):
    signals = make_signals(
        batch_size=2,
        channels=6,
        time_steps=129,
        device=device,
    ).requires_grad_(True)
    image = MedActivityGraph(img_size=32).to(device)(signals)
    assert_valid_image(image, (2, 3, 32, 32), signals.dtype, signals.device)

    horizontal_weights = torch.linspace(
        0.5,
        1.5,
        image.shape[-1],
        dtype=image.dtype,
        device=image.device,
    )
    (image * horizontal_weights).mean().backward()
    assert signals.grad is not None
    assert torch.isfinite(signals.grad).all()
    assert signals.grad.abs().sum().item() > 0.0


def test_granularity_bank():
    signals = make_signals(batch_size=2, channels=6, time_steps=129)
    single_scale_bank = TemporalGranularityGraphBank(
        patch_length_bank=((4,), (8,), (16,)),
        img_size=32,
        channel_mix=0.35,
        router_temperature=0.2,
        router_mix=0.5,
    )
    single_scale_candidates = single_scale_bank(signals)
    assert_valid_image(
        single_scale_candidates,
        (2, 3, 3, 32, 32),
        signals.dtype,
        "cpu",
    )
    for candidate_index in range(3):
        candidate = single_scale_candidates[:, candidate_index]
        torch.testing.assert_close(candidate[:, 0], candidate[:, 1])
        torch.testing.assert_close(candidate[:, 1], candidate[:, 2])
        direct = MedformerGraphRenderer(
            patch_lengths=((4,), (8,), (16,))[candidate_index],
            img_size=32,
            channel_mix=0.35,
            router_temperature=0.2,
            router_mix=0.5,
        )(signals)
        torch.testing.assert_close(candidate, direct, rtol=0.0, atol=0.0)
    for left_index in range(3):
        for right_index in range(left_index + 1, 3):
            assert (
                single_scale_candidates[:, left_index]
                - single_scale_candidates[:, right_index]
            ).abs().sum().item() > 0.0

    # The old three-scales-per-RGB candidate remains supported for historical
    # sample-level experiments, but patch_mindts v3 does not use this path.
    bank_config = ((1, 2, 4), (2, 4, 8), (4, 8, 16))
    bank = TemporalGranularityGraphBank(
        patch_length_bank=bank_config,
        img_size=32,
        channel_mix=0.35,
        router_temperature=0.2,
        router_mix=0.5,
    )
    legacy = MedActivityGraph(
        patch_lengths=(2, 4, 8),
        img_size=32,
        channel_mix=0.35,
        router_temperature=0.2,
        router_mix=0.5,
    )
    candidates = bank(signals)
    assert_valid_image(
        candidates,
        (2, 3, 3, 32, 32),
        signals.dtype,
        "cpu",
    )
    torch.testing.assert_close(
        candidates[:, 1],
        legacy(signals),
        rtol=0.0,
        atol=0.0,
    )

    separate = torch.cat([bank(sample) for sample in signals], dim=0)
    torch.testing.assert_close(candidates, separate, rtol=1e-6, atol=1e-6)

    # Actual EEG channel counts used by APAVA, ADFTD, and TDBRAIN.
    small_bank = TemporalGranularityGraphBank(
        patch_length_bank=bank_config,
        img_size=24,
    )
    for channels in (1, 16, 19, 33):
        eeg_signals = make_signals(
            batch_size=1,
            channels=channels,
            time_steps=256,
        )
        eeg_candidates = small_bank(eeg_signals)
        assert_valid_image(
            eeg_candidates,
            (1, 3, 3, 24, 24),
            torch.float32,
            "cpu",
        )

    constant = small_bank(torch.ones(1, 16, 256))
    torch.testing.assert_close(
        constant,
        torch.full_like(constant, 0.5),
        rtol=0.0,
        atol=0.0,
    )


def test_validation():
    try:
        MedformerGraphRenderer(patch_lengths=(2, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("Two patch lengths were accepted")

    try:
        MedformerGraphRenderer(patch_lengths=(4, 2, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("Unordered patch lengths were accepted")

    try:
        TemporalGranularityGraphBank(
            patch_length_bank=((1, 2, 4), (1, 2, 4))
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate granularity regimes were accepted")


def main():
    test_public_api_compatibility()
    test_neurosigvit_single_scale_bank_metadata()
    test_pair_covering_orders()
    test_forward_and_edge_cases()
    test_gradients()
    test_granularity_bank()
    if torch.cuda.is_available():
        test_gradients("cuda")
    test_validation()
    print("MED ACTIVITY GRAPH VALIDATION PASSED")


if __name__ == "__main__":
    main()
