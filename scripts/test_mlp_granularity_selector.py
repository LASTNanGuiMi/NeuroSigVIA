#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.medformer_graph import AdaptiveTemporalGranularitySelector  # noqa: E402
from src.mlp_classifier import (  # noqa: E402
    AdaptiveGranularitySelector as LegacyAdaptiveGranularitySelector,
    FusionModule,
    MLPClassifier,
    _extract_feature_split,
    forward_feature_batch,
)


def test_public_api_compatibility():
    assert AdaptiveTemporalGranularitySelector.__name__ == (
        "AdaptiveTemporalGranularitySelector"
    )
    assert AdaptiveTemporalGranularitySelector.__module__ == (
        "src.medformer_graph.selector"
    )
    assert (
        LegacyAdaptiveGranularitySelector
        is AdaptiveTemporalGranularitySelector
    )
    selector = AdaptiveTemporalGranularitySelector(
        feature_dim=4,
        num_granularities=3,
    )
    expected_keys = {
        "granularity_bias",
        "normalization.weight",
        "normalization.bias",
        "scorer.0.weight",
        "scorer.0.bias",
        "scorer.2.weight",
        "scorer.2.bias",
    }
    legacy_state = {
        key: value.detach().clone()
        for key, value in selector.state_dict().items()
    }
    assert set(legacy_state) == expected_keys
    restored = AdaptiveTemporalGranularitySelector(
        feature_dim=4,
        num_granularities=3,
    )
    incompatible = restored.load_state_dict(legacy_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_fresh_process_import_orders():
    statements = (
        (
            "from src.med_activity_graph import MedActivityGraph; "
            "from src.medformer_graph import MedformerGraphRenderer; "
            "assert MedActivityGraph is MedformerGraphRenderer"
        ),
        (
            "from src.mlp_classifier import AdaptiveGranularitySelector as Old; "
            "from src.medformer_graph import "
            "AdaptiveTemporalGranularitySelector as New; "
            "assert Old is New"
        ),
        (
            "import src.utils; "
            "import src.medformer_graph"
        ),
        (
            "import src.neurosigvit; "
            "import src.medformer_graph"
        ),
    )
    for statement in statements:
        completed = subprocess.run(
            [sys.executable, "-c", statement],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_selector_probabilities_and_gradients():
    torch.manual_seed(7)
    selector = AdaptiveTemporalGranularitySelector(
        feature_dim=12,
        num_granularities=3,
        hidden_dim=8,
        temperature=1.0,
        base_index=1,
        base_prior=0.9,
    )
    candidates = torch.randn(10, 3, 12)
    selected, weights = selector(candidates.reshape(10, -1))

    assert selected.shape == (10, 12)
    assert weights.shape == (10, 3)
    assert torch.isfinite(selected).all()
    assert torch.isfinite(weights).all()
    assert (weights >= 0.0).all()
    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones(10),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        weights.mean(dim=0),
        torch.tensor([0.05, 0.90, 0.05]),
        rtol=0.0,
        atol=0.01,
    )

    probe = torch.linspace(-1.0, 1.0, 12)
    loss = (selected * probe).sum() + 0.1 * weights[:, 0].sum()
    loss.backward()
    assert selector.scorer[-1].weight.grad is not None
    assert torch.isfinite(selector.scorer[-1].weight.grad).all()
    assert selector.scorer[-1].weight.grad.abs().sum().item() > 0.0
    assert selector.granularity_bias.grad is not None
    assert torch.isfinite(selector.granularity_bias.grad).all()
    assert selector.granularity_bias.grad.abs().sum().item() > 0.0


def test_selector_degenerate_fallback():
    selector = AdaptiveTemporalGranularitySelector(
        feature_dim=6,
        num_granularities=3,
        hidden_dim=4,
        base_index=1,
    )
    repeated = torch.randn(4, 1, 6).expand(-1, 3, -1).contiguous()
    selected, weights = selector(repeated.reshape(4, -1))
    expected = torch.zeros_like(weights)
    expected[:, 1] = 1.0
    torch.testing.assert_close(weights, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        selected,
        nn.functional.normalize(repeated[:, 1], dim=-1),
        rtol=1e-6,
        atol=1e-6,
    )

    non_finite = repeated.clone()
    non_finite[:, 0, 0] = float("nan")
    try:
        selector(non_finite.reshape(4, -1))
    except FloatingPointError as exc:
        assert "batch indices" in str(exc)
    else:
        raise AssertionError("Training silently accepted non-finite candidates")

    selector.eval()
    _, non_finite_weights = selector(non_finite.reshape(4, -1))
    torch.testing.assert_close(
        non_finite_weights,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert selector.last_fallback_codes.tolist() == [1, 1, 1, 1]
    assert selector.last_fallback_indices.tolist() == [1, 1, 1, 1]

    damaged_base = repeated.clone()
    damaged_base[:, 1, 0] = float("inf")
    _, damaged_base_weights = selector(damaged_base.reshape(4, -1))
    expected_first = torch.zeros_like(weights)
    expected_first[:, 0] = 1.0
    torch.testing.assert_close(
        damaged_base_weights, expected_first, rtol=0.0, atol=0.0
    )
    assert selector.last_fallback_indices.tolist() == [0, 0, 0, 0]

    all_damaged = torch.full_like(repeated, float("nan"))
    try:
        selector(all_damaged.reshape(4, -1))
    except FloatingPointError as exc:
        assert "complete nonzero candidate" in str(exc)
    else:
        raise AssertionError("Evaluation accepted samples with no finite candidate")

    clean = torch.randn(4, 3, 6)
    selector(clean.reshape(4, -1))
    assert selector.last_fallback_codes.tolist() == [0, 0, 0, 0]
    assert selector.last_fallback_indices.tolist() == [-1, -1, -1, -1]

    zero_candidate = clean.clone()
    zero_candidate[:, 0] = 0.0
    _, zero_weights = selector(zero_candidate.reshape(4, -1))
    assert selector.last_fallback_codes.tolist() == [1, 1, 1, 1]
    assert selector.last_fallback_indices.tolist() == [1, 1, 1, 1]
    torch.testing.assert_close(zero_weights, expected, rtol=0.0, atol=0.0)

    all_zero = torch.zeros_like(clean)
    try:
        selector(all_zero.reshape(4, -1))
    except FloatingPointError as exc:
        assert "complete nonzero candidate" in str(exc)
    else:
        raise AssertionError("Evaluation accepted all-zero candidates")


def test_selector_nonfinite_logits_fail_fast():
    selector = AdaptiveTemporalGranularitySelector(
        feature_dim=4,
        num_granularities=3,
        base_index=1,
    )
    with torch.no_grad():
        selector.granularity_bias[0] = float("inf")
    candidates = torch.randn(2, 3, 4)
    for training in (True, False):
        selector.train(training)
        try:
            selector(candidates.reshape(2, -1))
        except FloatingPointError as exc:
            assert "selector logits" in str(exc)
        else:
            raise AssertionError("Non-finite selector logits did not fail fast")


def test_selector_degenerate_mixture_policy():
    selector = AdaptiveTemporalGranularitySelector(
        feature_dim=2,
        num_granularities=2,
        hidden_dim=3,
        base_index=0,
        base_prior=0.5,
    )
    with torch.no_grad():
        for parameter in selector.scorer.parameters():
            parameter.zero_()
        selector.granularity_bias.zero_()
    candidates = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    try:
        selector(candidates.reshape(1, -1))
    except FloatingPointError as exc:
        assert "weighted candidate mixtures" in str(exc)
    else:
        raise AssertionError("Training accepted a zero-norm candidate mixture")

    selector.eval()
    selected, weights = selector(candidates.reshape(1, -1))
    torch.testing.assert_close(weights, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(selected, torch.tensor([[1.0, 0.0]]))
    assert selector.last_fallback_codes.tolist() == [4]
    assert selector.last_fallback_indices.tolist() == [0]


def test_granularity_regularization():
    fusion = FusionModule(
        branch_dims=[12],
        modal_interaction="concat",
        branch_names=["vision"],
        branch_granularity_counts=[3],
        branch_granularity_base_indices=[1],
        branch_granularity_labels=[("fine", "base", "coarse")],
    )

    cases = (
        (torch.full((6, 3), 1.0 / 3.0), 0.0, 1.0),
        (
            torch.tensor([[1.0, 0.0, 0.0]]).expand(6, -1).clone(),
            1.0,
            0.0,
        ),
        (torch.eye(3), 0.0, 0.0),
    )
    for weights, expected_balance, expected_entropy in cases:
        fusion._granularity_weights_for_loss = {"vision": weights}
        fusion._granularity_valid_for_loss = {
            "vision": torch.ones(len(weights), dtype=torch.bool)
        }
        total, balance, entropy = fusion.granularity_regularization_loss(
            balance_weight=1.0,
            entropy_weight=1.0,
        )
        assert torch.isfinite(total)
        torch.testing.assert_close(
            balance,
            torch.tensor(expected_balance),
            rtol=0.0,
            atol=5e-6,
        )
        torch.testing.assert_close(
            entropy,
            torch.tensor(expected_entropy),
            rtol=0.0,
            atol=5e-6,
        )

    torch.manual_seed(23)
    output = fusion([torch.randn(9, 12)])
    assert output.shape == (9, 4)
    assert not fusion.last_granularity_weights["vision"].requires_grad
    assert fusion._granularity_weights_for_loss["vision"].requires_grad
    regularization, _, _ = fusion.granularity_regularization_loss(
        balance_weight=0.01,
        entropy_weight=0.001,
    )
    regularization.backward()
    selector = fusion.granularity_selectors[0]
    assert selector.granularity_bias.grad is not None
    assert torch.isfinite(selector.granularity_bias.grad).all()
    assert selector.scorer[-1].weight.grad is not None
    assert torch.isfinite(selector.scorer[-1].weight.grad).all()

    identical = torch.randn(5, 1, 4).expand(-1, 3, -1).reshape(5, -1)
    fusion([identical])
    regularization, balance, entropy = fusion.granularity_regularization_loss()
    torch.testing.assert_close(regularization, regularization.new_zeros(()))
    torch.testing.assert_close(balance, balance.new_zeros(()))
    torch.testing.assert_close(entropy, entropy.new_zeros(()))


def test_fusion_selects_before_modal_attention():
    torch.manual_seed(11)
    fusion = FusionModule(
        branch_dims=[36, 8],
        modal_interaction="concat_attn",
        fusion_dim=16,
        fusion_heads=4,
        branch_names=["vision_model_1", "mantis_model"],
        branch_granularity_counts=[3, 1],
        branch_granularity_base_indices=[1, 0],
        branch_granularity_labels=[
            ("1-2-4", "2-4-8", "4-8-16"),
            ("single",),
        ],
        granularity_hidden_dim=8,
    )
    output = fusion([torch.randn(5, 36), torch.randn(5, 8)])
    assert output.shape == (5, 32)
    assert fusion.output_dim == 32
    assert set(fusion.last_granularity_weights) == {"vision_model_1"}
    output.square().mean().backward()
    selector = fusion.granularity_selectors[0]
    assert type(selector) is AdaptiveTemporalGranularitySelector
    selector_keys = {
        key
        for key in fusion.state_dict()
        if key.startswith("granularity_selectors.0.")
    }
    assert selector_keys == {
        f"granularity_selectors.0.{key}"
        for key in (
            "granularity_bias",
            "normalization.weight",
            "normalization.bias",
            "scorer.0.weight",
            "scorer.0.bias",
            "scorer.2.weight",
            "scorer.2.bias",
        )
    }
    assert selector.granularity_bias.grad is not None
    assert torch.isfinite(selector.granularity_bias.grad).all()

    concat = FusionModule(
        branch_dims=[36, 8],
        modal_interaction="concat",
        branch_names=["vision_model_1", "mantis_model"],
        branch_granularity_counts=[3, 1],
        branch_granularity_base_indices=[1, 0],
        branch_granularity_labels=[
            ("1-2-4", "2-4-8", "4-8-16"),
            ("single",),
        ],
    )
    assert concat.output_dim == 20
    assert concat([torch.randn(2, 36), torch.randn(2, 8)]).shape == (2, 20)


def test_two_visual_branches_have_independent_selectors():
    fusion = FusionModule(
        branch_dims=[18, 24],
        modal_interaction="concat",
        branch_names=["vision_model_1", "vision_model_2"],
        branch_granularity_counts=[3, 3],
        branch_granularity_base_indices=[1, 1],
        branch_granularity_labels=[
            ("fine", "base", "coarse"),
            ("fine", "base", "coarse"),
        ],
    )
    first, second = fusion.granularity_selectors
    assert first is not second
    assert first.granularity_bias.data_ptr() != second.granularity_bias.data_ptr()
    output = fusion([torch.randn(3, 18), torch.randn(3, 24)])
    assert output.shape == (3, 14)
    assert set(fusion.last_granularity_weights) == {
        "vision_model_1",
        "vision_model_2",
    }


def test_selector_rng_isolated_from_shared_head():
    torch.manual_seed(19)
    fixed_fusion = FusionModule(
        branch_dims=[12, 8],
        modal_interaction="concat_attn",
        fusion_dim=16,
        fusion_heads=4,
    )
    fixed_mlp = MLPClassifier(32, 10, 2, 0.1, 3)

    torch.manual_seed(19)
    adaptive_fusion = FusionModule(
        branch_dims=[36, 8],
        modal_interaction="concat_attn",
        fusion_dim=16,
        fusion_heads=4,
        branch_granularity_counts=[3, 1],
        branch_granularity_base_indices=[1, 0],
    )
    adaptive_mlp = MLPClassifier(32, 10, 2, 0.1, 3)

    for fixed, adaptive in zip(
        fixed_fusion.projections.state_dict().values(),
        adaptive_fusion.projections.state_dict().values(),
    ):
        torch.testing.assert_close(fixed, adaptive, rtol=0.0, atol=0.0)
    for fixed, adaptive in zip(
        fixed_fusion.attention.state_dict().values(),
        adaptive_fusion.attention.state_dict().values(),
    ):
        torch.testing.assert_close(fixed, adaptive, rtol=0.0, atol=0.0)
    for fixed, adaptive in zip(
        fixed_mlp.state_dict().values(),
        adaptive_mlp.state_dict().values(),
    ):
        torch.testing.assert_close(fixed, adaptive, rtol=0.0, atol=0.0)


class DummyAdaptiveVision(nn.Module):
    image_mode = "med_activity_graph"
    feature_granularity_count = 3
    feature_granularity_base_index = 1
    feature_granularity_labels = ("fine", "base", "coarse")

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_granularities(self, batch):
        summary = torch.stack(
            [
                batch.mean(dim=(1, 2)),
                batch.std(dim=(1, 2), unbiased=False),
                batch.amax(dim=(1, 2)),
                batch.amin(dim=(1, 2)),
            ],
            dim=1,
        )
        offsets = torch.tensor(
            [[0.1, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0]],
            dtype=summary.dtype,
            device=summary.device,
        )
        return (summary.unsqueeze(1) + offsets.unsqueeze(0)) * self.scale


def test_feature_layout_and_backbone_freeze():
    torch.manual_seed(13)
    model = DummyAdaptiveVision()
    batch = torch.randn(4, 16, 256)
    features = forward_feature_batch(
        batch=batch,
        channels=16,
        device="cpu",
        vision_model_1=model,
    )
    assert len(features) == 1
    assert features[0].shape == (4, 12)
    candidate_norms = features[0].reshape(4, 3, 4).norm(dim=-1)
    torch.testing.assert_close(
        candidate_norms,
        torch.ones_like(candidate_norms),
        rtol=1e-6,
        atol=1e-6,
    )

    loader = DataLoader(TensorDataset(batch), batch_size=2, shuffle=False)
    cached = _extract_feature_split(
        loader,
        {"vision_model_1": model},
        channels=16,
        device="cpu",
    )
    assert cached[0].shape == (4, 12)
    assert not model.scale.requires_grad
    assert model.scale.grad is None


def main():
    test_public_api_compatibility()
    test_fresh_process_import_orders()
    test_selector_probabilities_and_gradients()
    test_selector_degenerate_fallback()
    test_selector_nonfinite_logits_fail_fast()
    test_selector_degenerate_mixture_policy()
    test_granularity_regularization()
    test_fusion_selects_before_modal_attention()
    test_two_visual_branches_have_independent_selectors()
    test_selector_rng_isolated_from_shared_head()
    test_feature_layout_and_backbone_freeze()
    print("MLP ADAPTIVE GRANULARITY VALIDATION PASSED")


if __name__ == "__main__":
    main()
