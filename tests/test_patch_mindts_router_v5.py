#!/usr/bin/env python3
"""Focused invariants for the paper-aligned Patch-MindTS Router v5.

These tests deliberately exercise the scientific contract of the router, not
just tensor shapes: routing must be Line-Q/Activity-Graph-K, sparsity must be a
real shared Top-2 decision, both the sample-global and patch-local branches
must remain content dependent, and the small-data regularizers must retain a
gradient even for a single sample containing four temporal patches.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.patch_mindts import (  # noqa: E402
    PatchGranularityCrossAttention,
    PatchMindTSFusionModule,
    masked_v5_router_regularization_terms,
)


V5_CONFIGURATION_KEYS = {
    "router_top_k",
    "router_training_noise_std",
    "router_local_weight",
    "router_relation_hidden_dim",
    "router_relation_residual_scale",
    "router_key_adapter_scale",
    "router_value_adapter_scale",
    "router_route_budget_weight",
    "router_load_balance_weight",
}


def make_selector(**overrides):
    configuration = {
        "visual_dim": 8,
        "fusion_dim": 8,
        "num_heads": 2,
        "num_granularities": 3,
        "temperature": 0.7,
        "dropout": 0.0,
        "router_mode": "adaptive_v5",
        "router_top_k": 2,
        "router_training_noise_std": 0.0,
        "router_local_weight": 0.5,
        "router_relation_hidden_dim": 16,
        "router_relation_residual_scale": 0.25,
        "router_key_adapter_scale": 0.1,
        "router_value_adapter_scale": 0.1,
    }
    configuration.update(overrides)
    return PatchGranularityCrossAttention(**configuration)


def make_fusion(router_mode="adaptive_v5", **overrides):
    configuration = {
        "visual_dim": 8,
        "temporal_dim": 6,
        "num_channels": 4,
        "num_granularities": 3,
        "num_classes": 2,
        "fusion_dim": 8,
        "fusion_heads": 2,
        "dropout": 0.0,
        "classifier_hidden_dim": 8,
        "classifier_num_layers": 2,
        "channel_hidden_dim": 4,
        "granularity_temperature": 0.7,
        "alignment_dim": 4,
        "alignment_temperature": 0.2,
        "granularity_router_mode": router_mode,
        "granularity_local_mix_max": 0.8,
        "granularity_local_mix_init": 0.5,
        "granularity_global_mix_max": 0.75,
        "granularity_global_mix_init": 0.5,
        "granularity_evidence_half_saturation": 0.005,
        "granularity_minimum_weight": 0.0,
        "granularity_score_cap": 0.7,
        "granularity_scorer_hidden_dim": 8,
        "granularity_confidence_half_saturation": 0.05,
        "router_top_k": 2,
        "router_training_noise_std": 0.0,
        "router_local_weight": 0.5,
        "router_relation_hidden_dim": 16,
        "router_relation_residual_scale": 0.25,
        "router_key_adapter_scale": 0.1,
        "router_value_adapter_scale": 0.1,
    }
    configuration.update(overrides)
    return PatchMindTSFusionModule(**configuration)


def random_inputs(batch_size=2, patch_count=4):
    line = torch.randn(batch_size, patch_count, 8)
    graph = torch.randn(batch_size, patch_count, 3, 8)
    mantis = torch.randn(batch_size, patch_count, 4, 6)
    mask = torch.ones(batch_size, patch_count, dtype=torch.bool)
    fraction = torch.ones(batch_size, patch_count)
    return line, graph, mantis, mask, fraction


class RouterV5ScientificContractTests(unittest.TestCase):
    def test_route_logits_and_weights_depend_on_line_query_and_graph_keys(self):
        """A query shuffle must change routing while graph candidates stay fixed."""
        torch.manual_seed(501)
        selector = make_selector(
            visual_dim=4,
            fusion_dim=4,
            num_heads=1,
            router_local_weight=1.0,
            router_relation_residual_scale=0.0,
        ).eval()
        with torch.no_grad():
            selector.query_projection.weight.copy_(torch.eye(4))
            selector.key_projection.weight.copy_(torch.eye(4))
        candidates = torch.tensor(
            [
                [2.0, -1.0, 0.0, -1.0],
                [-1.0, 2.0, -1.0, 0.0],
                [0.0, -1.0, 2.0, -1.0],
            ]
        )
        line = candidates.unsqueeze(0)
        graph = candidates.view(1, 1, 3, 4).expand(1, 3, 3, 4).clone()

        outputs = selector(line, graph, return_router_diagnostics=True)
        weights = outputs[2]
        pre_topk_scores = outputs[5]["router_pre_topk_scores"]

        shuffled_line = line.roll(1, dims=1)
        shuffled = selector(
            shuffled_line,
            graph,
            return_router_diagnostics=True,
        )
        shuffled_weights = shuffled[2]
        shuffled_scores = shuffled[5]["router_pre_topk_scores"]
        self.assertGreater(
            (pre_topk_scores - shuffled_scores).abs().sum().item(),
            1.0e-5,
        )
        self.assertGreater(
            (weights - shuffled_weights).abs().sum().item(),
            1.0e-5,
        )

        changed_graph = graph.roll(1, dims=2)
        changed = selector(
            line,
            changed_graph,
            return_router_diagnostics=True,
        )
        self.assertGreater(
            (pre_topk_scores - changed[5]["router_pre_topk_scores"])
            .abs()
            .sum()
            .item(),
            1.0e-5,
        )
        self.assertGreater((weights - changed[2]).abs().sum().item(), 1.0e-5)

    def test_middle_scale_has_trainable_escape_from_centered_linear_constraint(self):
        """Scale 8 must receive gradients through both its adapter and scorer."""
        torch.manual_seed(502)
        selector = make_selector(
            visual_dim=4,
            fusion_dim=4,
            num_heads=1,
        ).train()
        # Construct *projected* graph keys with a nonzero exact midpoint.  The
        # identity test normalization and atanh inputs cancel the production
        # tanh projection, while retaining the real v5 adapter code path.
        selector.graph_input_norm = torch.nn.Identity()
        with torch.no_grad():
            selector.key_projection.weight.copy_(torch.eye(4))
        middle_key = torch.tensor([0.20, -0.15, 0.10, -0.05])
        offset = torch.tensor([0.08, 0.04, -0.06, 0.02])
        projected_key_targets = torch.stack(
            (middle_key - offset, middle_key, middle_key + offset)
        )
        graph = torch.atanh(projected_key_targets).view(1, 1, 3, 4)
        graph = graph.expand(2, 4, 3, 4).clone()
        projected_query = torch.tensor([0.4, -0.2, 0.3, -0.1])
        projected_query = projected_query.view(1, 1, 4).expand(2, 4, 4)

        routed_keys, _ = selector._project_graph(graph)
        torch.testing.assert_close(
            routed_keys[:, :, 1],
            0.5 * (routed_keys[:, :, 0] + routed_keys[:, :, 2]),
            atol=1.0e-6,
            rtol=0.0,
        )
        diagnostics = selector._v5_routing_distribution(
            projected_query,
            routed_keys,
        )[3]
        scores = diagnostics["router_pre_topk_scores"]
        middle_ranking_loss = F.softplus(scores[..., 0] - scores[..., 1]).mean()
        middle_ranking_loss = middle_ranking_loss + F.softplus(
            scores[..., 2] - scores[..., 1]
        ).mean()
        middle_ranking_loss.backward()

        self.assertIsNotNone(selector.v5_key_adapter_delta.grad)
        self.assertTrue(torch.isfinite(selector.v5_key_adapter_delta.grad).all())
        self.assertGreater(
            selector.v5_key_adapter_delta.grad[1].abs().sum().item(),
            0.0,
        )
        scorer_gradients = [
            parameter.grad
            for parameter in selector.v5_relation_scorer.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in scorer_gradients))
        self.assertGreater(
            sum(gradient.abs().sum().item() for gradient in scorer_gradients),
            0.0,
        )

        initial_middle_scores = scores[..., 1].detach().clone()
        with torch.no_grad():
            selector.v5_key_adapter_delta[1].fill_(2.0)
        adapted_keys, _ = selector._project_graph(graph)
        midpoint_residual = adapted_keys[:, :, 1] - 0.5 * (
            adapted_keys[:, :, 0] + adapted_keys[:, :, 2]
        )
        self.assertGreater(midpoint_residual.abs().sum().item(), 1.0e-4)
        adapted_diagnostics = selector._v5_routing_distribution(
            projected_query,
            adapted_keys,
        )[3]
        self.assertGreater(
            (
                adapted_diagnostics["router_pre_topk_scores"][..., 1]
                - initial_middle_scores
            )
            .abs()
            .sum()
            .item(),
            1.0e-5,
        )

    def test_shared_top2_has_exact_sparse_support_and_unit_mass(self):
        torch.manual_seed(503)
        selector = make_selector().eval()
        line = torch.randn(2, 4, 8)
        graph = torch.randn(2, 4, 3, 8)
        mask = torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        )
        outputs = selector(
            line,
            graph,
            patch_mask=mask,
            return_router_diagnostics=True,
        )
        weights = outputs[2]
        per_head_weights = outputs[3]
        diagnostics = outputs[5]

        nonzero = (weights[mask] > 0.0).sum(dim=-1)
        self.assertTrue((nonzero == 2).all())
        self.assertTrue((nonzero <= selector.router_top_k).all())
        torch.testing.assert_close(
            weights[mask].sum(dim=-1),
            torch.ones(int(mask.sum())),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        torch.testing.assert_close(weights[~mask], torch.zeros_like(weights[~mask]))
        self.assertTrue(
            (diagnostics["router_topk_mask"][mask].sum(dim=-1) == 2).all()
        )
        self.assertEqual(
            diagnostics["router_topk_indices"].shape,
            (2, 4, 2),
        )

        # There is one semantic route per patch.  Per-head values must use
        # exactly that same sparse decision, rather than silently selecting a
        # union of different experts whose mean has more than Top-2 support.
        expected_per_head = weights.unsqueeze(2).expand_as(per_head_weights)
        torch.testing.assert_close(
            per_head_weights,
            expected_per_head,
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    def test_zero_noise_router_is_identical_in_train_and_eval(self):
        torch.manual_seed(504)
        selector = make_selector(router_training_noise_std=0.0)
        line = torch.randn(2, 4, 8)
        graph = torch.randn(2, 4, 3, 8)

        selector.train()
        train_outputs = selector(
            line,
            graph,
            return_router_diagnostics=True,
        )
        selector.eval()
        eval_outputs = selector(
            line,
            graph,
            return_router_diagnostics=True,
        )
        for train_value, eval_value in zip(train_outputs[:5], eval_outputs[:5]):
            torch.testing.assert_close(
                train_value,
                eval_value,
                atol=0.0,
                rtol=0.0,
            )
        for name in (
            "router_pre_topk_scores",
            "router_topk_mask",
            "router_topk_indices",
            "sample_global_route_scores",
            "patch_local_route_scores",
            "router_noise_std",
        ):
            torch.testing.assert_close(
                train_outputs[5][name],
                eval_outputs[5][name],
                atol=0.0,
                rtol=0.0,
            )

    def test_sample_global_and_patch_local_scores_both_have_qk_gradients(self):
        torch.manual_seed(505)
        selector = make_selector().eval()
        line = torch.randn(2, 4, 8, requires_grad=True)
        graph = torch.randn(2, 4, 3, 8, requires_grad=True)
        diagnostics = selector(
            line,
            graph,
            return_router_diagnostics=True,
        )[5]

        for index, name in enumerate(
            ("sample_global_route_scores", "patch_local_route_scores")
        ):
            # Selecting one candidate avoids the zero-gradient identity that
            # can result from summing centered logits over all candidates.
            objective = diagnostics[name][..., index].sum()
            line_gradient, graph_gradient = torch.autograd.grad(
                objective,
                (line, graph),
                retain_graph=True,
            )
            self.assertTrue(torch.isfinite(line_gradient).all(), name)
            self.assertTrue(torch.isfinite(graph_gradient).all(), name)
            self.assertGreater(line_gradient.abs().sum().item(), 0.0, name)
            self.assertGreater(graph_gradient.abs().sum().item(), 0.0, name)

    def test_budget_and_load_losses_backpropagate_for_one_sample_four_patches(self):
        torch.manual_seed(506)
        selector = make_selector().train()
        # Repeat one content pair so a single four-patch sample has a clearly
        # non-uniform expert marginal and both penalties exercise their
        # differentiable paths.
        line = torch.randn(1, 1, 8).expand(1, 4, 8).clone()
        graph = torch.randn(1, 1, 3, 8).expand(1, 4, 3, 8).clone()
        mask = torch.ones(1, 4, dtype=torch.bool)
        outputs = selector(
            line,
            graph,
            patch_mask=mask,
            return_router_diagnostics=True,
        )
        diagnostics = outputs[5]
        terms = masked_v5_router_regularization_terms(
            outputs[2],
            diagnostics["router_pre_topk_weights"],
            diagnostics["router_topk_mask"],
            mask,
        )
        route_parameters = [
            selector.query_projection.weight,
            selector.key_projection.weight,
            selector.v5_key_adapter_delta,
            *selector.v5_relation_scorer.parameters(),
        ]

        for index, loss_name in enumerate(("route_budget_loss", "load_cv2_loss")):
            loss = terms[loss_name]
            self.assertEqual(loss.ndim, 0, loss_name)
            self.assertTrue(torch.isfinite(loss), loss_name)
            self.assertTrue(loss.requires_grad, loss_name)
            gradients = torch.autograd.grad(
                loss,
                route_parameters,
                allow_unused=True,
                retain_graph=index == 0,
            )
            finite_nonzero = [
                gradient
                for gradient in gradients
                if gradient is not None
                and torch.isfinite(gradient).all()
                and gradient.abs().sum().item() > 0.0
            ]
            self.assertTrue(finite_nonzero, loss_name)

    def test_v4_and_v41_checkpoint_state_remain_strictly_compatible(self):
        torch.manual_seed(507)
        line, graph, mantis, mask, fraction = random_inputs()
        for router_mode in ("adaptive_v4", "adaptive_v41"):
            with self.subTest(router_mode=router_mode):
                model = make_fusion(router_mode=router_mode).eval()
                state = model.state_dict()
                self.assertFalse(any(".v5_" in name for name in state))

                # Recreate the pre-v5 checkpoint metadata: new constructor
                # fields did not exist, so all must be optional for old modes.
                legacy_configuration = {
                    name: value
                    for name, value in model.constructor_configuration.items()
                    if name not in V5_CONFIGURATION_KEYS
                }
                restored = PatchMindTSFusionModule(
                    **legacy_configuration
                ).eval()
                restored.load_state_dict(state, strict=True)

                logits, auxiliary = model(line, graph, mantis, mask, fraction)
                restored_logits, restored_auxiliary = restored(
                    line,
                    graph,
                    mantis,
                    mask,
                    fraction,
                )
                torch.testing.assert_close(
                    restored_logits,
                    logits,
                    atol=0.0,
                    rtol=0.0,
                )
                for name in (
                    "granularity_weights",
                    "granularity_scores",
                    "sample_global_weights",
                    "patch_local_weights",
                ):
                    torch.testing.assert_close(
                        restored_auxiliary[name],
                        auxiliary[name],
                        atol=0.0,
                        rtol=0.0,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
