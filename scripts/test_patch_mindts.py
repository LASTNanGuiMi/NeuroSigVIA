#!/usr/bin/env python3
"""Fast CPU tests for the time-patch MindTS-style fusion components."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.patch_mindts import (  # noqa: E402
    ChannelAttentionPool,
    MaskedIntraSampleInfoNCE,
    PatchGranularityCrossAttention,
    PatchMindTSFusionModule,
    _aggregate_subject_predictions,
    _checkpoint_selection_key,
    _make_query_counterfactuals,
    _resolve_checkpoint_metric,
    compute_line_query_center,
    load_patch_feature_cache,
    make_temporal_patches,
    masked_granularity_regularization_terms,
    save_patch_feature_cache,
    valid_fraction_weighted_pool,
)


class TemporalPatchTests(unittest.TestCase):
    def _assert_patch_layout(self, length, expected_lengths):
        channels = 2
        signal = torch.arange(
            channels * length,
            dtype=torch.float32,
        ).reshape(1, channels, length)
        result = make_temporal_patches(
            signal,
            window_size=64,
            stride=64,
        )

        num_patches = len(expected_lengths)
        self.assertEqual(result.patches.shape, (1, num_patches, channels, 64))
        self.assertEqual(result.time_mask.shape, (1, num_patches, 64))
        self.assertEqual(result.patch_mask.shape, (1, num_patches))
        self.assertEqual(result.valid_lengths.shape, (1, num_patches))
        self.assertEqual(result.valid_fraction.shape, (1, num_patches))
        self.assertEqual(result.time_mask.dtype, torch.bool)
        self.assertEqual(result.patch_mask.dtype, torch.bool)
        torch.testing.assert_close(
            result.valid_lengths,
            torch.tensor([expected_lengths], dtype=result.valid_lengths.dtype),
        )
        torch.testing.assert_close(
            result.valid_fraction,
            torch.tensor(
                [[value / 64.0 for value in expected_lengths]],
                dtype=result.valid_fraction.dtype,
            ),
        )
        self.assertTrue(result.patch_mask.all())

        for patch_index, valid_length in enumerate(expected_lengths):
            start = patch_index * 64
            torch.testing.assert_close(
                result.patches[0, patch_index, :, :valid_length],
                signal[0, :, start : start + valid_length],
            )
            self.assertTrue(
                result.time_mask[0, patch_index, :valid_length].all()
            )
            if valid_length < 64:
                torch.testing.assert_close(
                    result.patches[0, patch_index, :, valid_length:],
                    torch.zeros(channels, 64 - valid_length),
                )
                self.assertFalse(
                    result.time_mask[0, patch_index, valid_length:].any()
                )

    def test_fixed_64_by_64_patch_counts(self):
        cases = (
            (256, [64, 64, 64, 64]),
            (130, [64, 64, 2]),
            (976, ([64] * 15) + [16]),
            (4096, [64] * 64),
        )
        for length, expected_lengths in cases:
            with self.subTest(length=length):
                self._assert_patch_layout(length, expected_lengths)

    def test_per_sample_lengths_mask_tail_and_empty_slots(self):
        signal = torch.randn(2, 3, 130)
        result = make_temporal_patches(
            signal,
            window_size=64,
            stride=64,
            lengths=torch.tensor([130, 70]),
        )

        torch.testing.assert_close(
            result.valid_lengths,
            torch.tensor([[64, 64, 2], [64, 6, 0]]),
        )
        torch.testing.assert_close(
            result.patch_mask,
            torch.tensor([[True, True, True], [True, True, False]]),
        )
        torch.testing.assert_close(
            result.valid_fraction,
            torch.tensor([[1.0, 1.0, 2.0 / 64.0], [1.0, 6.0 / 64.0, 0.0]]),
        )
        self.assertFalse(result.time_mask[1, 2].any())
        torch.testing.assert_close(
            result.patches[1, 2],
            torch.zeros_like(result.patches[1, 2]),
        )


class AdaptivePoolingAndSelectionTests(unittest.TestCase):
    def test_channel_attention_pool_probabilities_and_patch_mask(self):
        torch.manual_seed(3)
        pool = ChannelAttentionPool(
            input_dim=5,
            output_dim=7,
            num_channels=3,
            hidden_dim=11,
        )
        tokens = torch.randn(2, 4, 3, 5, requires_grad=True)
        patch_mask = torch.tensor(
            [[True, True, False, True], [True, False, False, True]]
        )
        pooled, weights = pool(tokens, patch_mask=patch_mask)

        self.assertEqual(pooled.shape, (2, 4, 7))
        self.assertEqual(weights.shape, (2, 4, 3))
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(
            weights[patch_mask].sum(dim=-1),
            torch.ones(int(patch_mask.sum())),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            weights[~patch_mask],
            torch.zeros_like(weights[~patch_mask]),
        )
        torch.testing.assert_close(
            pooled[~patch_mask],
            torch.zeros_like(pooled[~patch_mask]),
        )

        pooled.square().sum().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertGreater(tokens.grad[patch_mask].abs().sum().item(), 0.0)
        self.assertEqual(tokens.grad[~patch_mask].abs().sum().item(), 0.0)

    def test_soft_granularity_weights_top1_and_all_candidate_gradients(self):
        torch.manual_seed(5)
        selector = PatchGranularityCrossAttention(
            visual_dim=8,
            fusion_dim=8,
            num_heads=2,
            num_granularities=3,
            temperature=0.7,
            dropout=0.0,
        )
        line = torch.randn(2, 4, 8, requires_grad=True)
        graph = torch.randn(2, 4, 3, 8, requires_grad=True)
        patch_mask = torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        )
        (
            z_visual,
            alignment_visual,
            mean_weights,
            per_head_weights,
            scores,
        ) = selector(
            line,
            graph,
            patch_mask=patch_mask,
        )

        self.assertEqual(z_visual.shape, (2, 4, 8))
        self.assertEqual(mean_weights.shape, (2, 4, 3))
        self.assertEqual(per_head_weights.shape, (2, 4, 2, 3))
        self.assertEqual(scores.shape, (2, 4, 3))
        torch.testing.assert_close(alignment_visual, z_visual)
        self.assertTrue(torch.isfinite(z_visual).all())
        self.assertTrue(torch.isfinite(mean_weights).all())
        self.assertTrue(torch.isfinite(per_head_weights).all())
        self.assertTrue((mean_weights[patch_mask] > 0.0).all())
        self.assertTrue((mean_weights[patch_mask] < 1.0).all())
        torch.testing.assert_close(
            mean_weights[patch_mask].sum(dim=-1),
            torch.ones(int(patch_mask.sum())),
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            per_head_weights[patch_mask].sum(dim=-1),
            torch.ones(int(patch_mask.sum()), 2),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertGreater(
            (
                per_head_weights[patch_mask][:, 0]
                - per_head_weights[patch_mask][:, 1]
            ).abs().sum().item(),
            0.0,
        )
        torch.testing.assert_close(
            mean_weights[~patch_mask],
            torch.zeros_like(mean_weights[~patch_mask]),
        )
        torch.testing.assert_close(
            per_head_weights[~patch_mask],
            torch.zeros_like(per_head_weights[~patch_mask]),
        )

        top1 = mean_weights.argmax(dim=-1)
        self.assertEqual(top1.shape, patch_mask.shape)
        self.assertTrue(((top1 >= 0) & (top1 < 3)).all())

        loss = z_visual[patch_mask].square().sum()
        loss = loss + 0.1 * per_head_weights[patch_mask].square().sum()
        loss.backward()
        self.assertIsNotNone(line.grad)
        self.assertIsNotNone(graph.grad)
        self.assertGreater(line.grad[patch_mask].abs().sum().item(), 0.0)
        per_candidate_gradient = graph.grad[patch_mask].abs().sum(dim=(0, 2))
        self.assertTrue((per_candidate_gradient > 0.0).all())

    def test_selector_has_no_mantis_dependency(self):
        torch.manual_seed(11)
        model = PatchMindTSFusionModule(
            visual_dim=8,
            temporal_dim=6,
            num_channels=4,
            num_granularities=3,
            num_classes=2,
            fusion_dim=8,
            fusion_heads=2,
            dropout=0.0,
            classifier_hidden_dim=8,
            classifier_num_layers=2,
            channel_hidden_dim=4,
            granularity_temperature=1.0,
            alignment_dim=4,
            alignment_temperature=0.2,
        ).eval()
        line = torch.randn(1, 3, 8)
        graph = torch.randn(1, 3, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)
        fraction = torch.ones(1, 3)
        mantis_a = torch.randn(1, 3, 4, 6)
        mantis_b = 1000.0 * torch.randn(1, 3, 4, 6) + 999.0

        _, auxiliary_a = model(line, graph, mantis_a, mask, fraction)
        _, auxiliary_b = model(line, graph, mantis_b, mask, fraction)
        for name in (
            "granularity_weights",
            "granularity_weights_per_head",
            "granularity_scores",
            "granularity_top1",
        ):
            torch.testing.assert_close(
                auxiliary_a[name],
                auxiliary_b[name],
                rtol=0.0,
                atol=0.0,
            )

    def test_identical_candidates_fall_back_to_dataset_prior(self):
        torch.manual_seed(23)
        selector = PatchGranularityCrossAttention(
            visual_dim=8,
            fusion_dim=8,
            num_heads=2,
            num_granularities=3,
            temperature=1.0,
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            selector.dataset_prior_logits.copy_(
                torch.tensor([0.9, -0.2, -0.7])
            )
        candidate = torch.zeros(2, 3, 8)
        candidate[..., 0] = 1.0
        graph = candidate.unsqueeze(2).expand(-1, -1, 3, -1).clone()
        line_a = torch.randn(2, 3, 8)
        line_b = 100.0 * torch.randn(2, 3, 8) + 50.0

        visual_a, _, weights_a, _, scores_a, diagnostics_a = selector(
            line_a,
            graph,
            return_router_diagnostics=True,
        )
        visual_b, _, weights_b, _, scores_b, diagnostics_b = selector(
            line_b,
            graph,
            return_router_diagnostics=True,
        )

        expected = selector._bounded_distribution(
            selector.dataset_prior_logits.view(1, 1, 1, 3)
        ).squeeze(2).expand_as(weights_a)
        torch.testing.assert_close(weights_a, expected, rtol=0.0, atol=1e-7)
        torch.testing.assert_close(weights_b, expected, rtol=0.0, atol=1e-7)
        expected_scores = expected.log()
        torch.testing.assert_close(scores_a, expected_scores, atol=1e-7, rtol=0.0)
        torch.testing.assert_close(scores_b, expected_scores, atol=1e-7, rtol=0.0)
        torch.testing.assert_close(
            diagnostics_a["dataset_prior_weights"],
            expected,
            atol=1e-7,
            rtol=0.0,
        )
        torch.testing.assert_close(
            diagnostics_a["candidate_evidence"],
            torch.zeros_like(diagnostics_a["candidate_evidence"]),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            diagnostics_b["candidate_evidence"],
            torch.zeros_like(diagnostics_b["candidate_evidence"]),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(visual_a, visual_b, atol=1e-6, rtol=1e-6)

    def test_candidate_permutation_is_equivariant_and_output_invariant(self):
        torch.manual_seed(29)
        selector = PatchGranularityCrossAttention(
            visual_dim=8,
            fusion_dim=8,
            num_heads=2,
            num_granularities=3,
            temperature=0.2,
            dropout=0.0,
        ).eval()
        line = torch.randn(2, 4, 8)
        graph = torch.randn(2, 4, 3, 8)
        permutation = torch.tensor([2, 0, 1])

        visual, _, weights, _, scores = selector(line, graph)
        permuted_visual, _, permuted_weights, _, permuted_scores = selector(
            line,
            graph.index_select(2, permutation),
        )

        torch.testing.assert_close(
            permuted_weights,
            weights.index_select(2, permutation),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            permuted_scores,
            scores.index_select(2, permutation),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(permuted_visual, visual, atol=1e-5, rtol=1e-5)

    def test_rbf_router_gives_exact_middle_candidate_a_strict_decision_region(self):
        selector = PatchGranularityCrossAttention(
            visual_dim=3,
            fusion_dim=2,
            num_heads=1,
            num_granularities=3,
            temperature=1.0,
            dropout=0.0,
            local_mix_max=1.0,
            local_mix_init=0.999,
            global_mix_max=1.0,
            global_mix_init=0.999,
            evidence_half_saturation=1.0e-6,
        ).eval()
        projected_query = torch.zeros(1, 1, 2)
        relative_keys = torch.tensor(
            [[[[ -1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]]]
        )
        graph = torch.eye(3).view(1, 1, 3, 3)

        _, weights, _, diagnostics = selector._routing_distribution(
            projected_query,
            relative_keys,
            graph,
        )

        self.assertEqual(weights.argmax(dim=-1).item(), 1)
        self.assertGreater(weights[0, 0, 1].item(), weights[0, 0, 0].item())
        self.assertGreater(weights[0, 0, 1].item(), weights[0, 0, 2].item())
        self.assertEqual(
            diagnostics["patch_local_weights"].argmax(dim=-1).item(),
            1,
        )

    def test_bounded_router_has_finite_max_weight_and_entropy_floor(self):
        selector = PatchGranularityCrossAttention(
            visual_dim=4,
            fusion_dim=4,
            num_heads=1,
            num_granularities=3,
            temperature=1.0,
            dropout=0.0,
            score_cap=1.0,
        ).eval()
        extreme_logits = torch.tensor([[[[1000.0, -1000.0, -1000.0]]]])
        bounded_weights = selector._bounded_distribution(extreme_logits)

        candidate_count = 3
        theoretical_max = math.exp(1.0) / (
            math.exp(1.0) + (candidate_count - 1) * math.exp(-1.0)
        )
        theoretical_tail = (1.0 - theoretical_max) / (candidate_count - 1)
        theoretical_min_entropy = -(
            theoretical_max * math.log(theoretical_max)
            + (candidate_count - 1)
            * theoretical_tail
            * math.log(theoretical_tail)
        ) / math.log(candidate_count)

        self.assertLessEqual(
            bounded_weights.max().item(), theoretical_max + 1.0e-7
        )
        normalized_entropy = -torch.sum(
            bounded_weights * bounded_weights.clamp_min(1e-12).log(),
            dim=-1,
        ) / math.log(candidate_count)
        self.assertGreaterEqual(
            normalized_entropy.min().item(),
            theoretical_min_entropy - 1.0e-7,
        )

        torch.manual_seed(47)
        line = 100.0 * torch.randn(3, 5, 4)
        graph = 100.0 * torch.randn(3, 5, 3, 4)
        _, _, routed_weights, _, _ = selector(line, graph)
        routed_entropy = -torch.sum(
            routed_weights * routed_weights.clamp_min(1e-12).log(),
            dim=-1,
        ) / math.log(candidate_count)
        self.assertLessEqual(
            routed_weights.max().item(), theoretical_max + 1.0e-7
        )
        self.assertGreaterEqual(
            routed_entropy.min().item(),
            theoretical_min_entropy - 1.0e-6,
        )

    def test_hierarchical_router_diagnostics_have_masked_expected_shapes(self):
        torch.manual_seed(53)
        selector = PatchGranularityCrossAttention(
            visual_dim=8,
            fusion_dim=8,
            num_heads=2,
            num_granularities=3,
            temperature=1.0,
            dropout=0.0,
        ).eval()
        line = torch.randn(2, 4, 8)
        graph = torch.randn(2, 4, 3, 8)
        patch_mask = torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        )
        valid_fraction = torch.tensor(
            [[1.0, 1.0, 0.25, 0.0], [1.0, 0.5, 0.0, 0.0]]
        )
        (
            _,
            _,
            weights,
            per_head_weights,
            scores,
            diagnostics,
        ) = selector(
            line,
            graph,
            patch_mask=patch_mask,
            valid_fraction=valid_fraction,
            return_router_diagnostics=True,
        )

        self.assertEqual(weights.shape, (2, 4, 3))
        self.assertEqual(per_head_weights.shape, (2, 4, 2, 3))
        self.assertEqual(scores.shape, (2, 4, 3))
        for name in (
            "dataset_prior_weights",
            "sample_global_weights",
            "patch_local_weights",
        ):
            self.assertEqual(diagnostics[name].shape, (2, 4, 3), name)
            torch.testing.assert_close(
                diagnostics[name][~patch_mask],
                torch.zeros_like(diagnostics[name][~patch_mask]),
            )
        for name in (
            "candidate_evidence",
            "effective_global_mix",
            "effective_local_mix",
        ):
            self.assertEqual(diagnostics[name].shape, (2, 4), name)
            torch.testing.assert_close(
                diagnostics[name][~patch_mask],
                torch.zeros_like(diagnostics[name][~patch_mask]),
            )
        self.assertEqual(diagnostics["global_mix"].shape, ())
        self.assertEqual(diagnostics["local_mix"].shape, ())
        self.assertFalse(diagnostics["candidate_evidence"].requires_grad)
        torch.testing.assert_close(
            per_head_weights[patch_mask].mean(dim=1),
            weights[patch_mask],
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            scores[patch_mask].softmax(dim=-1),
            weights[patch_mask],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_router_learns_balanced_content_dependent_rule(self):
        torch.manual_seed(37)
        selector = PatchGranularityCrossAttention(
            visual_dim=6,
            fusion_dim=6,
            num_heads=1,
            num_granularities=3,
            temperature=1.0,
            dropout=0.0,
        )
        prototypes = torch.randn(3, 6)

        def make_split(repeats, noise_scale):
            targets = torch.arange(3).repeat_interleave(repeats)
            order = torch.randperm(len(targets))
            targets = targets[order]
            line = prototypes.index_select(0, targets)
            line = line + noise_scale * torch.randn_like(line)
            graph = prototypes.view(1, 1, 3, 6).expand(
                len(targets), 1, -1, -1
            ).clone()
            graph = graph + 0.02 * torch.randn_like(graph)
            return line.unsqueeze(1), graph, targets

        train_line, train_graph, train_targets = make_split(48, 0.08)
        optimizer = torch.optim.Adam(selector.parameters(), lr=0.03)
        selector.train()
        for _ in range(160):
            optimizer.zero_grad(set_to_none=True)
            _, _, weights, _, _ = selector(train_line, train_graph)
            loss = torch.nn.functional.nll_loss(
                weights[:, 0].clamp_min(1e-8).log(),
                train_targets,
            )
            loss.backward()
            optimizer.step()

        test_line, test_graph, test_targets = make_split(24, 0.08)
        selector.eval()
        with torch.no_grad():
            _, _, weights, _, _ = selector(test_line, test_graph)
        probabilities = weights[:, 0]
        predictions = probabilities.argmax(dim=-1)
        accuracy = (predictions == test_targets).float().mean()
        correct_weight = probabilities[
            torch.arange(len(test_targets)),
            test_targets,
        ].mean()
        entropy = -torch.sum(
            probabilities * probabilities.clamp_min(1e-8).log(),
            dim=-1,
        ).mean() / math.log(3)
        usage = torch.bincount(predictions, minlength=3).float() / len(predictions)

        self.assertGreaterEqual(accuracy.item(), 0.95)
        # v4 intentionally caps confidence and shrinks local routing toward
        # sample-global and dataset-level distributions.
        self.assertGreaterEqual(correct_weight.item(), 0.55)
        self.assertLessEqual(entropy.item(), 0.95)
        self.assertTrue(((usage >= 0.20) & (usage <= 0.45)).all())

    def test_query_shuffle_and_constant_controls_change_only_local_routing(self):
        selector = PatchGranularityCrossAttention(
            visual_dim=4,
            fusion_dim=4,
            num_heads=1,
            num_granularities=3,
            temperature=1.0,
            dropout=0.0,
            local_mix_max=1.0,
            local_mix_init=0.9,
            evidence_half_saturation=1.0e-4,
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
        graph = candidates.view(1, 1, 3, 4).expand(1, 3, -1, -1).clone()
        line = torch.stack((candidates[0], candidates[1], candidates[2])).unsqueeze(0)
        shuffled_line = line.roll(1, dims=1)
        constant_line = line.mean(dim=1, keepdim=True).expand_as(line)

        _, _, weights, _, _ = selector(line, graph)
        _, _, shuffled_weights, _, _ = selector(shuffled_line, graph)
        _, _, constant_weights, _, _ = selector(constant_line, graph)

        self.assertGreater((weights - shuffled_weights).abs().sum().item(), 0.1)
        torch.testing.assert_close(
            constant_weights[:, 0],
            constant_weights[:, 1],
            atol=1e-7,
            rtol=0.0,
        )
        torch.testing.assert_close(
            constant_weights[:, 1],
            constant_weights[:, 2],
            atol=1e-7,
            rtol=0.0,
        )

    def test_candidate_convergence_smoothly_approaches_uniform(self):
        selector = PatchGranularityCrossAttention(
            visual_dim=4,
            fusion_dim=4,
            num_heads=1,
            num_granularities=3,
            temperature=0.2,
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            selector.query_projection.weight.copy_(torch.eye(4))
            selector.key_projection.weight.copy_(torch.eye(4))
        line = torch.tensor([[[2.0, -1.0, 0.5, -1.5]]])
        base = torch.tensor([1.5, -0.7, 0.3, -1.1])
        residuals = torch.tensor(
            [
                [0.8, -0.4, 0.2, -0.6],
                [-0.3, 0.9, -0.7, 0.1],
                [-0.5, -0.5, 0.5, 0.5],
            ]
        )
        deviations = []
        for scale in (1.0e-1, 1.0e-3, 1.0e-5):
            graph = (base.unsqueeze(0) + scale * residuals).view(1, 1, 3, 4)
            _, _, weights, _, _ = selector(line, graph)
            deviations.append(
                (weights - torch.full_like(weights, 1.0 / 3.0)).abs().sum()
            )
        self.assertGreater(deviations[0].item(), deviations[1].item())
        self.assertGreater(deviations[1].item(), deviations[2].item())
        self.assertLess(deviations[2].item(), 1.0e-4)

    def test_task_loss_reaches_query_key_and_value_paths_without_weight_loss(self):
        torch.manual_seed(31)
        model = PatchMindTSFusionModule(
            visual_dim=8,
            temporal_dim=6,
            num_channels=2,
            num_granularities=3,
            num_classes=2,
            fusion_dim=8,
            fusion_heads=2,
            dropout=0.0,
            classifier_hidden_dim=8,
            classifier_num_layers=2,
            channel_hidden_dim=4,
            granularity_temperature=0.2,
            alignment_dim=4,
            alignment_temperature=0.2,
        )
        line = torch.randn(4, 3, 8)
        graph = torch.randn(4, 3, 3, 8)
        mantis = torch.randn(4, 3, 2, 6)
        mask = torch.ones(4, 3, dtype=torch.bool)
        fraction = torch.ones(4, 3)
        labels = torch.tensor([0, 1, 0, 1])

        logits, _ = model(line, graph, mantis, mask, fraction)
        torch.nn.functional.cross_entropy(logits, labels).backward()

        for name in (
            "query_projection",
            "key_projection",
            "value_projection",
            "output_projection",
        ):
            gradient = getattr(model.granularity_attention, name).weight.grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0.0, name)

    def test_alignment_loss_cannot_train_query_or_key_router(self):
        torch.manual_seed(41)
        model = PatchMindTSFusionModule(
            visual_dim=8,
            temporal_dim=6,
            num_channels=2,
            num_granularities=3,
            num_classes=2,
            fusion_dim=8,
            fusion_heads=2,
            dropout=0.0,
            classifier_hidden_dim=8,
            classifier_num_layers=2,
            channel_hidden_dim=4,
            granularity_temperature=0.2,
            alignment_dim=4,
            alignment_temperature=0.2,
        )
        line = torch.randn(4, 3, 8)
        graph = torch.randn(4, 3, 3, 8)
        mantis = torch.randn(4, 3, 2, 6)
        mask = torch.ones(4, 3, dtype=torch.bool)
        fraction = torch.ones(4, 3)

        _, auxiliary = model(line, graph, mantis, mask, fraction)
        auxiliary["alignment_loss"].backward()

        for name in ("query_projection", "key_projection"):
            gradient = getattr(model.granularity_attention, name).weight.grad
            self.assertTrue(
                gradient is None or gradient.abs().sum().item() == 0.0,
                name,
            )
        for name in ("value_projection", "output_projection"):
            gradient = getattr(model.granularity_attention, name).weight.grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0.0, name)

    def test_usage_ema_updates_only_for_training_soft_route(self):
        torch.manual_seed(43)
        model = PatchMindTSFusionModule(
            visual_dim=8,
            temporal_dim=6,
            num_channels=2,
            num_granularities=3,
            num_classes=2,
            fusion_dim=8,
            fusion_heads=2,
            dropout=0.0,
            classifier_hidden_dim=8,
            classifier_num_layers=2,
            channel_hidden_dim=4,
            granularity_temperature=0.2,
            alignment_dim=4,
            alignment_temperature=0.2,
            granularity_usage_ema_decay=0.5,
        )
        line = torch.randn(2, 3, 8)
        graph = torch.randn(2, 3, 3, 8)
        mantis = torch.randn(2, 3, 2, 6)
        mask = torch.ones(2, 3, dtype=torch.bool)
        fraction = torch.ones(2, 3)

        self.assertFalse(bool(model.granularity_usage_ema_initialized))
        model.train()
        _, auxiliary = model(line, graph, mantis, mask, fraction)
        torch.testing.assert_close(
            model.granularity_usage_ema,
            auxiliary["granularity_marginal"].detach(),
        )
        self.assertTrue(bool(model.granularity_usage_ema_initialized))

        first_usage = model.granularity_usage_ema.clone()
        _, second_auxiliary = model(line.roll(1, dims=1), graph, mantis, mask, fraction)
        expected = (
            0.5 * first_usage
            + 0.5 * second_auxiliary["granularity_marginal"].detach()
        )
        torch.testing.assert_close(model.granularity_usage_ema, expected)

        before_override = model.granularity_usage_ema.clone()
        uniform = torch.full((2, 3, 3), 1.0 / 3.0)
        model(
            line,
            graph,
            mantis,
            mask,
            fraction,
            granularity_weights_override=uniform,
        )
        torch.testing.assert_close(model.granularity_usage_ema, before_override)

        model.eval()
        model(line, graph, mantis, mask, fraction)
        torch.testing.assert_close(model.granularity_usage_ema, before_override)

    def test_uniform_router_is_exact_and_has_no_query_key_gradient_or_ema_update(self):
        torch.manual_seed(59)
        model = PatchMindTSFusionModule(
            visual_dim=8,
            temporal_dim=6,
            num_channels=2,
            num_granularities=3,
            num_classes=2,
            fusion_dim=8,
            fusion_heads=2,
            dropout=0.0,
            classifier_hidden_dim=8,
            classifier_num_layers=2,
            channel_hidden_dim=4,
            granularity_temperature=1.0,
            alignment_dim=4,
            alignment_temperature=0.2,
            granularity_router_mode="uniform",
            granularity_usage_ema_decay=0.5,
        )
        line = torch.randn(3, 4, 8)
        graph = torch.randn(3, 4, 3, 8)
        mantis = torch.randn(3, 4, 2, 6)
        mask = torch.tensor(
            [[True, True, True, True], [True, True, True, False], [True, False, False, False]]
        )
        fraction = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.25, 0.0], [0.5, 0.0, 0.0, 0.0]]
        )
        labels = torch.tensor([0, 1, 0])
        query_before = model.granularity_attention.query_projection.weight.detach().clone()
        key_before = model.granularity_attention.key_projection.weight.detach().clone()
        ema_before = model.granularity_usage_ema.clone()
        ema_initialized_before = model.granularity_usage_ema_initialized.clone()

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        logits, auxiliary = model(line, graph, mantis, mask, fraction)
        expected = torch.full(
            (int(mask.sum()), 3),
            1.0 / 3.0,
            dtype=auxiliary["granularity_weights"].dtype,
        )
        torch.testing.assert_close(
            auxiliary["granularity_weights"][mask],
            expected,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            auxiliary["granularity_weights"][~mask],
            torch.zeros_like(auxiliary["granularity_weights"][~mask]),
            rtol=0.0,
            atol=0.0,
        )
        loss = (
            torch.nn.functional.cross_entropy(logits, labels)
            + auxiliary["alignment_loss"]
            + auxiliary["granularity_balance_loss"]
            + auxiliary["granularity_entropy_loss"]
            + auxiliary["granularity_mix_shrinkage_loss"]
            + auxiliary["granularity_prior_kl_loss"]
        )
        loss.backward()
        for name in ("query_projection", "key_projection"):
            gradient = getattr(model.granularity_attention, name).weight.grad
            self.assertTrue(
                gradient is None or gradient.abs().sum().item() == 0.0,
                name,
            )
        optimizer.step()

        torch.testing.assert_close(
            model.granularity_attention.query_projection.weight,
            query_before,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            model.granularity_attention.key_projection.weight,
            key_before,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            model.granularity_usage_ema,
            ema_before,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            model.granularity_usage_ema_initialized,
            ema_initialized_before,
            rtol=0.0,
            atol=0.0,
        )


class GranularityRegularizationTests(unittest.TestCase):
    def test_entropy_floor_penalty_pushes_a_peaked_route_toward_higher_entropy(self):
        mask = torch.ones(1, 1, dtype=torch.bool)
        fraction = torch.ones(1, 1)
        peaked = torch.tensor(
            [[[0.98, 0.01, 0.01]]],
            requires_grad=True,
        )
        moderate = torch.tensor([[[0.70, 0.15, 0.15]]])
        uniform = torch.full((1, 1, 3), 1.0 / 3.0)

        _, peaked_floor, peaked_ceiling, _ = (
            masked_granularity_regularization_terms(
                peaked,
                mask,
                fraction,
                usage_floor=0.0,
                entropy_floor=0.55,
                entropy_ceiling=1.0,
                return_entropy_components=True,
            )
        )
        _, moderate_floor, _, _ = masked_granularity_regularization_terms(
            moderate,
            mask,
            fraction,
            usage_floor=0.0,
            entropy_floor=0.55,
            entropy_ceiling=1.0,
            return_entropy_components=True,
        )
        _, uniform_floor, _, _ = masked_granularity_regularization_terms(
            uniform,
            mask,
            fraction,
            usage_floor=0.0,
            entropy_floor=0.55,
            entropy_ceiling=1.0,
            return_entropy_components=True,
        )

        self.assertGreater(peaked_floor.item(), moderate_floor.item())
        torch.testing.assert_close(moderate_floor, torch.tensor(0.0))
        torch.testing.assert_close(uniform_floor, torch.tensor(0.0))
        torch.testing.assert_close(peaked_ceiling, torch.tensor(0.0))

        peaked_floor.backward()
        # A gradient-descent step lowers the dominant mass and raises both
        # tails, which is the intended anti-collapse direction.
        self.assertGreater(peaked.grad[0, 0, 0].item(), 0.0)
        self.assertLess(peaked.grad[0, 0, 1].item(), 0.0)
        self.assertLess(peaked.grad[0, 0, 2].item(), 0.0)

    def test_usage_floor_and_entropy_ceiling_only_activate_past_limits(self):
        mask = torch.ones(1, 3, dtype=torch.bool)
        fraction = torch.ones(1, 3)

        uniform = torch.full((1, 3, 3), 1.0 / 3.0)
        usage, entropy, marginal = masked_granularity_regularization_terms(
            uniform,
            mask,
            fraction,
        )
        torch.testing.assert_close(usage, torch.tensor(0.0), atol=1e-7, rtol=0.0)
        torch.testing.assert_close(entropy, torch.tensor(0.01), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(marginal, torch.full((3,), 1.0 / 3.0))

        balanced_one_hot = torch.eye(3).unsqueeze(0)
        usage, entropy, marginal = masked_granularity_regularization_terms(
            balanced_one_hot,
            mask,
            fraction,
        )
        torch.testing.assert_close(usage, torch.tensor(0.0), atol=1e-7, rtol=0.0)
        torch.testing.assert_close(entropy, torch.tensor(0.0), atol=1e-7, rtol=0.0)
        torch.testing.assert_close(marginal, torch.full((3,), 1.0 / 3.0))

        collapsed = torch.zeros(1, 3, 3)
        collapsed[..., 0] = 1.0
        usage, entropy, marginal = masked_granularity_regularization_terms(
            collapsed,
            mask,
            fraction,
        )
        torch.testing.assert_close(
            usage,
            torch.tensor(2.0 / 3.0),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(entropy, torch.tensor(0.0), atol=1e-7, rtol=0.0)
        torch.testing.assert_close(marginal, torch.tensor([1.0, 0.0, 0.0]))

        usage, entropy, _ = masked_granularity_regularization_terms(
            balanced_one_hot,
            mask,
            fraction,
            usage_reference=torch.tensor([0.90, 0.06, 0.04]),
            usage_floor=0.05,
            entropy_ceiling=1.0,
        )
        torch.testing.assert_close(
            usage,
            torch.tensor((0.01 ** 2) / (3.0 * 0.05 ** 2)),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(entropy, torch.tensor(0.0), atol=1e-7, rtol=0.0)

    def test_partial_tail_has_fractional_mass_and_invalid_patch_is_ignored(self):
        weights = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [99.0, -7.0, 3.0]]]
        )
        mask = torch.tensor([[True, True, False]])
        fraction = torch.tensor([[1.0, 0.25, 0.0]])

        _, _, marginal = masked_granularity_regularization_terms(
            weights,
            mask,
            fraction,
        )
        torch.testing.assert_close(marginal, torch.tensor([0.8, 0.2, 0.0]))


class AlignmentLossTests(unittest.TestCase):
    def _make_loss(self):
        torch.manual_seed(17)
        return MaskedIntraSampleInfoNCE(
            temporal_dim=6,
            visual_dim=8,
            projection_dim=5,
            temperature=0.2,
        )

    def test_batch_loss_equals_mean_of_independent_samples(self):
        loss_module = self._make_loss()
        temporal = torch.randn(2, 3, 6)
        visual = torch.randn(2, 3, 8)
        patch_mask = torch.ones(2, 3, dtype=torch.bool)

        combined = loss_module(temporal, visual, patch_mask)
        separate = torch.stack(
            [
                loss_module(
                    temporal[index : index + 1],
                    visual[index : index + 1],
                    patch_mask[index : index + 1],
                )
                for index in range(2)
            ]
        ).mean()
        torch.testing.assert_close(combined, separate, rtol=1e-6, atol=1e-6)

    def test_single_patch_and_masked_positions_are_safe(self):
        loss_module = self._make_loss()
        single_temporal = torch.randn(3, 1, 6, requires_grad=True)
        single_visual = torch.randn(3, 1, 8, requires_grad=True)
        single_mask = torch.ones(3, 1, dtype=torch.bool)
        single_loss = loss_module(
            single_temporal,
            single_visual,
            single_mask,
        )
        self.assertTrue(torch.isfinite(single_loss))
        torch.testing.assert_close(single_loss, torch.zeros_like(single_loss))

        temporal = torch.randn(1, 3, 6)
        visual = torch.randn(1, 3, 8)
        patch_mask = torch.tensor([[True, True, False]])
        valid_fraction = torch.tensor([[1.0, 0.5, 0.0]])
        reference = loss_module(
            temporal,
            visual,
            patch_mask,
            valid_fraction=valid_fraction,
        )
        changed_temporal = temporal.clone()
        changed_visual = visual.clone()
        changed_temporal[:, 2] = 1.0e6
        changed_visual[:, 2] = -1.0e6
        changed = loss_module(
            changed_temporal,
            changed_visual,
            patch_mask,
            valid_fraction=valid_fraction,
        )
        self.assertTrue(torch.isfinite(reference))
        torch.testing.assert_close(reference, changed, rtol=0.0, atol=0.0)

    def test_zero_logits_have_unit_random_baseline_for_different_patch_counts(self):
        loss_module = MaskedIntraSampleInfoNCE(
            temporal_dim=3,
            visual_dim=3,
            projection_dim=3,
            temperature=0.2,
        )
        with torch.no_grad():
            loss_module.temporal_projection.weight.zero_()
            loss_module.temporal_projection.bias.fill_(1.0)
            loss_module.visual_projection.weight.zero_()
            loss_module.visual_projection.bias.fill_(1.0)
        for patch_count in (2, 4, 8):
            temporal = torch.randn(1, patch_count, 3)
            visual = torch.randn(1, patch_count, 3)
            mask = torch.ones(1, patch_count, dtype=torch.bool)
            loss = loss_module(temporal, visual, mask)
            torch.testing.assert_close(loss, torch.tensor(1.0), atol=1e-6, rtol=0.0)


class LineQueryCenterTests(unittest.TestCase):
    def test_center_uses_only_training_indices_mask_and_valid_time(self):
        line = torch.tensor(
            [
                [
                    [2.0, 0.0, -1.0, 1.0],
                    [-1.0, 3.0, 0.0, 2.0],
                    [100.0, -90.0, 80.0, -70.0],
                ],
                [
                    [1000.0, -800.0, 600.0, -400.0],
                    [-900.0, 700.0, -500.0, 300.0],
                    [800.0, -600.0, 400.0, -200.0],
                ],
            ]
        )
        bundle = {
            "line_tokens": line.clone(),
            "patch_mask": torch.tensor(
                [[True, True, False], [True, True, True]]
            ),
            "valid_fraction": torch.tensor(
                [[1.0, 0.25, 0.0], [1.0, 1.0, 1.0]]
            ),
        }
        center = compute_line_query_center(bundle, [0], chunk_size=1)
        normalized = torch.nn.functional.layer_norm(line[0, :2], (4,))
        expected = (normalized[0] + 0.25 * normalized[1]) / 1.25
        torch.testing.assert_close(center, expected, atol=1e-6, rtol=1e-6)

        bundle["line_tokens"][1].mul_(1.0e6).add_(12345.0)
        changed = compute_line_query_center(bundle, [0], chunk_size=1)
        torch.testing.assert_close(changed, center, atol=0.0, rtol=0.0)


class QueryCounterfactualTests(unittest.TestCase):
    def test_shuffle_respects_sample_length_groups_and_reports_zero_coverage(self):
        line = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        lengths = torch.tensor([[64, 64, 16, 0], [64, 16, 0, 0]])
        mask = lengths > 0
        fraction = lengths.float() / 64.0
        shuffled, constant, permutation, moved = _make_query_counterfactuals(
            line,
            lengths,
            mask,
            fraction,
        )

        torch.testing.assert_close(permutation[0], torch.tensor([1, 0, 2, 3]))
        torch.testing.assert_close(permutation[1], torch.tensor([0, 1, 2, 3]))
        torch.testing.assert_close(moved[0], torch.tensor([True, True, False, False]))
        self.assertFalse(moved[1].any())
        torch.testing.assert_close(shuffled[0, 0], line[0, 1])
        torch.testing.assert_close(shuffled[0, 1], line[0, 0])
        expected_mean = (
            line[0, 0] + line[0, 1] + 0.25 * line[0, 2]
        ) / 2.25
        torch.testing.assert_close(constant[0, 0], expected_mean)
        torch.testing.assert_close(constant[0, 3], expected_mean)


class PatchPoolingTests(unittest.TestCase):
    def test_partial_tail_uses_valid_fraction_weight(self):
        patch_features = torch.tensor([[[2.0], [10.0]]])
        patch_mask = torch.tensor([[True, True]])
        valid_fraction = torch.tensor([[1.0, 0.25]])
        pooled = valid_fraction_weighted_pool(
            patch_features,
            patch_mask,
            valid_fraction,
        )
        torch.testing.assert_close(pooled, torch.tensor([[3.6]]))


class PatchFeatureCacheTests(unittest.TestCase):
    def _make_bundle(self):
        patch_mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, False],
                [True, True, False, False],
            ]
        )
        return {
            "line_tokens": torch.randn(3, 4, 8),
            "graph_tokens": torch.randn(3, 4, 3, 8),
            "mantis_channel_tokens": torch.randn(3, 4, 2, 5),
            "patch_mask": patch_mask,
            "valid_fraction": torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 0.25, 0.0],
                    [1.0, 0.5, 0.0, 0.0],
                ]
            ),
            "valid_lengths": torch.tensor(
                [[64, 64, 64, 64], [64, 64, 16, 0], [64, 32, 0, 0]]
            ),
        }

    def test_cache_round_trip_and_metadata_rejection(self):
        torch.manual_seed(23)
        bundle = self._make_bundle()
        labels = torch.tensor([0, 2, 1])
        signature = "patch-mindts-test-signature-v1"

        with tempfile.TemporaryDirectory(prefix="patch_mindts_cache_") as temp_dir:
            cache_path = Path(temp_dir) / "features.pt"
            save_patch_feature_cache(
                cache_path,
                bundle,
                labels,
                signature,
            )
            loaded = load_patch_feature_cache(
                cache_path,
                labels,
                signature,
                expected_num_granularities=3,
            )
            self.assertEqual(set(loaded), set(bundle))
            for key, expected in bundle.items():
                torch.testing.assert_close(loaded[key], expected)

            with self.assertRaises((ValueError, RuntimeError)):
                load_patch_feature_cache(
                    cache_path,
                    labels,
                    "different-signature",
                    expected_num_granularities=3,
                )
            with self.assertRaises((ValueError, RuntimeError)):
                load_patch_feature_cache(
                    cache_path,
                    labels,
                    signature,
                    expected_num_granularities=2,
                )
            with self.assertRaises((ValueError, RuntimeError)):
                load_patch_feature_cache(
                    cache_path,
                    torch.tensor([0, 1, 2]),
                    signature,
                    expected_num_granularities=3,
                )


class SubjectCheckpointTests(unittest.TestCase):
    def test_subject_aggregation_aligns_shuffled_indices_and_equalizes_subjects(self):
        sample_subject_ids = np.asarray(["a", "a", "a", "b"])
        sample_indices = np.asarray([3, 1, 0, 2])
        y_true = np.asarray([1, 0, 0, 0])
        y_score = np.asarray(
            [
                [0.10, 0.90],
                [0.70, 0.30],
                [0.90, 0.10],
                [0.80, 0.20],
            ]
        )
        metrics, details = _aggregate_subject_predictions(
            y_true,
            y_score,
            sample_indices,
            sample_subject_ids,
            classes=np.asarray([0, 1]),
        )
        np.testing.assert_array_equal(details["subject_id"], ["a", "b"])
        np.testing.assert_array_equal(details["subject_window_count"], [3, 1])
        np.testing.assert_allclose(
            details["subject_y_score"],
            [[0.80, 0.20], [0.10, 0.90]],
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertTrue(math.isfinite(metrics["macro_log_loss"]))

    def test_subject_aggregation_rejects_inconsistent_labels(self):
        with self.assertRaisesRegex(ValueError, "inconsistent labels"):
            _aggregate_subject_predictions(
                np.asarray([0, 1]),
                np.asarray([[0.8, 0.2], [0.2, 0.8]]),
                np.asarray([0, 1]),
                np.asarray(["same", "same"]),
                classes=np.asarray([0, 1]),
            )

    def test_checkpoint_metric_requires_available_disjoint_subjects(self):
        train_ids = np.asarray(["train_a", "train_b"])
        validation_ids = np.asarray(["val_a", "val_b"])
        indices = [0, 1]
        self.assertEqual(
            _resolve_checkpoint_metric(
                "auto", train_ids, validation_ids, indices, indices
            ),
            "subject_macro_f1",
        )
        self.assertEqual(
            _resolve_checkpoint_metric(
                "auto", train_ids, train_ids, indices, indices
            ),
            "window_macro_f1",
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            _resolve_checkpoint_metric(
                "subject_macro_f1", train_ids, train_ids, indices, indices
            )

    def test_subject_selection_key_uses_subject_log_loss_not_window_f1(self):
        first = {
            "macro_f1": 0.99,
            "subject_macro_f1": 0.50,
            "subject_macro_log_loss": 0.40,
        }
        second = {
            "macro_f1": 0.10,
            "subject_macro_f1": 0.50,
            "subject_macro_log_loss": 0.30,
        }
        self.assertGreater(
            _checkpoint_selection_key(second, "subject_macro_f1"),
            _checkpoint_selection_key(first, "subject_macro_f1"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
