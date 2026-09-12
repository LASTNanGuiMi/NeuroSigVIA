"""Checks for the no-fusion ablation and legacy zero-visual checkpoints.

Run: python -m unittest discover -s tests -p 'test_numeric_ablation.py' -v
These tests use synthetic data only and never inspect experiment test labels.
"""

import inspect
import io
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adaptive_graph_training import NeuroSigVIAClassifier
from src.numeric_ablation import (
    ARCHITECTURE,
    LEGACY_ARCHITECTURE,
    NumericOnlyClassifier,
    ZeroVisualSlotClassifier,
    numeric_model_from_checkpoint,
)
from src.utils import set_random_seed


CONFIG = dict(
    visual_dim=8,
    temporal_dim=6,
    num_channels=4,
    num_classes=3,
    fusion_dim=128,
    fusion_heads=8,
    classifier_hidden_dim=32,
    classifier_num_layers=2,
    channel_hidden_dim=16,
    alignment_dim=8,
    dropout=0.1,
)


class ZeroVisualOutput(nn.Module):
    """Intervene only at the visual-fusion output of the original full path."""

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, line, graph, *, patch_mask, return_attention_weights=False):
        zeros = line.new_zeros(*line.shape[:2], self.feature_dim)
        return (zeros, None) if return_attention_weights else zeros


def synthetic_batch():
    generator = torch.Generator().manual_seed(990)
    tokens = torch.randn(3, 4, 4, 6, generator=generator)
    fractions = torch.tensor([
        [1.0, 0.5, 0.0, 0.0],
        [1.0, 1.0, 0.25, 0.0],
        [0.25, 0.0, 0.0, 0.0],
    ])
    return tokens, fractions > 0, fractions


class SingleThreadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)


class ZeroVisualSlotTests(SingleThreadTests):
    def fresh_numeric(self, seed=42):
        set_random_seed(seed)
        return ZeroVisualSlotClassifier.from_config(CONFIG)

    def test_shared_initial_weights_and_rng_match_full_constructor_all_seeds(self):
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                set_random_seed(seed)
                full = NeuroSigVIAClassifier.from_config(CONFIG)
                expected_rng = torch.get_rng_state().clone()
                numeric = self.fresh_numeric(seed)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                expected = {
                    key: value
                    for key, value in full.fusion.state_dict().items()
                    if key.split('.', 1)[0] in {'channel_pool', 'temporal_visual_fusion', 'classifier'}
                }
                self.assertEqual(set(numeric.state_dict()), set(expected))
                self.assertEqual(numeric.model_config, full.get_config())
                for name, tensor in numeric.state_dict().items():
                    torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0, msg=f'{seed}: {name}')

    def test_original_full_fusion_with_zero_visual_output_is_exactly_equivalent(self):
        tokens, mask, fractions = synthetic_batch()
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                set_random_seed(seed)
                full = NeuroSigVIAClassifier.from_config(CONFIG).eval()
                full.fusion.visual_cross_attention = ZeroVisualOutput(CONFIG['fusion_dim'])
                numeric = self.fresh_numeric(seed).eval()
                line = torch.randn(3, 4, CONFIG['visual_dim'])
                graph = torch.randn(3, 4, 49, CONFIG['visual_dim'])
                with torch.no_grad():
                    expected, _ = full.fusion(line, graph, tokens, mask, fractions)
                    actual = numeric(tokens, mask, fractions)
                    unrelated_images, _ = full.fusion(100 * line + 17, -13 * graph, tokens, mask, fractions)
                self.assertEqual(tuple(actual.shape), (3, CONFIG['num_classes']))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(unrelated_images, actual, rtol=0, atol=0)

    def test_no_visual_gate_or_alignment_parameters_but_exact_two_slot_head(self):
        model = self.fresh_numeric()
        self.assertEqual(set(dict(model.named_children())), {'channel_pool', 'temporal_visual_fusion', 'classifier'})
        self.assertEqual(tuple(inspect.signature(model.forward).parameters), ('mantis_channel_tokens', 'patch_mask', 'valid_fraction'))
        self.assertEqual(model.temporal_visual_fusion.modal_interaction, 'concat_attn')
        self.assertEqual(model.temporal_visual_fusion.branch_dims, [128, 128])
        self.assertEqual(model.temporal_visual_fusion.output_dim, 256)
        self.assertEqual(model.classifier.network[0].in_features, 256)
        for name in dict(model.named_parameters()):
            self.assertFalse(any(forbidden in name for forbidden in ('renderer', 'visual_cross_attention', 'alignment', 'gate')), name)
        # The original visual projection bias is deliberately retained: this is
        # a sample-independent slot, not a claim that every attention token is zero.
        projection = model.temporal_visual_fusion.projections[0]
        self.assertIsNotNone(projection.bias)
        torch.testing.assert_close(projection(torch.zeros(2, 128)), projection.bias.expand(2, -1), rtol=0, atol=0)

    def test_classification_gradient_reaches_numeric_fusion_and_head_only(self):
        model = self.fresh_numeric().train()
        tokens, mask, fractions = synthetic_batch()
        tokens.requires_grad_()
        loss = F.cross_entropy(model(tokens, mask, fractions), torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(tokens.grad[mask].abs().sum().item(), 0)
        self.assertEqual(tokens.grad[~mask].abs().sum().item(), 0)
        for module_name in ('channel_pool', 'temporal_visual_fusion', 'classifier'):
            gradients = [p.grad for p in getattr(model, module_name).parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients), module_name)
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0, module_name)
        # A zero visual input gives no gradient to its input projection weights,
        # while the bias can learn as part of the unchanged fusion mechanism.
        self.assertEqual(model.temporal_visual_fusion.projections[0].weight.grad.abs().sum().item(), 0)

    def test_duration_weighting_and_masked_patch_values(self):
        model = self.fresh_numeric().eval()
        tokens, mask, fractions = synthetic_batch()
        captured = []
        hook = model.classifier.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            with torch.no_grad():
                expected_logits = model(tokens, mask, fractions)
                temporal, _ = model.channel_pool(tokens, patch_mask=mask)
                expected_pooled = []
                for sample in range(len(tokens)):
                    valid = temporal[sample, mask[sample]]
                    patch_features = model.temporal_visual_fusion([torch.zeros_like(valid), valid])
                    durations = fractions[sample, mask[sample]]
                    expected_pooled.append((patch_features * durations[:, None]).sum(0) / durations.sum())
                torch.testing.assert_close(captured[0], torch.stack(expected_pooled), rtol=1e-5, atol=1e-7)
                changed_padding = tokens.clone()
                changed_padding[~mask] = 12345
                torch.testing.assert_close(model(changed_padding, mask, fractions), expected_logits, rtol=0, atol=0)
                changed_durations = fractions.clone()
                changed_durations[0, 1] = 1.0
                changed_logits = model(tokens, mask, changed_durations)
                self.assertGreater((changed_logits[0] - expected_logits[0]).abs().max().item(), 1e-6)
                torch.testing.assert_close(changed_logits[1:], expected_logits[1:], rtol=0, atol=0)
        finally:
            hook.remove()

    def test_invalid_patch_metadata_is_rejected(self):
        model = self.fresh_numeric()
        tokens, mask, fractions = synthetic_batch()
        invalid = []
        for value in (float('nan'), float('inf'), -0.5, 1.5, 0.0):
            changed = fractions.clone()
            changed[0, 0] = value
            invalid.append((mask, changed))
        empty_mask, empty_fractions = mask.clone(), fractions.clone()
        empty_mask[0] = False
        empty_fractions[0] = 0
        invalid.append((empty_mask, empty_fractions))
        for changed_mask, changed_fractions in invalid:
            with self.subTest(fractions=changed_fractions[0].tolist()):
                with self.assertRaises(ValueError):
                    model(tokens, changed_mask, changed_fractions)
        with self.assertRaisesRegex(ValueError, 'shape'):
            model(tokens, mask[:, :2], fractions)
        with self.assertRaisesRegex(ValueError, 'Mantis tokens'):
            model(tokens[:, 0], mask, fractions)

    def test_strict_numeric_checkpoint_roundtrip(self):
        model = self.fresh_numeric().eval()
        tokens, mask, fractions = synthetic_batch()
        with torch.no_grad():
            expected = model(tokens, mask, fractions)
        serialized = io.BytesIO()
        torch.save({'model_config': model.model_config, 'model_state_dict': model.state_dict()}, serialized)
        serialized.seek(0)
        payload = torch.load(serialized, map_location='cpu', weights_only=True)
        restored = ZeroVisualSlotClassifier.from_config(payload['model_config']).eval()
        restored.load_state_dict(payload['model_state_dict'], strict=True)
        with torch.no_grad():
            actual = restored(tokens, mask, fractions)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class NoFusionTests(SingleThreadTests):
    def fresh_numeric(self, seed=42):
        set_random_seed(seed)
        return NumericOnlyClassifier.from_config(CONFIG)

    def test_default_has_no_fusion_and_uses_single_numeric_width(self):
        model = self.fresh_numeric()
        self.assertEqual(ARCHITECTURE, 'neurosigvia_numeric_only_no_fusion_v2')
        self.assertEqual(set(dict(model.named_children())), {'channel_pool', 'classifier'})
        self.assertEqual(tuple(inspect.signature(model.forward).parameters),
                         ('mantis_channel_tokens', 'patch_mask', 'valid_fraction'))
        self.assertEqual(model.classifier_input_dim, CONFIG['fusion_dim'])
        first = model.classifier.network[0]
        self.assertEqual((first.in_features, first.out_features), (128, 32))
        self.assertIsInstance(model.classifier.network[1], nn.ReLU)
        self.assertIsInstance(model.classifier.network[2], nn.Dropout)
        self.assertEqual(model.classifier.network[2].p, CONFIG['dropout'])
        for name in dict(model.named_parameters()):
            self.assertFalse(any(part in name for part in
                                 ('fusion', 'visual', 'alignment', 'renderer', 'gate')), name)

    def test_shared_initialization_and_new_projection_are_seed_reproducible(self):
        first_weights = []
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                set_random_seed(seed)
                full = NeuroSigVIAClassifier.from_config(CONFIG)
                numeric = self.fresh_numeric(seed)
                self.assertEqual(numeric.model_config, full.get_config())
                for name, value in numeric.channel_pool.state_dict().items():
                    torch.testing.assert_close(value, full.fusion.channel_pool.state_dict()[name],
                                               rtol=0, atol=0, msg=name)
                # Only the first classifier layer changes shape/initialization.
                for name, value in numeric.classifier.state_dict().items():
                    if not name.startswith('network.0.'):
                        torch.testing.assert_close(value, full.fusion.classifier.state_dict()[name],
                                                   rtol=0, atol=0, msg=name)
                repeated = self.fresh_numeric(seed)
                for name, value in numeric.state_dict().items():
                    torch.testing.assert_close(value, repeated.state_dict()[name], rtol=0, atol=0)
                first_weights.append(numeric.classifier.network[0].weight.detach().clone())
        self.assertFalse(torch.equal(first_weights[0], first_weights[1]))
        self.assertFalse(torch.equal(first_weights[1], first_weights[2]))

    def test_forward_is_direct_duration_pool_then_classifier(self):
        model = self.fresh_numeric().eval()
        tokens, mask, fractions = synthetic_batch()
        captured = []
        hook = model.classifier.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            with torch.no_grad():
                actual = model(tokens, mask, fractions)
                temporal, _ = model.channel_pool(tokens, patch_mask=mask)
                expected_pool = torch.stack([
                    (temporal[i, mask[i]] * fractions[i, mask[i], None]).sum(0)
                    / fractions[i, mask[i]].sum()
                    for i in range(len(tokens))
                ])
                self.assertEqual(tuple(captured[0].shape), (3, CONFIG['fusion_dim']))
                torch.testing.assert_close(captured[0], expected_pool, rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(actual, model.classifier(expected_pool), rtol=1e-5, atol=1e-7)
                changed_padding = tokens.clone()
                changed_padding[~mask] = 12345
                torch.testing.assert_close(model(changed_padding, mask, fractions), actual, rtol=0, atol=0)
                changed_durations = fractions.clone()
                changed_durations[0, 1] = 1.0
                changed = model(tokens, mask, changed_durations)
                self.assertGreater((changed[0] - actual[0]).abs().max().item(), 1e-6)
                torch.testing.assert_close(changed[1:], actual[1:], rtol=0, atol=0)
        finally:
            hook.remove()

    def test_gradient_reaches_numeric_pool_and_head_not_padding(self):
        model = self.fresh_numeric().train()
        tokens, mask, fractions = synthetic_batch()
        tokens.requires_grad_()
        loss = F.cross_entropy(model(tokens, mask, fractions), torch.tensor([0, 1, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(tokens.grad[mask].abs().sum().item(), 0)
        self.assertEqual(tokens.grad[~mask].abs().sum().item(), 0)
        for module_name in ('channel_pool', 'classifier'):
            gradients = [p.grad for p in getattr(model, module_name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), module_name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, module_name)

    def test_invalid_patch_metadata_is_rejected(self):
        # Both implementations must enforce the same input/cache contract.
        ZeroVisualSlotTests.test_invalid_patch_metadata_is_rejected(self)

    def test_versioned_checkpoint_dispatch_is_strict_and_roundtrips(self):
        self.assertEqual(LEGACY_ARCHITECTURE, 'neurosigvia_numeric_only_zero_visual_slot_v1')
        tokens, mask, fractions = synthetic_batch()
        for architecture, model_type in (
            (LEGACY_ARCHITECTURE, ZeroVisualSlotClassifier),
            (ARCHITECTURE, NumericOnlyClassifier),
        ):
            with self.subTest(architecture=architecture):
                set_random_seed(42)
                model = model_type.from_config(CONFIG).eval()
                payload = dict(architecture=architecture, model_config=model.model_config,
                               model_state_dict=model.state_dict())
                serialized = io.BytesIO()
                torch.save(payload, serialized)
                serialized.seek(0)
                saved = torch.load(serialized, map_location='cpu', weights_only=True)
                restored = numeric_model_from_checkpoint(saved).eval()
                self.assertIs(type(restored), model_type)
                with torch.no_grad():
                    torch.testing.assert_close(restored(tokens, mask, fractions),
                                               model(tokens, mask, fractions), rtol=0, atol=0)
                broken = dict(saved, model_state_dict=dict(saved['model_state_dict']))
                broken['model_state_dict'].pop(next(iter(broken['model_state_dict'])))
                with self.assertRaises(RuntimeError):
                    numeric_model_from_checkpoint(broken)
                wrong_version = dict(saved, architecture=(
                    ARCHITECTURE if architecture == LEGACY_ARCHITECTURE else LEGACY_ARCHITECTURE))
                with self.assertRaises(RuntimeError):
                    numeric_model_from_checkpoint(wrong_version)
        with self.assertRaises(ValueError):
            numeric_model_from_checkpoint(dict(payload, architecture='unknown_future_architecture'))


if __name__ == '__main__':
    unittest.main()
