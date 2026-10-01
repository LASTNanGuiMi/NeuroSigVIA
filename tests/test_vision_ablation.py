"""Scientific isolation and gradient checks for the visual-only ablation."""
from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.adaptive_graph_training import NeuroSigVIAClassifier
from src.patch_fusion import make_temporal_patches, valid_fraction_weighted_pool
from src.vision_ablation import (
    VISUAL_STATIC_KEYS, VisionOnlyClassifier, extract_visual_split, validate_visual_bundle,
)


class TinyFrozenVision(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.vit = torch.nn.Conv2d(3, width, 1)
        self.aggregation = 'mean'
        self.requires_grad_(False)

    def forward_vit(self, images):
        # Pool before the linear 1x1 map to keep the 1024-dimension shape test small.
        tokens = self.vit(F.adaptive_avg_pool2d(images, (4, 4))).flatten(2).transpose(1, 2)
        return torch.cat((tokens.mean(1, keepdim=True), tokens), dim=1)

    def project_pooled_representation(self, values):
        return values

    def aggregate_hidden_representations(self, hidden, aggregation='mean'):
        if aggregation != 'mean':
            raise ValueError('test encoder supports mean only')
        return hidden[:, 1:].mean(dim=1)


def temporal_args(batch=2, channels=4, patches=1, dim=8):
    return (
        torch.randn(batch, patches, channels, 64),
        torch.randn(batch, patches, dim),
        torch.ones(batch, patches, dtype=torch.bool),
        torch.ones(batch, patches),
        torch.full((batch, patches), 64, dtype=torch.long),
    )


class VisionAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_adftd_axes_and_f128_classifier_shape(self):
        torch.manual_seed(42)
        model = VisionOnlyClassifier(1024, 19, 3, dropout=0).eval()
        vision = TinyFrozenVision(1024)
        raw, line, mask, fraction, lengths = temporal_args(1, 19, 4, 1024)
        with torch.no_grad():
            logits, details = model(raw, line, mask, fraction, lengths, vision,
                                    encode_batch_size=1, return_attention_weights=True)
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertEqual(tuple(details['visual_tokens'].shape), (1, 4, 128))
        self.assertEqual(tuple(details['sample_features'].shape), (1, 128))
        self.assertEqual(tuple(details['cross_attention_weights'].shape), (1, 4, 2, 1, 16))

    def test_shared_visual_weights_match_full_method_visual_subpath(self):
        torch.manual_seed(43)
        full = NeuroSigVIAClassifier(8, 6, 4, 3, fusion_dim=8, fusion_heads=2,
                                    classifier_hidden_dim=8, alignment_dim=4, dropout=0).eval()
        visual = VisionOnlyClassifier(8, 4, 3, fusion_dim=8, classifier_hidden_dim=8, dropout=0).eval()
        visual.renderer.load_state_dict(full.renderer.state_dict())
        visual.fusion.visual_cross_attention.load_state_dict(full.fusion.visual_cross_attention.state_dict())
        inputs = temporal_args()
        vision = TinyFrozenVision()
        with torch.no_grad():
            _, reference = full(inputs[0], inputs[1], torch.randn(2, 1, 4, 6),
                                inputs[2], inputs[3], inputs[4], vision)
            _, actual = visual(*inputs, vision)
        torch.testing.assert_close(actual['visual_tokens'], reference['visual_tokens'])
        torch.testing.assert_close(actual['sample_features'],
                                   valid_fraction_weighted_pool(reference['visual_tokens'], inputs[2], inputs[3]))

    def test_classification_gradient_reaches_gate_with_frozen_vision(self):
        torch.manual_seed(44)
        model = VisionOnlyClassifier(8, 4, 3, fusion_dim=8, classifier_hidden_dim=8, dropout=0)
        vision = TinyFrozenVision()
        logits, details = model(*temporal_args(), vision)
        # Deliberately omit balance loss: classification itself must train the gate.
        F.cross_entropy(logits, torch.tensor([0, 1])).backward()
        gradients = [p.grad for p in model.renderer.gate.region_cls.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertTrue(all(p.grad is None for p in vision.parameters()))
        self.assertTrue(torch.isfinite(details['selector_balance_loss']))

    def test_prohibited_modules_are_never_constructed_or_extracted(self):
        poison = AssertionError('prohibited numeric/cross-modal branch touched')
        targets = [
            'src.multimodal_fusion.ChannelAttentionPool',
            'src.multimodal_fusion.MaskedIntraSampleInfoNCE',
            'src.multimodal_fusion.FusionModule',
            'src.patch_fusion._extract_mantis_channel_tokens',
            'src.adaptive_graph_training._extract_mantis_channel_tokens',
            'mantis.architecture.Mantis8M',
        ]
        with ExitStack() as stack:
            for target in targets:
                stack.enter_context(patch(target, side_effect=poison))
            model = VisionOnlyClassifier(8, 4, 3, fusion_dim=8, classifier_hidden_dim=8, dropout=0)
            vision = TinyFrozenVision()
            logits, _ = model(*temporal_args(), vision)
            F.cross_entropy(logits, torch.tensor([0, 1])).backward()
            # Isolate line extraction in this test; the cold extraction has a
            # separate real line-renderer/encoder compatibility test below.
            stack.enter_context(patch('src.vision_ablation._extract_line_tokens',
                                      side_effect=lambda x, *a: torch.ones(len(x), 8)))
            bundle = extract_visual_split(DataLoader(TensorDataset(torch.randn(3, 4, 256)), batch_size=2),
                                          vision, 'cpu')
        self.assertEqual(set(bundle), set(VISUAL_STATIC_KEYS))
        self.assertFalse(any(any(word in key for word in ('mantis', 'channel_pool', 'alignment', 'temporal_visual'))
                             for key in model.state_dict()))

    def test_real_cold_line_renderer_and_encoder_skip_mantis(self):
        torch.manual_seed(42)
        signal = torch.randn(2, 4, 64)
        with patch('mantis.architecture.Mantis8M', side_effect=AssertionError('Mantis constructed')), \
             patch('src.patch_fusion._extract_mantis_channel_tokens', side_effect=AssertionError('Mantis extracted')), \
             patch('src.adaptive_graph_training._extract_mantis_channel_tokens', side_effect=AssertionError('Mantis extracted')):
            bundle = extract_visual_split(DataLoader(TensorDataset(signal), batch_size=1),
                                          TinyFrozenVision(), 'cpu', encode_batch_size=1)
        self.assertEqual(tuple(bundle['line_tokens'].shape), (2, 1, 8))
        self.assertTrue(torch.isfinite(bundle['line_tokens']).all())
        torch.testing.assert_close(bundle['raw_windows'][:, 0], signal)
        self.assertTrue(all(not value.requires_grad for value in bundle.values()))

    def test_cold_cache_preserves_source_order_patches_and_partial_tail(self):
        signal = torch.randn(3, 4, 256)
        lengths = torch.tensor([256, 129, 64])
        loader = DataLoader(TensorDataset(signal, lengths), batch_size=2, shuffle=False)
        # Line token values encode the raw patch mean so order errors are visible.
        def fake_line(windows, valid_lengths, *_):
            return torch.stack([window[:, :int(n)].mean().repeat(8)
                                for window, n in zip(windows, valid_lengths)])
        with patch('src.vision_ablation._extract_line_tokens', side_effect=fake_line):
            bundle = extract_visual_split(loader, TinyFrozenVision(), 'cpu')
        expected = make_temporal_patches(signal, window_size=64, stride=64, lengths=lengths)
        torch.testing.assert_close(bundle['raw_windows'], expected.patches)
        torch.testing.assert_close(bundle['patch_mask'], expected.patch_mask)
        torch.testing.assert_close(bundle['valid_lengths'], expected.valid_lengths)
        torch.testing.assert_close(bundle['valid_fraction'].float(), expected.valid_fraction)
        self.assertEqual(tuple(bundle['line_tokens'].shape), (3, 4, 8))
        self.assertTrue(torch.equal(bundle['line_tokens'][~bundle['patch_mask']],
                                    torch.zeros_like(bundle['line_tokens'][~bundle['patch_mask']])))
        invalid = dict(bundle, mantis_channel_tokens=torch.zeros(3, 4, 4, 6))
        with self.assertRaises(ValueError):
            validate_visual_bundle(invalid)
        with self.assertRaises(ValueError):
            validate_visual_bundle(dict(bundle, valid_fraction=torch.zeros_like(bundle['valid_fraction'])))
        with self.assertRaises(ValueError):
            extract_visual_split(DataLoader(TensorDataset(signal), shuffle=True), TinyFrozenVision(), 'cpu')

    def test_config_and_checkpoint_reconstruction_roundtrip(self):
        model = VisionOnlyClassifier(8, 4, 3, fusion_dim=8, classifier_hidden_dim=8).eval()
        restored = VisionOnlyClassifier.from_config(model.get_config()).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        inputs, vision = temporal_args(), TinyFrozenVision()
        with torch.no_grad():
            first, _ = model(*inputs, vision)
            second, _ = restored(*inputs, vision)
        torch.testing.assert_close(first, second)
        self.assertEqual(restored.ARCHITECTURE, model.ARCHITECTURE)


if __name__ == '__main__':
    unittest.main()
