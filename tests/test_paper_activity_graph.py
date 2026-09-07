"""Paper examples, waveform preservation and actual training gradient checks."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.activity_graph import (
    ActivityGraphRenderer, TemporalGranularityGraphBank,
    paper_signal_order, paper_multicolumn_indices, render_paper_activity_graph,
)
from src.adaptive_activity_graph import AdaptiveActivityGraphRenderer
from src.adaptive_graph_training import (
    NeuroSigVIAClassifier, NEUROSIGVIA_ARCHITECTURE,
    NEUROSIGVIA_CHECKPOINT_SCHEMA_VERSION, _encoder_contract,
    load_neurosigvia_checkpoint,
)


class TinyFrozenVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = torch.nn.Conv2d(3, 8, 1)
        self.requires_grad_(False)

    def forward_vit(self, images):
        spatial = F.adaptive_avg_pool2d(self.vit(images), (4, 4)).flatten(2).transpose(1, 2)
        return torch.cat((spatial.mean(1, keepdim=True), spatial), dim=1)

    def project_pooled_representation(self, tokens):
        return tokens


class PaperGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_published_examples_and_pair_coverage(self):
        self.assertEqual(paper_signal_order(4), (0, 1, 2, 3, 0, 2, 3, 1))
        self.assertEqual(paper_signal_order(6), (0, 1, 2, 3, 4, 5, 0, 2, 4, 0, 3, 1, 4, 5, 1, 2, 5, 3))
        self.assertEqual(paper_multicolumn_indices(4)[0], (1, 0, 1))
        self.assertEqual(paper_multicolumn_indices(4)[-1], (3, 1, 0))
        for channels in range(1, 65):
            order = paper_signal_order(channels)
            covered = {tuple(sorted(pair)) for pair in zip(order, order[1:]) if pair[0] != pair[1]}
            self.assertEqual(len(covered), channels * (channels - 1) // 2)

    def test_narrow_unsampled_spike_changes_full_waveform(self):
        original = torch.zeros(1, 1, 239)
        original[..., 100] = 1
        spiked = original.clone()
        spiked[..., 31] = 1  # missed by the previous 120-point resampling
        self.assertGreater((render_paper_activity_graph(original) - render_paper_activity_graph(spiked)).abs().max().item(), 0.1)

    def test_padding_singletons_and_finite_rgb(self):
        torch.manual_seed(41)
        for channels in (1, 4, 6, 19, 28, 32, 33):
            with self.subTest(channels=channels):
                signal = torch.randn(1, channels, 37)
                padded = F.pad(signal, (0, 27), value=1e6)
                expected = render_paper_activity_graph(signal)
                actual = render_paper_activity_graph(padded, valid_lengths=torch.tensor([37]))
                torch.testing.assert_close(actual, expected)
                self.assertEqual(actual.shape, (1, 3, 224, 224))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual[:, 0], actual[:, 1])
                self.assertGreater(actual.std().item(), 1e-4)
        one = torch.ones(1, 4, 1)
        torch.testing.assert_close(render_paper_activity_graph(one), render_paper_activity_graph(F.pad(one, (0, 63)), valid_lengths=torch.tensor([1])))

    def test_no_legacy_rgb_experts_and_distinct_waveform_scales(self):
        with self.assertRaisesRegex(ValueError, "legacy RGB"):
            ActivityGraphRenderer(patch_lengths=(2, 4, 8))
        images = TemporalGranularityGraphBank()(torch.randn(1, 4, 64))
        self.assertFalse(torch.allclose(images[:, 0], images[:, 1]))

    def test_adaptive_eval_tail_invariance(self):
        torch.manual_seed(43)
        model = AdaptiveActivityGraphRenderer().eval()
        signal = torch.randn(1, 6, 37)
        with torch.no_grad():
            expected = model(signal)
            padded = model(F.pad(signal, (0, 27), value=1e6), valid_lengths=torch.tensor([37]))
            torch.testing.assert_close(expected, padded)
            torch.testing.assert_close(expected, model(signal))

    def test_classification_loss_reaches_gate_through_frozen_vision(self):
        torch.manual_seed(44)
        model = NeuroSigVIAClassifier(8, 6, 4, 2, fusion_dim=8, fusion_heads=2,
                                     classifier_hidden_dim=8, alignment_dim=4, dropout=0)
        vision = TinyFrozenVision()
        logits, details = model(torch.randn(2, 1, 4, 64), torch.randn(2, 1, 8),
                                torch.randn(2, 1, 4, 6), torch.ones(2, 1, dtype=torch.bool),
                                torch.ones(2, 1), torch.full((2, 1), 64), vision)
        F.cross_entropy(logits, torch.tensor([0, 1])).backward()
        gradients = [p.grad for p in model.renderer.gate.region_cls.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertTrue(all(p.grad is None for p in vision.parameters()))
        self.assertTrue(torch.isfinite(details['selector_balance_loss']))

    def test_constant_axes_and_mixed_lengths_have_finite_input_gradients(self):
        torch.manual_seed(43)
        signal = torch.randn(2, 4, 64)
        signal[:, 0] = 3
        signal.requires_grad_()
        model = AdaptiveActivityGraphRenderer().eval()
        image = model(signal, valid_lengths=torch.tensor([1, 37]))
        image.square().mean().backward()
        self.assertTrue(torch.isfinite(signal.grad).all())
        self.assertEqual(signal.grad[0, :, 1:].abs().sum().item(), 0)
        self.assertEqual(signal.grad[1, :, 37:].abs().sum().item(), 0)

    def test_checkpoint_roundtrip_and_old_schema_rejection(self):
        model = NeuroSigVIAClassifier(8, 6, 4, 2, fusion_dim=8, fusion_heads=2, classifier_hidden_dim=8)
        vision = TinyFrozenVision()
        payload = dict(schema_version=NEUROSIGVIA_CHECKPOINT_SCHEMA_VERSION,
                       architecture=NEUROSIGVIA_ARCHITECTURE,
                       model_constructor_configuration=model.get_config(),
                       model_state_dict=model.state_dict(),
                       external_encoder_contracts={'vision': _encoder_contract(vision, 'vision')})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(payload, path)
            restored, _ = load_neurosigvia_checkpoint(path, vision_model=vision)
            self.assertEqual(restored.get_config(), model.get_config())
            payload['schema_version'] = 2
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'schema mismatch'):
                load_neurosigvia_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
