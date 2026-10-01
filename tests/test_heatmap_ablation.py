import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from src.heatmap_ablation import (
    HeatmapClassifier, HeatmapFusionModule, JointProjectedInfoNCE,
    extract_heatmap_feature_batch, heatmap_matrix, render_heatmap,
)
from src.patch_fusion import MaskedIntraSampleInfoNCE


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(3))
        self.aggregation = "mean"
        self.calls = 0

    def forward_vit(self, images):
        self.calls += len(images)
        return images.mean((-1, -2)).matmul(self.weight)[:, None, :]

    def aggregate_hidden_representations(self, hidden, aggregation):
        return hidden.mean(1)


class TinyMantis(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, data):
        return torch.cat([data.mean(-1), data.std(-1), data.amax(-1)], 1) * self.scale


class HeatmapTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2419)
        torch.set_num_threads(1)

    def inputs(self):
        visual = torch.randn(2, 3, 6)
        temporal = torch.randn(2, 3, 2, 5)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        fraction = torch.tensor([[1.0, 0.5, 0.0], [1.0, 1.0, 0.25]])
        return visual, temporal, mask, fraction

    def module(self):
        return HeatmapFusionModule(6, 5, 2, 3, fusion_dim=8, fusion_heads=2,
                                   alignment_dim=4, channel_hidden_dim=4,
                                   classifier_hidden_dim=7, dropout=0)

    def test_channel_major_patch_arrangement_and_tail_isolation(self):
        window = torch.tensor([[0., 1., 2., 3., 4., 999.],
                               [10., 11., 12., 13., 14., float("nan")]])
        matrix, mask = heatmap_matrix(window, valid_length=5,
                                     image_representation="patch_heatmap",
                                     heatmap_patch_size=3)
        expected = torch.tensor([[0., 1., 2.], [3., 4., 0.],
                                 [10., 11., 12.], [13., 14., 0.]])
        self.assertTrue(torch.equal(matrix, expected))
        self.assertEqual(mask.sum().item(), 10)
        self.assertFalse(mask[1, 2])
        self.assertFalse(mask[3, 2])
        for mode in ("patch_heatmap", "multivariate_heatmap"):
            a = render_heatmap(window, valid_length=5, image_representation=mode,
                               heatmap_patch_size=3)
            b = render_heatmap(window[:, :5], image_representation=mode,
                               heatmap_patch_size=3)
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.isfinite(a).all())
            self.assertEqual(tuple(a.shape), (3, 224, 224))

    def test_multivariate_order_and_constant_values(self):
        source = torch.tensor([[0., 1., 2.], [3., 4., 5.]])
        matrix, mask = heatmap_matrix(source)
        self.assertTrue(torch.equal(source, matrix))
        self.assertTrue(mask.all())
        image = render_heatmap(torch.ones(2, 8))
        self.assertTrue(torch.isfinite(image).all())
        self.assertTrue(torch.equal(image[:, 0, 0], image[:, -1, -1]))

    def test_exact_reference_infonce_loss_and_gradients(self):
        ref = MaskedIntraSampleInfoNCE(5, 6, 4, temperature=0.17)
        actual = JointProjectedInfoNCE(5, 6, 4, temperature=0.17)
        actual.load_state_dict(ref.state_dict(), strict=True)
        t, v = torch.randn(3, 4, 5), torch.randn(3, 4, 6)
        mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]]).bool()
        weights = torch.tensor([[1., 1., .25, 0.], [1., 0., 0., 0.], [1., .5, .5, 1.]])
        for fractions in (None, weights):
            ref.zero_grad(); actual.zero_grad()
            expected = ref(t, v, mask, fractions)
            loss, aligned_t, aligned_v = actual(t, v, mask, fractions, return_tokens=True)
            self.assertTrue(torch.equal(expected, loss))
            self.assertTrue(torch.equal(aligned_t, F.normalize(actual.temporal_projection(t), dim=-1, eps=1e-8)))
            self.assertTrue(torch.equal(aligned_v, F.normalize(actual.visual_projection(v), dim=-1, eps=1e-8)))
            expected.backward(); loss.backward()
            for left, right in zip(ref.parameters(), actual.parameters()):
                self.assertTrue(torch.equal(left.grad, right.grad))
        # A one-window sample has no eligible negatives, per the old protocol.
        self.assertEqual(actual(t[:, :1], v[:, :1], mask[:, :1]).item(), 0.0)

    def test_fusion_consumes_joint_alignment_outputs_exactly_once(self):
        model = self.module()
        captured, projection_calls = {}, {"v": 0, "t": 0}
        handles = [model.temporal_visual_fusion.register_forward_pre_hook(
            lambda module, args: captured.update(branches=args[0]))]
        def count(name):
            def hook(*_):
                projection_calls[name] += 1
            return hook
        handles.extend([
            model.alignment.visual_projection.register_forward_hook(count("v")),
            model.alignment.temporal_projection.register_forward_hook(count("t"))])
        v, t, mask, fraction = self.inputs()
        _, info = model(v, t, mask, fraction)
        self.assertTrue(torch.equal(captured["branches"][0], info["aligned_visual_tokens"][mask]))
        self.assertTrue(torch.equal(captured["branches"][1], info["aligned_temporal_tokens"][mask]))
        self.assertEqual(projection_calls, {"v": 1, "t": 1})
        self.assertEqual(captured["branches"][0].shape[-1], 4)
        self.assertNotEqual(captured["branches"][0].shape[-1], v.shape[-1])
        for handle in handles:
            handle.remove()

    def test_supervised_ce_reaches_both_alignment_projections(self):
        model = self.module()
        logits, _ = model(*self.inputs())
        F.cross_entropy(logits, torch.tensor([0, 2])).backward()
        for branch in (model.alignment.visual_projection, model.alignment.temporal_projection):
            self.assertIsNotNone(branch.weight.grad)
            self.assertTrue(torch.isfinite(branch.weight.grad).all())
            self.assertGreater(branch.weight.grad.abs().sum().item(), 0)

    def test_alignment_loss_reaches_both_modalities_and_channel_pool(self):
        model = self.module()
        _, info = model(*self.inputs())
        info["alignment_loss"].backward()
        for weight in (model.alignment.visual_projection.weight,
                       model.alignment.temporal_projection.weight,
                       model.channel_pool.value_projection.weight):
            self.assertGreater(weight.grad.abs().sum().item(), 0)

    def test_invalid_windows_excluded_from_pool_and_logits(self):
        model = self.module().eval()
        v, t, mask, fraction = self.inputs()
        a, info = model(v, t, mask, fraction)
        v[~mask] = float("nan")
        t[~mask] = float("nan")
        b, info_b = model(v, t, mask, fraction)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(info["alignment_loss"], info_b["alignment_loss"]))
        self.assertEqual(info_b["patch_features"][~mask].abs().sum().item(), 0)
        expected_pool = (info_b["patch_features"] * fraction[..., None]).sum(1) / fraction.sum(1, keepdim=True)
        self.assertTrue(torch.equal(info_b["sample_features"], expected_pool))
        bad_fraction = fraction.clone(); bad_fraction[0, 2] = 1
        with self.assertRaises(ValueError):
            model(v, t, mask, bad_fraction)

    def test_classifier_strict_reload_without_old_graph_or_online_vision(self):
        model = HeatmapClassifier(6, 5, 2, 3, fusion_dim=8, fusion_heads=2,
                                   alignment_dim=4, dropout=0,
                                   image_representation="patch_heatmap").eval()
        clone = HeatmapClassifier.from_config(model.get_config()).eval()
        clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        v, t, mask, fraction = self.inputs()
        # None proves forward doesn't require raw windows or an online encoder.
        a, info = model(None, v, t, mask, fraction, None, None)
        b, _ = clone(None, v, t, mask, fraction, None, None)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(hasattr(model, "renderer"))
        self.assertFalse(hasattr(model.fusion, "visual_cross_attention"))
        self.assertFalse(any("gate" in key for key in model.state_dict()))
        self.assertEqual(info["selector_valid_count"].item(), 0)
        self.assertEqual(model.configuration["fusion_input"], "aligned_projected_features")

    def test_extraction_only_heatmaps_with_frozen_encoders_and_padding(self):
        vision, mantis = TinyVision(), TinyMantis()
        raw = torch.randn(2, 2, 12)
        raw[1, :, 6:] = float("nan")
        bundle = extract_heatmap_feature_batch(
            raw, vision, mantis, "cpu", window_size=4, stride=4,
            lengths=[12, 6], encode_batch_size=2,
            image_representation="patch_heatmap", heatmap_patch_size=3)
        self.assertEqual(set(bundle), {"raw_windows", "line_tokens", "mantis_channel_tokens",
                                      "patch_mask", "valid_fraction", "valid_lengths"})
        self.assertEqual(vision.calls, 5)
        self.assertEqual(tuple(bundle["line_tokens"].shape), (2, 3, 3))
        self.assertEqual(tuple(bundle["mantis_channel_tokens"].shape), (2, 3, 2, 3))
        self.assertEqual(bundle["valid_lengths"].tolist(), [[4, 4, 4], [4, 2, 0]])
        self.assertEqual(bundle["valid_fraction"].tolist(), [[1, 1, 1], [1, .5, 0]])
        for name, values in bundle.items():
            self.assertTrue(torch.isfinite(values).all(), name)
        self.assertEqual(bundle["line_tokens"][1, 2].abs().sum().item(), 0)
        self.assertFalse(vision.training)
        self.assertFalse(mantis.training)
        self.assertFalse(any(p.requires_grad for p in vision.parameters()))
        self.assertFalse(any(p.requires_grad for p in mantis.parameters()))


if __name__ == "__main__":
    unittest.main()
