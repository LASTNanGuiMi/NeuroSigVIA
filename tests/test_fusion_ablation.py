"""Controlled fusion replacement tests without encoder/model downloads."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.multimodal_fusion import AdaptiveGranularityFusionModule


class FusionAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_model(self, mode="concat_attn", alignment=True, mask_prob=0.3):
        return AdaptiveGranularityFusionModule(
            8, 6, 4, 2, fusion_dim=8, fusion_heads=2, dropout=0,
            classifier_hidden_dim=8, alignment_dim=4,
            temporal_visual_fusion=mode, alignment_enabled=alignment,
            mask_prob=mask_prob,
        )

    def inputs(self):
        torch.manual_seed(201)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        fractions = torch.tensor([[1., .5, 0.], [1., 1., .25]])
        return (torch.randn(2, 3, 8), torch.randn(2, 3, 16, 8),
                torch.randn(2, 3, 4, 6), mask, fractions)

    def test_default_matches_explicit_and_accepts_legacy_config(self):
        torch.manual_seed(41)
        default = AdaptiveGranularityFusionModule(
            8, 6, 4, 2, fusion_dim=8, fusion_heads=2, dropout=0,
            classifier_hidden_dim=8, alignment_dim=4,
        ).eval()
        torch.manual_seed(41)
        explicit = self.make_model().eval()
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        for key, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(default(*self.inputs())[0], explicit(*self.inputs())[0], rtol=0, atol=0)
        legacy_config = default.get_config()
        for key in ("temporal_visual_fusion", "alignment_enabled", "mask_prob"):
            legacy_config.pop(key)
        restored = AdaptiveGranularityFusionModule.from_config(legacy_config).eval()
        restored.load_state_dict(default.state_dict(), strict=True)
        torch.testing.assert_close(default(*self.inputs())[0], restored(*self.inputs())[0], rtol=0, atol=0)

    def test_all_modes_shapes_no_alignment_gradients_and_roundtrip(self):
        for mode in ("concat_attn", "concat", "masked_pretrain"):
            with self.subTest(mode=mode):
                torch.manual_seed(42)
                model = self.make_model(mode, alignment=False)
                self.assertIsNone(model.alignment)
                self.assertFalse(any(name.startswith("alignment.") for name, _ in model.named_parameters()))
                logits, details = model(*self.inputs())
                self.assertEqual(tuple(logits.shape), (2, 2))
                width = 8 if mode == "masked_pretrain" else 16
                self.assertEqual(tuple(details["patch_features"].shape), (2, 3, width))
                self.assertEqual(details["alignment_loss"].item(), 0.)
                self.assertFalse(details["reconstruction_applied"])
                self.assertTrue(torch.isfinite(logits).all())
                F.cross_entropy(logits, torch.tensor([0, 1])).backward()
                for name in ("channel_pool", "visual_cross_attention", "classifier"):
                    grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
                    self.assertTrue(grads, name)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads), name)
                    self.assertGreater(sum(g.abs().sum().item() for g in grads), 0., name)
                restored = AdaptiveGranularityFusionModule.from_config(model.get_config())
                restored.load_state_dict(model.state_dict(), strict=True)
                self.assertEqual(restored.configuration, model.configuration)
                torch.testing.assert_close(restored(*self.inputs())[0], logits, rtol=0, atol=0)

    def test_concat_is_exact_legacy_raw_branch_concatenation(self):
        model = self.make_model("concat", alignment=False).eval()
        args = self.inputs()
        _, details = model(*args)
        expected = torch.cat([details["visual_tokens"], details["temporal_tokens"]], dim=-1)
        expected = expected * args[3].unsqueeze(-1)
        torch.testing.assert_close(details["patch_features"], expected, rtol=0, atol=0)
        self.assertEqual(len(model.temporal_visual_fusion.projections), 0)

    def test_invalid_patch_values_do_not_affect_logits_or_reconstruction(self):
        for mode in ("concat", "masked_pretrain"):
            with self.subTest(mode=mode):
                model = self.make_model(mode, alignment=False, mask_prob=1.).eval()
                args = self.inputs()
                changed = [arg.clone() for arg in args]
                for tokens in changed[:3]:
                    tokens[~args[3]] = 1e4
                torch.manual_seed(44)
                expected, before = model(*args, pretrain=mode == "masked_pretrain")
                torch.manual_seed(44)
                actual, after = model(*changed, pretrain=mode == "masked_pretrain")
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(after["reconstruction_loss"], before["reconstruction_loss"], rtol=0, atol=0)

    def test_masked_pretraining_updates_only_fusion_and_preserves_target_gradient(self):
        model = self.make_model("masked_pretrain", alignment=False, mask_prob=1.)
        model.requires_grad_(False)
        fusion = model.temporal_visual_fusion
        fusion.requires_grad_(True)
        saved = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=1e-3)
        reconstruct = fusion.reconstruct_masked
        def fixed_branch(branches, mask_prob):
            self.assertEqual(branches[0].shape[0], 5)
            return reconstruct(branches, mask_prob=mask_prob, mask_index=0)
        with patch.object(fusion, "reconstruct_masked", side_effect=fixed_branch):
            _, details = model(*self.inputs(), pretrain=True)
        self.assertTrue(details["reconstruction_applied"])
        self.assertEqual(details["masked_branch_index"], 0)
        self.assertTrue(torch.isfinite(details["reconstruction_loss"]))
        self.assertGreater(details["reconstruction_loss"].item(), 0.)
        details["reconstruction_loss"].backward()
        # Masked branch enters MSE only as its projected target: a nonzero
        # gradient here verifies preservation of the legacy nondetached target.
        self.assertGreater(fusion.projections[0].weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters()
                            if not name.startswith("temporal_visual_fusion.")))
        optimizer.step()
        changed_fusion = 0
        for name, tensor in model.state_dict().items():
            if name.startswith("temporal_visual_fusion."):
                changed_fusion += int(not torch.equal(tensor, saved[name]))
            else:
                torch.testing.assert_close(tensor, saved[name], rtol=0, atol=0)
        self.assertGreater(changed_fusion, 0)

    def test_skipped_mask_returns_explicit_no_update(self):
        model = self.make_model("masked_pretrain", alignment=False, mask_prob=0.)
        _, details = model(*self.inputs(), pretrain=True)
        self.assertFalse(details["reconstruction_applied"])
        self.assertIsNone(details["masked_branch_index"])
        self.assertEqual(details["reconstruction_loss"].item(), 0.)
        self.assertFalse(details["reconstruction_loss"].requires_grad)

    def test_invalid_configuration_fails(self):
        with self.assertRaises(ValueError):
            self.make_model("unknown")
        for probability in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                self.make_model(mask_prob=probability)
        with self.assertRaises(TypeError):
            self.make_model(alignment=0)
        with self.assertRaises(ValueError):
            self.make_model("concat", alignment=False)(*self.inputs(), pretrain=True)


if __name__ == "__main__":
    unittest.main()
