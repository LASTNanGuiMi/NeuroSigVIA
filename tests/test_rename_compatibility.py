"""Check renamed CLI and checkpoint identities using small CPU encoders."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.adaptive_graph_training import _assert_encoder_contract, _encoder_contract
from src.compatibility import matches_checkpoint_architecture, normalize_cli_arguments


class _TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = torch.nn.Linear(2, 2)
        self.register_buffer("scale", torch.ones(2))
        self.processor = SimpleNamespace(transforms=["resize=224", "normalize"])
        self.layer_idx = 14
        self.aggregation = "mean"
        self.image_mode = "med_activity_graph"


def encoder(previous=False, suffix="OpenCLIP", module=None):
    prefix = "NeuroSigViT" if previous else "NeuroSigVIA"
    wrapper = type(prefix + "_" + suffix, (_TinyEncoder,), {
        "__module__": module or (
            "src.neurosigvit" if previous else "src.neurosigvia"
        ),
    })
    return wrapper()


class RenameCompatibilityTests(unittest.TestCase):
    def test_previous_encoder_contract_accepts_same_state_without_mutation(self):
        for suffix in ("OpenCLIP", "HF"):
            with self.subTest(suffix=suffix):
                previous = encoder(previous=True, suffix=suffix)
                current = encoder(suffix=suffix)
                current.load_state_dict(previous.state_dict())
                expected = _encoder_contract(previous, "vision")
                unchanged = copy.deepcopy(expected)
                actual = _assert_encoder_contract(expected, current, "vision")
                self.assertEqual(actual, _encoder_contract(current, "vision"))
                self.assertEqual(expected, unchanged)
                self.assertNotEqual(actual["wrapper_class"], expected["wrapper_class"])
                self.assertNotEqual(actual["sampled_state_sha256"], expected["sampled_state_sha256"])
                self.assertEqual(_assert_encoder_contract(actual, current, "vision"), actual)

    def test_previous_encoder_contract_still_checks_state_and_configuration(self):
        previous = encoder(previous=True)
        expected = _encoder_contract(previous, "vision")
        for changed in ("weight", "buffer", "shape", "transforms", "layer", "aggregation", "image_mode"):
            with self.subTest(changed=changed):
                current = encoder()
                current.load_state_dict(previous.state_dict())
                with torch.no_grad():
                    if changed == "weight":
                        current.vit.weight[0, 0] += 1
                    elif changed == "buffer":
                        current.scale[0] += 1
                    elif changed == "shape":
                        current.vit = torch.nn.Linear(2, 3)
                    elif changed == "transforms":
                        current.processor.transforms.append("crop=192")
                    elif changed == "layer":
                        current.layer_idx = 13
                    elif changed == "aggregation":
                        current.aggregation = "cls_token"
                    else:
                        current.image_mode = "line_plot"
                with self.assertRaisesRegex(ValueError, "checkpoint contract"):
                    _assert_encoder_contract(expected, current, "vision")

    def test_wrapper_aliases_reject_unknown_or_different_encoder_classes(self):
        previous = encoder(previous=True)
        expected = _encoder_contract(previous, "vision")
        for current in (encoder(module="src.other"), encoder(suffix="HF")):
            current.load_state_dict(previous.state_dict())
            with self.subTest(wrapper=type(current).__module__ + "." + type(current).__name__):
                with self.assertRaisesRegex(ValueError, "wrapper_class"):
                    _assert_encoder_contract(expected, current, "vision")
        forged = dict(expected, wrapper_class="src.other.NeuroSigViT_OpenCLIP")
        current = encoder()
        current.load_state_dict(previous.state_dict())
        with self.assertRaisesRegex(ValueError, "wrapper_class"):
            _assert_encoder_contract(forged, current, "vision")

    def test_queued_cli_normalizes_space_and_equals_syntax_without_renaming_paths(self):
        for equals in (False, True):
            with self.subTest(equals=equals):
                pairs = [
                    ("--modal_interaction", "patch_timemosaic_graph"),
                    ("--timemosaic_gate_temperature", "0.5"),
                    ("--timemosaic_selector_balance_weight", "0.001"),
                    ("--timemosaic_graph_token_grid", "4"),
                    ("--timemosaic_gate_checkpoint", "/tmp/timemosaic/run=42/gate.pt"),
                    ("--result_dir", "patch_timemosaic_graph"),
                ]
                expected_pairs = [
                    ("--modal_interaction", "adaptive_granularity"),
                    ("--granularity_gate_temperature", "0.5"),
                    ("--granularity_balance_weight", "0.001"),
                    ("--granularity_graph_token_grid", "4"),
                    ("--granularity_gate_checkpoint", "/tmp/timemosaic/run=42/gate.pt"),
                    ("--result_dir", "patch_timemosaic_graph"),
                ]
                arguments = [option + "=" + value for option, value in pairs] if equals else [token for pair in pairs for token in pair]
                expected = [option + "=" + value for option, value in expected_pairs] if equals else [token for pair in expected_pairs for token in pair]
                arguments.append("--timemosaic_freeze_gate")
                expected.append("--granularity_freeze_gate")
                self.assertEqual(normalize_cli_arguments(arguments), expected)
                self.assertEqual(normalize_cli_arguments(expected), expected)

    def test_only_matching_previous_checkpoint_architecture_is_accepted(self):
        previous = "timemosaic_adaptive_graph_crossattn_concatattn_v2"
        current = "neurosigvia_adaptive_graph_crossattn_concatattn_v2"
        self.assertTrue(matches_checkpoint_architecture(previous, current))
        self.assertTrue(matches_checkpoint_architecture(current, current))
        self.assertFalse(matches_checkpoint_architecture(previous, "other_architecture"))
        self.assertFalse(matches_checkpoint_architecture("timemosaic_adaptive_graph_concat_mlp_v1", current))
        self.assertFalse(matches_checkpoint_architecture(None, current))


if __name__ == "__main__":
    unittest.main()
