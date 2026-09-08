"""TimesNet integration checks: python -m unittest discover -s tests -p 'test_timesnet_integration.py'."""
import contextlib
import io
from pathlib import Path
import re
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runners import baselines as baseline


TIMESNET_SETTINGS = {
    "split_seed": 42,
    "d_model": 128,
    "d_ff": 256,
    "e_layers": 2,
    "n_heads": 8,
    "dropout": 0.1,
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "train_epochs": 100,
    "patience": 12,
    "warmup_epochs": 10,
    "min_delta": 0.002,
}
MEDFORMER_EFFECTIVE_SETTINGS = {
    "d_model": 128,
    "d_ff": 256,
    "e_layers": 6,
    "n_heads": 8,
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "train_epochs": 100,
    "patience": 10,
}
OLD_MODELS = (
    "Medformer", "Crossformer", "FEDformer", "Autoformer", "PatchTST", "Transformer",
)


def arguments(model="TimesNet", dataset="apava", *extra):
    args = baseline.parser().parse_args([
        "--model", model, "--dataset", dataset, "--device", "cpu",
        "--result_dir", "unused-test-result", *extra,
    ])
    baseline.validate_args(args)
    return args


def script_numeric_setting(text, name):
    match = re.search(r"--" + re.escape(name) + r"\s+([0-9.eE+-]+)(?=\s|$)", text)
    if match is None:
        raise AssertionError(f"Missing explicit --{name} in launcher")
    return float(match.group(1))


@contextlib.contextmanager
def isolated_vendor_imports():
    """Isolate only upstream package names; do not unload torch's lazy modules."""
    packages = ("models", "layers", "utils")

    def is_vendor_module(name):
        return any(name == package or name.startswith(package + ".") for package in packages)

    previous = {key: value for key, value in sys.modules.copy().items() if is_vendor_module(key)}
    previous_path = sys.path[:]
    for key in previous:
        sys.modules.pop(key, None)
    try:
        yield
    finally:
        for key in list(sys.modules):
            if is_vendor_module(key):
                sys.modules.pop(key, None)
        sys.modules.update(previous)
        sys.path[:] = previous_path


class TimesNetProtocolTests(unittest.TestCase):
    def test_vendor_routing_preserves_six_existing_baselines(self):
        self.assertEqual(set(baseline.MODELS), set(OLD_MODELS) | {"TimesNet"})
        for name in (*OLD_MODELS, "TimesNet"):
            with self.subTest(model=name):
                args = arguments(name)
                expected = "timesnet" if name == "TimesNet" else "medformer"
                self.assertEqual(args.vendor_root.resolve(), (ROOT / "third_party" / expected).resolve())
        explicit = arguments("TimesNet", "apava", "--vendor_root", "explicit-vendor")
        self.assertEqual(explicit.vendor_root, Path("explicit-vendor"))

    def test_parser_defaults_remain_timesnet_compatible(self):
        batches = {"adftd": 8, "tdbrain": 8, "apava": 8, "shimmer10": 1, "pads11": 4}
        for dataset, batch_size in batches.items():
            with self.subTest(dataset=dataset):
                timesnet = arguments("TimesNet", dataset)
                self.assertEqual(timesnet.batch_size, batch_size)
                for name, expected in TIMESNET_SETTINGS.items():
                    self.assertEqual(getattr(timesnet, name), expected, name)
                self.assertEqual(timesnet.random_seed, 42)
                self.assertEqual(timesnet.top_k, 3)
                self.assertEqual(timesnet.num_kernels, 6)

    def test_launcher_uses_four_datasets_three_seeds_and_unchanged_settings(self):
        script = (ROOT / "scripts" / "TimesNet.sh").read_text(encoding="utf-8")
        for name, expected in TIMESNET_SETTINGS.items():
            self.assertEqual(script_numeric_setting(script, name), expected, name)
        self.assertEqual(script_numeric_setting(script, "top_k"), 3)
        self.assertEqual(script_numeric_setting(script, "num_kernels"), 6)
        for name, expected in (
            ("DATASETS", ["tdbrain", "apava", "shimmer10", "pads11"]),
            ("SEEDS", ["42", "43", "44"]),
            ("GPUS", ["0", "1", "2", "3"]),
        ):
            match = re.search(r"^" + name + r'=\"([^\"]+)\"\s*$', script, re.MULTILINE)
            self.assertIsNotNone(match, name)
            self.assertEqual(match.group(1).split(), expected, name)
        for dataset, expected in (("tdbrain", 8), ("apava", 8), ("shimmer10", 1), ("pads11", 4)):
            match = re.search(r"\[" + dataset + r"\]=(\d+)", script)
            self.assertIsNotNone(match, dataset)
            self.assertEqual(int(match.group(1)), expected, dataset)
        self.assertIn("--progress", script)

    def test_six_medformer_family_launchers_follow_upstream_effective_settings(self):
        for model in OLD_MODELS:
            with self.subTest(model=model):
                script = (ROOT / "scripts" / f"{model}.sh").read_text(encoding="utf-8")
                for name, expected in MEDFORMER_EFFECTIVE_SETTINGS.items():
                    self.assertEqual(script_numeric_setting(script, name), expected, name)
                for dataset, expected in (
                    ("adftd", 128), ("tdbrain", 32), ("apava", 32),
                    ("shimmer10", 1), ("pads11", 4),
                ):
                    match = re.search(r"\[" + dataset + r"\]=(\d+)", script)
                    self.assertIsNotNone(match, dataset)
                    self.assertEqual(int(match.group(1)), expected, dataset)

        medformer = (ROOT / "scripts" / "Medformer.sh").read_text(encoding="utf-8")
        for expected in (
            'patch_len_list="2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32"',
            'augmentations="drop0.5"',
            'patch_len_list="8,8,8,16,16,16"',
            'augmentations="none,drop0.25"',
            'patch_len_list="2,2,2,4,4,4,16,16,16,16,32,32,32,32,32"',
            'augmentations="none,drop0.35"',
            '--patch_len_list "$patch_len_list" --augmentations "$augmentations" --swa',
        ):
            self.assertIn(expected, medformer)
        for model in OLD_MODELS:
            script = (ROOT / "scripts" / f"{model}.sh").read_text(encoding="utf-8")
            self.assertEqual("--swa" in script, model == "Medformer", model)

    def test_swa_is_medformer_only_and_opt_in(self):
        self.assertFalse(arguments("Medformer").swa)
        self.assertTrue(arguments("Medformer", "apava", "--swa").swa)
        with self.assertRaisesRegex(ValueError, "Medformer"):
            arguments("TimesNet", "apava", "--swa")


class TimesNetModelTests(unittest.TestCase):
    def test_official_imports_forward_backward_and_serialized_checkpoint(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with isolated_vendor_imports(), torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                args = arguments(
                    "TimesNet", "apava", "--d_model", "8", "--d_ff", "8",
                    "--e_layers", "1", "--top_k", "3", "--num_kernels", "2",
                )
                config = baseline.model_config(args, sequence_length=64, channels=3, num_classes=2)
                model_class = baseline.import_model("TimesNet", args.vendor_root)
                for name, relative in (
                    ("models.TimesNet", "models/TimesNet.py"),
                    ("layers.Embed", "layers/Embed.py"),
                    ("layers.Conv_Blocks", "layers/Conv_Blocks.py"),
                ):
                    self.assertEqual(Path(sys.modules[name].__file__).resolve(), (args.vendor_root / relative).resolve())
                self.assertIs(model_class, sys.modules["models.TimesNet"].Model)
                self.assertIs(sys.modules["models.TimesNet"].DataEmbedding, sys.modules["layers.Embed"].DataEmbedding)
                self.assertIs(sys.modules["models.TimesNet"].Inception_Block_V1,
                              sys.modules["layers.Conv_Blocks"].Inception_Block_V1)

                model = model_class(config).float()
                x = torch.randn(2, 3, 64)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                model.train()
                logits = baseline.forward(model, x, torch.device("cpu"), 2)
                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertTrue(torch.isfinite(logits).all().item())
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
                loss.backward()
                for name, parameter in (
                    ("classification projection", model.projection.weight),
                    ("period convolution", model.model[0].conv[0].kernels[0].weight),
                ):
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    expected = baseline.forward(model, x, torch.device("cpu"), 2)
                    direct = model(x.transpose(1, 2).contiguous(), torch.ones(2, 64), None, None)
                torch.testing.assert_close(expected, direct, rtol=0, atol=0)
                serialized = io.BytesIO()
                torch.save({"model_state_dict": model.state_dict()}, serialized)
                serialized.seek(0)
                restored = model_class(config).float().eval()
                checkpoint = torch.load(serialized, map_location="cpu", weights_only=True)
                restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
                with torch.no_grad():
                    actual = baseline.forward(restored, x, torch.device("cpu"), 2)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)


class MedformerModelConfigTests(unittest.TestCase):
    def test_upstream_eeg_patch_and_augmentation_configs_construct_and_restore(self):
        cases = {
            "adftd": ("2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32", "drop0.5"),
            "tdbrain": ("8,8,8,16,16,16", "none,drop0.25"),
            "apava": ("2,2,2,4,4,4,16,16,16,16,32,32,32,32,32", "none,drop0.35"),
        }
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        reformer_stub = ModuleType("reformer_pytorch")
        reformer_stub.LSHSelfAttention = torch.nn.Identity
        try:
            with patch.dict(sys.modules, {"reformer_pytorch": reformer_stub}), \
                 isolated_vendor_imports(), torch.random.fork_rng(devices=[]):
                model_class = baseline.import_model("Medformer", ROOT / "third_party" / "medformer")
                x = torch.randn(2, 3, 64)
                for dataset, (patch_lengths, augmentations) in cases.items():
                    with self.subTest(dataset=dataset):
                        args = arguments(
                            "Medformer", dataset, "--d_model", "8", "--d_ff", "16",
                            "--e_layers", "1", "--n_heads", "2", "--patch_len_list", patch_lengths,
                            "--augmentations", augmentations, "--swa",
                        )
                        config = baseline.model_config(args, sequence_length=64, channels=3, num_classes=2)
                        model = model_class(config).float().eval()
                        with torch.no_grad():
                            expected = baseline.forward(model, x, torch.device("cpu"), 2)
                        averaged = torch.optim.swa_utils.AveragedModel(model)
                        averaged.update_parameters(model)
                        restored = model_class(config).float().eval()
                        restored.load_state_dict(averaged.module.state_dict(), strict=True)
                        with torch.no_grad():
                            actual = baseline.forward(restored, x, torch.device("cpu"), 2)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
