"""TimesNet integration checks: python -m unittest discover -s tests -p 'test_timesnet_integration.py'."""
import contextlib
import io
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
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


def launcher_commands(model):
    """Read the commands users will execute without creating a run or importing a model."""
    bash = os.environ.get("BASH_BIN")
    if not bash and os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if candidate.is_file():
            bash = str(candidate)
    bash = bash or shutil.which("bash")
    if not bash:
        raise unittest.SkipTest("Bash is required to verify shell launchers")
    with tempfile.TemporaryDirectory(prefix="launcher test ") as directory:
        sandbox = Path(directory)
        shutil.copytree(ROOT / "scripts", sandbox / "scripts")
        env = {**os.environ, "DRY_RUN": "1", "RUN_TAG": "launcher-test", "PYTHON_BIN": "python"}
        result = subprocess.run(
            [bash, f"scripts/{model}.sh"], cwd=sandbox, env=env,
            check=True, text=True, encoding="utf-8", capture_output=True, timeout=30,
        )
    commands = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        command = shlex.split(line)
        if command[0] != "env" or not command[1].startswith("CUDA_VISIBLE_DEVICES="):
            raise AssertionError(f"Unexpected launcher output: {line}")
        if command[2:6] != ["python", "-u", "-m", "runners.baselines"]:
            raise AssertionError(f"Unexpected launcher entrypoint: {line}")
        argv = command[6:]
        args = baseline.parser().parse_args(argv)
        # The launcher must remain a CUDA command; validation does not need a real GPU.
        with patch.object(torch.cuda, "is_available", return_value=True):
            baseline.validate_args(args)
        commands.append((int(command[1].split("=", 1)[1]), args, argv))
    return commands


def expected_launcher_arguments(model, dataset, seed, batch_size, result_dir):
    """The complete effective baseline protocol, including intentional parser defaults."""
    settings = dict(TIMESNET_SETTINGS)
    if model != "TimesNet":
        settings.update(MEDFORMER_EFFECTIVE_SETTINGS)
    if model == "TimesNet" and dataset in ("apava", "shimmer10"):
        settings["dropout"] = 0.3
    settings.update({
        "task_name": "classification", "model": model, "dataset": dataset,
        "random_seed": seed, "batch_size": batch_size, "result_dir": result_dir,
        "vendor_root": ROOT / "third_party" / ("timesnet" if model == "TimesNet" else "medformer"),
        "checkpoint_metric": "window_macro_f1", "device": "cuda", "gpu": 0,
        "patch_len_list": "2,4,8", "augmentations": "none", "swa": model == "Medformer",
        "patch_len": 16, "stride": 8, "top_k": 3, "num_kernels": 6,
        "progress": model == "TimesNet", "smoke": False, "smoke_samples_per_class": 2,
    })
    if model == "Medformer":
        patch_settings = {
            "adftd": ("2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32", "drop0.5"),
            "tdbrain": ("8,8,8,16,16,16", "none,drop0.25"),
            "apava": ("2,2,2,4,4,4,16,16,16,16,32,32,32,32,32", "none,drop0.35"),
        }
        settings["patch_len_list"], settings["augmentations"] = patch_settings.get(dataset, ("2,4,8", "none"))
    return settings


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
        datasets = {"tdbrain": (0, 8), "apava": (1, 8), "shimmer10": (2, 1), "pads11": (3, 4)}
        commands = launcher_commands("TimesNet")
        self.assertEqual(len(commands), 12)
        self.assertCountEqual(
            [(args.dataset, args.random_seed) for _, args, _ in commands],
            [(dataset, seed) for dataset in datasets for seed in (42, 43, 44)],
        )
        self.assertEqual(len({args.result_dir for _, args, _ in commands}), 12)
        for gpu, args, argv in commands:
            with self.subTest(dataset=args.dataset, seed=args.random_seed):
                expected_gpu, batch_size = datasets[args.dataset]
                self.assertEqual(gpu, expected_gpu)
                self.assertEqual(vars(args), expected_launcher_arguments(
                    "TimesNet", args.dataset, args.random_seed, batch_size, args.result_dir,
                ))
                for name in (*TIMESNET_SETTINGS, "top_k", "num_kernels", "batch_size", "random_seed", "checkpoint_metric"):
                    self.assertIn(f"--{name}", argv)
                self.assertIn("launcher-test", args.result_dir.as_posix())
                self.assertEqual(args.result_dir.name, args.dataset)
                self.assertEqual(args.result_dir.parent.name, f"seed{args.random_seed}")

    def test_six_medformer_family_launchers_follow_upstream_effective_settings(self):
        datasets = {"adftd": (0, 128), "tdbrain": (1, 32), "apava": (2, 32), "shimmer10": (3, 1), "pads11": (4, 4)}
        for model in OLD_MODELS:
            with self.subTest(model=model):
                commands = launcher_commands(model)
                self.assertEqual(len(commands), 15)
                self.assertCountEqual(
                    [(args.dataset, args.random_seed) for _, args, _ in commands],
                    [(dataset, seed) for dataset in datasets for seed in (42, 43, 44)],
                )
                self.assertEqual(len({args.result_dir for _, args, _ in commands}), 15)
                for gpu, args, argv in commands:
                    with self.subTest(dataset=args.dataset, seed=args.random_seed):
                        expected_gpu, batch_size = datasets[args.dataset]
                        self.assertEqual(gpu, expected_gpu)
                        self.assertEqual(vars(args), expected_launcher_arguments(
                            model, args.dataset, args.random_seed, batch_size, args.result_dir,
                        ))
                        for name in (*TIMESNET_SETTINGS, "batch_size", "random_seed", "checkpoint_metric"):
                            self.assertIn(f"--{name}", argv)
                        if model == "Medformer":
                            for option in ("--patch_len_list", "--augmentations", "--swa"):
                                self.assertIn(option, argv)
                        self.assertIn("launcher-test", args.result_dir.as_posix())
                        self.assertEqual(args.result_dir.name, args.dataset)
                        self.assertEqual(args.result_dir.parent.name, f"seed{args.random_seed}")

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
