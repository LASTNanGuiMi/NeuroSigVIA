"""Run without ML dependencies: python -m unittest discover -s tests -p 'test_explicit_launchers.py'."""
from collections import Counter
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("Medformer", "Crossformer", "FEDformer", "Autoformer", "PatchTST", "Transformer", "TimesNet")
METHODS = (*BASELINES, "NeuroSigVIA", "NeuroSigVIA_NumericOnly")
DATASETS = ("adftd", "tdbrain", "apava", "shimmer10", "pads11")
MAIN_DATASET_NAMES = ("ADFTD", "TDBRAIN", "APAVA", "Shimmer_10_session10_AFC", "PADS_11_task08_TouchIndex")


def file_snapshot(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        for path in root.rglob("*")
    }


def bash_executable():
    configured = os.environ.get("BASH_BIN")
    if configured:
        return configured
    if os.name == "nt":
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if git_bash.is_file():
            return str(git_bash)
    bash = shutil.which("bash")
    if not bash:
        raise unittest.SkipTest("Bash is required to verify shell launchers")
    return bash


class ExplicitLauncherTests(unittest.TestCase):
    def test_public_launchers_list_commands_without_generating_dataset_or_seed_arguments(self):
        for model in METHODS:
            with self.subTest(model=model):
                script = (ROOT / "scripts" / f"{model}.sh").read_text(encoding="utf-8")
                code = "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))
                self.assertNotRegex(code, r"\b(?:train_one|configure_dataset|run_experiments)\b")
                self.assertNotRegex(code, r"(?m)^\s*(?:for|while|until|select)\b")
                self.assertNotRegex(code, r"\b(?:DATASETS|SEEDS|GPUS|BATCH_SIZES)=")

    def test_dry_run_lists_all_129_commands_and_creates_no_files_or_directories(self):
        bash = bash_executable()
        with tempfile.TemporaryDirectory(prefix="explicit launcher test ") as directory:
            sandbox = Path(directory)
            shutil.copytree(ROOT / "scripts", sandbox / "scripts")
            before = file_snapshot(sandbox)
            env = {**os.environ, "DRY_RUN": "1", "RUN_TAG": "launcher-test", "PYTHON_BIN": "python"}
            total = 0
            for model in METHODS:
                with self.subTest(model=model):
                    result = subprocess.run(
                        [bash, f"scripts/{model}.sh"], cwd=sandbox, env=env,
                        check=True, text=True, encoding="utf-8", capture_output=True, timeout=30,
                    )
                    self.assertEqual(file_snapshot(sandbox), before, "DRY_RUN modified the temporary checkout")
                    commands = [shlex.split(line) for line in result.stdout.splitlines() if line.strip()]
                    numeric = model == "NeuroSigVIA_NumericOnly"
                    self.assertEqual(len(commands), 12 if model == "TimesNet" or numeric else 15)
                    total += len(commands)
                    covered = []
                    result_paths = []
                    cache_paths = []
                    dataset_values = MAIN_DATASET_NAMES if model == "NeuroSigVIA" else DATASETS
                    if model == "TimesNet":
                        dataset_values = DATASETS[1:]
                    if numeric:
                        dataset_values = ("shimmer10", "pads11", "apava", "tdbrain")
                    for command in commands:
                        self.assertEqual(command[0], "env")
                        self.assertRegex(command[1], r"^CUDA_VISIBLE_DEVICES=[0-9]+$")
                        self.assertEqual(command[2:5], ["python", "-u", "-m"])
                        expected_module = "runners.neurosigvia" if model == "NeuroSigVIA" else "runners.baselines"
                        if numeric:
                            expected_module = "runners.numeric_ablation"
                        self.assertEqual(command[5], expected_module)
                        argv = command[6:]
                        options = [token for token in argv if token.startswith("--")]
                        self.assertFalse([option for option, count in Counter(options).items() if count > 1], "Repeated option in a launch command")
                        dataset_option = "--dataset_names" if model == "NeuroSigVIA" else "--dataset"
                        dataset = argv[argv.index(dataset_option) + 1]
                        seed = int(argv[argv.index("--seed" if numeric else "--random_seed") + 1])
                        covered.append((dataset, seed))
                        self.assertEqual(int(command[1].split("=", 1)[1]), dataset_values.index(dataset) + (2 if numeric else 0))
                        result_path = argv[argv.index("--output" if numeric else "--result_dir") + 1]
                        self.assertIn("launcher-test", result_path)
                        self.assertIn(f"seed{seed}/", result_path)
                        result_paths.append(result_path)
                        if model == "NeuroSigVIA":
                            cache_paths.append(argv[argv.index("--feature_cache_dir") + 1])
                            self.assertEqual(argv[argv.index("--patch_checkpoint_metric") + 1], "window_macro_f1")
                        elif numeric:
                            self.assertIn("--reference-run", argv)
                            self.assertEqual(argv[argv.index("--checkpoint-metric") + 1], "window_macro_f1")
                        else:
                            self.assertEqual(argv[argv.index("--model") + 1], model)
                    self.assertCountEqual(covered, [(dataset, seed) for dataset in dataset_values for seed in (42, 43, 44)])
                    self.assertEqual(len(set(result_paths)), len(commands))
                    if model == "NeuroSigVIA":
                        self.assertEqual(len(set(cache_paths)), 15)
            self.assertEqual(total, 129)


if __name__ == "__main__":
    unittest.main()
