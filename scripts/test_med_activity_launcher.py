#!/usr/bin/env python3
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = PROJECT_ROOT / "scripts" / "run_med_activity_multimodal.sh"
EEG_LAUNCHER = PROJECT_ROOT / "scripts" / "run_eeg_medformer_multimodal.sh"


def assert_adaptive_command(tokens, dataset_group, dataset_name):
    assert tokens.count("--image_mode") == 1
    assert tokens[tokens.index("--image_mode") + 1] == "med_activity_graph"
    assert tokens.count("--med_activity_adaptive_granularity") == 1
    assert tokens[tokens.index("--med_activity_granularity_bank") + 1] == (
        "1,2,4;2,4,8;4,8,16"
    )
    assert tokens[tokens.index("--med_activity_granularity_hidden_dim") + 1] == "64"
    assert tokens[tokens.index("--med_activity_granularity_temperature") + 1] == "1.0"
    assert tokens[tokens.index("--med_activity_granularity_base_prior") + 1] == "0.9"
    assert tokens[tokens.index("--med_activity_granularity_balance_weight") + 1] == "0.01"
    assert tokens[tokens.index("--med_activity_granularity_entropy_weight") + 1] == "0.001"
    assert tokens.count("--random_seed") == 1
    assert tokens[tokens.index("--random_seed") + 1] == "42"
    assert tokens[tokens.index("--datasets") + 1] == dataset_group
    assert tokens[tokens.index("--dataset_names") + 1] == dataset_name
    assert "adaptive_granularity" in tokens[tokens.index("--result_dir") + 1]
    assert "adaptive_granularity" in tokens[tokens.index("--feature_cache_dir") + 1]
    assert "seed42" in tokens[tokens.index("--result_dir") + 1]
    assert "seed42" in tokens[tokens.index("--feature_cache_dir") + 1]
    assert tokens[0:2] == ["env", "CUDA_VISIBLE_DEVICES=7"]


def dry_run(launcher, key):
    env = dict(os.environ)
    env.update(
        {
            "DRY_RUN": "1",
            "SEED": "42",
            "GPU": "7",
            "PYTHON_BIN": "/abs/python",
        }
    )
    completed = subprocess.run(
        [str(launcher), key],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return shlex.split(completed.stdout.strip().splitlines()[-1])


def main():
    if os.name == "nt":
        print("SKIP launcher test on Windows; bash validation runs on the server")
        return

    expected = {
        "shimmer10": ("aaai27", "Shimmer_10_session10_AFC"),
        "pads11": ("aaai27", "PADS_11_task08_TouchIndex"),
        "ucihar": ("uci", "UCIHAR"),
        "falltl": ("falltl", "FallTL"),
    }
    for key, (dataset_group, dataset_name) in expected.items():
        assert_adaptive_command(
            dry_run(LAUNCHER, key),
            dataset_group,
            dataset_name,
        )

    eeg_expected = {
        "tdbrain": "TDBRAIN",
        "apava": "APAVA",
        "adftd": "ADFTD",
    }
    for key, dataset_name in eeg_expected.items():
        tokens = dry_run(EEG_LAUNCHER, key)
        assert_adaptive_command(tokens, "eeg", dataset_name)
        assert tokens[tokens.index("--eeg_protocol") + 1] == (
            "medformer_code_exact"
        )
    print("MED ACTIVITY LAUNCHER VALIDATION PASSED")


if __name__ == "__main__":
    main()
