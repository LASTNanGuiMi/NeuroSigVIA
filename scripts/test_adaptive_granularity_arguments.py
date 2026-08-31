#!/usr/bin/env python3
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arguments import (  # noqa: E402
    parse_args,
    parse_med_activity_granularity_bank,
)


def parse(argv):
    previous = sys.argv
    try:
        sys.argv = ["main.py", *argv]
        return parse_args()
    finally:
        sys.argv = previous


def assert_rejected(argv):
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parse(argv)
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError(f"Invalid arguments were accepted: {argv}")


def main():
    bank = parse_med_activity_granularity_bank(
        "1,2,4;2,4,8;4,8,16"
    )
    assert bank == ((1, 2, 4), (2, 4, 8), (4, 8, 16))

    required = ["--data_dir", "/data", "--result_dir", "/results"]
    common = [
        "--vit_1_name",
        "/models/clip",
        "--image_mode",
        "med_activity_graph",
        "--classifier_type",
        "mlp",
        "--med_activity_adaptive_granularity",
        *required,
    ]
    args = parse(common)
    assert args.med_activity_adaptive_granularity
    assert args.med_activity_granularity_bank == bank
    assert args.med_activity_granularity_hidden_dim == 64
    assert args.med_activity_granularity_temperature == 1.0
    assert args.med_activity_granularity_base_prior == 0.9
    assert args.med_activity_granularity_balance_weight == 0.01
    assert args.med_activity_granularity_entropy_weight == 0.001

    assert_rejected(
        [
            "--vit_1_name",
            "/models/clip",
            "--image_mode",
            "line_plot",
            "--classifier_type",
            "mlp",
            "--med_activity_adaptive_granularity",
            *required,
        ]
    )
    assert_rejected(
        [
            *common,
            "--med_activity_granularity_balance_weight",
            "-0.1",
        ]
    )
    assert_rejected(
        [
            *common,
            "--med_activity_granularity_entropy_weight",
            "-0.1",
        ]
    )
    for option in (
        "--med_activity_granularity_balance_weight",
        "--med_activity_granularity_entropy_weight",
        "--med_activity_granularity_temperature",
    ):
        for value in ("nan", "inf"):
            assert_rejected([*common, option, value])
    assert_rejected(
        [
            "--vit_1_name",
            "/models/clip",
            "--image_mode",
            "med_activity_graph",
            "--classifier_type",
            "logistic_regression",
            "--med_activity_adaptive_granularity",
            *required,
        ]
    )
    assert_rejected(
        [
            "--vit_1_name",
            "/models/clip",
            "--image_mode",
            "med_activity_graph",
            "--classifier_type",
            "mlp",
            "--med_activity_adaptive_granularity",
            "--med_activity_granularity_bank",
            "1,2,4;4,8,16",
            *required,
        ]
    )
    print("ADAPTIVE GRANULARITY ARGUMENT VALIDATION PASSED")


if __name__ == "__main__":
    main()
