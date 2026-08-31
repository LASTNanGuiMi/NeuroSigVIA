#!/usr/bin/env python3
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arguments import parse_args  # noqa: E402


REQUIRED_PATHS = [
    "--data_dir",
    "/data",
    "--result_dir",
    "/results",
]

PATCH_MINDTS_COMMAND = [
    "--vit_1_name",
    "/models/clip",
    "--vit_1_layer",
    "14",
    "--aggregation",
    "mean",
    "--image_mode",
    "med_activity_graph",
    "--classifier_type",
    "mlp",
    "--med_activity_adaptive_granularity",
    "--mantis",
    "--modal_interaction",
    "patch_mindts",
    *REQUIRED_PATHS,
]


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


def replace_option(argv, option, *replacement):
    values = list(argv)
    index = values.index(option)
    del values[index : index + 2]
    values[index:index] = replacement
    return values


def without_flag(argv, flag):
    values = list(argv)
    values.remove(flag)
    return values


def main():
    args = parse(PATCH_MINDTS_COMMAND)
    assert args.modal_interaction == "patch_mindts"
    assert args.outer_patch_size == 64
    assert args.outer_patch_stride == 64
    assert args.classifier_type == "mlp"
    assert args.image_mode == "med_activity_graph"
    assert args.med_activity_adaptive_granularity
    assert args.vit_1_name == "/models/clip"
    assert args.vit_2_name is None
    assert args.mantis
    assert args.moment is None
    assert args.med_activity_granularity_bank == ((4,), (8,), (16,))
    assert args.med_activity_granularity_temperature == 1.0
    assert args.patch_granularity_router_mode == "adaptive_v4"
    assert args.patch_checkpoint_metric == "auto"
    assert args.mlp_early_stop_strategy == "raw_selection_key"
    assert args.mlp_early_stop_min_epochs == 0
    assert args.mlp_early_stop_ema_decay == 0.6
    assert args.mlp_early_stop_min_delta == 0.0
    assert args.med_activity_granularity_balance_weight == 0.01
    assert args.med_activity_granularity_entropy_weight == 0.01
    assert args.med_activity_granularity_mix_shrinkage_weight == 0.005
    assert args.med_activity_granularity_prior_kl_weight == 0.001
    assert args.med_activity_granularity_usage_floor == 0.05
    assert args.med_activity_granularity_usage_ema_decay == 0.95
    assert args.med_activity_granularity_entropy_floor == 0.55
    assert args.med_activity_granularity_entropy_ceiling == 1.0
    assert args.med_activity_granularity_local_mix_max == 0.50
    assert args.med_activity_granularity_local_mix_init == 0.10
    assert args.med_activity_granularity_global_mix_max == 0.75
    assert args.med_activity_granularity_global_mix_init == 0.50
    assert args.med_activity_granularity_evidence_half_saturation == 0.05
    assert args.med_activity_granularity_minimum_weight == 0.0
    assert args.med_activity_granularity_score_cap == 1.0
    assert args.med_activity_granularity_scorer_hidden_dim == 32
    assert args.med_activity_granularity_confidence_half_saturation == 0.05

    v41 = parse(
        [
            *PATCH_MINDTS_COMMAND,
            "--patch_granularity_router_mode",
            "adaptive_v41",
            "--med_activity_granularity_scorer_hidden_dim",
            "16",
            "--med_activity_granularity_confidence_half_saturation",
            "0.1",
        ]
    )
    assert v41.patch_granularity_router_mode == "adaptive_v41"
    assert v41.med_activity_granularity_scorer_hidden_dim == 16
    assert v41.med_activity_granularity_confidence_half_saturation == 0.1

    v5 = parse(
        [
            *PATCH_MINDTS_COMMAND,
            "--patch_granularity_router_mode",
            "adaptive_v5",
        ]
    )
    assert v5.patch_granularity_router_mode == "adaptive_v5"
    assert v5.patch_router_top_k == 2
    assert v5.patch_router_training_noise_std == 0.0
    assert v5.patch_router_local_weight == 0.5
    assert v5.patch_router_relation_hidden_dim == 16
    assert v5.patch_router_relation_residual_scale == 0.25
    assert v5.patch_router_key_adapter_scale == 0.1
    assert v5.patch_router_value_adapter_scale == 0.1
    assert v5.patch_router_route_budget_weight == 0.005
    assert v5.patch_router_load_balance_weight == 0.005

    uniform = parse(
        [
            *PATCH_MINDTS_COMMAND,
            "--patch_granularity_router_mode",
            "uniform",
            "--patch_checkpoint_metric",
            "window_macro_f1",
        ]
    )
    assert uniform.patch_granularity_router_mode == "uniform"
    assert uniform.patch_checkpoint_metric == "window_macro_f1"

    subject_selected = parse(
        [
            *PATCH_MINDTS_COMMAND,
            "--patch_checkpoint_metric",
            "subject_macro_f1",
        ]
    )
    assert subject_selected.patch_checkpoint_metric == "subject_macro_f1"

    smoothed_early_stop = parse(
        [
            *PATCH_MINDTS_COMMAND,
            "--mlp_epochs",
            "40",
            "--mlp_early_stop_patience",
            "10",
            "--mlp_early_stop_strategy",
            "ema_primary",
            "--mlp_early_stop_min_epochs",
            "30",
            "--mlp_early_stop_ema_decay",
            "0.6",
            "--mlp_early_stop_min_delta",
            "0.001",
        ]
    )
    assert smoothed_early_stop.mlp_early_stop_patience == 10
    assert smoothed_early_stop.mlp_early_stop_strategy == "ema_primary"
    assert smoothed_early_stop.mlp_early_stop_min_epochs == 30
    assert smoothed_early_stop.mlp_early_stop_ema_decay == 0.6
    assert smoothed_early_stop.mlp_early_stop_min_delta == 0.001

    composite = parse(
        [
            *PATCH_MINDTS_COMMAND,
            "--med_activity_granularity_bank",
            "1,2,4;2,4,8;4,8,16",
        ]
    )
    assert composite.med_activity_granularity_bank == (
        (1, 2, 4),
        (2, 4, 8),
        (4, 8, 16),
    )

    # Every required branch/mode is checked independently.  Replacing ViT-1
    # with ViT-2 lets the legacy adaptive-granularity validation pass first,
    # so this also witnesses the patch_mindts-specific ViT-1 requirement.
    assert_rejected(
        replace_option(
            PATCH_MINDTS_COMMAND,
            "--classifier_type",
            "--classifier_type",
            "logistic_regression",
        )
    )
    assert_rejected(
        replace_option(
            PATCH_MINDTS_COMMAND,
            "--image_mode",
            "--image_mode",
            "line_plot",
        )
    )
    assert_rejected(
        without_flag(
            PATCH_MINDTS_COMMAND,
            "--med_activity_adaptive_granularity",
        )
    )
    assert_rejected(
        replace_option(
            PATCH_MINDTS_COMMAND,
            "--vit_1_name",
            "--vit_2_name",
            "/models/clip-2",
        )
    )
    assert_rejected(without_flag(PATCH_MINDTS_COMMAND, "--mantis"))
    assert_rejected(
        replace_option(PATCH_MINDTS_COMMAND, "--aggregation")
    )
    assert_rejected(
        replace_option(PATCH_MINDTS_COMMAND, "--vit_1_layer")
    )

    assert_rejected(
        [
            *PATCH_MINDTS_COMMAND,
            "--vit_2_name",
            "/models/clip-2",
        ]
    )
    assert_rejected([*PATCH_MINDTS_COMMAND, "--moment", "small"])

    # Patch routing accepts either three single-scale candidates or three
    # legacy RGB composites, but never mixed-width or duplicate candidates.
    for bank in (
        "4;8",
        "1,2,4;2,4,8;4,8,16;8,16,32",
        "4;8,16,32;16",
        "4;4;16",
    ):
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                "--med_activity_granularity_bank",
                bank,
            ]
        )

    for option, invalid_choice in (
        ("--patch_granularity_router_mode", "adaptive_v3"),
        ("--patch_checkpoint_metric", "sample_macro_f1"),
        ("--mlp_early_stop_strategy", "median_primary"),
    ):
        assert_rejected([*PATCH_MINDTS_COMMAND, option, invalid_choice])

    for option, values in (
        ("--mlp_early_stop_ema_decay", ("-0.1", "1", "nan", "inf")),
        ("--mlp_early_stop_min_delta", ("-0.1", "nan", "inf")),
    ):
        for value in values:
            assert_rejected([*PATCH_MINDTS_COMMAND, option, value])
    for value in ("-1", "21"):
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                "--mlp_early_stop_min_epochs",
                value,
            ]
        )

    # The outer time patch must contain more samples than the largest internal
    # graph aggregation scale.  The default bank has maximum scale 16.
    for outer_size in ("16", "1", "0", "-1"):
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                "--outer_patch_size",
                outer_size,
            ]
        )
    boundary = parse(
        [
            *PATCH_MINDTS_COMMAND,
            "--outer_patch_size",
            "17",
            "--outer_patch_stride",
            "17",
        ]
    )
    assert boundary.outer_patch_size == 17

    for stride in ("0", "-1"):
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                "--outer_patch_stride",
                stride,
            ]
        )
    assert_rejected(
        [
            *PATCH_MINDTS_COMMAND,
            "--outer_patch_stride",
            "65",
        ]
    )

    for temperature in ("0", "-0.1", "nan", "inf"):
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                "--patch_alignment_temperature",
                temperature,
            ]
        )
    for weight in ("-0.1", "nan", "inf"):
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                "--patch_alignment_weight",
                weight,
            ]
        )
    zero_weight = parse(
        [
            *PATCH_MINDTS_COMMAND,
            "--patch_alignment_weight",
            "0",
        ]
    )
    assert zero_weight.patch_alignment_weight == 0.0

    for option, values in (
        (
            "--med_activity_granularity_usage_floor",
            ("-0.1", "0.34", "1", "nan"),
        ),
        ("--med_activity_granularity_usage_ema_decay", ("-0.1", "1", "nan")),
        ("--med_activity_granularity_entropy_floor", ("-0.1", "1.1", "nan")),
        ("--med_activity_granularity_entropy_ceiling", ("-0.1", "1.1", "nan")),
        ("--med_activity_granularity_minimum_weight", ("-0.1", "0.34", "nan")),
    ):
        for value in values:
            assert_rejected([*PATCH_MINDTS_COMMAND, option, value])

    for option in (
        "--med_activity_granularity_mix_shrinkage_weight",
        "--med_activity_granularity_prior_kl_weight",
    ):
        for value in ("-0.1", "nan", "inf"):
            assert_rejected([*PATCH_MINDTS_COMMAND, option, value])

    for option in (
        "--med_activity_granularity_evidence_half_saturation",
        "--med_activity_granularity_score_cap",
        "--med_activity_granularity_confidence_half_saturation",
    ):
        for value in ("0", "-0.1", "nan", "inf"):
            assert_rejected([*PATCH_MINDTS_COMMAND, option, value])

    for value in ("0", "-1"):
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                "--med_activity_granularity_scorer_hidden_dim",
                value,
            ]
        )

    for value in ("0", "-1"):
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                "--patch_router_relation_hidden_dim",
                value,
            ]
        )
    for value in ("0", "-1", "4"):
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                "--patch_granularity_router_mode",
                "adaptive_v5",
                "--patch_router_top_k",
                value,
            ]
        )
    for value in ("-0.1", "1.1", "nan", "inf"):
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                "--patch_router_local_weight",
                value,
            ]
        )
    for option in (
        "--patch_router_training_noise_std",
        "--patch_router_relation_residual_scale",
        "--patch_router_key_adapter_scale",
        "--patch_router_value_adapter_scale",
        "--patch_router_route_budget_weight",
        "--patch_router_load_balance_weight",
    ):
        for value in ("-0.1", "nan", "inf"):
            assert_rejected([*PATCH_MINDTS_COMMAND, option, value])

    assert_rejected(
        [
            *PATCH_MINDTS_COMMAND,
            "--med_activity_granularity_entropy_floor",
            "0.8",
            "--med_activity_granularity_entropy_ceiling",
            "0.7",
        ]
    )
    for prefix in ("local", "global"):
        max_option = f"--med_activity_granularity_{prefix}_mix_max"
        init_option = f"--med_activity_granularity_{prefix}_mix_init"
        for value in ("0", "-0.1", "1.1", "nan"):
            assert_rejected([*PATCH_MINDTS_COMMAND, max_option, value])
        for value in ("0", "-0.1", "nan"):
            assert_rejected([*PATCH_MINDTS_COMMAND, init_option, value])
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                max_option,
                "0.5",
                init_option,
                "0.5",
            ]
        )

    for microbatch in ("0", "-1"):
        assert_rejected(
            [
                *PATCH_MINDTS_COMMAND,
                "--visual_encode_batch_size",
                microbatch,
            ]
        )
    one_image = parse(
        [
            *PATCH_MINDTS_COMMAND,
            "--visual_encode_batch_size",
            "1",
        ]
    )
    assert one_image.visual_encode_batch_size == 1

    for heads in ("0", "-1"):
        assert_rejected(
            [*PATCH_MINDTS_COMMAND, "--fusion_heads", heads]
        )

    # The historical sample-level ATGS command remains legal and does not
    # acquire patch_mindts-only Mantis/ViT-1 constraints.
    legacy = parse(
        [
            "--vit_2_name",
            "/models/legacy-clip",
            "--image_mode",
            "med_activity_graph",
            "--classifier_type",
            "mlp",
            "--med_activity_adaptive_granularity",
            "--modal_interaction",
            "concat",
            *REQUIRED_PATHS,
        ]
    )
    assert legacy.modal_interaction == "concat"
    assert legacy.med_activity_adaptive_granularity
    assert legacy.vit_1_name is None
    assert legacy.vit_2_name == "/models/legacy-clip"
    assert not legacy.mantis
    assert legacy.med_activity_granularity_bank == (
        (1, 2, 4),
        (2, 4, 8),
        (4, 8, 16),
    )

    print("PATCH MINDTS ARGUMENT VALIDATION PASSED")


if __name__ == "__main__":
    main()
