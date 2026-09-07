import argparse
import math
import sys

from src.compatibility import normalize_cli_arguments

VIT_NAME = [
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "google/siglip2-so400m-patch14-224",
    "facebook/vit-mae-base",
    "facebook/vit-mae-large",
    "facebook/vit-mae-huge",
]


def parse_med_activity_patch_lengths(value):
    try:
        patch_lengths = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Med activity patch lengths must be comma-separated integers."
        ) from exc

    if len(patch_lengths) != 1:
        raise argparse.ArgumentTypeError(
            "Paper waveform graphs accept one smoothing scale (1 means raw waveform)."
        )
    if any(length <= 0 for length in patch_lengths):
        raise argparse.ArgumentTypeError("Med activity patch lengths must be positive.")
    if tuple(sorted(set(patch_lengths))) != patch_lengths:
        raise argparse.ArgumentTypeError(
            "Med activity patch lengths must be unique and strictly increasing."
        )
    return patch_lengths


def parse_med_activity_granularity_bank(value):
    regimes = []
    for raw_regime in value.split(";"):
        raw_regime = raw_regime.strip()
        if not raw_regime:
            raise argparse.ArgumentTypeError(
                "Granularity-bank regimes must be separated by semicolons."
            )
        try:
            patch_lengths = tuple(
                int(item.strip()) for item in raw_regime.split(",")
            )
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Granularity experts must contain comma-separated integers."
            ) from exc
        if len(patch_lengths) != 1:
            raise argparse.ArgumentTypeError(
                "Each paper waveform expert must contain one smoothing scale."
            )
        if any(length <= 0 for length in patch_lengths):
            raise argparse.ArgumentTypeError(
                "Granularity expert patch lengths must be positive."
            )
        if (
            len(patch_lengths) == 3
            and tuple(sorted(set(patch_lengths))) != patch_lengths
        ):
            raise argparse.ArgumentTypeError(
                "A legacy RGB expert requires three unique increasing lengths."
            )
        regimes.append(patch_lengths)
    granularity_bank = tuple(regimes)
    if len(granularity_bank) < 2:
        raise argparse.ArgumentTypeError(
            "Adaptive granularity requires at least two candidate regimes."
        )
    if len(set(granularity_bank)) != len(granularity_bank):
        raise argparse.ArgumentTypeError(
            "Adaptive granularity candidate regimes must be unique."
        )
    return granularity_bank


def parse_args():
    parser = argparse.ArgumentParser(description="NeuroSigVIA.")

    parser.add_argument(
        "--vit_1_name",
        type=str,
        help="Pretrained weights of vision backbone, either a HuggingFace id or a local path",
    )

    parser.add_argument(
        "--vit_2_name",
        type=str,
        help="Pretrained weights of vision backbone, either a HuggingFace id or a local path",
    )

    parser.add_argument(
        "--vit_1_layer",
        type=int,
        help="Layer of vision backbone from which we extract representations",
    )

    parser.add_argument(
        "--vit_2_layer",
        type=int,
        help="Layer of vision backbone from which we extract representations",
    )

    parser.add_argument(
        "--mantis",
        action="store_true",
        help="Use time series foundation model Mantis",
    )

    parser.add_argument(
        "--mantis_name",
        type=str,
        default="paris-noah/Mantis-8M",
        help="Pretrained weights of Mantis, either a HuggingFace id or a local path",
    )

    parser.add_argument(
        "--moment",
        type=str,
        choices=["small", "base", "large"],
        help="Use time series foundation model MOMENT",
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        help="Aggregation of hidden representations",
    )

    parser.add_argument(
        "--image_mode",
        type=str,
        choices=[
            "line_plot",
            "multichannel_line_plot",
            "activity_graph",
            "med_activity_graph",
            "activity_matrix",
            "segment",
        ],
        default="line_plot",
        help="How to convert each time series into an image for the vision backbone",
    )

    parser.add_argument(
        "--med_activity_patch_lengths",
        type=parse_med_activity_patch_lengths,
        default=(1,),
        help=(
            "One fixed smoothing block length for paper waveform graphs; 1 uses raw signals"
        ),
    )

    parser.add_argument(
        "--med_activity_channel_mix",
        type=float,
        default=0.35,
        help="Deprecated compatibility option; paper Activity Graph does not propagate channel values",
    )

    parser.add_argument(
        "--med_activity_router_temperature",
        type=float,
        default=0.2,
        help="Deprecated compatibility option; unused by the paper waveform renderer",
    )

    parser.add_argument(
        "--med_activity_router_mix",
        type=float,
        default=0.5,
        help="Deprecated compatibility option; unused by the paper waveform renderer",
    )

    parser.add_argument(
        "--med_activity_adaptive_granularity",
        action="store_true",
        help=(
            "Extract a bank of full RGB med-activity graphs and learn a "
            "sample-wise granularity selector in the MLP fusion head"
        ),
    )

    parser.add_argument(
        "--med_activity_granularity_bank",
        type=parse_med_activity_granularity_bank,
        default=None,
        help=(
            "Semicolon-separated fixed waveform smoothing scales, e.g. 4;8;16."
        ),
    )

    parser.add_argument(
        "--med_activity_granularity_hidden_dim",
        type=int,
        default=64,
        help=(
            "Hidden width of the Mantis channel-attention pool in patch_mindts; "
            "also retained for the legacy sample-level selector"
        ),
    )

    parser.add_argument(
        "--med_activity_granularity_temperature",
        type=float,
        default=None,
        help=(
            "Softmax temperature of the granularity selector. Defaults to "
            "1.0 for bounded-RBF patch_mindts v4 and the legacy sample-level "
            "ATGS path"
        ),
    )

    parser.add_argument(
        "--patch_granularity_router_mode",
        choices=["adaptive_v4", "adaptive_v41", "adaptive_v5", "uniform"],
        default="adaptive_v4",
        help=(
            "Patch-level graph router: legacy bounded-RBF v4, confidence-gated "
            "scale-specific v4.1, project-specific line-Q/graph-KV v5 with "
            "adaptive-granularity patch-local decisions and multi-expert "
            "sparse top-k experts, or a formal fixed-uniform baseline"
        ),
    )

    parser.add_argument(
        "--patch_checkpoint_metric",
        choices=["auto", "subject_macro_f1", "window_macro_f1"],
        default="auto",
        help=(
            "Checkpoint unit. auto selects subject macro-F1 only when validated "
            "disjoint per-sample subject IDs are available"
        ),
    )

    parser.add_argument(
        "--med_activity_granularity_base_prior",
        type=float,
        default=0.9,
        help="Initial selector probability assigned to the legacy patch regime",
    )

    parser.add_argument(
        "--med_activity_granularity_balance_weight",
        type=float,
        default=0.01,
        help=(
            "Weight of the router usage regularizer. patch_mindts uses an EMA "
            "minimum-usage floor; legacy ATGS retains batch balance"
        ),
    )

    parser.add_argument(
        "--med_activity_granularity_entropy_weight",
        type=float,
        default=None,
        help=(
            "Weight of the router entropy-band regularizer. Defaults to 0.01 "
            "for patch_mindts and 0.001 for legacy sample-level ATGS"
        ),
    )

    parser.add_argument(
        "--med_activity_granularity_mix_shrinkage_weight",
        type=float,
        default=0.005,
        help="Weight shrinking v4 sample-global and patch-local routing mixtures",
    )

    parser.add_argument(
        "--med_activity_granularity_prior_kl_weight",
        type=float,
        default=0.001,
        help="Weight keeping the v4 dataset-level routing prior near uniform",
    )

    parser.add_argument(
        "--med_activity_granularity_usage_floor",
        type=float,
        default=0.05,
        help="Minimum long-run EMA usage assigned to every graph expert",
    )

    parser.add_argument(
        "--med_activity_granularity_usage_ema_decay",
        type=float,
        default=0.95,
        help="EMA decay retained for patch_mindts long-run usage diagnostics",
    )

    parser.add_argument(
        "--med_activity_granularity_entropy_floor",
        type=float,
        default=None,
        help=(
            "Minimum normalized mean routing entropy allowed without a penalty; "
            "defaults to 0.55 for patch_mindts and 0 for legacy routing"
        ),
    )

    parser.add_argument(
        "--med_activity_granularity_entropy_ceiling",
        type=float,
        default=1.0,
        help="Normalized mean Softmax entropy allowed without a penalty",
    )

    parser.add_argument(
        "--med_activity_granularity_local_mix_max",
        type=float,
        default=0.50,
        help="Upper bound on the v4 patch-local routing mixture",
    )
    parser.add_argument(
        "--med_activity_granularity_local_mix_init",
        type=float,
        default=0.10,
        help="Initial v4 patch-local routing mixture",
    )
    parser.add_argument(
        "--med_activity_granularity_global_mix_max",
        type=float,
        default=0.75,
        help="Upper bound on the v4 sample-global routing mixture",
    )
    parser.add_argument(
        "--med_activity_granularity_global_mix_init",
        type=float,
        default=0.50,
        help="Initial v4 sample-global routing mixture",
    )
    parser.add_argument(
        "--med_activity_granularity_evidence_half_saturation",
        type=float,
        default=0.05,
        help="Raw candidate dissimilarity yielding a 0.5 v4 evidence gate",
    )
    parser.add_argument(
        "--med_activity_granularity_minimum_weight",
        type=float,
        default=0.0,
        help="Optional explicit minimum probability for every graph expert",
    )
    parser.add_argument(
        "--med_activity_granularity_score_cap",
        type=float,
        default=1.0,
        help="Absolute cap applied to centered v4 routing logits",
    )
    parser.add_argument(
        "--med_activity_granularity_scorer_hidden_dim",
        type=int,
        default=32,
        help="Hidden width of every scale-specific nonlinear v4.1 scorer",
    )
    parser.add_argument(
        "--med_activity_granularity_confidence_half_saturation",
        type=float,
        default=0.05,
        help=(
            "Top1-minus-top2 probability margin yielding a 0.5 confidence "
            "gate in the v4.1 router"
        ),
    )
    parser.add_argument(
        "--patch_router_top_k",
        type=int,
        default=2,
        help=(
            "Number of graph-scale experts retained by adaptive_v5. The sparse "
            "top-k expert pattern is multi-expert; the scores remain the "
            "project's line-Q/graph-KV relation scores"
        ),
    )
    parser.add_argument(
        "--patch_router_training_noise_std",
        type=float,
        default=0.0,
        help=(
            "Optional standard deviation of training-only v5 router-logit noise. "
            "multi-expert noisy gating is opt-in; the default 0 keeps train "
            "and inference routing identical"
        ),
    )
    parser.add_argument(
        "--patch_router_local_weight",
        type=float,
        default=0.5,
        help=(
            "Weight of patch-local line-Q/graph-KV relation evidence in the v5 "
            "route; the patch-local decision scope is adaptive-granularity"
        ),
    )
    parser.add_argument(
        "--patch_router_relation_hidden_dim",
        type=int,
        default=16,
        help="Hidden width of the adaptive_v5 line-Q/graph-KV relation scorer",
    )
    parser.add_argument(
        "--patch_router_relation_residual_scale",
        type=float,
        default=0.25,
        help="Residual scale of the adaptive_v5 learned Q/K relation score",
    )
    parser.add_argument(
        "--patch_router_key_adapter_scale",
        type=float,
        default=0.1,
        help="Residual scale of each adaptive_v5 graph-key adapter",
    )
    parser.add_argument(
        "--patch_router_value_adapter_scale",
        type=float,
        default=0.1,
        help="Residual scale of each adaptive_v5 graph-value expert adapter",
    )
    parser.add_argument(
        "--patch_router_route_budget_weight",
        type=float,
        default=0.005,
        help=(
            "Weight of adaptive_v5 MSE between the sample-equal post-top-k "
            "expert marginal and uniform usage; paired with load balance, the "
            "default total router regularization is 0.01"
        ),
    )
    parser.add_argument(
        "--patch_router_load_balance_weight",
        type=float,
        default=0.005,
        help=(
            "Weight of the multi-expert adaptive_v5 CV-squared load "
            "penalty on the sample-equal pre-top-k clean-softmax marginal"
        ),
    )

    parser.add_argument(
        "--patch_size",
        type=str,
        choices=["sqrt", "linspace"],
        help="How to find the patch size for 2D segmentation",
    )

    parser.add_argument(
        "--stride",
        type=float,
        help="Stride as a fraction of patch size",
    )

    parser.add_argument(
        "--classifier_type",
        type=str,
        choices=[
            "logistic_regression",
            "nearest_centroid",
            "random_forest",
            "mlp",
        ],
        help="Classifier type",
    )

    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of the MLP classifier",
    )

    parser.add_argument(
        "--mlp_num_layers",
        type=int,
        default=2,
        help="Number of linear layers in the MLP classifier",
    )

    parser.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.1,
        help="Dropout probability of the MLP classifier",
    )

    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=1e-4,
        help="Learning rate for MLP/fusion-head training",
    )

    parser.add_argument(
        "--mlp_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for MLP/fusion-head training",
    )

    parser.add_argument(
        "--mlp_class_weight",
        choices=["none", "balanced"],
        default="none",
        help="Optional inverse-frequency class weighting for MLP training",
    )

    parser.add_argument(
        "--mlp_epochs",
        type=int,
        default=20,
        help="Number of training epochs for the MLP classifier",
    )

    parser.add_argument(
        "--mlp_early_stop_patience",
        type=int,
        default=0,
        help=(
            "Stop MLP training after this many epochs without improvement in "
            "the effective window- or subject-level checkpoint criterion"
        ),
    )

    parser.add_argument(
        "--mlp_early_stop_strategy",
        choices=["raw_selection_key", "raw_primary", "ema_primary"],
        default="raw_selection_key",
        help=(
            "Early-stopping monitor. raw_selection_key preserves the legacy "
            "checkpoint-key behavior; raw_primary applies min_delta to the "
            "unsmoothed primary validation metric; ema_primary smooths the primary "
            "validation metric while checkpoint saving still uses the raw "
            "criterion"
        ),
    )

    parser.add_argument(
        "--mlp_early_stop_warmup_epochs",
        type=int,
        default=0,
        help=(
            "Number of initial MLP epochs excluded from early-stopping and "
            "ReduceLROnPlateau monitoring"
        ),
    )

    parser.add_argument(
        "--mlp_early_stop_min_epochs",
        type=int,
        default=0,
        help=(
            "Minimum number of MLP epochs to run before an early-stop decision "
            "is allowed"
        ),
    )

    parser.add_argument(
        "--mlp_early_stop_ema_decay",
        type=float,
        default=0.6,
        help=(
            "EMA decay for --mlp_early_stop_strategy ema_primary; ignored by "
            "raw_selection_key"
        ),
    )

    parser.add_argument(
        "--mlp_early_stop_min_delta",
        type=float,
        default=0.0,
        help=(
            "Minimum increase in the smoothed primary validation metric that "
            "resets early-stopping patience"
        ),
    )

    parser.add_argument(
        "--mlp_lr_scheduler",
        choices=["none", "reduce_on_plateau"],
        default="none",
        help="Optional validation-metric learning-rate scheduler for MLP training",
    )

    parser.add_argument(
        "--mlp_lr_scheduler_patience",
        type=int,
        default=4,
        help="ReduceLROnPlateau patience after monitor warmup",
    )

    parser.add_argument(
        "--mlp_lr_scheduler_factor",
        type=float,
        default=0.5,
        help="Multiplicative ReduceLROnPlateau learning-rate factor",
    )

    parser.add_argument(
        "--mlp_lr_scheduler_min_lr",
        type=float,
        default=1.0e-6,
        help="Minimum learning rate used by ReduceLROnPlateau",
    )

    parser.add_argument(
        "--modal_interaction",
        type=str,
        choices=[
            "concat",
            "concat_attn",
            "cross_attn_gate",
            "masked_pretrain",
            "patch_mindts",
            "adaptive_granularity",
        ],
        default="concat",
        help=(
            "How to fuse branch embeddings in the MLP path. "
            "patch_mindts retains the historical post-encoding selector; "
            "adaptive_granularity selects 4/8/16 while constructing one "
            "Activity Graph, applies Line-Q/Graph-KV cross-attention, and "
            "uses concat_attn for final visual-Mantis fusion."
        ),
    )

    parser.add_argument(
        "--outer_patch_size",
        type=int,
        default=64,
        help=(
            "Length of the shared temporal patch used by line, Activity Graph, "
            "and Mantis branches in patch_mindts"
        ),
    )

    parser.add_argument(
        "--outer_patch_stride",
        type=int,
        default=64,
        help="Stride of the shared temporal patch in patch_mindts",
    )

    parser.add_argument(
        "--patch_alignment_dim",
        type=int,
        default=256,
        help="Projection width of the within-sample patch alignment objective",
    )

    parser.add_argument(
        "--patch_alignment_temperature",
        type=float,
        default=0.1,
        help="Temperature of the within-sample N-by-N patch alignment objective",
    )

    parser.add_argument(
        "--patch_alignment_weight",
        type=float,
        default=0.1,
        help="Weight of patch alignment relative to classification loss",
    )

    parser.add_argument(
        "--visual_encode_batch_size",
        type=int,
        default=16,
        help=(
            "Micro-batch size for frozen line/Activity-Graph image encoding in "
            "patch_mindts"
        ),
    )

    parser.add_argument(
        "--granularity_gate_temperature",
        type=float,
        default=0.5,
        help=(
            "Training temperature of the hard straight-through adaptive-granularity "
            "4/8/16 region gate used by adaptive_granularity"
        ),
    )

    parser.add_argument(
        "--granularity_balance_weight",
        type=float,
        default=0.001,
        help=(
            "Weight of the train-split gate-usage balance term in "
            "adaptive_granularity"
        ),
    )

    parser.add_argument(
        "--granularity_graph_token_grid",
        type=int,
        default=4,
        help=(
            "Side length of the retained Activity Graph spatial-token grid; "
            "the default 4 keeps 16 K/V tokens"
        ),
    )

    parser.add_argument(
        "--activity_graph_canvas_size",
        type=int,
        default=360,
        help=(
            "Square reference raster used for the Yang et al. Activity Graph; "
            "the paper reports 360 by 360"
        ),
    )

    parser.add_argument(
        "--activity_graph_line_width",
        type=float,
        default=1.0,
        help="Waveform line width in reference-canvas pixels",
    )

    parser.add_argument(
        "--activity_graph_vertical_margin",
        type=float,
        default=0.05,
        help=(
            "Explicit per-lane plotting margin used because the paper does not "
            "report waveform axis bounds"
        ),
    )

    parser.add_argument(
        "--granularity_gate_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional trusted checkpoint containing adaptive granularity region_cls "
            "weights. Omit to train the gate jointly."
        ),
    )

    parser.add_argument(
        "--granularity_freeze_gate",
        action="store_true",
        help="Freeze a loaded adaptive granularity region gate during classifier training",
    )

    parser.add_argument(
        "--fusion_dim",
        type=int,
        default=512,
        help="Shared fusion dimension for non-concat interaction modes",
    )

    parser.add_argument(
        "--fusion_heads",
        type=int,
        default=4,
        help="Number of attention heads used by fusion interaction modes",
    )

    parser.add_argument(
        "--cross_attn_query",
        type=str,
        choices=["ts", "visual"],
        default="ts",
        help="Which branch acts as the query in cross_attn_gate",
    )

    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.3,
        help="Mask probability for masked_pretrain branch reconstruction",
    )

    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=10,
        help="Number of masked_pretrain self-supervised pretraining epochs",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        choices=[
            "ucr",
            "uea",
            "uci",
            "flaap",
            "falltl",
            "feng",
            "wearable",
            "eeg",
        ],
        help="Time series classification benchmark",
    )

    parser.add_argument(
        "--har_channels",
        type=str,
        choices=["all", "acc_gyro"],
        default="all",
        help=(
            "Channel subset for FLAAP/UCI-HAR. acc_gyro uses Acc XYZ plus "
            "Gyro XYZ; for UCI-HAR, Acc XYZ means total_acc XYZ."
        ),
    )

    parser.add_argument(
        "--uci_protocol",
        choices=["official_subject", "legacy_resplit"],
        default="official_subject",
        help=(
            "UCI-HAR split protocol. official_subject preserves the published "
            "subject-disjoint test partition and draws a fixed seed-42, "
            "subject-disjoint validation split only from official training subjects. "
            "legacy_resplit is retained only to inspect "
            "historical runs that merged and randomly re-split both partitions."
        ),
    )

    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        help="Optional dataset names to run within the selected benchmark",
    )

    parser.add_argument(
        "--wearable_label_mode",
        choices=[
            "original",
            "shimmer_hc_vs_pd",
            "pads_pd_vs_hc",
            "pads_pd_vs_omd",
        ],
        default="original",
        help=(
            "Explicit clinical endpoint for PADS/Shimmer. Shimmer merges MildPD "
            "and ModeratePD against HC; PADS supports PD-vs-healthy or "
            "PD-vs-other-movement-disorders without collapsing all diagnoses."
        ),
    )

    parser.add_argument(
        "--eeg_protocol",
        choices=["medformer_code_exact"],
        default="medformer_code_exact",
        help=(
            "Fixed Medformer processed-data subject splits for TDBRAIN, APAVA, "
            "and ADFTD. TDBRAIN uses the pinned official legacy-50 cohort."
        ),
    )

    parser.add_argument(
        "--eeg_normalization",
        choices=["per_window_per_channel_standard_scaler_ddof0"],
        default="per_window_per_channel_standard_scaler_ddof0",
        help=(
            "Normalize every one-second EEG window independently, per channel "
            "over its 256 time points, using population standard deviation."
        ),
    )

    parser.add_argument(
        "--falltl_protocol",
        choices=["legacy_windows", "comparison_binary"],
        default="comparison_binary",
        help=(
            "FallTL data protocol. comparison_binary uses one variable-length "
            "sequence per CSV, D/F binary labels, linear interpolation, "
            "training-split channel statistics, zero padding, and a fixed "
            "seed-42 60/20/20 stratified file-level split. legacy_windows is "
            "retained only to inspect historical window-level runs."
        ),
    )

    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        help=(
            "Optional directory for frozen model-layer features. Cached split "
            "features are reused by subsequent runs with matching labels and branches."
        ),
    )
    parser.add_argument(
        "--reuse_static_cache_dir",
        default=None,
        help="Optional existing cache root to search for strictly validated raw/line/Mantis cache reuse",
    )

    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for dataloader"
    )

    parser.add_argument(
        "--aeon",
        action="store_true",
        help="Activate aeon data preprocessing",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory where datasets are stored",
    )

    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to the directory where results are stored",
    )

    parser.add_argument(
        "--measure_alignment",
        action="store_true",
        help="Measure the alignment of model representations",
    )

    parser.add_argument(
        "--get_intrinsic_dimension",
        action="store_true",
        help="Compute the intrinsic dimension of representations",
    )

    parser.add_argument(
        "--get_principal_components",
        action="store_true",
        help="Compute the number of principal components required to cover 95 percent of the representation variance",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Change random seed for experiments",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.25,
        help=(
            "Validation ratio sampled from the training split "
            "(0.25 gives an overall 60/20/20 split when test ratio is 0.2)"
        ),
    )

    parser.add_argument(
        "--custom_test_ratio",
        type=float,
        default=0.2,
        help="Test ratio used to split the combined dataset",
    )

    parser.add_argument(
        "--falltl_target_length",
        type=int,
        default=2048,
        help=(
            "Common full-trial length used by FallTL comparison_binary after "
            "linear resampling; the same value is applied to every split"
        ),
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=200,
        help="Sliding-window length for custom CSV datasets",
    )

    parser.add_argument(
        "--window_stride",
        type=int,
        default=100,
        help="Sliding-window stride for custom CSV datasets",
    )

    parser.add_argument(
        "--max_windows_per_file",
        type=int,
        help=(
            "Optional maximum number of windows to keep from each Feng or "
            "legacy FallTL CSV file"
        ),
    )

    parser.add_argument(
        "--save_activity_graph_samples",
        type=int,
        default=0,
        help="Number of activity graph sample images to save per dataset",
    )

    parser.add_argument(
        "--save_activity_lineplot_samples",
        type=int,
        default=0,
        help="Number of activity line plot sample images to save per dataset",
    )

    args = parser.parse_args(normalize_cli_arguments(sys.argv[1:]))
    if args.med_activity_granularity_bank is None:
        args.med_activity_granularity_bank = ((4,), (8,), (16,))
    if args.med_activity_granularity_temperature is None:
        args.med_activity_granularity_temperature = 1.0
    if args.med_activity_granularity_entropy_weight is None:
        args.med_activity_granularity_entropy_weight = (
            0.01 if args.modal_interaction == "patch_mindts" else 0.001
        )
    if args.med_activity_granularity_entropy_floor is None:
        args.med_activity_granularity_entropy_floor = (
            0.55 if args.modal_interaction == "patch_mindts" else 0.0
        )
    if not 0.0 <= args.med_activity_channel_mix <= 1.0:
        parser.error("--med_activity_channel_mix must be in [0, 1]")
    if (
        not math.isfinite(args.med_activity_router_temperature)
        or args.med_activity_router_temperature <= 0.0
    ):
        parser.error("--med_activity_router_temperature must be positive")
    if not 0.0 <= args.med_activity_router_mix <= 1.0:
        parser.error("--med_activity_router_mix must be in [0, 1]")
    if args.med_activity_granularity_hidden_dim <= 0:
        parser.error("--med_activity_granularity_hidden_dim must be positive")
    if (
        not math.isfinite(args.med_activity_granularity_temperature)
        or args.med_activity_granularity_temperature <= 0.0
    ):
        parser.error("--med_activity_granularity_temperature must be positive")
    if not 0.0 < args.med_activity_granularity_base_prior < 1.0:
        parser.error("--med_activity_granularity_base_prior must be in (0, 1)")
    if (
        not math.isfinite(args.med_activity_granularity_balance_weight)
        or args.med_activity_granularity_balance_weight < 0.0
    ):
        parser.error(
            "--med_activity_granularity_balance_weight must be non-negative"
        )
    if (
        not math.isfinite(args.med_activity_granularity_entropy_weight)
        or args.med_activity_granularity_entropy_weight < 0.0
    ):
        parser.error(
            "--med_activity_granularity_entropy_weight must be non-negative"
        )
    for option_name in (
        "med_activity_granularity_mix_shrinkage_weight",
        "med_activity_granularity_prior_kl_weight",
    ):
        option_value = getattr(args, option_name)
        if not math.isfinite(option_value) or option_value < 0.0:
            parser.error(f"--{option_name} must be finite and non-negative")
    if (
        not math.isfinite(args.med_activity_granularity_usage_floor)
        or not 0.0 <= args.med_activity_granularity_usage_floor < 1.0
    ):
        parser.error("--med_activity_granularity_usage_floor must be in [0, 1)")
    if (
        not math.isfinite(args.med_activity_granularity_usage_ema_decay)
        or not 0.0 <= args.med_activity_granularity_usage_ema_decay < 1.0
    ):
        parser.error(
            "--med_activity_granularity_usage_ema_decay must be in [0, 1)"
        )
    if (
        not math.isfinite(args.med_activity_granularity_entropy_floor)
        or not 0.0 <= args.med_activity_granularity_entropy_floor <= 1.0
    ):
        parser.error(
            "--med_activity_granularity_entropy_floor must be in [0, 1]"
        )
    if (
        not math.isfinite(args.med_activity_granularity_entropy_ceiling)
        or not 0.0 <= args.med_activity_granularity_entropy_ceiling <= 1.0
    ):
        parser.error(
            "--med_activity_granularity_entropy_ceiling must be in [0, 1]"
        )
    if (
        args.med_activity_granularity_entropy_floor
        > args.med_activity_granularity_entropy_ceiling
    ):
        parser.error(
            "--med_activity_granularity_entropy_floor cannot exceed the ceiling"
        )
    for prefix in ("local", "global"):
        maximum = getattr(args, f"med_activity_granularity_{prefix}_mix_max")
        initial = getattr(args, f"med_activity_granularity_{prefix}_mix_init")
        if not math.isfinite(maximum) or not 0.0 < maximum <= 1.0:
            parser.error(
                f"--med_activity_granularity_{prefix}_mix_max must be in (0, 1]"
            )
        if not math.isfinite(initial) or not 0.0 < initial < maximum:
            parser.error(
                f"--med_activity_granularity_{prefix}_mix_init must be in (0, max)"
            )
    if (
        not math.isfinite(
            args.med_activity_granularity_evidence_half_saturation
        )
        or args.med_activity_granularity_evidence_half_saturation <= 0.0
    ):
        parser.error(
            "--med_activity_granularity_evidence_half_saturation must be positive"
        )
    if (
        not math.isfinite(args.med_activity_granularity_minimum_weight)
        or args.med_activity_granularity_minimum_weight < 0.0
    ):
        parser.error(
            "--med_activity_granularity_minimum_weight must be non-negative"
        )
    if (
        not math.isfinite(args.med_activity_granularity_score_cap)
        or args.med_activity_granularity_score_cap <= 0.0
    ):
        parser.error("--med_activity_granularity_score_cap must be positive")
    if args.med_activity_granularity_scorer_hidden_dim <= 0:
        parser.error(
            "--med_activity_granularity_scorer_hidden_dim must be positive"
        )
    if (
        not math.isfinite(
            args.med_activity_granularity_confidence_half_saturation
        )
        or args.med_activity_granularity_confidence_half_saturation <= 0.0
    ):
        parser.error(
            "--med_activity_granularity_confidence_half_saturation must be positive"
        )
    if args.patch_router_top_k <= 0:
        parser.error("--patch_router_top_k must be positive")
    if (
        args.patch_granularity_router_mode == "adaptive_v5"
        and args.patch_router_top_k > len(args.med_activity_granularity_bank)
    ):
        parser.error(
            "--patch_router_top_k cannot exceed the number of graph experts "
            "for adaptive_v5"
        )
    if (
        not math.isfinite(args.patch_router_training_noise_std)
        or args.patch_router_training_noise_std < 0.0
    ):
        parser.error(
            "--patch_router_training_noise_std must be finite and non-negative"
        )
    if (
        not math.isfinite(args.patch_router_local_weight)
        or not 0.0 <= args.patch_router_local_weight <= 1.0
    ):
        parser.error("--patch_router_local_weight must be in [0, 1]")
    if args.patch_router_relation_hidden_dim <= 0:
        parser.error("--patch_router_relation_hidden_dim must be positive")
    for option_name in (
        "patch_router_relation_residual_scale",
        "patch_router_key_adapter_scale",
        "patch_router_value_adapter_scale",
        "patch_router_route_budget_weight",
        "patch_router_load_balance_weight",
    ):
        option_value = getattr(args, option_name)
        if not math.isfinite(option_value) or option_value < 0.0:
            parser.error(f"--{option_name} must be finite and non-negative")
    if args.mlp_epochs <= 0:
        parser.error("--mlp_epochs must be positive")
    if args.mlp_early_stop_patience < 0:
        parser.error("--mlp_early_stop_patience must be non-negative")
    if (
        args.mlp_early_stop_warmup_epochs < 0
        or args.mlp_early_stop_warmup_epochs > args.mlp_epochs
    ):
        parser.error(
            "--mlp_early_stop_warmup_epochs must lie in [0, --mlp_epochs]"
        )
    if (
        args.mlp_early_stop_min_epochs < 0
        or args.mlp_early_stop_min_epochs > args.mlp_epochs
    ):
        parser.error("--mlp_early_stop_min_epochs must lie in [0, --mlp_epochs]")
    if (
        not math.isfinite(args.mlp_early_stop_ema_decay)
        or not 0.0 <= args.mlp_early_stop_ema_decay < 1.0
    ):
        parser.error("--mlp_early_stop_ema_decay must lie in [0, 1)")
    if (
        not math.isfinite(args.mlp_early_stop_min_delta)
        or args.mlp_early_stop_min_delta < 0.0
    ):
        parser.error("--mlp_early_stop_min_delta must be finite and non-negative")
    if args.mlp_lr_scheduler_patience < 0:
        parser.error("--mlp_lr_scheduler_patience must be non-negative")
    if (
        not math.isfinite(args.mlp_lr_scheduler_factor)
        or not 0.0 < args.mlp_lr_scheduler_factor < 1.0
    ):
        parser.error("--mlp_lr_scheduler_factor must lie in (0, 1)")
    if (
        not math.isfinite(args.mlp_lr_scheduler_min_lr)
        or args.mlp_lr_scheduler_min_lr < 0.0
        or args.mlp_lr_scheduler_min_lr > args.mlp_lr
    ):
        parser.error(
            "--mlp_lr_scheduler_min_lr must be finite and lie in [0, --mlp_lr]"
        )
    if args.pretrain_epochs < 0:
        parser.error("--pretrain_epochs must be non-negative")
    if args.outer_patch_size <= 0:
        parser.error("--outer_patch_size must be positive")
    if args.outer_patch_stride <= 0:
        parser.error("--outer_patch_stride must be positive")
    if args.patch_alignment_dim <= 0:
        parser.error("--patch_alignment_dim must be positive")
    if (
        not math.isfinite(args.patch_alignment_temperature)
        or args.patch_alignment_temperature <= 0.0
    ):
        parser.error("--patch_alignment_temperature must be positive and finite")
    if (
        not math.isfinite(args.patch_alignment_weight)
        or args.patch_alignment_weight < 0.0
    ):
        parser.error("--patch_alignment_weight must be finite and non-negative")
    if args.visual_encode_batch_size <= 0:
        parser.error("--visual_encode_batch_size must be positive")
    if (
        not math.isfinite(args.granularity_gate_temperature)
        or args.granularity_gate_temperature <= 0.0
    ):
        parser.error("--granularity_gate_temperature must be positive and finite")
    if (
        not math.isfinite(args.granularity_balance_weight)
        or args.granularity_balance_weight < 0.0
    ):
        parser.error(
            "--granularity_balance_weight must be finite and non-negative"
        )
    if args.granularity_graph_token_grid < 2:
        parser.error(
            "--granularity_graph_token_grid must be at least 2 so cross-attention "
            "has more than one graph K/V token"
        )
    if args.activity_graph_canvas_size < 3 or args.activity_graph_canvas_size % 3:
        parser.error("--activity_graph_canvas_size must be a multiple of three")
    if (
        not math.isfinite(args.activity_graph_line_width)
        or args.activity_graph_line_width <= 0.0
    ):
        parser.error("--activity_graph_line_width must be positive and finite")
    if (
        not math.isfinite(args.activity_graph_vertical_margin)
        or args.activity_graph_vertical_margin < 0.0
        or args.activity_graph_vertical_margin >= 0.5
    ):
        parser.error("--activity_graph_vertical_margin must lie in [0, 0.5)")
    if args.granularity_freeze_gate and not args.granularity_gate_checkpoint:
        parser.error(
            "--granularity_freeze_gate requires --granularity_gate_checkpoint"
        )
    if args.med_activity_adaptive_granularity:
        if args.image_mode != "med_activity_graph":
            parser.error(
                "--med_activity_adaptive_granularity requires "
                "--image_mode med_activity_graph"
            )
        if args.classifier_type != "mlp":
            parser.error(
                "--med_activity_adaptive_granularity requires "
                "--classifier_type mlp"
            )
        if not (args.vit_1_name or args.vit_2_name):
            parser.error(
                "--med_activity_adaptive_granularity requires at least one ViT branch"
            )
        if (
            any(len(regime) != 1 for regime in args.med_activity_granularity_bank)
            and args.med_activity_granularity_bank.count(
                args.med_activity_patch_lengths
            )
            != 1
        ):
            parser.error(
                "--med_activity_granularity_bank must contain "
                "--med_activity_patch_lengths exactly once"
            )
    if args.modal_interaction == "patch_mindts":
        if args.classifier_type != "mlp":
            parser.error("--modal_interaction patch_mindts requires --classifier_type mlp")
        if args.image_mode != "med_activity_graph":
            parser.error(
                "--modal_interaction patch_mindts requires "
                "--image_mode med_activity_graph"
            )
        if not args.med_activity_adaptive_granularity:
            parser.error(
                "--modal_interaction patch_mindts requires "
                "--med_activity_adaptive_granularity"
            )
        if not args.vit_1_name:
            parser.error("--modal_interaction patch_mindts requires --vit_1_name")
        if args.aggregation not in {"mean", "cls_token"}:
            parser.error(
                "patch_mindts requires --aggregation mean or --aggregation cls_token"
            )
        if args.vit_1_layer is None or (
            args.vit_1_layer != -1 and args.vit_1_layer <= 0
        ):
            parser.error(
                "patch_mindts requires --vit_1_layer to be a positive integer or -1"
            )
        if args.vit_2_name:
            parser.error(
                "patch_mindts uses one shared visual encoder; --vit_2_name is not supported"
            )
        if not args.mantis:
            parser.error("--modal_interaction patch_mindts requires --mantis")
        if args.moment:
            parser.error("patch_mindts does not use MOMENT in its first implementation")
        if len(args.med_activity_granularity_bank) != 3:
            parser.error("patch_mindts requires exactly three Activity Graph regimes")
        regime_widths = {
            len(regime) for regime in args.med_activity_granularity_bank
        }
        if regime_widths != {1}:
            parser.error(
                "patch_mindts requires three single-scale waveform experts"
            )
        if len(set(args.med_activity_granularity_bank)) != 3:
            parser.error(
                "patch_mindts Activity Graph experts must be distinct"
            )
        if args.med_activity_granularity_usage_floor > 1.0 / len(
            args.med_activity_granularity_bank
        ):
            parser.error(
                "--med_activity_granularity_usage_floor cannot exceed "
                "uniform per-expert usage in patch_mindts"
            )
        if args.med_activity_granularity_minimum_weight >= 1.0 / len(
            args.med_activity_granularity_bank
        ):
            parser.error(
                "--med_activity_granularity_minimum_weight must be below "
                "uniform per-expert usage"
            )
        maximum_internal_scale = max(
            max(regime) for regime in args.med_activity_granularity_bank
        )
        if args.outer_patch_size <= maximum_internal_scale:
            parser.error(
                "--outer_patch_size must be larger than the largest internal "
                f"Activity Graph scale ({maximum_internal_scale})"
            )
        if args.outer_patch_stride > args.outer_patch_size:
            parser.error(
                "--outer_patch_stride cannot exceed --outer_patch_size because "
                "that would leave time points uncovered"
            )
        if args.fusion_dim <= 0:
            parser.error("--fusion_dim must be positive in patch_mindts")
        if args.fusion_heads <= 0:
            parser.error("--fusion_heads must be positive in patch_mindts")
        if args.fusion_dim % args.fusion_heads != 0:
            parser.error(
                "--fusion_dim must be divisible by --fusion_heads in patch_mindts"
            )
    if args.modal_interaction == "adaptive_granularity":
        if args.classifier_type != "mlp":
            parser.error(
                "--modal_interaction adaptive_granularity requires "
                "--classifier_type mlp"
            )
        if args.image_mode != "med_activity_graph":
            parser.error(
                "--modal_interaction adaptive_granularity requires "
                "--image_mode med_activity_graph"
            )
        if args.aggregation not in {"mean", "cls_token"}:
            parser.error(
                "adaptive_granularity requires --aggregation mean or cls_token"
            )
        if args.med_activity_adaptive_granularity:
            parser.error(
                "adaptive_granularity performs its own pre-render selection; "
                "do not enable the historical --med_activity_adaptive_granularity bank"
            )
        if not args.vit_1_name:
            parser.error(
                "--modal_interaction adaptive_granularity requires --vit_1_name"
            )
        if args.vit_1_layer is None or (
            args.vit_1_layer != -1 and args.vit_1_layer <= 0
        ):
            parser.error(
                "adaptive_granularity requires --vit_1_layer to be a positive "
                "integer or -1"
            )
        if args.vit_2_name:
            parser.error(
                "adaptive_granularity uses one shared visual encoder; "
                "--vit_2_name is not supported"
            )
        if not args.mantis:
            parser.error("adaptive_granularity requires --mantis")
        if args.moment:
            parser.error("adaptive_granularity does not use MOMENT")
        if args.outer_patch_size != 64:
            parser.error(
                "adaptive_granularity currently defines each outer window as "
                "exactly 64 samples"
            )
        if args.granularity_graph_token_grid != 4:
            parser.error(
                "adaptive_granularity currently fixes the retained graph token "
                "grid at 4x4 (16 K/V tokens)"
            )
        if args.outer_patch_stride > args.outer_patch_size:
            parser.error(
                "--outer_patch_stride cannot exceed --outer_patch_size because "
                "that would leave time points uncovered"
            )
        if args.fusion_dim <= 0 or args.fusion_heads <= 0:
            parser.error(
                "--fusion_dim and --fusion_heads must be positive in "
                "adaptive_granularity"
            )
        if args.fusion_dim % args.fusion_heads != 0:
            parser.error(
                "--fusion_dim must be divisible by --fusion_heads in "
                "adaptive_granularity"
            )
    if args.falltl_target_length <= 0:
        parser.error("--falltl_target_length must be positive")
    return args
