#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

MODEL_DIR="${MODEL_DIR:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-../Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
EEG_DATA_DIR="${EEG_DATA_DIR:-$PROJECT_DIR/data/eeg/processed}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VISUAL_BATCH_SIZE="${VISUAL_BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
DRY_RUN="${DRY_RUN:-0}"

ROUTER_MODE="${ROUTER_MODE:-adaptive_v4}"
CANDIDATE_MODE="${CANDIDATE_MODE:-single}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-auto}"
GRANULARITY_TEMPERATURE="${GRANULARITY_TEMPERATURE:-1.0}"
GRANULARITY_USAGE_FLOOR="${GRANULARITY_USAGE_FLOOR:-0.05}"
GRANULARITY_USAGE_EMA_DECAY="${GRANULARITY_USAGE_EMA_DECAY:-0.95}"
GRANULARITY_ENTROPY_FLOOR="${GRANULARITY_ENTROPY_FLOOR:-0.55}"
GRANULARITY_ENTROPY_CEILING="${GRANULARITY_ENTROPY_CEILING:-1.0}"
GRANULARITY_LOCAL_MIX_MAX="${GRANULARITY_LOCAL_MIX_MAX:-0.50}"
GRANULARITY_LOCAL_MIX_INIT="${GRANULARITY_LOCAL_MIX_INIT:-0.10}"
GRANULARITY_GLOBAL_MIX_MAX="${GRANULARITY_GLOBAL_MIX_MAX:-0.75}"
GRANULARITY_GLOBAL_MIX_INIT="${GRANULARITY_GLOBAL_MIX_INIT:-0.50}"
GRANULARITY_EVIDENCE_HALF_SATURATION="${GRANULARITY_EVIDENCE_HALF_SATURATION:-0.05}"
GRANULARITY_MINIMUM_WEIGHT="${GRANULARITY_MINIMUM_WEIGHT:-0.0}"
GRANULARITY_SCORE_CAP="${GRANULARITY_SCORE_CAP:-1.0}"

case "$ROUTER_MODE" in
  adaptive_v4)
    DEFAULT_GRANULARITY_BALANCE_WEIGHT=0.01
    DEFAULT_GRANULARITY_ENTROPY_WEIGHT=0.01
    DEFAULT_GRANULARITY_MIX_SHRINKAGE_WEIGHT=0.005
    DEFAULT_GRANULARITY_PRIOR_KL_WEIGHT=0.001
    ;;
  uniform)
    # A formal uniform baseline has exact 1/3 routing and no router regularizers.
    DEFAULT_GRANULARITY_BALANCE_WEIGHT=0.0
    DEFAULT_GRANULARITY_ENTROPY_WEIGHT=0.0
    DEFAULT_GRANULARITY_MIX_SHRINKAGE_WEIGHT=0.0
    DEFAULT_GRANULARITY_PRIOR_KL_WEIGHT=0.0
    ;;
  *)
    echo "ROUTER_MODE must be adaptive_v4 or uniform, got: $ROUTER_MODE" >&2
    exit 2
    ;;
esac

GRANULARITY_BALANCE_WEIGHT="${GRANULARITY_BALANCE_WEIGHT:-$DEFAULT_GRANULARITY_BALANCE_WEIGHT}"
GRANULARITY_ENTROPY_WEIGHT="${GRANULARITY_ENTROPY_WEIGHT:-$DEFAULT_GRANULARITY_ENTROPY_WEIGHT}"
GRANULARITY_MIX_SHRINKAGE_WEIGHT="${GRANULARITY_MIX_SHRINKAGE_WEIGHT:-$DEFAULT_GRANULARITY_MIX_SHRINKAGE_WEIGHT}"
GRANULARITY_PRIOR_KL_WEIGHT="${GRANULARITY_PRIOR_KL_WEIGHT:-$DEFAULT_GRANULARITY_PRIOR_KL_WEIGHT}"

case "$CANDIDATE_MODE" in
  single)
    GRANULARITY_BANK="4;8;16"
    ;;
  composite)
    GRANULARITY_BANK="1,2,4;2,4,8;4,8,16"
    ;;
  *)
    echo "CANDIDATE_MODE must be single or composite, got: $CANDIDATE_MODE" >&2
    exit 2
    ;;
esac

DATASET_KEY="${1:-${DATASET_KEY:-}}"
if [[ $# -gt 0 ]]; then
  shift
fi
case "$DATASET_KEY" in
  tdbrain) DATASET_NAME="TDBRAIN" ;;
  apava) DATASET_NAME="APAVA" ;;
  adftd) DATASET_NAME="ADFTD" ;;
  *)
    echo "Usage: $0 {tdbrain|apava|adftd} [extra main.py arguments]" >&2
    exit 2
    ;;
esac

RESULT_DIR="${RESULT_DIR:-results/eeg_patch_mindts_${CANDIDATE_MODE}_${ROUTER_MODE}_seed${SEED}/${DATASET_KEY}}"
# Router-independent frozen features are intentionally shared by uniform and v4.
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-feature_cache/eeg_patch_mindts_${CANDIDATE_MODE}_seed${SEED}/${DATASET_KEY}}"

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU must be non-negative" >&2; exit 1; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE must be positive" >&2; exit 1; }
[[ "$VISUAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "VISUAL_BATCH_SIZE must be positive" >&2; exit 1; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be positive" >&2; exit 1; }
[[ "$PATIENCE" =~ ^[0-9]+$ ]] || { echo "PATIENCE must be non-negative" >&2; exit 1; }

command=(
  "$PYTHON_BIN" main.py
  --vit_1_name "$MODEL_DIR"
  --vit_1_layer 14
  --aggregation mean
  --image_mode med_activity_graph
  --med_activity_patch_lengths 4,8,16
  --med_activity_adaptive_granularity
  --med_activity_granularity_bank "$GRANULARITY_BANK"
  --med_activity_granularity_temperature "$GRANULARITY_TEMPERATURE"
  --med_activity_granularity_balance_weight "$GRANULARITY_BALANCE_WEIGHT"
  --med_activity_granularity_entropy_weight "$GRANULARITY_ENTROPY_WEIGHT"
  --med_activity_granularity_mix_shrinkage_weight "$GRANULARITY_MIX_SHRINKAGE_WEIGHT"
  --med_activity_granularity_prior_kl_weight "$GRANULARITY_PRIOR_KL_WEIGHT"
  --med_activity_granularity_usage_floor "$GRANULARITY_USAGE_FLOOR"
  --med_activity_granularity_usage_ema_decay "$GRANULARITY_USAGE_EMA_DECAY"
  --med_activity_granularity_entropy_floor "$GRANULARITY_ENTROPY_FLOOR"
  --med_activity_granularity_entropy_ceiling "$GRANULARITY_ENTROPY_CEILING"
  --med_activity_granularity_local_mix_max "$GRANULARITY_LOCAL_MIX_MAX"
  --med_activity_granularity_local_mix_init "$GRANULARITY_LOCAL_MIX_INIT"
  --med_activity_granularity_global_mix_max "$GRANULARITY_GLOBAL_MIX_MAX"
  --med_activity_granularity_global_mix_init "$GRANULARITY_GLOBAL_MIX_INIT"
  --med_activity_granularity_evidence_half_saturation "$GRANULARITY_EVIDENCE_HALF_SATURATION"
  --med_activity_granularity_minimum_weight "$GRANULARITY_MINIMUM_WEIGHT"
  --med_activity_granularity_score_cap "$GRANULARITY_SCORE_CAP"
  --mantis
  --mantis_name "$MANTIS_DIR"
  --classifier_type mlp
  --modal_interaction patch_mindts
  --patch_granularity_router_mode "$ROUTER_MODE"
  --patch_checkpoint_metric "$CHECKPOINT_METRIC"
  --outer_patch_size 64
  --outer_patch_stride 64
  --patch_alignment_dim 256
  --patch_alignment_temperature 0.1
  --patch_alignment_weight 0.1
  --visual_encode_batch_size "$VISUAL_BATCH_SIZE"
  --fusion_dim 128
  --fusion_heads 2
  --mlp_hidden_dim 128
  --mlp_num_layers 2
  --mlp_dropout 0.1
  --mlp_lr 3e-4
  --mlp_weight_decay 1e-3
  --mlp_class_weight balanced
  --mlp_epochs "$EPOCHS"
  --mlp_early_stop_patience "$PATIENCE"
  --batch_size "$BATCH_SIZE"
  --data_dir "$EEG_DATA_DIR"
  --datasets eeg
  --dataset_names "$DATASET_NAME"
  --eeg_protocol medformer_code_exact
  --eeg_normalization per_window_per_channel_standard_scaler_ddof0
  --random_seed "$SEED"
  --feature_cache_dir "$FEATURE_CACHE_DIR"
  --result_dir "$RESULT_DIR"
  "$@"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU" "${command[@]}"
  printf '\n'
  exit 0
fi

for path in "$MODEL_DIR" "$MANTIS_DIR" "$EEG_DATA_DIR/$DATASET_NAME/Feature"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 1; }

mkdir -p "$RESULT_DIR" "$FEATURE_CACHE_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/neurosigvit_patch_mindts_${DATASET_KEY}_mpl}"
export HF_HOME="${HF_HOME:-/tmp/neurosigvit_patch_mindts_${DATASET_KEY}_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
exec "${command[@]}"
