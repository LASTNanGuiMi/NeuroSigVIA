#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

# Runtime locations are intentionally configurable. The NEUROSIGVIT_* aliases
# match the portable reproduction bundle; the shorter names match old launchers.
MODEL_DIR="${MODEL_DIR:-${NEUROSIGVIT_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}"
MANTIS_DIR="${MANTIS_DIR:-${NEUROSIGVIT_MANTIS_PATH:-../models/Mantis-8M}}"
EEG_DATA_DIR="${EEG_DATA_DIR:-${NEUROSIGVIT_EEG_ROOT:-$PROJECT_DIR/data/eeg/processed}}"
WEARABLE_DATA_ROOT="${WEARABLE_DATA_ROOT:-${NEUROSIGVIT_WEARABLE_ROOT:-$PROJECT_DIR/data/wearable}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
VISUAL_BATCH_SIZE="${VISUAL_BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-12}"
GATE_TEMPERATURE="${GATE_TEMPERATURE:-0.5}"
SELECTOR_BALANCE_WEIGHT="${SELECTOR_BALANCE_WEIGHT:-0.001}"
CHANNEL_MIX="${CHANNEL_MIX:-0.35}"
ALIGNMENT_WEIGHT="${ALIGNMENT_WEIGHT:-0.1}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-auto}"
GATE_CHECKPOINT="${GATE_CHECKPOINT:-}"
FREEZE_GATE="${FREEZE_GATE:-0}"
DRY_RUN="${DRY_RUN:-0}"

DATASET_KEY="${1:-${DATASET_KEY:-}}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$DATASET_KEY" in
  adftd)
    DATASET_GROUP="eeg"
    DATASET_NAME="ADFTD"
    DATA_DIR="$EEG_DATA_DIR"
    DEFAULT_BATCH_SIZE=8
    DATASET_ARGS=(
      --eeg_protocol medformer_code_exact
      --eeg_normalization per_window_per_channel_standard_scaler_ddof0
    )
    ;;
  tdbrain)
    DATASET_GROUP="eeg"
    DATASET_NAME="TDBRAIN"
    DATA_DIR="$EEG_DATA_DIR"
    DEFAULT_BATCH_SIZE=8
    DATASET_ARGS=(
      --eeg_protocol medformer_code_exact
      --eeg_normalization per_window_per_channel_standard_scaler_ddof0
    )
    ;;
  apava)
    DATASET_GROUP="eeg"
    DATASET_NAME="APAVA"
    DATA_DIR="$EEG_DATA_DIR"
    DEFAULT_BATCH_SIZE=8
    DATASET_ARGS=(
      --eeg_protocol medformer_code_exact
      --eeg_normalization per_window_per_channel_standard_scaler_ddof0
    )
    ;;
  shimmer10)
    DATASET_GROUP="wearable"
    DATASET_NAME="Shimmer_10_session10_AFC"
    DATA_DIR="$WEARABLE_DATA_ROOT"
    DEFAULT_BATCH_SIZE=1
    DATASET_ARGS=(--wearable_label_mode shimmer_hc_vs_pd)
    ;;
  pads11)
    DATASET_GROUP="wearable"
    DATASET_NAME="PADS_11_task08_TouchIndex"
    DATA_DIR="$WEARABLE_DATA_ROOT"
    DEFAULT_BATCH_SIZE=4
    DATASET_ARGS=(--wearable_label_mode pads_pd_vs_hc)
    ;;
  *)
    echo "Usage: $0 {adftd|tdbrain|apava|shimmer10|pads11} [extra main.py arguments]" >&2
    exit 2
    ;;
esac

BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
RESULT_DIR="${RESULT_DIR:-results/timemosaic_graph_seed${SEED}/${DATASET_KEY}}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-feature_cache/timemosaic_graph_seed${SEED}/${DATASET_KEY}}"

[[ -n "$GPU" ]] || { echo "GPU must not be empty" >&2; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be non-negative" >&2; exit 1; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE must be positive" >&2; exit 1; }
[[ "$VISUAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "VISUAL_BATCH_SIZE must be positive" >&2; exit 1; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be positive" >&2; exit 1; }
[[ "$PATIENCE" =~ ^[0-9]+$ ]] || { echo "PATIENCE must be non-negative" >&2; exit 1; }
case "$FREEZE_GATE" in
  0|1) ;;
  *) echo "FREEZE_GATE must be 0 or 1" >&2; exit 1 ;;
esac
if [[ "$FREEZE_GATE" == "1" && -z "$GATE_CHECKPOINT" ]]; then
  echo "FREEZE_GATE=1 requires GATE_CHECKPOINT" >&2
  exit 1
fi

command=(
  "$PYTHON_BIN" main.py
  --vit_1_name "$MODEL_DIR"
  --vit_1_layer 14
  --aggregation mean
  --image_mode med_activity_graph
  --med_activity_channel_mix "$CHANNEL_MIX"
  --mantis
  --mantis_name "$MANTIS_DIR"
  --classifier_type mlp
  --modal_interaction patch_timemosaic_graph
  --outer_patch_size 64
  --outer_patch_stride 64
  --timemosaic_gate_temperature "$GATE_TEMPERATURE"
  --timemosaic_selector_balance_weight "$SELECTOR_BALANCE_WEIGHT"
  --timemosaic_graph_token_grid 4
  --patch_alignment_dim 256
  --patch_alignment_temperature 0.1
  --patch_alignment_weight "$ALIGNMENT_WEIGHT"
  --patch_checkpoint_metric "$CHECKPOINT_METRIC"
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
  --mlp_early_stop_strategy raw_primary
  --mlp_early_stop_warmup_epochs 10
  --mlp_early_stop_patience "$PATIENCE"
  --mlp_early_stop_min_delta 0.002
  --mlp_lr_scheduler reduce_on_plateau
  --mlp_lr_scheduler_patience 4
  --mlp_lr_scheduler_factor 0.5
  --mlp_lr_scheduler_min_lr 1e-6
  --batch_size "$BATCH_SIZE"
  --data_dir "$DATA_DIR"
  --datasets "$DATASET_GROUP"
  --dataset_names "$DATASET_NAME"
  "${DATASET_ARGS[@]}"
  --random_seed "$SEED"
  --feature_cache_dir "$FEATURE_CACHE_DIR"
  --result_dir "$RESULT_DIR"
)

if [[ -n "$GATE_CHECKPOINT" ]]; then
  command+=(--timemosaic_gate_checkpoint "$GATE_CHECKPOINT")
fi
if [[ "$FREEZE_GATE" == "1" ]]; then
  command+=(--timemosaic_freeze_gate)
fi
command+=("$@")

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU" "${command[@]}"
  printf '\n'
  exit 0
fi

for path in "$MODEL_DIR" "$MANTIS_DIR" "$DATA_DIR"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$RESULT_DIR" "$FEATURE_CACHE_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/neurosigvit_timemosaic_${DATASET_KEY}_mpl}"
export HF_HOME="${HF_HOME:-/tmp/neurosigvit_timemosaic_${DATASET_KEY}_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
exec "${command[@]}"
