#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

MODEL_DIR="${MODEL_DIR:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-../Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
CLINICAL_DATA_DIR="${CLINICAL_DATA_DIR:-data}"
SHARED_DATA_DIR="${SHARED_DATA_DIR:-../../data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
FALLTL_TARGET_LENGTH="${FALLTL_TARGET_LENGTH:-2048}"
DRY_RUN="${DRY_RUN:-0}"

DATASET_KEY="${1:-${DATASET_KEY:-}}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$DATASET_KEY" in
  shimmer10)
    DATASET_GROUP="aaai27"
    DATASET_NAME="Shimmer_10_session10_AFC"
    DATA_DIR="$CLINICAL_DATA_DIR"
    PROTOCOL="shimmer_hc_vs_pd; subject_split_seed42"
    dataset_args=(--aaai27_label_mode shimmer_hc_vs_pd)
    required_data_paths=(
      "$DATA_DIR/Neuro/AAAI_Data/$DATASET_NAME/Feature"
      "$DATA_DIR/Neuro/AAAI_Data/$DATASET_NAME/Label/label.npy"
      "$DATA_DIR/Neuro/AAAI_Data/$DATASET_NAME/Meta/subject_map.csv"
    )
    ;;
  pads11)
    DATASET_GROUP="aaai27"
    DATASET_NAME="PADS_11_task08_TouchIndex"
    DATA_DIR="$CLINICAL_DATA_DIR"
    PROTOCOL="pads_pd_vs_hc; subject_split_seed42"
    dataset_args=(--aaai27_label_mode pads_pd_vs_hc)
    required_data_paths=(
      "$DATA_DIR/Neuro/AAAI_Data/$DATASET_NAME/Feature"
      "$DATA_DIR/Neuro/AAAI_Data/$DATASET_NAME/Label/label.npy"
      "$DATA_DIR/Neuro/AAAI_Data/$DATASET_NAME/Meta/subject_map.csv"
    )
    ;;
  ucihar)
    DATASET_GROUP="uci"
    DATASET_NAME="UCIHAR"
    DATA_DIR="$SHARED_DATA_DIR"
    PROTOCOL="official_subject; all_9_channels"
    dataset_args=(--uci_protocol official_subject --har_channels all)
    required_data_paths=("$DATA_DIR/UCI HAR Dataset")
    ;;
  falltl)
    DATASET_GROUP="falltl"
    DATASET_NAME="FallTL"
    DATA_DIR="$SHARED_DATA_DIR"
    PROTOCOL="comparison_binary; event_trial_group_split_seed42"
    dataset_args=(
      --falltl_protocol comparison_binary
      --falltl_target_length "$FALLTL_TARGET_LENGTH"
    )
    required_data_paths=("$DATA_DIR/FallTL/Readme.txt")
    ;;
  *)
    echo "Usage: $0 {shimmer10|pads11|ucihar|falltl} [extra main.py arguments]" >&2
    exit 2
    ;;
esac

RESULT_DIR="${RESULT_DIR:-results/med_activity_graph_adaptive_granularity_seed${SEED}/${DATASET_KEY}}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-feature_cache/med_activity_graph_adaptive_granularity_seed${SEED}/${DATASET_KEY}}"

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer" >&2; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 1; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE must be positive" >&2; exit 1; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be positive" >&2; exit 1; }
[[ "$PATIENCE" =~ ^[1-9][0-9]*$ ]] || { echo "PATIENCE must be positive" >&2; exit 1; }
[[ "$FALLTL_TARGET_LENGTH" =~ ^[1-9][0-9]*$ ]] || {
  echo "FALLTL_TARGET_LENGTH must be positive" >&2
  exit 1
}

command=(
  "$PYTHON_BIN" main.py
  --vit_1_name "$MODEL_DIR"
  --vit_1_layer 14
  --aggregation mean
  --image_mode med_activity_graph
  --med_activity_patch_lengths 2,4,8
  --med_activity_channel_mix 0.35
  --med_activity_router_temperature 0.2
  --med_activity_router_mix 0.5
  --med_activity_adaptive_granularity
  --med_activity_granularity_bank "1,2,4;2,4,8;4,8,16"
  --med_activity_granularity_hidden_dim 64
  --med_activity_granularity_temperature 1.0
  --med_activity_granularity_base_prior 0.9
  --med_activity_granularity_balance_weight 0.01
  --med_activity_granularity_entropy_weight 0.001
  --save_activity_graph_samples 4
  --mantis
  --mantis_name "$MANTIS_DIR"
  --classifier_type mlp
  --modal_interaction concat_attn
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
  --data_dir "$DATA_DIR"
  --datasets "$DATASET_GROUP"
  --dataset_names "$DATASET_NAME"
  "${dataset_args[@]}"
  --random_seed "$SEED"
  --feature_cache_dir "$FEATURE_CACHE_DIR"
  --result_dir "$RESULT_DIR"
  "$@"
)

echo "Dataset protocol: key=$DATASET_KEY dataset=$DATASET_NAME protocol=$PROTOCOL adaptive_granularity=1 bank=1-2-4,2-4-8,4-8-16"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU" "${command[@]}"
  printf '\n'
  exit 0
fi

for path in "$MODEL_DIR" "$MANTIS_DIR" "${required_data_paths[@]}"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done

if [[ -d "$MODEL_DIR" ]]; then
  openclip_checkpoint_found=0
  for candidate in \
    "$MODEL_DIR/open_clip_pytorch_model.bin" \
    "$MODEL_DIR/open_clip_model.bin" \
    "$MODEL_DIR/pytorch_model.bin" \
    "$MODEL_DIR/model.safetensors" \
    "$MODEL_DIR/model.bin" \
    "$MODEL_DIR/model.pt" \
    "$MODEL_DIR/model.pth"; do
    if [[ -f "$candidate" ]]; then
      openclip_checkpoint_found=1
      break
    fi
  done
  if [[ "$openclip_checkpoint_found" -ne 1 ]]; then
    echo "No readable OpenCLIP checkpoint found in MODEL_DIR: $MODEL_DIR" >&2
    echo "Check for a missing file or a broken checkpoint symlink before launching." >&2
    exit 1
  fi
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] || {
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
}

mkdir -p "$RESULT_DIR" "$FEATURE_CACHE_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/neurosigvit_med_${DATASET_KEY}_mpl_gpu_${GPU}}"
export HF_HOME="${HF_HOME:-/tmp/neurosigvit_med_${DATASET_KEY}_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
"${command[@]}"
