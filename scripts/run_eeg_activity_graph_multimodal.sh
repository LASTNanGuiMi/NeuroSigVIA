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
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
DRY_RUN="${DRY_RUN:-0}"

DATASET_KEY="${1:-${DATASET_KEY:-}}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$DATASET_KEY" in
  tdbrain)
    DATASET_NAME="TDBRAIN"
    PROTOCOL="medformer_code_exact_official_legacy50_only"
    ;;
  apava)
    DATASET_NAME="APAVA"
    PROTOCOL="medformer_code_exact_apava_subject_split"
    ;;
  adftd)
    DATASET_NAME="ADFTD"
    PROTOCOL="medformer_code_exact_label_row_order_60_20_20"
    ;;
  *)
    echo "Usage: $0 {tdbrain|apava|adftd} [extra main.py arguments]" >&2
    exit 2
    ;;
esac

RESULT_DIR="${RESULT_DIR:-results/eeg_activity_graph_seed${SEED}/${DATASET_KEY}}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-feature_cache/eeg_activity_graph_seed${SEED}/${DATASET_KEY}}"

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer" >&2; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 1; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE must be positive" >&2; exit 1; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be positive" >&2; exit 1; }
[[ "$PATIENCE" =~ ^[1-9][0-9]*$ ]] || { echo "PATIENCE must be positive" >&2; exit 1; }

command=(
  "$PYTHON_BIN" main.py
  --vit_1_name "$MODEL_DIR"
  --vit_1_layer 14
  --aggregation mean
  --image_mode activity_graph
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

echo "Dataset protocol: key=$DATASET_KEY dataset=$DATASET_NAME protocol=$PROTOCOL normalization=per_window_per_channel_standard_scaler_ddof0 metric_unit=window image_mode=activity_graph medformer_graph=0 adaptive_granularity=0"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU" "${command[@]}"
  printf '\n'
  exit 0
fi

for path in \
  "$MODEL_DIR" \
  "$MANTIS_DIR" \
  "$EEG_DATA_DIR/$DATASET_NAME/Feature" \
  "$EEG_DATA_DIR/$DATASET_NAME/Label/label.npy"; do
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
    exit 1
  fi
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] || {
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
}

mkdir -p "$RESULT_DIR" "$FEATURE_CACHE_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/neurosigvit_eeg_activity_${DATASET_KEY}_mpl_gpu_${GPU}}"
export HF_HOME="${HF_HOME:-/tmp/neurosigvit_eeg_activity_${DATASET_KEY}_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
"${command[@]}"
