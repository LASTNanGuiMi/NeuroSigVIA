#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

MODEL_DIR="${MODEL_DIR:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-../Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data}"
WEARABLE_DATA_ROOT="${WEARABLE_DATA_ROOT:-$DATA_DIR/wearable}"
GPU="${GPU:-0}"
SEED="${SEED:-2022}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
DRY_RUN="${DRY_RUN:-0}"

DATASET_KEY="${1:-${DATASET_KEY:-}}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$DATASET_KEY" in
  shimmer10)
    DATASET_GROUP="wearable"
    DATASET_NAME="Shimmer_10_session10_AFC"
    LABEL_PROTOCOL="shimmer_hc_vs_pd"
    dataset_args=(--wearable_label_mode shimmer_hc_vs_pd)
    required_data_paths=("$WEARABLE_DATA_ROOT/$DATASET_NAME")
    ;;
  pads11)
    DATASET_GROUP="wearable"
    DATASET_NAME="PADS_11_task08_TouchIndex"
    LABEL_PROTOCOL="pads_pd_vs_hc"
    dataset_args=(--wearable_label_mode pads_pd_vs_hc)
    required_data_paths=("$WEARABLE_DATA_ROOT/$DATASET_NAME")
    ;;
  *)
    echo "Usage: $0 {shimmer10|pads11} [extra main.py arguments]" >&2
    exit 2
    ;;
esac

RESULT_DIR="${RESULT_DIR:-results/${DATASET_KEY}_seed${SEED}}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-feature_cache/${DATASET_KEY}}"

for path in "$MODEL_DIR" "$MANTIS_DIR" "${required_data_paths[@]}"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
if [[ "$DATASET_GROUP" == "wearable" ]]; then
  for path in \
    "$WEARABLE_DATA_ROOT/$DATASET_NAME/Feature" \
    "$WEARABLE_DATA_ROOT/$DATASET_NAME/Label/label.npy" \
    "$WEARABLE_DATA_ROOT/$DATASET_NAME/Meta/subject_map.csv" \
    "data_loading/split_reference_seed42.csv"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
  done
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] || {
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
}
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer" >&2; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 1; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be positive" >&2; exit 1; }
[[ "$PATIENCE" =~ ^[1-9][0-9]*$ ]] || { echo "PATIENCE must be positive" >&2; exit 1; }

command=(
  "$PYTHON_BIN" "main.py"
  --vit_1_name "$MODEL_DIR"
  --vit_1_layer 14
  --aggregation mean
  --image_mode activity_graph
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
  --batch_size 16
  --data_dir "$WEARABLE_DATA_ROOT"
  --datasets "$DATASET_GROUP"
  --dataset_names "$DATASET_NAME"
  "${dataset_args[@]}"
  --random_seed "$SEED"
  --feature_cache_dir "$FEATURE_CACHE_DIR"
  --result_dir "$RESULT_DIR"
  "$@"
)

echo "Dataset protocol: key=$DATASET_KEY dataset=$DATASET_NAME protocol=$LABEL_PROTOCOL"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU" "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$RESULT_DIR" "$FEATURE_CACHE_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/neurosigvit_${DATASET_KEY}_mpl_gpu_${GPU}}"
export HF_HOME="${HF_HOME:-/tmp/neurosigvit_${DATASET_KEY}_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
"${command[@]}"
