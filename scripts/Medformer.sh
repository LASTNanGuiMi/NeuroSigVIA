#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/lib/experiments.sh

# Run: bash scripts/Medformer.sh
# Edit experiment settings here; training hyperparameters are below.
model_name=Medformer
DATASETS="adftd tdbrain apava shimmer10 pads11"
SEEDS="42 43 44"
GPUS="0 1 2 3 4"
# Each dataset uses one GPU; its three seeds run sequentially.
declare -A BATCH_SIZES=(
  [adftd]=8 [tdbrain]=8 [apava]=8 [shimmer10]=1 [pads11]=4
)

train_one() {
  local dataset="$1" seed="$2" result="$3" cache="$4"
  shift 4
  configure_dataset "$dataset"
  # Original model implementations are copied from the Medformer project.
  # These are study-specific starting configurations, not paper-tuned results.
  run_python run_baseline.py \
    --task_name classification \
    --model "$model_name" \
    --dataset "$dataset" \
    --split_seed 42 \
    --d_model 128 --d_ff 256 --e_layers 2 --n_heads 8 \
    --dropout 0.1 \
    --learning_rate 3e-4 --weight_decay 1e-3 \
    --train_epochs 100 --patience 12 \
    --warmup_epochs 10 --min_delta 0.002 \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --patch_len_list 2,4,8 --augmentations none \
    --random_seed "$seed" \
    --result_dir "$result" \
    "$@"
}

run_experiments "$model_name" "$@"
