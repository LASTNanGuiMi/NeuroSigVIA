#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/lib/experiments.sh

# Run: bash scripts/PatchTST.sh
# Edit experiment settings here; training hyperparameters are below.
model_name=PatchTST
DATASETS="adftd tdbrain apava shimmer10 pads11"
SEEDS="42 43 44"
GPUS="0 1 2 3 4"
# Each dataset uses one GPU; its three seeds run sequentially.
declare -A BATCH_SIZES=(
  [adftd]=128 [tdbrain]=32 [apava]=32 [shimmer10]=1 [pads11]=4
)

train_one() {
  local dataset="$1" seed="$2" result="$3" cache="$4"
  shift 4
  configure_dataset "$dataset"
  # Effective model settings follow Medformer scripts/classification/PatchTST.sh.
  # Medformer has no Shimmer/PADS entries, so their local batch fallbacks stay 1/4.
  run_python -m runners.baselines \
    --task_name classification \
    --model "$model_name" \
    --dataset "$dataset" \
    --split_seed 42 \
    --d_model 128 --d_ff 256 --e_layers 6 --n_heads 8 \
    --dropout 0.1 \
    --learning_rate 1e-4 --weight_decay 1e-3 \
    --train_epochs 100 --patience 10 \
    --warmup_epochs 10 --min_delta 0.002 \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --patch_len 16 --stride 8 \
    --random_seed "$seed" \
    --result_dir "$result" \
    "$@"
}

run_experiments "$model_name" "$@"
