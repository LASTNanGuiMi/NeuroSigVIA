#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/lib/experiments.sh

# Activate the training environment, then run: bash scripts/TimesNet.sh
# Four non-ADFTD datasets; each GPU runs its three initialization seeds in order.
model_name=TimesNet
DATASETS="tdbrain apava shimmer10 pads11"
SEEDS="42 43 44"
GPUS="0 1 2 3"
declare -A BATCH_SIZES=(
  [adftd]=8 [tdbrain]=8 [apava]=8 [shimmer10]=1 [pads11]=4
)

train_one() {
  local dataset="$1" seed="$2" result="$3" cache="$4"
  shift 4
  configure_dataset "$dataset"
  # Same shared hyperparameters and protocol as the six existing baselines.
  # TimesNet-only defaults: top_k=3, num_kernels=6 (official classification setup).
  run_python -m runners.baselines \
    --task_name classification \
    --model "$model_name" \
    --dataset "$dataset" \
    --split_seed 42 \
    --d_model 128 --d_ff 256 --e_layers 2 --n_heads 8 \
    --dropout 0.1 \
    --learning_rate 3e-4 --weight_decay 1e-3 \
    --train_epochs 100 --patience 12 \
    --warmup_epochs 10 --min_delta 0.002 \
    --top_k 3 --num_kernels 6 \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --random_seed "$seed" \
    --result_dir "$result" \
    --progress \
    "$@"
}

run_experiments "$model_name" "$@"
