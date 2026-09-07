#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/lib/experiments.sh

# Five datasets: ADFTD, TDBRAIN, APAVA, Shimmer AFC, PADS TouchIndex.
# Defaults: SEEDS="42 43 44", GPUS="0 1 2 3 4".
# Usage: bash scripts/Crossformer.sh
#        SEEDS="43 44" DATASETS="adftd apava" GPUS="0 1" bash scripts/Crossformer.sh
model_name=Crossformer

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
    --d_model 128 --d_ff 256 --e_layers 2 --n_heads 8 \
    --dropout 0.1 \
    --learning_rate 3e-4 --weight_decay 1e-3 \
    --train_epochs "${EPOCHS:-100}" --patience "${PATIENCE:-12}" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --random_seed "$seed" \
    --result_dir "$result" \
    "$@"
}

run_experiments "$model_name" "$@"
