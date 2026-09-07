#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/lib/experiments.sh

# Run: bash scripts/NeuroSigVIA.sh
# Edit experiment settings here; training hyperparameters are below.
model_name=NeuroSigVIA
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
  run_python -m runners.neurosigvia \
    --datasets "$DATASET_GROUP" \
    --dataset_names "$DATASET_NAME" \
    --data_dir "$DATA_ROOT" \
    "${DATA_ARGS[@]}" \
    --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
    --vit_1_layer 14 \
    --aggregation mean \
    --image_mode med_activity_graph \
    --med_activity_channel_mix 0.35 \
    --med_activity_granularity_hidden_dim 64 \
    --mantis --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
    --classifier_type mlp \
    --modal_interaction adaptive_granularity \
    --outer_patch_size 64 --outer_patch_stride 64 \
    --granularity_gate_temperature 0.5 \
    --granularity_balance_weight 0.001 \
    --granularity_graph_token_grid 4 \
    --patch_alignment_dim 256 --patch_alignment_temperature 0.1 \
    --patch_alignment_weight 0.1 \
    --patch_checkpoint_metric subject_macro_f1 \
    --visual_encode_batch_size 4 \
    --fusion_dim 128 --fusion_heads 2 --mlp_hidden_dim 128 \
    --mlp_num_layers 2 --mlp_dropout 0.1 \
    --mlp_lr 3e-4 --mlp_weight_decay 1e-3 --mlp_class_weight balanced \
    --mlp_epochs 100 \
    --mlp_early_stop_strategy raw_primary \
    --mlp_early_stop_warmup_epochs 10 \
    --mlp_early_stop_min_epochs 0 \
    --mlp_early_stop_patience 12 \
    --mlp_early_stop_min_delta 0.002 \
    --mlp_lr_scheduler reduce_on_plateau \
    --mlp_lr_scheduler_patience 4 --mlp_lr_scheduler_factor 0.5 --mlp_lr_scheduler_min_lr 1e-6 \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --random_seed "$seed" \
    --feature_cache_dir "$cache" \
    --result_dir "$result" \
    "$@"
}

run_experiments "$model_name" "$@"
