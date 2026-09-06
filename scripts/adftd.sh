#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# ADFTD: fixed Medformer row-order 60/20/20 split, seed 42.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/adftd.sh
python -u main.py \
  --datasets eeg \
  --dataset_names ADFTD \
  --data_dir ./data/eeg/processed \
  --vit_1_name ../models/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode med_activity_graph \
  --mantis \
  --mantis_name ../models/Mantis-8M \
  --classifier_type mlp \
  --modal_interaction patch_timemosaic_graph \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --timemosaic_gate_temperature 0.5 \
  --timemosaic_selector_balance_weight 0.001 \
  --patch_alignment_weight 0.1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_lr 3e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --batch_size 8 \
  --random_seed 42 \
  --feature_cache_dir ./feature_cache/timemosaic_graph_seed42/adftd \
  --result_dir ./results/timemosaic_graph_seed42/adftd \
  "$@"
