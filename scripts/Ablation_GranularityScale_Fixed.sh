#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 粒度尺度下的固定粒度消融：P=256、R=128、Q={32,64,128}，与 Ablation_GranularityScale.sh 的命令相比
# 只多 --fixed_granularity（32/64/128），门控冻结为单一粒度；自适应版本即 Ablation_GranularityScale.sh。
# 所有数据集与 seed 的训练参数均在下方逐条列出，可直接修改对应命令。
# 运行前激活 Python 环境；PYTHON_BIN 可指定解释器，DRY_RUN=1 只打印命令。
# 每个（数据集，粒度）占一张 GPU（共六张），三个 seed 依次运行；六组之间并行。
METHOD=Ablation_GranularityScale_Fixed
source scripts/lib/explicit_experiments.sh

# Subject-Independent
# Shimmer10 Dataset — 固定粒度 q=32
(
export CUDA_VISIBLE_DEVICES=0

# Shimmer10 — P=256, R=128, 固定粒度 q=32 — seed 42
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names Shimmer_10_session10_AFC \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode shimmer_hc_vs_pd \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric subject_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.1 \
  --mlp_lr 3e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 1 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 32 \
  --random_seed 42 \
  --feature_cache_dir "$CACHE_ROOT/seed42/q32_shimmer10" \
  --result_dir "$RESULT_ROOT/seed42/q32_shimmer10" \
  "$@"

# Shimmer10 — P=256, R=128, 固定粒度 q=32 — seed 43
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names Shimmer_10_session10_AFC \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode shimmer_hc_vs_pd \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric subject_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.1 \
  --mlp_lr 3e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 1 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 32 \
  --random_seed 43 \
  --feature_cache_dir "$CACHE_ROOT/seed43/q32_shimmer10" \
  --result_dir "$RESULT_ROOT/seed43/q32_shimmer10" \
  "$@"

# Shimmer10 — P=256, R=128, 固定粒度 q=32 — seed 44
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names Shimmer_10_session10_AFC \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode shimmer_hc_vs_pd \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric subject_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.1 \
  --mlp_lr 3e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 1 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 32 \
  --random_seed 44 \
  --feature_cache_dir "$CACHE_ROOT/seed44/q32_shimmer10" \
  --result_dir "$RESULT_ROOT/seed44/q32_shimmer10" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# Shimmer10 Dataset — 固定粒度 q=64
(
export CUDA_VISIBLE_DEVICES=1

# Shimmer10 — P=256, R=128, 固定粒度 q=64 — seed 42
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names Shimmer_10_session10_AFC \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode shimmer_hc_vs_pd \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric subject_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.1 \
  --mlp_lr 3e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 1 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 64 \
  --random_seed 42 \
  --feature_cache_dir "$CACHE_ROOT/seed42/q64_shimmer10" \
  --result_dir "$RESULT_ROOT/seed42/q64_shimmer10" \
  "$@"

# Shimmer10 — P=256, R=128, 固定粒度 q=64 — seed 43
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names Shimmer_10_session10_AFC \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode shimmer_hc_vs_pd \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric subject_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.1 \
  --mlp_lr 3e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 1 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 64 \
  --random_seed 43 \
  --feature_cache_dir "$CACHE_ROOT/seed43/q64_shimmer10" \
  --result_dir "$RESULT_ROOT/seed43/q64_shimmer10" \
  "$@"

# Shimmer10 — P=256, R=128, 固定粒度 q=64 — seed 44
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names Shimmer_10_session10_AFC \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode shimmer_hc_vs_pd \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric subject_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.1 \
  --mlp_lr 3e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 1 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 64 \
  --random_seed 44 \
  --feature_cache_dir "$CACHE_ROOT/seed44/q64_shimmer10" \
  --result_dir "$RESULT_ROOT/seed44/q64_shimmer10" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# Shimmer10 Dataset — 固定粒度 q=128
(
export CUDA_VISIBLE_DEVICES=2

# Shimmer10 — P=256, R=128, 固定粒度 q=128 — seed 42
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names Shimmer_10_session10_AFC \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode shimmer_hc_vs_pd \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric subject_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.1 \
  --mlp_lr 3e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 1 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 128 \
  --random_seed 42 \
  --feature_cache_dir "$CACHE_ROOT/seed42/q128_shimmer10" \
  --result_dir "$RESULT_ROOT/seed42/q128_shimmer10" \
  "$@"

# Shimmer10 — P=256, R=128, 固定粒度 q=128 — seed 43
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names Shimmer_10_session10_AFC \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode shimmer_hc_vs_pd \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric subject_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.1 \
  --mlp_lr 3e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 1 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 128 \
  --random_seed 43 \
  --feature_cache_dir "$CACHE_ROOT/seed43/q128_shimmer10" \
  --result_dir "$RESULT_ROOT/seed43/q128_shimmer10" \
  "$@"

# Shimmer10 — P=256, R=128, 固定粒度 q=128 — seed 44
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names Shimmer_10_session10_AFC \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode shimmer_hc_vs_pd \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric subject_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.1 \
  --mlp_lr 3e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 1 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 128 \
  --random_seed 44 \
  --feature_cache_dir "$CACHE_ROOT/seed44/q128_shimmer10" \
  --result_dir "$RESULT_ROOT/seed44/q128_shimmer10" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# PADS11 Dataset — 固定粒度 q=32
(
export CUDA_VISIBLE_DEVICES=3

# PADS11 — P=256, R=128, 固定粒度 q=32 — seed 42
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names PADS_11_task08_TouchIndex \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode pads_pd_vs_hc \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.05 \
  --patch_checkpoint_metric window_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.2 \
  --mlp_lr 1e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 16 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 32 \
  --random_seed 42 \
  --feature_cache_dir "$CACHE_ROOT/seed42/q32_pads11" \
  --result_dir "$RESULT_ROOT/seed42/q32_pads11" \
  "$@"

# PADS11 — P=256, R=128, 固定粒度 q=32 — seed 43
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names PADS_11_task08_TouchIndex \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode pads_pd_vs_hc \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.05 \
  --patch_checkpoint_metric window_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.2 \
  --mlp_lr 1e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 16 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 32 \
  --random_seed 43 \
  --feature_cache_dir "$CACHE_ROOT/seed43/q32_pads11" \
  --result_dir "$RESULT_ROOT/seed43/q32_pads11" \
  "$@"

# PADS11 — P=256, R=128, 固定粒度 q=32 — seed 44
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names PADS_11_task08_TouchIndex \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode pads_pd_vs_hc \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.05 \
  --patch_checkpoint_metric window_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.2 \
  --mlp_lr 1e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 16 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 32 \
  --random_seed 44 \
  --feature_cache_dir "$CACHE_ROOT/seed44/q32_pads11" \
  --result_dir "$RESULT_ROOT/seed44/q32_pads11" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# PADS11 Dataset — 固定粒度 q=64
(
export CUDA_VISIBLE_DEVICES=4

# PADS11 — P=256, R=128, 固定粒度 q=64 — seed 42
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names PADS_11_task08_TouchIndex \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode pads_pd_vs_hc \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.05 \
  --patch_checkpoint_metric window_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.2 \
  --mlp_lr 1e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 16 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 64 \
  --random_seed 42 \
  --feature_cache_dir "$CACHE_ROOT/seed42/q64_pads11" \
  --result_dir "$RESULT_ROOT/seed42/q64_pads11" \
  "$@"

# PADS11 — P=256, R=128, 固定粒度 q=64 — seed 43
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names PADS_11_task08_TouchIndex \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode pads_pd_vs_hc \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.05 \
  --patch_checkpoint_metric window_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.2 \
  --mlp_lr 1e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 16 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 64 \
  --random_seed 43 \
  --feature_cache_dir "$CACHE_ROOT/seed43/q64_pads11" \
  --result_dir "$RESULT_ROOT/seed43/q64_pads11" \
  "$@"

# PADS11 — P=256, R=128, 固定粒度 q=64 — seed 44
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names PADS_11_task08_TouchIndex \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode pads_pd_vs_hc \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.05 \
  --patch_checkpoint_metric window_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.2 \
  --mlp_lr 1e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 16 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 64 \
  --random_seed 44 \
  --feature_cache_dir "$CACHE_ROOT/seed44/q64_pads11" \
  --result_dir "$RESULT_ROOT/seed44/q64_pads11" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# PADS11 Dataset — 固定粒度 q=128
(
export CUDA_VISIBLE_DEVICES=5

# PADS11 — P=256, R=128, 固定粒度 q=128 — seed 42
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names PADS_11_task08_TouchIndex \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode pads_pd_vs_hc \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.05 \
  --patch_checkpoint_metric window_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.2 \
  --mlp_lr 1e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 16 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 128 \
  --random_seed 42 \
  --feature_cache_dir "$CACHE_ROOT/seed42/q128_pads11" \
  --result_dir "$RESULT_ROOT/seed42/q128_pads11" \
  "$@"

# PADS11 — P=256, R=128, 固定粒度 q=128 — seed 43
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names PADS_11_task08_TouchIndex \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode pads_pd_vs_hc \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.05 \
  --patch_checkpoint_metric window_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.2 \
  --mlp_lr 1e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 16 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 128 \
  --random_seed 43 \
  --feature_cache_dir "$CACHE_ROOT/seed43/q128_pads11" \
  --result_dir "$RESULT_ROOT/seed43/q128_pads11" \
  "$@"

# PADS11 — P=256, R=128, 固定粒度 q=128 — seed 44
python \
  -u \
  -m runners.neurosigvia \
  --datasets wearable \
  --dataset_names PADS_11_task08_TouchIndex \
  --data_dir "$NEUROSIGVIA_WEARABLE_ROOT" \
  --wearable_label_mode pads_pd_vs_hc \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode multiscale_activity_graph \
  --activity_graph_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 256 \
  --outer_patch_stride 256 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.05 \
  --patch_checkpoint_metric window_macro_f1 \
  --visual_encode_batch_size 4 \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.2 \
  --mlp_lr 1e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 100 \
  --mlp_early_stop_strategy raw_primary \
  --mlp_early_stop_warmup_epochs 10 \
  --mlp_early_stop_min_epochs 0 \
  --mlp_early_stop_patience 12 \
  --mlp_early_stop_min_delta 0.002 \
  --mlp_lr_scheduler reduce_on_plateau \
  --mlp_lr_scheduler_patience 4 \
  --mlp_lr_scheduler_factor 0.5 \
  --mlp_lr_scheduler_min_lr 1e-6 \
  --batch_size 16 \
  --granularity_region_length 128 \
  --granularity_candidates 32,64,128 \
  --fixed_granularity 128 \
  --random_seed 44 \
  --feature_cache_dir "$CACHE_ROOT/seed44/q128_pads11" \
  --result_dir "$RESULT_ROOT/seed44/q128_pads11" \
  "$@"

) &
PIDS+=("$!")

wait_for_jobs "${PIDS[@]}"
