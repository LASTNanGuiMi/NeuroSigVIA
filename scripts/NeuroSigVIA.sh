#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 所有数据集与 seed 的训练参数均在下方逐条列出，可直接修改对应命令。
# 运行前激活 Python 环境；PYTHON_BIN 可指定解释器，DRY_RUN=1 只打印命令。
# 每个数据集占一张 GPU，三个种子依次运行；数据集之间并行。
METHOD=NeuroSigVIA
source scripts/lib/explicit_experiments.sh

# TDBRAIN 调用链：runners.neurosigvia -> get_eeg_medformer_dataloaders -> train_neurosigvia_classifier。
# TDBRAIN 每批 [8,33,256] -> 内部 [8,4,33,64] -> 窗口 logits [8,2]。
# checkpoint 按验证集窗口 Macro-F1 选择；编码批大小与输入 batch_size 分开。

# Subject-Independent
# ADFTD Dataset
(
export CUDA_VISIBLE_DEVICES=0

# ADFTD — seed 42
python \
  -u \
  -m runners.neurosigvia \
  --datasets eeg \
  --dataset_names ADFTD \
  --data_dir "$NEUROSIGVIA_EEG_ROOT" \
  --eeg_protocol medformer_code_exact \
  --eeg_normalization per_window_per_channel_standard_scaler_ddof0 \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 8 \
  --random_seed 42 \
  --feature_cache_dir "$CACHE_ROOT/seed42/adftd" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed42/adftd" \
  "$@"

# ADFTD — seed 43
python \
  -u \
  -m runners.neurosigvia \
  --datasets eeg \
  --dataset_names ADFTD \
  --data_dir "$NEUROSIGVIA_EEG_ROOT" \
  --eeg_protocol medformer_code_exact \
  --eeg_normalization per_window_per_channel_standard_scaler_ddof0 \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 8 \
  --random_seed 43 \
  --feature_cache_dir "$CACHE_ROOT/seed43/adftd" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed43/adftd" \
  "$@"

# ADFTD — seed 44
python \
  -u \
  -m runners.neurosigvia \
  --datasets eeg \
  --dataset_names ADFTD \
  --data_dir "$NEUROSIGVIA_EEG_ROOT" \
  --eeg_protocol medformer_code_exact \
  --eeg_normalization per_window_per_channel_standard_scaler_ddof0 \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 8 \
  --random_seed 44 \
  --feature_cache_dir "$CACHE_ROOT/seed44/adftd" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed44/adftd" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# TDBRAIN Dataset
(
export CUDA_VISIBLE_DEVICES=1

# TDBRAIN — seed 42
python \
  -u \
  -m runners.neurosigvia \
  --datasets eeg \
  --dataset_names TDBRAIN \
  --data_dir "$NEUROSIGVIA_EEG_ROOT" \
  --eeg_protocol medformer_code_exact \
  --eeg_normalization per_window_per_channel_standard_scaler_ddof0 \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 8 \
  --random_seed 42 \
  --feature_cache_dir "$CACHE_ROOT/seed42/tdbrain" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed42/tdbrain" \
  "$@"

# TDBRAIN — seed 43
python \
  -u \
  -m runners.neurosigvia \
  --datasets eeg \
  --dataset_names TDBRAIN \
  --data_dir "$NEUROSIGVIA_EEG_ROOT" \
  --eeg_protocol medformer_code_exact \
  --eeg_normalization per_window_per_channel_standard_scaler_ddof0 \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 8 \
  --random_seed 43 \
  --feature_cache_dir "$CACHE_ROOT/seed43/tdbrain" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed43/tdbrain" \
  "$@"

# TDBRAIN — seed 44
python \
  -u \
  -m runners.neurosigvia \
  --datasets eeg \
  --dataset_names TDBRAIN \
  --data_dir "$NEUROSIGVIA_EEG_ROOT" \
  --eeg_protocol medformer_code_exact \
  --eeg_normalization per_window_per_channel_standard_scaler_ddof0 \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 8 \
  --random_seed 44 \
  --feature_cache_dir "$CACHE_ROOT/seed44/tdbrain" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed44/tdbrain" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# APAVA Dataset
(
export CUDA_VISIBLE_DEVICES=2

# APAVA — seed 42
python \
  -u \
  -m runners.neurosigvia \
  --datasets eeg \
  --dataset_names APAVA \
  --data_dir "$NEUROSIGVIA_EEG_ROOT" \
  --eeg_protocol medformer_code_exact \
  --eeg_normalization per_window_per_channel_standard_scaler_ddof0 \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 8 \
  --random_seed 42 \
  --feature_cache_dir "$CACHE_ROOT/seed42/apava" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed42/apava" \
  "$@"

# APAVA — seed 43
python \
  -u \
  -m runners.neurosigvia \
  --datasets eeg \
  --dataset_names APAVA \
  --data_dir "$NEUROSIGVIA_EEG_ROOT" \
  --eeg_protocol medformer_code_exact \
  --eeg_normalization per_window_per_channel_standard_scaler_ddof0 \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 8 \
  --random_seed 43 \
  --feature_cache_dir "$CACHE_ROOT/seed43/apava" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed43/apava" \
  "$@"

# APAVA — seed 44
python \
  -u \
  -m runners.neurosigvia \
  --datasets eeg \
  --dataset_names APAVA \
  --data_dir "$NEUROSIGVIA_EEG_ROOT" \
  --eeg_protocol medformer_code_exact \
  --eeg_normalization per_window_per_channel_standard_scaler_ddof0 \
  --vit_1_name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 8 \
  --random_seed 44 \
  --feature_cache_dir "$CACHE_ROOT/seed44/apava" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed44/apava" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# Shimmer10 Dataset
(
export CUDA_VISIBLE_DEVICES=3

# Shimmer10 — seed 42
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
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --random_seed 42 \
  --feature_cache_dir "$CACHE_ROOT/seed42/shimmer10" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed42/shimmer10" \
  "$@"

# Shimmer10 — seed 43
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
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --random_seed 43 \
  --feature_cache_dir "$CACHE_ROOT/seed43/shimmer10" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed43/shimmer10" \
  "$@"

# Shimmer10 — seed 44
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
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --random_seed 44 \
  --feature_cache_dir "$CACHE_ROOT/seed44/shimmer10" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed44/shimmer10" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# PADS11 Dataset
(
export CUDA_VISIBLE_DEVICES=4

# PADS11 — seed 42
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
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 4 \
  --random_seed 42 \
  --feature_cache_dir "$CACHE_ROOT/seed42/pads11" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed42/pads11" \
  "$@"

# PADS11 — seed 43
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
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 4 \
  --random_seed 43 \
  --feature_cache_dir "$CACHE_ROOT/seed43/pads11" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed43/pads11" \
  "$@"

# PADS11 — seed 44
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
  --image_mode med_activity_graph \
  --med_activity_granularity_hidden_dim 64 \
  --mantis \
  --mantis_name "${MANTIS_DIR:-${NEUROSIGVIA_MANTIS_PATH:-../models/Mantis-8M}}" \
  --classifier_type mlp \
  --modal_interaction adaptive_granularity \
  --outer_patch_size 64 \
  --outer_patch_stride 64 \
  --granularity_gate_temperature 0.5 \
  --granularity_balance_weight 0.001 \
  --granularity_graph_token_grid 4 \
  --activity_graph_canvas_size 360 \
  --activity_graph_line_width 1.0 \
  --activity_graph_vertical_margin 0.05 \
  --patch_alignment_dim 256 \
  --patch_alignment_temperature 0.1 \
  --patch_alignment_weight 0.1 \
  --patch_checkpoint_metric window_macro_f1 \
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
  --batch_size 4 \
  --random_seed 44 \
  --feature_cache_dir "$CACHE_ROOT/seed44/pads11" \
  --reuse_static_cache_dir "${REUSE_STATIC_CACHE_DIR:-feature_cache}" \
  --result_dir "$RESULT_ROOT/seed44/pads11" \
  "$@"

) &
PIDS+=("$!")

wait_for_jobs "${PIDS[@]}"
