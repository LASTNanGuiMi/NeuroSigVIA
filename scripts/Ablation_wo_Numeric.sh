#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 分支消融 w/o Numeric（单视觉）：去掉 Mantis 数值分支，只保留折线图与自适应 Activity Graph 的视觉路径。
# 所有数据集与 seed 的训练参数均在下方逐条列出，可直接修改对应命令。
# 运行前激活 Python 环境；PYTHON_BIN 可指定解释器，DRY_RUN=1 只打印命令。
# 每个数据集占一张 GPU，各条命令依次运行；数据集之间并行。
METHOD=Ablation_wo_Numeric
source scripts/lib/explicit_experiments.sh

# 单视觉消融与一次已完成的主方法运行配对：REFERENCE_RUN 指向该运行的 status/<RUN_TAG>/manifest.tsv。
REFERENCE_RUN="${REFERENCE_RUN:-status/MAIN_RUN_TAG/manifest.tsv}"
[[ "${DRY_RUN:-0}" == 1 || -f "$REFERENCE_RUN" ]] || {
  printf 'Set REFERENCE_RUN to status/<RUN_TAG>/manifest.tsv of a completed scripts/NeuroSigVIA.sh run.\n' >&2
  exit 2
}

# Subject-Independent
# TDBRAIN Dataset
(
export CUDA_VISIBLE_DEVICES=0

# TDBRAIN — seed 42
python \
  -u \
  -m runners.vision_ablation \
  --dataset tdbrain \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/tdbrain" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 8 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

# TDBRAIN — seed 43
python \
  -u \
  -m runners.vision_ablation \
  --dataset tdbrain \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/tdbrain" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 8 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

# TDBRAIN — seed 44
python \
  -u \
  -m runners.vision_ablation \
  --dataset tdbrain \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/tdbrain" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 8 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# APAVA Dataset
(
export CUDA_VISIBLE_DEVICES=1

# APAVA — seed 42
python \
  -u \
  -m runners.vision_ablation \
  --dataset apava \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/apava" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 8 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

# APAVA — seed 43
python \
  -u \
  -m runners.vision_ablation \
  --dataset apava \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/apava" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 8 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

# APAVA — seed 44
python \
  -u \
  -m runners.vision_ablation \
  --dataset apava \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/apava" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 8 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# Shimmer10 Dataset
(
export CUDA_VISIBLE_DEVICES=2

# Shimmer10 — seed 42
python \
  -u \
  -m runners.vision_ablation \
  --dataset shimmer10 \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/shimmer10" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 1 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

# Shimmer10 — seed 43
python \
  -u \
  -m runners.vision_ablation \
  --dataset shimmer10 \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/shimmer10" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 1 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

# Shimmer10 — seed 44
python \
  -u \
  -m runners.vision_ablation \
  --dataset shimmer10 \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/shimmer10" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 1 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# PADS11 Dataset
(
export CUDA_VISIBLE_DEVICES=3

# PADS11 — seed 42
python \
  -u \
  -m runners.vision_ablation \
  --dataset pads11 \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/pads11" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 4 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

# PADS11 — seed 43
python \
  -u \
  -m runners.vision_ablation \
  --dataset pads11 \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/pads11" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 4 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

# PADS11 — seed 44
python \
  -u \
  -m runners.vision_ablation \
  --dataset pads11 \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/pads11" \
  --reference-run "$REFERENCE_RUN" \
  --vit-1-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --checkpoint-metric window_macro_f1 \
  --mlp-lr 0.0003 \
  --mlp-weight-decay 0.001 \
  --mlp-dropout 0.1 \
  --granularity-gate-temperature 0.5 \
  --granularity-balance-weight 0.001 \
  --activity-graph-line-width 1.0 \
  --activity-graph-vertical-margin 0.05 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.6 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-06 \
  --batch-size 4 \
  --vit-1-layer 14 \
  --fusion-dim 128 \
  --fusion-heads 2 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-epochs 100 \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --visual-encode-batch-size 4 \
  --granularity-graph-token-grid 4 \
  --activity-graph-canvas-size 360 \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-lr-scheduler-patience 4 \
  --aggregation mean \
  --image-mode multiscale_activity_graph \
  --eeg-protocol medformer_code_exact \
  --eeg-normalization per_window_per_channel_standard_scaler_ddof0 \
  --mlp-class-weight balanced \
  --mlp-early-stop-strategy raw_primary \
  --mlp-lr-scheduler reduce_on_plateau \
  "$@"

) &
PIDS+=("$!")

wait_for_jobs "${PIDS[@]}"
