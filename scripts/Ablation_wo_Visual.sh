#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 分支消融 w/o Visual（单数值）：去掉视觉分支，只保留 Mantis 数值分支与分类器。
METHOD=Ablation_wo_Visual
source scripts/lib/explicit_experiments.sh

(
export CUDA_VISIBLE_DEVICES=2

# shimmer10 — seed 42
python \
  -u \
  -m runners.numeric_ablation \
  --dataset shimmer10 \
  --seed 42 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 1 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed42/shimmer10" \
  "$@"

# shimmer10 — seed 43
python \
  -u \
  -m runners.numeric_ablation \
  --dataset shimmer10 \
  --seed 43 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 1 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed43/shimmer10" \
  "$@"

# shimmer10 — seed 44
python \
  -u \
  -m runners.numeric_ablation \
  --dataset shimmer10 \
  --seed 44 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 1 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed44/shimmer10" \
  "$@"
) &
PIDS+=("$!")

(
export CUDA_VISIBLE_DEVICES=3

# pads11 — seed 42
python \
  -u \
  -m runners.numeric_ablation \
  --dataset pads11 \
  --seed 42 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 4 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed42/pads11" \
  "$@"

# pads11 — seed 43
python \
  -u \
  -m runners.numeric_ablation \
  --dataset pads11 \
  --seed 43 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 4 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed43/pads11" \
  "$@"

# pads11 — seed 44
python \
  -u \
  -m runners.numeric_ablation \
  --dataset pads11 \
  --seed 44 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 4 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed44/pads11" \
  "$@"
) &
PIDS+=("$!")

(
export CUDA_VISIBLE_DEVICES=4

# apava — seed 42
python \
  -u \
  -m runners.numeric_ablation \
  --dataset apava \
  --seed 42 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 8 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed42/apava" \
  "$@"

# apava — seed 43
python \
  -u \
  -m runners.numeric_ablation \
  --dataset apava \
  --seed 43 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 8 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed43/apava" \
  "$@"

# apava — seed 44
python \
  -u \
  -m runners.numeric_ablation \
  --dataset apava \
  --seed 44 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 8 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed44/apava" \
  "$@"
) &
PIDS+=("$!")

(
export CUDA_VISIBLE_DEVICES=5

# tdbrain — seed 42
python \
  -u \
  -m runners.numeric_ablation \
  --dataset tdbrain \
  --seed 42 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 8 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed42/tdbrain" \
  "$@"

# tdbrain — seed 43
python \
  -u \
  -m runners.numeric_ablation \
  --dataset tdbrain \
  --seed 43 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 8 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed43/tdbrain" \
  "$@"

# tdbrain — seed 44
python \
  -u \
  -m runners.numeric_ablation \
  --dataset tdbrain \
  --seed 44 \
  --mantis-name ../models/Mantis-8M \
  --outer-patch-size 64 \
  --outer-patch-stride 64 \
  --encode-batch-size 4 \
  --fusion-dim 128 \
  --channel-hidden-dim 64 \
  --mlp-hidden-dim 128 \
  --mlp-num-layers 2 \
  --mlp-dropout 0.1 \
  --mlp-lr 3e-4 \
  --mlp-weight-decay 1e-3 \
  --mlp-class-weight balanced \
  --mlp-epochs 100 \
  --mlp-early-stop-strategy raw_primary \
  --mlp-early-stop-warmup-epochs 10 \
  --mlp-early-stop-min-epochs 0 \
  --mlp-early-stop-patience 12 \
  --mlp-early-stop-min-delta 0.002 \
  --mlp-early-stop-ema-decay 0.9 \
  --mlp-lr-scheduler reduce_on_plateau \
  --mlp-lr-scheduler-patience 4 \
  --mlp-lr-scheduler-factor 0.5 \
  --mlp-lr-scheduler-min-lr 1e-6 \
  --batch-size 8 \
  --checkpoint-metric window_macro_f1 \
  --output "$RESULT_ROOT/seed44/tdbrain" \
  "$@"
) &
PIDS+=("$!")

wait_for_jobs "${PIDS[@]}"
