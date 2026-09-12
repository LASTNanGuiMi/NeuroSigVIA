#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 所有数据集与 seed 的训练参数均在下方逐条列出，可直接修改对应命令。
# 运行前激活 Python 环境；PYTHON_BIN 可指定解释器，DRY_RUN=1 只打印命令。
# 每个数据集占一张 GPU，三个种子依次运行；数据集之间并行。
METHOD=PatchTST
source scripts/lib/explicit_experiments.sh

# Subject-Independent
# ADFTD Dataset
(
export CUDA_VISIBLE_DEVICES=0

# ADFTD — seed 42
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset adftd \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 128 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 42 \
  --result_dir "$RESULT_ROOT/seed42/adftd" \
  "$@"

# ADFTD — seed 43
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset adftd \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 128 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 43 \
  --result_dir "$RESULT_ROOT/seed43/adftd" \
  "$@"

# ADFTD — seed 44
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset adftd \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 128 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 44 \
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
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset tdbrain \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 32 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 42 \
  --result_dir "$RESULT_ROOT/seed42/tdbrain" \
  "$@"

# TDBRAIN — seed 43
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset tdbrain \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 32 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 43 \
  --result_dir "$RESULT_ROOT/seed43/tdbrain" \
  "$@"

# TDBRAIN — seed 44
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset tdbrain \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 32 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 44 \
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
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset apava \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 32 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 42 \
  --result_dir "$RESULT_ROOT/seed42/apava" \
  "$@"

# APAVA — seed 43
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset apava \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 32 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 43 \
  --result_dir "$RESULT_ROOT/seed43/apava" \
  "$@"

# APAVA — seed 44
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset apava \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 32 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 44 \
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
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset shimmer10 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 1 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 42 \
  --result_dir "$RESULT_ROOT/seed42/shimmer10" \
  "$@"

# Shimmer10 — seed 43
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset shimmer10 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 1 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 43 \
  --result_dir "$RESULT_ROOT/seed43/shimmer10" \
  "$@"

# Shimmer10 — seed 44
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset shimmer10 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 1 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 44 \
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
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset pads11 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 4 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 42 \
  --result_dir "$RESULT_ROOT/seed42/pads11" \
  "$@"

# PADS11 — seed 43
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset pads11 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 4 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 43 \
  --result_dir "$RESULT_ROOT/seed43/pads11" \
  "$@"

# PADS11 — seed 44
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model PatchTST \
  --dataset pads11 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 6 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 10 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --batch_size 4 \
  --patch_len 16 \
  --stride 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 44 \
  --result_dir "$RESULT_ROOT/seed44/pads11" \
  "$@"

) &
PIDS+=("$!")

wait_for_jobs "${PIDS[@]}"
