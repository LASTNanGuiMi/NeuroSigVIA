#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 所有数据集与 seed 的训练参数均在下方逐条列出，可直接修改对应命令。
# 运行前激活 Python 环境；PYTHON_BIN 可指定解释器，DRY_RUN=1 只打印命令。
# 每个数据集占一张 GPU，三个种子依次运行；数据集之间并行。
METHOD=TimesNet
source scripts/lib/explicit_experiments.sh

# 用户指定的四数据集对照组合：APAVA/Shimmer dropout=0.3，TDBRAIN/PADS dropout=0.1。
# 对应已有截图/敏感性实验；其他超参数保持原 TimesNet 配置。

# Subject-Independent
# TDBRAIN Dataset
(
export CUDA_VISIBLE_DEVICES=0

# TDBRAIN — seed 42
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset tdbrain \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 42 \
  --result_dir "$RESULT_ROOT/seed42/tdbrain" \
  --progress \
  "$@"

# TDBRAIN — seed 43
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset tdbrain \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 43 \
  --result_dir "$RESULT_ROOT/seed43/tdbrain" \
  --progress \
  "$@"

# TDBRAIN — seed 44
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset tdbrain \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 44 \
  --result_dir "$RESULT_ROOT/seed44/tdbrain" \
  --progress \
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
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset apava \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.3 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 42 \
  --result_dir "$RESULT_ROOT/seed42/apava" \
  --progress \
  "$@"

# APAVA — seed 43
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset apava \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.3 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 43 \
  --result_dir "$RESULT_ROOT/seed43/apava" \
  --progress \
  "$@"

# APAVA — seed 44
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset apava \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.3 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 8 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 44 \
  --result_dir "$RESULT_ROOT/seed44/apava" \
  --progress \
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
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset shimmer10 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.3 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 1 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 42 \
  --result_dir "$RESULT_ROOT/seed42/shimmer10" \
  --progress \
  "$@"

# Shimmer10 — seed 43
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset shimmer10 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.3 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 1 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 43 \
  --result_dir "$RESULT_ROOT/seed43/shimmer10" \
  --progress \
  "$@"

# Shimmer10 — seed 44
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset shimmer10 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.3 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 1 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 44 \
  --result_dir "$RESULT_ROOT/seed44/shimmer10" \
  --progress \
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
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset pads11 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 4 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 42 \
  --result_dir "$RESULT_ROOT/seed42/pads11" \
  --progress \
  "$@"

# PADS11 — seed 43
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset pads11 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 4 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 43 \
  --result_dir "$RESULT_ROOT/seed43/pads11" \
  --progress \
  "$@"

# PADS11 — seed 44
python \
  -u \
  -m runners.baselines \
  --task_name classification \
  --model TimesNet \
  --dataset pads11 \
  --split_seed 42 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 8 \
  --dropout 0.1 \
  --learning_rate 3e-4 \
  --weight_decay 1e-3 \
  --train_epochs 100 \
  --patience 12 \
  --warmup_epochs 10 \
  --min_delta 0.002 \
  --top_k 3 \
  --num_kernels 6 \
  --batch_size 4 \
  --checkpoint_metric window_macro_f1 \
  --random_seed 44 \
  --result_dir "$RESULT_ROOT/seed44/pads11" \
  --progress \
  "$@"

) &
PIDS+=("$!")

wait_for_jobs "${PIDS[@]}"
