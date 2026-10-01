#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# TeCh 对比方法：上游 TeCh 模型（third_party/tech）在本项目固定受试者划分上训练；各数据集配置写在 runners/tech.py。
# 所有数据集与 seed 的训练参数均在下方逐条列出，可直接修改对应命令。
# 运行前激活 Python 环境；PYTHON_BIN 可指定解释器，DRY_RUN=1 只打印命令。
# 每个数据集占一张 GPU，各条命令依次运行；数据集之间并行。
METHOD=TeCh
source scripts/lib/explicit_experiments.sh

# Subject-Independent
# TDBRAIN Dataset
(
export CUDA_VISIBLE_DEVICES=0

# TDBRAIN — seed 42
python \
  -u \
  -m runners.tech \
  --dataset tdbrain \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/tdbrain" \
  "$@"

# TDBRAIN — seed 43
python \
  -u \
  -m runners.tech \
  --dataset tdbrain \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/tdbrain" \
  "$@"

# TDBRAIN — seed 44
python \
  -u \
  -m runners.tech \
  --dataset tdbrain \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/tdbrain" \
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
  -m runners.tech \
  --dataset apava \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/apava" \
  "$@"

# APAVA — seed 43
python \
  -u \
  -m runners.tech \
  --dataset apava \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/apava" \
  "$@"

# APAVA — seed 44
python \
  -u \
  -m runners.tech \
  --dataset apava \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/apava" \
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
  -m runners.tech \
  --dataset shimmer10 \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/shimmer10" \
  "$@"

# Shimmer10 — seed 43
python \
  -u \
  -m runners.tech \
  --dataset shimmer10 \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/shimmer10" \
  "$@"

# Shimmer10 — seed 44
python \
  -u \
  -m runners.tech \
  --dataset shimmer10 \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/shimmer10" \
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
  -m runners.tech \
  --dataset pads11 \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/pads11" \
  "$@"

# PADS11 — seed 43
python \
  -u \
  -m runners.tech \
  --dataset pads11 \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/pads11" \
  "$@"

# PADS11 — seed 44
python \
  -u \
  -m runners.tech \
  --dataset pads11 \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/pads11" \
  "$@"

) &
PIDS+=("$!")

wait_for_jobs "${PIDS[@]}"
