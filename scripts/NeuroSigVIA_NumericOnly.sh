#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 所有数据集与 seed 的训练参数均在下方逐条列出，可直接修改对应命令。
# 运行前激活 Python 环境；PYTHON_BIN 可指定解释器，DRY_RUN=1 只打印命令。
# 每个数据集占一张 GPU，三个种子依次运行；数据集之间并行。
METHOD=NeuroSigVIA_NumericOnly_NoFusion
source scripts/lib/explicit_experiments.sh

# NumericOnly 与 reference-run 中对应种子配对；模型和训练配置继承该参考实验。
# 本脚本显式列出数据集、种子、参考路径、输出路径和窗口级 checkpoint 指标。

# Subject-Independent
# Shimmer10 Dataset
(
export CUDA_VISIBLE_DEVICES=2

# Shimmer10 — seed 42
python \
  -u \
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset shimmer10 \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/shimmer10" \
  --checkpoint-metric window_macro_f1 \
  "$@"

# Shimmer10 — seed 43
python \
  -u \
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset shimmer10 \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/shimmer10" \
  --checkpoint-metric window_macro_f1 \
  "$@"

# Shimmer10 — seed 44
python \
  -u \
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset shimmer10 \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/shimmer10" \
  --checkpoint-metric window_macro_f1 \
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
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset pads11 \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/pads11" \
  --checkpoint-metric window_macro_f1 \
  "$@"

# PADS11 — seed 43
python \
  -u \
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset pads11 \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/pads11" \
  --checkpoint-metric window_macro_f1 \
  "$@"

# PADS11 — seed 44
python \
  -u \
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset pads11 \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/pads11" \
  --checkpoint-metric window_macro_f1 \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# APAVA Dataset
(
export CUDA_VISIBLE_DEVICES=4

# APAVA — seed 42
python \
  -u \
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset apava \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/apava" \
  --checkpoint-metric window_macro_f1 \
  "$@"

# APAVA — seed 43
python \
  -u \
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset apava \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/apava" \
  --checkpoint-metric window_macro_f1 \
  "$@"

# APAVA — seed 44
python \
  -u \
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset apava \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/apava" \
  --checkpoint-metric window_macro_f1 \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# TDBRAIN Dataset
(
export CUDA_VISIBLE_DEVICES=5

# TDBRAIN — seed 42
python \
  -u \
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset tdbrain \
  --seed 42 \
  --output "$RESULT_ROOT/seed42/tdbrain" \
  --checkpoint-metric window_macro_f1 \
  "$@"

# TDBRAIN — seed 43
python \
  -u \
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset tdbrain \
  --seed 43 \
  --output "$RESULT_ROOT/seed43/tdbrain" \
  --checkpoint-metric window_macro_f1 \
  "$@"

# TDBRAIN — seed 44
python \
  -u \
  -m runners.numeric_ablation \
  --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
  --dataset tdbrain \
  --seed 44 \
  --output "$RESULT_ROOT/seed44/tdbrain" \
  --checkpoint-metric window_macro_f1 \
  "$@"

) &
PIDS+=("$!")

wait_for_jobs "${PIDS[@]}"

# 汇总脚本使用真实解释器，不走训练任务记录器。
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  command "$PYTHON_BIN" scripts/summarize_numeric_ablation.py \
    --run "$RESULT_ROOT" \
    --reference-run "${REFERENCE_RUN:-results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550}" \
    --metric-level window
fi
