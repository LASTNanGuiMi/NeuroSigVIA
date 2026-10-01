#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 数值分支特征提取器消融：完整 NeuroSigVIA 的数值分支由 Mantis 换成冻结的对比方法编码器
# （TeCh / PatchTST / Medformer / TimesNet）；视觉分支、融合和训练设置与主方法一致。
# 所有数据集与 seed 的训练参数均在下方逐条列出，可直接修改对应命令。
# 运行前激活 Python 环境；PYTHON_BIN 可指定解释器，DRY_RUN=1 只打印命令。
# 每个数据集占一张 GPU，各条命令依次运行；数据集之间并行。
METHOD=Ablation_NumericBackbone
source scripts/lib/explicit_experiments.sh

# 冻结骨干的参数来自本仓库训练出的对比方法运行；先运行对应脚本，再把运行目录传入。
TECH_RUN="${TECH_RUN:-results/TECH_RUN_TAG}"
[[ "${DRY_RUN:-0}" == 1 || -d "$TECH_RUN" ]] || {
  printf 'Set TECH_RUN to a completed scripts/TeCh.sh run directory, e.g. results/<RUN_TAG>.\n' >&2
  exit 2
}
PATCHTST_RUN="${PATCHTST_RUN:-results/PATCHTST_RUN_TAG}"
[[ "${DRY_RUN:-0}" == 1 || -d "$PATCHTST_RUN" ]] || {
  printf 'Set PATCHTST_RUN to a completed scripts/PatchTST.sh run directory, e.g. results/<RUN_TAG>.\n' >&2
  exit 2
}
MEDFORMER_RUN="${MEDFORMER_RUN:-results/MEDFORMER_RUN_TAG}"
[[ "${DRY_RUN:-0}" == 1 || -d "$MEDFORMER_RUN" ]] || {
  printf 'Set MEDFORMER_RUN to a completed scripts/Medformer.sh run directory, e.g. results/<RUN_TAG>.\n' >&2
  exit 2
}
TIMESNET_RUN="${TIMESNET_RUN:-results/TIMESNET_RUN_TAG}"
[[ "${DRY_RUN:-0}" == 1 || -d "$TIMESNET_RUN" ]] || {
  printf 'Set TIMESNET_RUN to a completed scripts/TimesNet.sh run directory, e.g. results/<RUN_TAG>.\n' >&2
  exit 2
}

# Subject-Independent
# TDBRAIN Dataset
(
export CUDA_VISIBLE_DEVICES=0

# TDBRAIN — 数值分支换成冻结的 TeCh — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 42 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed42/TeCh_tdbrain" \
  "$@"

# TDBRAIN — 数值分支换成冻结的 TeCh — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 43 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed43/TeCh_tdbrain" \
  "$@"

# TDBRAIN — 数值分支换成冻结的 TeCh — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 44 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed44/TeCh_tdbrain" \
  "$@"

# TDBRAIN — 数值分支换成冻结的 PatchTST — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 42 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed42/PatchTST_tdbrain" \
  "$@"

# TDBRAIN — 数值分支换成冻结的 PatchTST — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 43 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed43/PatchTST_tdbrain" \
  "$@"

# TDBRAIN — 数值分支换成冻结的 PatchTST — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 44 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed44/PatchTST_tdbrain" \
  "$@"

# TDBRAIN — 数值分支换成冻结的 Medformer — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 42 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed42/Medformer_tdbrain" \
  "$@"

# TDBRAIN — 数值分支换成冻结的 Medformer — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 43 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed43/Medformer_tdbrain" \
  "$@"

# TDBRAIN — 数值分支换成冻结的 Medformer — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 44 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed44/Medformer_tdbrain" \
  "$@"

# TDBRAIN — 数值分支换成冻结的 TimesNet — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 42 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed42/TimesNet_tdbrain" \
  "$@"

# TDBRAIN — 数值分支换成冻结的 TimesNet — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 43 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed43/TimesNet_tdbrain" \
  "$@"

# TDBRAIN — 数值分支换成冻结的 TimesNet — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/tdbrain" \
  --dataset tdbrain \
  --seed 44 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed44/TimesNet_tdbrain" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# APAVA Dataset
(
export CUDA_VISIBLE_DEVICES=1

# APAVA — 数值分支换成冻结的 TeCh — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 42 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed42/TeCh_apava" \
  "$@"

# APAVA — 数值分支换成冻结的 TeCh — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 43 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed43/TeCh_apava" \
  "$@"

# APAVA — 数值分支换成冻结的 TeCh — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 44 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed44/TeCh_apava" \
  "$@"

# APAVA — 数值分支换成冻结的 PatchTST — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 42 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed42/PatchTST_apava" \
  "$@"

# APAVA — 数值分支换成冻结的 PatchTST — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 43 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed43/PatchTST_apava" \
  "$@"

# APAVA — 数值分支换成冻结的 PatchTST — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 44 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed44/PatchTST_apava" \
  "$@"

# APAVA — 数值分支换成冻结的 Medformer — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 42 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed42/Medformer_apava" \
  "$@"

# APAVA — 数值分支换成冻结的 Medformer — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 43 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed43/Medformer_apava" \
  "$@"

# APAVA — 数值分支换成冻结的 Medformer — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 44 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed44/Medformer_apava" \
  "$@"

# APAVA — 数值分支换成冻结的 TimesNet — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 42 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed42/TimesNet_apava" \
  "$@"

# APAVA — 数值分支换成冻结的 TimesNet — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 43 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed43/TimesNet_apava" \
  "$@"

# APAVA — 数值分支换成冻结的 TimesNet — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/apava" \
  --dataset apava \
  --seed 44 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 8 \
  --output "$RESULT_ROOT/seed44/TimesNet_apava" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# Shimmer10 Dataset
(
export CUDA_VISIBLE_DEVICES=2

# Shimmer10 — 数值分支换成冻结的 TeCh — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 42 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed42/TeCh_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

# Shimmer10 — 数值分支换成冻结的 TeCh — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 43 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed43/TeCh_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

# Shimmer10 — 数值分支换成冻结的 TeCh — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 44 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed44/TeCh_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

# Shimmer10 — 数值分支换成冻结的 PatchTST — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 42 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed42/PatchTST_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

# Shimmer10 — 数值分支换成冻结的 PatchTST — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 43 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed43/PatchTST_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

# Shimmer10 — 数值分支换成冻结的 PatchTST — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 44 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed44/PatchTST_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

# Shimmer10 — 数值分支换成冻结的 Medformer — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 42 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed42/Medformer_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

# Shimmer10 — 数值分支换成冻结的 Medformer — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 43 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed43/Medformer_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

# Shimmer10 — 数值分支换成冻结的 Medformer — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 44 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed44/Medformer_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

# Shimmer10 — 数值分支换成冻结的 TimesNet — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 42 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed42/TimesNet_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

# Shimmer10 — 数值分支换成冻结的 TimesNet — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 43 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed43/TimesNet_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

# Shimmer10 — 数值分支换成冻结的 TimesNet — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/shimmer10" \
  --dataset shimmer10 \
  --seed 44 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 1 \
  --output "$RESULT_ROOT/seed44/TimesNet_shimmer10" \
  --checkpoint-metric subject_macro_f1 \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# PADS11 Dataset
(
export CUDA_VISIBLE_DEVICES=3

# PADS11 — 数值分支换成冻结的 TeCh — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 42 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed42/TeCh_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

# PADS11 — 数值分支换成冻结的 TeCh — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 43 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed43/TeCh_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

# PADS11 — 数值分支换成冻结的 TeCh — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TECH_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 44 \
  --backbone TeCh \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed44/TeCh_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

# PADS11 — 数值分支换成冻结的 PatchTST — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 42 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed42/PatchTST_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

# PADS11 — 数值分支换成冻结的 PatchTST — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 43 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed43/PatchTST_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

# PADS11 — 数值分支换成冻结的 PatchTST — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$PATCHTST_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 44 \
  --backbone PatchTST \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed44/PatchTST_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

# PADS11 — 数值分支换成冻结的 Medformer — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 42 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed42/Medformer_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

# PADS11 — 数值分支换成冻结的 Medformer — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 43 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed43/Medformer_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

# PADS11 — 数值分支换成冻结的 Medformer — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$MEDFORMER_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 44 \
  --backbone Medformer \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed44/Medformer_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

# PADS11 — 数值分支换成冻结的 TimesNet — seed 42
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 42 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed42/TimesNet_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

# PADS11 — 数值分支换成冻结的 TimesNet — seed 43
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 43 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed43/TimesNet_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

# PADS11 — 数值分支换成冻结的 TimesNet — seed 44
python \
  -u \
  -m runners.numeric_backbone_frozen \
  --frozen-run "$TIMESNET_RUN/seed{seed}/pads11" \
  --dataset pads11 \
  --seed 44 \
  --backbone TimesNet \
  --vision-name "${MODEL_DIR:-${NEUROSIGVIA_CLIP_PATH:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K}}" \
  --batch-size 16 \
  --output "$RESULT_ROOT/seed44/TimesNet_pads11" \
  --lr 1e-4 \
  --dropout 0.2 \
  --alignment-weight 0.05 \
  "$@"

) &
PIDS+=("$!")

wait_for_jobs "${PIDS[@]}"
