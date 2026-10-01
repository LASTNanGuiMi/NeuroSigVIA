#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 扩展对比（表中带 * 的列）：TiViT 灰度图视觉分支（ViT-H-14 第 32 层）+ 冻结的对比方法编码器，
# 两路特征各自 L2 归一化后拼接，用 StandardScaler + LogisticRegression 分类；编码器不再训练。
# 所有数据集与 seed 的训练参数均在下方逐条列出，可直接修改对应命令。
# 运行前激活 Python 环境；PYTHON_BIN 可指定解释器，DRY_RUN=1 只打印命令。
# 每个数据集占一张 GPU，各条命令依次运行；数据集之间并行。
METHOD=TiViT_FrozenBackbone
source scripts/lib/explicit_experiments.sh

# 冻结骨干的参数来自本仓库训练出的对比方法运行；先运行对应脚本，再把运行目录传入。
MEDFORMER_RUN="${MEDFORMER_RUN:-results/MEDFORMER_RUN_TAG}"
[[ "${DRY_RUN:-0}" == 1 || -d "$MEDFORMER_RUN" ]] || {
  printf 'Set MEDFORMER_RUN to a completed scripts/Medformer.sh run directory, e.g. results/<RUN_TAG>.\n' >&2
  exit 2
}
PATCHTST_RUN="${PATCHTST_RUN:-results/PATCHTST_RUN_TAG}"
[[ "${DRY_RUN:-0}" == 1 || -d "$PATCHTST_RUN" ]] || {
  printf 'Set PATCHTST_RUN to a completed scripts/PatchTST.sh run directory, e.g. results/<RUN_TAG>.\n' >&2
  exit 2
}
TIMESNET_RUN="${TIMESNET_RUN:-results/TIMESNET_RUN_TAG}"
[[ "${DRY_RUN:-0}" == 1 || -d "$TIMESNET_RUN" ]] || {
  printf 'Set TIMESNET_RUN to a completed scripts/TimesNet.sh run directory, e.g. results/<RUN_TAG>.\n' >&2
  exit 2
}
TECH_RUN="${TECH_RUN:-results/TECH_RUN_TAG}"
[[ "${DRY_RUN:-0}" == 1 || -d "$TECH_RUN" ]] || {
  printf 'Set TECH_RUN to a completed scripts/TeCh.sh run directory, e.g. results/<RUN_TAG>.\n' >&2
  exit 2
}
OPENCLIP_WEIGHTS="${OPENCLIP_WEIGHTS:-../models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin}"

# Subject-Independent
# TDBRAIN Dataset
(
export CUDA_VISIBLE_DEVICES=0

# TDBRAIN — TiViT + Medformer — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/tdbrain" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/Medformer_tdbrain" \
  "$@"

# TDBRAIN — TiViT + Medformer — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/tdbrain" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/Medformer_tdbrain" \
  "$@"

# TDBRAIN — TiViT + Medformer — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/tdbrain" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/Medformer_tdbrain" \
  "$@"

# TDBRAIN — TiViT + PatchTST — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/tdbrain" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/PatchTST_tdbrain" \
  "$@"

# TDBRAIN — TiViT + PatchTST — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/tdbrain" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/PatchTST_tdbrain" \
  "$@"

# TDBRAIN — TiViT + PatchTST — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/tdbrain" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/PatchTST_tdbrain" \
  "$@"

# TDBRAIN — TiViT + TimesNet — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/tdbrain" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/TimesNet_tdbrain" \
  "$@"

# TDBRAIN — TiViT + TimesNet — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/tdbrain" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/TimesNet_tdbrain" \
  "$@"

# TDBRAIN — TiViT + TimesNet — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/tdbrain" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/TimesNet_tdbrain" \
  "$@"

# TDBRAIN — TiViT + TeCh — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/tdbrain" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/TeCh_tdbrain" \
  "$@"

# TDBRAIN — TiViT + TeCh — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/tdbrain" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/TeCh_tdbrain" \
  "$@"

# TDBRAIN — TiViT + TeCh — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset tdbrain \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/tdbrain" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/TeCh_tdbrain" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# APAVA Dataset
(
export CUDA_VISIBLE_DEVICES=1

# APAVA — TiViT + Medformer — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/apava" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/Medformer_apava" \
  "$@"

# APAVA — TiViT + Medformer — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/apava" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/Medformer_apava" \
  "$@"

# APAVA — TiViT + Medformer — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/apava" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/Medformer_apava" \
  "$@"

# APAVA — TiViT + PatchTST — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/apava" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/PatchTST_apava" \
  "$@"

# APAVA — TiViT + PatchTST — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/apava" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/PatchTST_apava" \
  "$@"

# APAVA — TiViT + PatchTST — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/apava" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/PatchTST_apava" \
  "$@"

# APAVA — TiViT + TimesNet — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/apava" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/TimesNet_apava" \
  "$@"

# APAVA — TiViT + TimesNet — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/apava" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/TimesNet_apava" \
  "$@"

# APAVA — TiViT + TimesNet — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/apava" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/TimesNet_apava" \
  "$@"

# APAVA — TiViT + TeCh — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/apava" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/TeCh_apava" \
  "$@"

# APAVA — TiViT + TeCh — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/apava" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/TeCh_apava" \
  "$@"

# APAVA — TiViT + TeCh — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset apava \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/apava" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/TeCh_apava" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# Shimmer10 Dataset
(
export CUDA_VISIBLE_DEVICES=2

# Shimmer10 — TiViT + Medformer — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/shimmer10" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/Medformer_shimmer10" \
  "$@"

# Shimmer10 — TiViT + Medformer — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/shimmer10" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/Medformer_shimmer10" \
  "$@"

# Shimmer10 — TiViT + Medformer — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/shimmer10" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/Medformer_shimmer10" \
  "$@"

# Shimmer10 — TiViT + PatchTST — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/shimmer10" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/PatchTST_shimmer10" \
  "$@"

# Shimmer10 — TiViT + PatchTST — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/shimmer10" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/PatchTST_shimmer10" \
  "$@"

# Shimmer10 — TiViT + PatchTST — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/shimmer10" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/PatchTST_shimmer10" \
  "$@"

# Shimmer10 — TiViT + TimesNet — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/shimmer10" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/TimesNet_shimmer10" \
  "$@"

# Shimmer10 — TiViT + TimesNet — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/shimmer10" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/TimesNet_shimmer10" \
  "$@"

# Shimmer10 — TiViT + TimesNet — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/shimmer10" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/TimesNet_shimmer10" \
  "$@"

# Shimmer10 — TiViT + TeCh — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/shimmer10" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/TeCh_shimmer10" \
  "$@"

# Shimmer10 — TiViT + TeCh — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/shimmer10" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/TeCh_shimmer10" \
  "$@"

# Shimmer10 — TiViT + TeCh — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset shimmer10 \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/shimmer10" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/TeCh_shimmer10" \
  "$@"

) &
PIDS+=("$!")

# Subject-Independent
# PADS11 Dataset
(
export CUDA_VISIBLE_DEVICES=3

# PADS11 — TiViT + Medformer — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/pads11" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/Medformer_pads11" \
  "$@"

# PADS11 — TiViT + Medformer — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/pads11" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/Medformer_pads11" \
  "$@"

# PADS11 — TiViT + Medformer — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model Medformer \
  --numeric-run "$MEDFORMER_RUN/seed{seed}/pads11" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/Medformer_pads11" \
  "$@"

# PADS11 — TiViT + PatchTST — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/pads11" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/PatchTST_pads11" \
  "$@"

# PADS11 — TiViT + PatchTST — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/pads11" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/PatchTST_pads11" \
  "$@"

# PADS11 — TiViT + PatchTST — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model PatchTST \
  --numeric-run "$PATCHTST_RUN/seed{seed}/pads11" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/PatchTST_pads11" \
  "$@"

# PADS11 — TiViT + TimesNet — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/pads11" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/TimesNet_pads11" \
  "$@"

# PADS11 — TiViT + TimesNet — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/pads11" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/TimesNet_pads11" \
  "$@"

# PADS11 — TiViT + TimesNet — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model TimesNet \
  --numeric-run "$TIMESNET_RUN/seed{seed}/pads11" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/TimesNet_pads11" \
  "$@"

# PADS11 — TiViT + TeCh — seed 42
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/pads11" \
  --seeds 42 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed42/TeCh_pads11" \
  "$@"

# PADS11 — TiViT + TeCh — seed 43
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/pads11" \
  --seeds 43 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed43/TeCh_pads11" \
  "$@"

# PADS11 — TiViT + TeCh — seed 44
python \
  -u \
  -m runners.tivit_numeric \
  --dataset pads11 \
  --numeric-model TeCh \
  --numeric-run "$TECH_RUN/seed{seed}/pads11" \
  --seeds 44 \
  --weights "$OPENCLIP_WEIGHTS" \
  --image-mode grayscale \
  --layer 32 \
  --aggregation mean \
  --patch-size sqrt \
  --stride 0.1 \
  --batch-size 32 \
  --chunk-size 128 \
  --output "$RESULT_ROOT/seed44/TeCh_pads11" \
  "$@"

) &
PIDS+=("$!")

wait_for_jobs "${PIDS[@]}"
