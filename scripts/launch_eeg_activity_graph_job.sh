#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

DATASET_KEY="${1:?Usage: $0 DATASET_KEY GPU_ID RUN_TAG}"
GPU_ID="${2:?Usage: $0 DATASET_KEY GPU_ID RUN_TAG}"
RUN_TAG="${3:?Usage: $0 DATASET_KEY GPU_ID RUN_TAG}"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-500}"
SEED="${SEED:-42}"

[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be non-negative" >&2; exit 2; }
[[ "$GPU_FREE_THRESHOLD_MIB" =~ ^[1-9][0-9]*$ ]] || {
  echo "GPU_FREE_THRESHOLD_MIB must be positive" >&2
  exit 2
}

LOG_ROOT="logs/$RUN_TAG"
LOG_FILE="$LOG_ROOT/${DATASET_KEY}_gpu${GPU_ID}.log"
STATUS_FILE="$LOG_ROOT/${DATASET_KEY}_gpu${GPU_ID}.status"
RESULT_DIR="results/$RUN_TAG/$DATASET_KEY"
FEATURE_CACHE_DIR="feature_cache/$RUN_TAG/$DATASET_KEY"
mkdir -p "$LOG_ROOT"

memory_used="$({
  nvidia-smi -i "$GPU_ID" \
    --query-gpu=memory.used \
    --format=csv,noheader,nounits
} | tr -d '[:space:]')"
if [[ ! "$memory_used" =~ ^[0-9]+$ ]]; then
  echo "Could not parse GPU $GPU_ID memory usage: $memory_used" | tee -a "$LOG_FILE" >&2
  printf '%s\n' 2 > "$STATUS_FILE"
  exit 2
fi
if (( memory_used >= GPU_FREE_THRESHOLD_MIB )); then
  echo "GPU $GPU_ID is not free: ${memory_used} MiB used" | tee -a "$LOG_FILE" >&2
  printf '%s\n' 75 > "$STATUS_FILE"
  exit 75
fi

echo "[$(date --iso-8601=seconds)] launching: dataset=$DATASET_KEY gpu=$GPU_ID image_mode=activity_graph medformer_graph=0 adaptive_granularity=0" | tee -a "$LOG_FILE"
set +e
set -o pipefail
GPU="$GPU_ID" \
SEED="$SEED" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
EPOCHS="${EPOCHS:-40}" \
PATIENCE="${PATIENCE:-8}" \
RESULT_DIR="$RESULT_DIR" \
FEATURE_CACHE_DIR="$FEATURE_CACHE_DIR" \
bash scripts/run_eeg_activity_graph_multimodal.sh "$DATASET_KEY" 2>&1 | tee -a "$LOG_FILE"
exit_status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$exit_status" > "$STATUS_FILE"
echo "[$(date --iso-8601=seconds)] finished: dataset=$DATASET_KEY gpu=$GPU_ID exit_status=$exit_status" | tee -a "$LOG_FILE"
exit "$exit_status"
