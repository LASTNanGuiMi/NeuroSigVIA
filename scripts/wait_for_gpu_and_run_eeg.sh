#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

DATASET_KEY="${1:?Usage: $0 DATASET_KEY GPU_ID}"
GPU_ID="${2:?Usage: $0 DATASET_KEY GPU_ID}"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-500}"
POLL_SECONDS="${POLL_SECONDS:-30}"
SEED="${SEED:-42}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-logs/eeg_medformer_adaptive_granularity_seed${SEED}}"

[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be non-negative" >&2; exit 2; }
[[ "$GPU_FREE_THRESHOLD_MIB" =~ ^[1-9][0-9]*$ ]] || {
  echo "GPU_FREE_THRESHOLD_MIB must be positive" >&2
  exit 2
}
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "POLL_SECONDS must be positive" >&2
  exit 2
}

mkdir -p "$RUN_LOG_ROOT"
LOG_FILE="$RUN_LOG_ROOT/${DATASET_KEY}_gpu${GPU_ID}.log"
STATUS_FILE="$RUN_LOG_ROOT/${DATASET_KEY}_gpu${GPU_ID}.status"
exec >> "$LOG_FILE" 2>&1

echo "[$(date --iso-8601=seconds)] waiting: dataset=$DATASET_KEY gpu=$GPU_ID threshold=${GPU_FREE_THRESHOLD_MIB}MiB"
poll_count=0
while true; do
  memory_used="$({
    nvidia-smi -i "$GPU_ID" \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits
  } | tr -d '[:space:]')"
  [[ "$memory_used" =~ ^[0-9]+$ ]] || {
    echo "Could not parse GPU $GPU_ID memory usage: $memory_used" >&2
    exit 1
  }
  if (( memory_used < GPU_FREE_THRESHOLD_MIB )); then
    break
  fi
  if (( poll_count % 10 == 0 )); then
    echo "[$(date --iso-8601=seconds)] still waiting: gpu=$GPU_ID memory_used=${memory_used}MiB"
  fi
  poll_count=$((poll_count + 1))
  sleep "$POLL_SECONDS"
done

echo "[$(date --iso-8601=seconds)] launching: dataset=$DATASET_KEY gpu=$GPU_ID"
set +e
GPU="$GPU_ID" \
SEED="$SEED" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
EPOCHS="${EPOCHS:-40}" \
PATIENCE="${PATIENCE:-8}" \
bash scripts/run_eeg_medformer_multimodal.sh "$DATASET_KEY"
exit_status=$?
set -e
printf '%s\n' "$exit_status" > "$STATUS_FILE"
echo "[$(date --iso-8601=seconds)] finished: dataset=$DATASET_KEY gpu=$GPU_ID exit_status=$exit_status"
exit "$exit_status"
