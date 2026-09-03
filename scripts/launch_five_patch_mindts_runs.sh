#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-patch_mindts_run_seed${SEED}_$(date +%Y%m%d)}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
VISUAL_BATCH_SIZE="${VISUAL_BATCH_SIZE:-32}"
START_GAP_SECONDS="${START_GAP_SECONDS:-10}"

LOG_DIR="$PROJECT_DIR/logs/$RUN_TAG"
RESULT_BASE="$PROJECT_DIR/results/$RUN_TAG"
CACHE_BASE="$PROJECT_DIR/feature_cache/patch_mindts_seed${SEED}"
MANIFEST_PATH="$LOG_DIR/launch_manifest.txt"

[[ "$SEED" =~ ^[0-9]+$ ]] || {
  echo "SEED must be a non-negative integer, got: $SEED" >&2
  exit 2
}
for value_name in EPOCHS VISUAL_BATCH_SIZE START_GAP_SECONDS; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  }
done
[[ "$PATIENCE" =~ ^[0-9]+$ ]] || {
  echo "PATIENCE must be a non-negative integer, got: $PATIENCE" >&2
  exit 2
}

command -v screen >/dev/null
command -v sha256sum >/dev/null
mkdir -p "$LOG_DIR" "$RESULT_BASE" "$CACHE_BASE"
if [[ -e "$MANIFEST_PATH" ]]; then
  echo "Refusing to reuse an existing run tag: $RUN_TAG" >&2
  exit 73
fi

SESSION_SUFFIX="s${SEED}"
SESSIONS=(
  "pmrun_adftd_${SESSION_SUFFIX}"
  "pmrun_tdbrain_${SESSION_SUFFIX}"
  "pmrun_apava_${SESSION_SUFFIX}"
  "pmrun_shimmer_${SESSION_SUFFIX}"
  "pmrun_pads_${SESSION_SUFFIX}"
)
for session in "${SESSIONS[@]}"; do
  if screen -ls 2>/dev/null | grep -Fq ".$session"; then
    echo "Screen session already exists: $session" >&2
    exit 73
  fi
done

{
  printf 'run_tag=%s\n' "$RUN_TAG"
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'git_head=%s\n' "$(git rev-parse HEAD)"
  printf 'seed=%s epochs_max=%s early_stop_patience=%s visual_batch_size=%s\n' \
    "$SEED" "$EPOCHS" "$PATIENCE" "$VISUAL_BATCH_SIZE"
  printf 'outer_patch_size=64 outer_patch_stride=64\n'
  printf 'adaptive_granularity_bank=4;8;16\n'
  printf 'gpu_map=0:ADFTD,1:TDBRAIN,2:APAVA,3:Shimmer10,4:PADS11\n'
  printf 'environment_spec=tivit_env@8ec636c6\n'
  printf '\n[git status --short]\n'
  git status --short
  printf '\n[source sha256]\n'
  sha256sum \
    main.py \
    src/arguments.py \
    src/neurosigvit.py \
    src/patch_mindts.py \
    src/medformer_graph/renderer.py \
    scripts/run_eeg_patch_mindts.sh \
    scripts/run_wearable_patch_mindts.sh \
    scripts/run_logged_experiment.sh \
    scripts/launch_five_patch_mindts_runs.sh
} > "$MANIFEST_PATH"

launch_job() {
  local session="$1"
  local gpu="$2"
  local key="$3"
  local batch_size="$4"
  local launcher="$5"
  local dataset_key="$6"
  local log_path="$LOG_DIR/${key}.log"
  local exit_path="$LOG_DIR/${key}.exit"

  if [[ -e "$log_path" || -e "$exit_path" ]]; then
    echo "Run artifacts already exist for $key" >&2
    return 73
  fi

  screen -dmS "$session" \
    bash "$PROJECT_DIR/scripts/run_logged_experiment.sh" \
      "$log_path" "$exit_path" \
      env \
        OMP_NUM_THREADS=8 \
        MKL_NUM_THREADS=8 \
        OPENBLAS_NUM_THREADS=8 \
        PYTHONUNBUFFERED=1 \
        GPU="$gpu" \
        SEED="$SEED" \
        EPOCHS="$EPOCHS" \
        PATIENCE="$PATIENCE" \
        BATCH_SIZE="$batch_size" \
        VISUAL_BATCH_SIZE="$VISUAL_BATCH_SIZE" \
        RESULT_DIR="$RESULT_BASE/$key" \
        FEATURE_CACHE_DIR="$CACHE_BASE/$key" \
        bash "$PROJECT_DIR/$launcher" "$dataset_key"

  sleep "$START_GAP_SECONDS"
  if ! screen -ls 2>/dev/null | grep -Fq ".$session"; then
    echo "Screen $session did not remain alive after launch" >&2
    [[ -f "$log_path" ]] && tail -n 40 "$log_path" >&2
    [[ -f "$exit_path" ]] && echo "Recorded exit: $(<"$exit_path")" >&2
    return 1
  fi
  printf 'launched session=%s gpu=%s dataset=%s log=%s\n' \
    "$session" "$gpu" "$key" "$log_path"
}

launch_job "pmrun_adftd_${SESSION_SUFFIX}" 0 adftd 8 scripts/run_eeg_patch_mindts.sh adftd
launch_job "pmrun_tdbrain_${SESSION_SUFFIX}" 1 tdbrain 8 scripts/run_eeg_patch_mindts.sh tdbrain
launch_job "pmrun_apava_${SESSION_SUFFIX}" 2 apava 8 scripts/run_eeg_patch_mindts.sh apava
launch_job "pmrun_shimmer_${SESSION_SUFFIX}" 3 shimmer10 1 scripts/run_wearable_patch_mindts.sh shimmer10
launch_job "pmrun_pads_${SESSION_SUFFIX}" 4 pads11 4 scripts/run_wearable_patch_mindts.sh pads11

printf '\nActive sessions:\n'
screen -ls
printf '\nManifest: %s\n' "$MANIFEST_PATH"
