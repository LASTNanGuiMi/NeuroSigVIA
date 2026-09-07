#!/usr/bin/env bash
# Shared scheduling/logging only. Model commands live in the seven method scripts.

configure_dataset() {
  DATA_ARGS=()
  case "$1" in
    adftd) DATASET_NAME=ADFTD; DATASET_GROUP=eeg ;;
    tdbrain) DATASET_NAME=TDBRAIN; DATASET_GROUP=eeg ;;
    apava) DATASET_NAME=APAVA; DATASET_GROUP=eeg ;;
    shimmer10)
      DATASET_NAME=Shimmer_10_session10_AFC; DATASET_GROUP=wearable
      DATA_ARGS=(--wearable_label_mode shimmer_hc_vs_pd) ;;
    pads11)
      DATASET_NAME=PADS_11_task08_TouchIndex; DATASET_GROUP=wearable
      DATA_ARGS=(--wearable_label_mode pads_pd_vs_hc) ;;
    *) printf 'Unknown dataset: %s\n' "$1" >&2; return 2 ;;
  esac
  if [[ "$DATASET_GROUP" == eeg ]]; then
    DATA_ROOT="${EEG_DATA_DIR:-${NEUROSIGVIT_EEG_ROOT:-$PROJECT_DIR/data/eeg/processed}}"
    DATA_ARGS=(--eeg_protocol medformer_code_exact --eeg_normalization per_window_per_channel_standard_scaler_ddof0)
  else
    DATA_ROOT="${WEARABLE_DATA_ROOT:-${NEUROSIGVIT_WEARABLE_ROOT:-$PROJECT_DIR/data/wearable}}"
  fi
  TRAIN_BATCH_SIZE="${BATCH_SIZES[$1]:-}"
  if [[ ! "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Set a positive BATCH_SIZES[%s] in the method script.\n' "$1" >&2
    return 2
  fi
}

run_python() {
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -u "$@"
    printf '\n'
  else
    exec "$PYTHON_BIN" -u "$@"
  fi
}

write_job_status() {
  local target="$1" state="$2" dataset="$3" seed="$4" gpu="$5" exit_code="$6"
  {
    printf 'state=%s\nmethod=%s\ndataset=%s\nseed=%s\ngpu=%s\n' "$state" "$METHOD" "$dataset" "$seed" "$gpu"
    printf 'worker_pid=%s\nexit_code=%s\nupdated_at=%s\n' "$BASHPID" "$exit_code" "$(date -Is)"
  } > "$target.tmp.$BASHPID"
  mv -- "$target.tmp.$BASHPID" "$target"
}

gpu_is_free() {
  local used
  used="$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)"
  used="${used//[[:space:]]/}"
  [[ "$used" =~ ^[0-9]+$ && "$used" -lt 500 ]]
}

run_worker() {
  local worker_index="$1" gpu="$2"; shift 2
  local idx dataset='' seed='' code status_file='' result cache log failures=0 child_pid=''
  # A signal records an interrupted job and terminates only this worker's child.
  trap 'if [[ -n "$child_pid" ]]; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; fi; if [[ -n "$status_file" ]]; then write_job_status "$status_file" INTERRUPTED "$dataset" "$seed" "$gpu" 143; fi; exit 143' TERM INT
  exec {lock_fd}>"$PROJECT_DIR/.aris/gpu_locks/gpu_$gpu.lock"
  if [[ "${WAIT_FOR_GPUS:-0}" == 1 ]]; then
    flock "$lock_fd"
  elif ! flock -n "$lock_fd"; then
    printf 'GPU %s is reserved by another method script.\n' "$gpu" >&2
    for ((idx=worker_index; idx<${#DATASET_KEYS[@]}; idx+=${#GPU_IDS[@]})); do
      for seed in "${SEED_VALUES[@]}"; do
        write_job_status "$STATUS_ROOT/${DATASET_KEYS[idx]}_seed$seed.status" FAILED "${DATASET_KEYS[idx]}" "$seed" "$gpu" 75
      done
    done
    return 75
  fi
  for ((idx=worker_index; idx<${#DATASET_KEYS[@]}; idx+=${#GPU_IDS[@]})); do
    dataset="${DATASET_KEYS[idx]}"
    for seed in "${SEED_VALUES[@]}"; do
      status_file="$STATUS_ROOT/${dataset}_seed$seed.status"
      result="$RESULT_ROOT/seed$seed/$dataset"
      cache="$CACHE_ROOT/seed$seed/$dataset"
      log="$LOG_ROOT/${dataset}_seed$seed.log"
      if [[ "${WAIT_FOR_GPUS:-0}" == 1 ]]; then
        write_job_status "$status_file" WAITING_FOR_GPU "$dataset" "$seed" "$gpu" ''
        until gpu_is_free "$gpu"; do sleep 15; done
      else
        # CUDA contexts may take a few seconds to disappear after the last seed.
        local attempt
        for attempt in {1..15}; do
          if gpu_is_free "$gpu"; then break; fi
          sleep 2
        done
      fi
      if ! gpu_is_free "$gpu"; then
        printf 'GPU %s occupied; did not start %s seed %s.\n' "$gpu" "$dataset" "$seed" >&2
        write_job_status "$status_file" FAILED "$dataset" "$seed" "$gpu" 75
        failures=1
        continue
      fi
      write_job_status "$status_file" RUNNING "$dataset" "$seed" "$gpu" ''
      printf '[RUNNING] %s %s seed=%s GPU=%s log=%s\n' "$METHOD" "$dataset" "$seed" "$gpu" "$log"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
        export MPLCONFIGDIR="$STATUS_ROOT/runtime/${dataset}_seed$seed/mpl"
        export HF_HOME="${HF_HOME:-$STATUS_ROOT/runtime/${dataset}_seed$seed/hf}"
        export NEUROSIGVIT_EEG_ROOT="${EEG_DATA_DIR:-${NEUROSIGVIT_EEG_ROOT:-$PROJECT_DIR/data/eeg/processed}}"
        export NEUROSIGVIT_WEARABLE_ROOT="${WEARABLE_DATA_ROOT:-${NEUROSIGVIT_WEARABLE_ROOT:-$PROJECT_DIR/data/wearable}}"
        train_one "$dataset" "$seed" "$result" "$cache" "$@"
      ) >"$log" 2>&1 &
      child_pid=$!
      if wait "$child_pid"; then code=0; else code=$?; fi
      child_pid=''
      if [[ "$code" == 0 ]]; then
        write_job_status "$status_file" COMPLETED "$dataset" "$seed" "$gpu" 0
        printf '[COMPLETED] %s %s seed=%s\n' "$METHOD" "$dataset" "$seed"
      else
        write_job_status "$status_file" FAILED "$dataset" "$seed" "$gpu" "$code"
        printf '[FAILED exit=%s] %s %s seed=%s log=%s\n' "$code" "$METHOD" "$dataset" "$seed" "$log" >&2
        failures=1
      fi
    done
  done
  return "$failures"
}

run_experiments() {
  METHOD="$1"; shift
  PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
  cd -- "$PROJECT_DIR"
  PYTHON_BIN="${PYTHON_BIN:-python}"
  read -r -a SEED_VALUES <<< "$SEEDS"
  read -r -a DATASET_KEYS <<< "$DATASETS"
  local gpu_list="$GPUS"
  read -r -a GPU_IDS <<< "${gpu_list//,/ }"
  [[ ${#SEED_VALUES[@]} -gt 0 && ${#DATASET_KEYS[@]} -gt 0 && ${#GPU_IDS[@]} -gt 0 ]] || return 2
  local value seen=' ' kind
  for kind in SEED_VALUES GPU_IDS; do
    local -n values="$kind"
    seen=' '
    for value in "${values[@]}"; do
      [[ "$value" =~ ^[0-9]+$ && "$seen" != *" $value "* ]] || { printf 'Invalid/duplicate %s: %s\n' "$kind" "$value" >&2; return 2; }
      seen+="$value "
    done
  done
  seen=' '
  for value in "${DATASET_KEYS[@]}"; do
    configure_dataset "$value" || return 2
    [[ "$seen" != *" $value "* ]] || { printf 'Duplicate dataset: %s\n' "$value" >&2; return 2; }
    seen+="$value "
  done
  for value in "$@"; do
    case "$value" in
      --random_seed|--random_seed=*|--result_dir|--result_dir=*|--feature_cache_dir|--feature_cache_dir=*|--dataset|--dataset=*|--datasets|--datasets=*|--dataset_names|--dataset_names=*|--model|--model=*|--data_dir|--data_dir=*|--wearable_label_mode|--wearable_label_mode=*|--eeg_protocol|--eeg_protocol=*|--eeg_normalization|--eeg_normalization=*)
        printf 'Edit DATASETS/SEEDS in the method script; output paths use RUN_TAG. Reserved argument: %s\n' "$value" >&2; return 2 ;;
    esac
  done
  local seed_label
  seed_label="$(IFS=-; printf '%s' "${SEED_VALUES[*]}")"
  RUN_TAG="${RUN_TAG:-${METHOD}_s${seed_label}_$(date +%Y%m%d_%H%M%S)_$$}"
  [[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { printf 'Invalid RUN_TAG\n' >&2; return 2; }
  STATUS_ROOT="$PROJECT_DIR/status/$RUN_TAG"
  LOG_ROOT="$PROJECT_DIR/logs/$RUN_TAG"
  RESULT_ROOT="$PROJECT_DIR/results/$RUN_TAG"
  CACHE_ROOT="$PROJECT_DIR/feature_cache/$RUN_TAG"
  local idx seed gpu result cache dataset
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    for idx in "${!DATASET_KEYS[@]}"; do
      dataset="${DATASET_KEYS[idx]}"; gpu="${GPU_IDS[idx % ${#GPU_IDS[@]}]}"
      for seed in "${SEED_VALUES[@]}"; do
        ( export CUDA_VISIBLE_DEVICES="$gpu"; train_one "$dataset" "$seed" "$RESULT_ROOT/seed$seed/$dataset" "$CACHE_ROOT/seed$seed/$dataset" "$@" )
      done
    done
    return 0
  fi
  command -v "$PYTHON_BIN" >/dev/null
  command -v flock >/dev/null
  for value in "$STATUS_ROOT" "$LOG_ROOT" "$RESULT_ROOT" "$CACHE_ROOT"; do
    [[ ! -e "$value" ]] || { printf 'Refusing to overwrite existing run: %s\n' "$value" >&2; return 2; }
  done
  for gpu in "${GPU_IDS[@]}"; do
    nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader >/dev/null || return 75
    if [[ "${WAIT_FOR_GPUS:-0}" != 1 ]]; then
      gpu_is_free "$gpu" || { printf 'GPU %s is occupied; edit GPUS in the method script or set WAIT_FOR_GPUS=1.\n' "$gpu" >&2; return 75; }
    fi
  done
  mkdir -p -- "$PROJECT_DIR/status"
  mkdir -- "$STATUS_ROOT" || { printf 'RUN_TAG already claimed: %s\n' "$RUN_TAG" >&2; return 2; }
  mkdir -p -- "$LOG_ROOT" "$RESULT_ROOT" "$CACHE_ROOT" "$PROJECT_DIR/.aris/gpu_locks"
  git rev-parse HEAD > "$STATUS_ROOT/git_commit.txt"
  git diff --binary > "$STATUS_ROOT/tracked_changes.patch"
  # Include newly added entrypoints/runner/vendor hashes, which git diff omits.
  find scripts third_party -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$STATUS_ROOT/source_sha256.txt"
  sha256sum main.py run_baseline.py src/datautils.py data_loading/datasets.py >> "$STATUS_ROOT/source_sha256.txt"
  {
    printf 'method=%s\nrun_tag=%s\ntraining_seeds=%s\nsplit_seed=42\npython=%s\nstarted_at=%s\n' "$METHOD" "$RUN_TAG" "${SEED_VALUES[*]}" "$PYTHON_BIN" "$(date -Is)"
    printf 'extra_arguments='; printf '%q ' "$@"; printf '\n'
    printf 'wait_for_gpus=%s\n' "${WAIT_FOR_GPUS:-0}"
  } > "$STATUS_ROOT/run_config.txt"
  printf 'method\tdataset\tseed\tgpu\tstatus\tlog\tresult\tcache\n' > "$STATUS_ROOT/manifest.tsv"
  for idx in "${!DATASET_KEYS[@]}"; do
    dataset="${DATASET_KEYS[idx]}"; gpu="${GPU_IDS[idx % ${#GPU_IDS[@]}]}"
    for seed in "${SEED_VALUES[@]}"; do
      write_job_status "$STATUS_ROOT/${dataset}_seed$seed.status" QUEUED "$dataset" "$seed" "$gpu" ''
      cache='-'
      [[ "$METHOD" != NeuroSigViT ]] || cache="$CACHE_ROOT/seed$seed/$dataset"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$METHOD" "$dataset" "$seed" "$gpu" \
        "$STATUS_ROOT/${dataset}_seed$seed.status" "$LOG_ROOT/${dataset}_seed$seed.log" \
        "$RESULT_ROOT/seed$seed/$dataset" "$cache" >> "$STATUS_ROOT/manifest.tsv"
    done
  done
  printf 'Run: %s\nManifest: %s/manifest.tsv\n' "$RUN_TAG" "$STATUS_ROOT"
  local -a workers=()
  local pid code=0
  trap 'for pid in "${workers[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; wait; exit 143' TERM INT
  for idx in "${!GPU_IDS[@]}"; do
    [[ $idx -lt ${#DATASET_KEYS[@]} ]] || break
    ( run_worker "$idx" "${GPU_IDS[idx]}" "$@" ) &
    workers+=("$!")
  done
  for pid in "${workers[@]}"; do
    if ! wait "$pid"; then code=1; fi
  done
  trap - TERM INT
  printf 'Batch finished, exit=%s; status directory: %s\n' "$code" "$STATUS_ROOT"
  return "$code"
}
