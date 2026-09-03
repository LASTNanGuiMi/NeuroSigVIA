#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

SEED="${SEED:-42}"
ROUTER_MODE="${ROUTER_MODE:-adaptive_v4}"
CANDIDATE_MODE="${CANDIDATE_MODE:-single}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-auto}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
EARLY_STOP_STRATEGY="${EARLY_STOP_STRATEGY:-raw_selection_key}"
EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-0}"
EARLY_STOP_EMA_DECAY="${EARLY_STOP_EMA_DECAY:-0.6}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
VISUAL_BATCH_SIZE="${VISUAL_BATCH_SIZE:-32}"
START_GAP_SECONDS="${START_GAP_SECONDS:-10}"
DRY_RUN="${DRY_RUN:-0}"
ROUTER_TOP_K="${ROUTER_TOP_K:-2}"
ROUTER_TRAINING_NOISE_STD="${ROUTER_TRAINING_NOISE_STD:-0.0}"
ROUTER_LOCAL_WEIGHT="${ROUTER_LOCAL_WEIGHT:-0.5}"
ROUTER_RELATION_HIDDEN_DIM="${ROUTER_RELATION_HIDDEN_DIM:-16}"
ROUTER_RELATION_RESIDUAL_SCALE="${ROUTER_RELATION_RESIDUAL_SCALE:-0.25}"
ROUTER_KEY_ADAPTER_SCALE="${ROUTER_KEY_ADAPTER_SCALE:-0.1}"
ROUTER_VALUE_ADAPTER_SCALE="${ROUTER_VALUE_ADAPTER_SCALE:-0.1}"
ROUTER_ROUTE_BUDGET_WEIGHT="${ROUTER_ROUTE_BUDGET_WEIGHT:-0.005}"
ROUTER_LOAD_BALANCE_WEIGHT="${ROUTER_LOAD_BALANCE_WEIGHT:-0.005}"

# The default mapping deliberately leaves GPU0 available for the in-flight ADFTD run.
DATASETS="${DATASETS:-tdbrain,apava,shimmer10,pads11}"
GPU_LIST="${GPU_LIST:-1,2,3,4}"
RUN_TAG="${RUN_TAG:-patch_mindts_${CANDIDATE_MODE}_${ROUTER_MODE}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_SESSION_FAMILY="pm4"
if [[ "$ROUTER_MODE" == "adaptive_v5" ]]; then
  DEFAULT_SESSION_FAMILY="pm5"
fi
SESSION_PREFIX="${SESSION_PREFIX:-${DEFAULT_SESSION_FAMILY}_${CANDIDATE_MODE}_${ROUTER_MODE}_s${SEED}}"

LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/$RUN_TAG}"
RESULT_BASE="${RESULT_BASE:-$PROJECT_DIR/results/$RUN_TAG}"
# This path intentionally excludes ROUTER_MODE so all modes share frozen features.
CACHE_BASE="${CACHE_BASE:-$PROJECT_DIR/feature_cache/patch_mindts_${CANDIDATE_MODE}_seed${SEED}}"
MANIFEST_PATH="$LOG_DIR/launch_manifest.txt"

case "$ROUTER_MODE" in
  adaptive_v4|adaptive_v41|adaptive_v5|uniform) ;;
  *)
    echo "ROUTER_MODE must be adaptive_v4, adaptive_v41, adaptive_v5, or uniform, got: $ROUTER_MODE" >&2
    exit 2
    ;;
esac
case "$CANDIDATE_MODE" in
  single|composite) ;;
  *)
    echo "CANDIDATE_MODE must be single or composite, got: $CANDIDATE_MODE" >&2
    exit 2
    ;;
esac
case "$CHECKPOINT_METRIC" in
  auto|subject_macro_f1|window_macro_f1) ;;
  *)
    echo "CHECKPOINT_METRIC must be auto, subject_macro_f1, or window_macro_f1" >&2
    exit 2
    ;;
esac
case "$DRY_RUN" in
  0|1) ;;
  *)
    echo "DRY_RUN must be 0 or 1, got: $DRY_RUN" >&2
    exit 2
    ;;
esac
case "$EARLY_STOP_STRATEGY" in
  raw_selection_key|ema_primary) ;;
  *)
    echo "EARLY_STOP_STRATEGY must be raw_selection_key or ema_primary" >&2
    exit 2
    ;;
esac

[[ "$SEED" =~ ^[0-9]+$ ]] || {
  echo "SEED must be a non-negative integer, got: $SEED" >&2
  exit 2
}
for value_name in EPOCHS VISUAL_BATCH_SIZE; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  }
done
for value_name in ROUTER_TOP_K ROUTER_RELATION_HIDDEN_DIM; do
  value="${!value_name}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  }
done
for value_name in PATIENCE START_GAP_SECONDS; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "$value_name must be a non-negative integer, got: $value" >&2
    exit 2
  }
done
[[ "$EARLY_STOP_MIN_EPOCHS" =~ ^[0-9]+$ ]] || {
  echo "EARLY_STOP_MIN_EPOCHS must be a non-negative integer" >&2
  exit 2
}
(( EARLY_STOP_MIN_EPOCHS <= EPOCHS )) || {
  echo "EARLY_STOP_MIN_EPOCHS must not exceed EPOCHS" >&2
  exit 2
}
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "RUN_TAG may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
}
[[ "$SESSION_PREFIX" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "SESSION_PREFIX may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
}

IFS=',' read -r -a DATASET_KEYS <<< "$DATASETS"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#DATASET_KEYS[@]} -eq 0 || ${#DATASET_KEYS[@]} -gt 4 ]]; then
  echo "DATASETS must contain between one and four supported dataset keys" >&2
  exit 2
fi
if [[ ${#DATASET_KEYS[@]} -ne ${#GPUS[@]} ]]; then
  echo "GPU_LIST must contain one GPU for each DATASETS entry" >&2
  exit 2
fi

declare -A SEEN_DATASETS=()
declare -A SEEN_GPUS=()
for index in "${!DATASET_KEYS[@]}"; do
  key="${DATASET_KEYS[$index]}"
  gpu="${GPUS[$index]}"
  case "$key" in
    tdbrain|apava|shimmer10|pads11) ;;
    *)
      echo "Unsupported dataset key: $key" >&2
      exit 2
      ;;
  esac
  [[ "$gpu" =~ ^[0-9]+$ ]] || {
    echo "GPU entries must be non-negative integers, got: $gpu" >&2
    exit 2
  }
  if [[ -n "${SEEN_DATASETS[$key]:-}" ]]; then
    echo "Duplicate dataset key: $key" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "Duplicate GPU assignment: $gpu" >&2
    exit 2
  fi
  SEEN_DATASETS[$key]=1
  SEEN_GPUS[$gpu]=1
done

dataset_launcher() {
  case "$1" in
    tdbrain|apava) printf '%s' "scripts/run_eeg_patch_mindts.sh" ;;
    shimmer10|pads11) printf '%s' "scripts/run_wearable_patch_mindts.sh" ;;
  esac
}

dataset_batch_size() {
  case "$1" in
    tdbrain|apava) printf '%s' "8" ;;
    shimmer10) printf '%s' "1" ;;
    pads11) printf '%s' "4" ;;
  esac
}

emit_dry_run() {
  local gpu="$1"
  local key="$2"
  local launcher
  local batch_size
  launcher="$(dataset_launcher "$key")"
  batch_size="$(dataset_batch_size "$key")"
  env \
    GPU="$gpu" \
    SEED="$SEED" \
    EPOCHS="$EPOCHS" \
    PATIENCE="$PATIENCE" \
    EARLY_STOP_STRATEGY="$EARLY_STOP_STRATEGY" \
    EARLY_STOP_MIN_EPOCHS="$EARLY_STOP_MIN_EPOCHS" \
    EARLY_STOP_EMA_DECAY="$EARLY_STOP_EMA_DECAY" \
    EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
    BATCH_SIZE="$batch_size" \
    VISUAL_BATCH_SIZE="$VISUAL_BATCH_SIZE" \
    ROUTER_MODE="$ROUTER_MODE" \
    CANDIDATE_MODE="$CANDIDATE_MODE" \
    CHECKPOINT_METRIC="$CHECKPOINT_METRIC" \
    ROUTER_TOP_K="$ROUTER_TOP_K" \
    ROUTER_TRAINING_NOISE_STD="$ROUTER_TRAINING_NOISE_STD" \
    ROUTER_LOCAL_WEIGHT="$ROUTER_LOCAL_WEIGHT" \
    ROUTER_RELATION_HIDDEN_DIM="$ROUTER_RELATION_HIDDEN_DIM" \
    ROUTER_RELATION_RESIDUAL_SCALE="$ROUTER_RELATION_RESIDUAL_SCALE" \
    ROUTER_KEY_ADAPTER_SCALE="$ROUTER_KEY_ADAPTER_SCALE" \
    ROUTER_VALUE_ADAPTER_SCALE="$ROUTER_VALUE_ADAPTER_SCALE" \
    ROUTER_ROUTE_BUDGET_WEIGHT="$ROUTER_ROUTE_BUDGET_WEIGHT" \
    ROUTER_LOAD_BALANCE_WEIGHT="$ROUTER_LOAD_BALANCE_WEIGHT" \
    RESULT_DIR="$RESULT_BASE/$key" \
    FEATURE_CACHE_DIR="$CACHE_BASE/$key" \
    DRY_RUN=1 \
    bash "$PROJECT_DIR/$launcher" "$key"
}

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'run_tag=%s router=%s candidates=%s cache_base=%s\n' \
    "$RUN_TAG" "$ROUTER_MODE" "$CANDIDATE_MODE" "$CACHE_BASE"
  for index in "${!DATASET_KEYS[@]}"; do
    emit_dry_run "${GPUS[$index]}" "${DATASET_KEYS[$index]}"
  done
  exit 0
fi

command -v screen >/dev/null
command -v sha256sum >/dev/null
mkdir -p "$LOG_DIR" "$RESULT_BASE" "$CACHE_BASE"
if [[ -e "$MANIFEST_PATH" ]]; then
  echo "Refusing to reuse an existing run tag: $RUN_TAG" >&2
  exit 73
fi

SESSIONS=()
for key in "${DATASET_KEYS[@]}"; do
  session="${SESSION_PREFIX}_${key}"
  SESSIONS+=("$session")
  if screen -ls 2>/dev/null | grep -Fq ".$session"; then
    echo "Screen session already exists: $session" >&2
    exit 73
  fi
done

{
  printf 'run_tag=%s\n' "$RUN_TAG"
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'host=%s\n' "$(hostname)"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'git_head=%s\n' "$(git rev-parse HEAD)"
  else
    printf 'git_head=unavailable\n'
  fi
  printf 'seed=%s epochs_max=%s early_stop_patience=%s visual_batch_size=%s\n' \
    "$SEED" "$EPOCHS" "$PATIENCE" "$VISUAL_BATCH_SIZE"
  printf 'early_stop_strategy=%s min_epochs=%s ema_decay=%s min_delta=%s checkpoint_metric_is_raw=true\n' \
    "$EARLY_STOP_STRATEGY" "$EARLY_STOP_MIN_EPOCHS" \
    "$EARLY_STOP_EMA_DECAY" "$EARLY_STOP_MIN_DELTA"
  printf 'router_mode=%s checkpoint_metric=%s candidate_mode=%s\n' \
    "$ROUTER_MODE" "$CHECKPOINT_METRIC" "$CANDIDATE_MODE"
  if [[ "$CANDIDATE_MODE" == "single" ]]; then
    printf 'adaptive_granularity_bank=4;8;16\n'
  else
    printf 'adaptive_granularity_bank=1,2,4;2,4,8;4,8,16\n'
  fi
  if [[ "$ROUTER_MODE" == "adaptive_v41" ]]; then
    printf 'granularity_temperature=0.5 entropy_floor=0.0 entropy_ceiling=1.0\n'
    printf 'local_mix_max=0.80 local_mix_init=0.50 global_mix_max=0.75 global_mix_init=0.50\n'
    printf 'evidence_half_saturation=0.005 minimum_weight=0.01 score_cap=0.70\n'
    printf 'scorer_hidden_dim=32 confidence_half_saturation=0.05\n'
  elif [[ "$ROUTER_MODE" == "adaptive_v5" ]]; then
    printf 'granularity_temperature=1.0 entropy_floor=0.0 entropy_ceiling=1.0\n'
    printf 'v5_relation=line_q_graph_kv local_weight=%s relation_hidden_dim=%s relation_residual_scale=%s\n' \
      "$ROUTER_LOCAL_WEIGHT" "$ROUTER_RELATION_HIDDEN_DIM" "$ROUTER_RELATION_RESIDUAL_SCALE"
    printf 'v5_top_k=%s training_noise_std=%s key_adapter_scale=%s value_adapter_scale=%s\n' \
      "$ROUTER_TOP_K" "$ROUTER_TRAINING_NOISE_STD" "$ROUTER_KEY_ADAPTER_SCALE" "$ROUTER_VALUE_ADAPTER_SCALE"
    printf 'v5_route_budget_weight=%s definition=mse_sample_equal_post_topk_marginal_to_uniform\n' \
      "$ROUTER_ROUTE_BUDGET_WEIGHT"
    printf 'v5_load_balance_weight=%s definition=cv_squared_sample_equal_pre_topk_clean_softmax_marginal\n' \
      "$ROUTER_LOAD_BALANCE_WEIGHT"
    printf 'v5_inspiration=time_mosaic_patch_local_scope+pathformer_sparse_topk project_core=line_q_graph_kv\n'
  else
    printf 'granularity_temperature=1.0 entropy_floor=0.55 entropy_ceiling=1.0\n'
    printf 'local_mix_max=0.50 local_mix_init=0.10 global_mix_max=0.75 global_mix_init=0.50\n'
    printf 'evidence_half_saturation=0.05 minimum_weight=0.0 score_cap=1.0\n'
  fi
  if [[ "$ROUTER_MODE" == "uniform" ]]; then
    printf 'usage_weight=0 entropy_weight=0 mix_weight=0 prior_kl_weight=0\n'
  elif [[ "$ROUTER_MODE" == "adaptive_v41" ]]; then
    printf 'usage_weight=0 entropy_weight=0 mix_weight=0 prior_kl_weight=0\n'
  elif [[ "$ROUTER_MODE" == "adaptive_v5" ]]; then
    printf 'usage_weight=0 entropy_weight=0 mix_weight=0 prior_kl_weight=0\n'
  else
    printf 'usage_weight=0.01 entropy_weight=0.01 mix_weight=0.005 prior_kl_weight=0.001\n'
  fi
  printf 'cache_scope=representation_only cache_base=%s\n' "$CACHE_BASE"
  printf 'datasets=%s gpu_list=%s\n' "$DATASETS" "$GPU_LIST"
  printf 'outer_patch_size=64 outer_patch_stride=64\n'
  printf 'environment_spec=tivit_env@8ec636c6\n'
  printf '\n[git status --short]\n'
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git status --short
  else
    printf 'unavailable\n'
  fi
  printf '\n[source sha256]\n'
  sha256sum \
    main.py \
    src/arguments.py \
    src/datautils.py \
    src/neurosigvit.py \
    src/patch_mindts.py \
    src/medformer_graph/renderer.py \
    scripts/run_eeg_patch_mindts.sh \
    scripts/run_wearable_patch_mindts.sh \
    scripts/run_logged_experiment.sh \
    scripts/launch_four_patch_mindts_v4_runs.sh
} > "$MANIFEST_PATH"

launch_job() {
  local session="$1"
  local gpu="$2"
  local key="$3"
  local launcher
  local batch_size
  local log_path="$LOG_DIR/${key}.log"
  local exit_path="$LOG_DIR/${key}.exit"
  launcher="$(dataset_launcher "$key")"
  batch_size="$(dataset_batch_size "$key")"

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
        EARLY_STOP_STRATEGY="$EARLY_STOP_STRATEGY" \
        EARLY_STOP_MIN_EPOCHS="$EARLY_STOP_MIN_EPOCHS" \
        EARLY_STOP_EMA_DECAY="$EARLY_STOP_EMA_DECAY" \
        EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
        BATCH_SIZE="$batch_size" \
        VISUAL_BATCH_SIZE="$VISUAL_BATCH_SIZE" \
        ROUTER_MODE="$ROUTER_MODE" \
        CANDIDATE_MODE="$CANDIDATE_MODE" \
        CHECKPOINT_METRIC="$CHECKPOINT_METRIC" \
        ROUTER_TOP_K="$ROUTER_TOP_K" \
        ROUTER_TRAINING_NOISE_STD="$ROUTER_TRAINING_NOISE_STD" \
        ROUTER_LOCAL_WEIGHT="$ROUTER_LOCAL_WEIGHT" \
        ROUTER_RELATION_HIDDEN_DIM="$ROUTER_RELATION_HIDDEN_DIM" \
        ROUTER_RELATION_RESIDUAL_SCALE="$ROUTER_RELATION_RESIDUAL_SCALE" \
        ROUTER_KEY_ADAPTER_SCALE="$ROUTER_KEY_ADAPTER_SCALE" \
        ROUTER_VALUE_ADAPTER_SCALE="$ROUTER_VALUE_ADAPTER_SCALE" \
        ROUTER_ROUTE_BUDGET_WEIGHT="$ROUTER_ROUTE_BUDGET_WEIGHT" \
        ROUTER_LOAD_BALANCE_WEIGHT="$ROUTER_LOAD_BALANCE_WEIGHT" \
        RESULT_DIR="$RESULT_BASE/$key" \
        FEATURE_CACHE_DIR="$CACHE_BASE/$key" \
        bash "$PROJECT_DIR/$launcher" "$key"

  if (( START_GAP_SECONDS > 0 )); then
    sleep "$START_GAP_SECONDS"
  fi
  if ! screen -ls 2>/dev/null | grep -Fq ".$session"; then
    if [[ -f "$exit_path" && "$(<"$exit_path")" == "0" ]]; then
      printf 'completed session=%s gpu=%s dataset=%s log=%s\n' \
        "$session" "$gpu" "$key" "$log_path"
      return 0
    fi
    echo "Screen $session did not remain alive after launch" >&2
    [[ -f "$log_path" ]] && tail -n 40 "$log_path" >&2
    [[ -f "$exit_path" ]] && echo "Recorded exit: $(<"$exit_path")" >&2
    return 1
  fi
  printf 'launched session=%s gpu=%s dataset=%s log=%s\n' \
    "$session" "$gpu" "$key" "$log_path"
}

for index in "${!DATASET_KEYS[@]}"; do
  launch_job \
    "${SESSIONS[$index]}" \
    "${GPUS[$index]}" \
    "${DATASET_KEYS[$index]}"
done

printf '\nActive sessions:\n'
screen -ls
printf '\nManifest: %s\n' "$MANIFEST_PATH"
