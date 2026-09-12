#!/usr/bin/env bash
# 仅负责运行目录、GPU 锁、日志和进程管理；模型参数由各方法脚本逐条写出。

[[ -n "${METHOD:-}" && "$METHOD" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  printf 'Set a valid METHOD before sourcing explicit_experiments.sh.\n' >&2
  return 2
}

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-${METHOD}_s42-43-44_$(date +%Y%m%d_%H%M%S)_$$}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  printf 'Invalid RUN_TAG: %s\n' "$RUN_TAG" >&2
  return 2
}
[[ "${DRY_RUN:-0}" == 0 || "${DRY_RUN:-0}" == 1 ]] || return 2
[[ "${WAIT_FOR_GPUS:-0}" == 0 || "${WAIT_FOR_GPUS:-0}" == 1 ]] || return 2

RESULT_ROOT="$PROJECT_DIR/results/$RUN_TAG"
LOG_ROOT="$PROJECT_DIR/logs/$RUN_TAG"
STATUS_ROOT="$PROJECT_DIR/status/$RUN_TAG"
CACHE_ROOT="$PROJECT_DIR/feature_cache/$RUN_TAG"
EXPLICIT_OWNER_PID="$BASHPID"
EXPLICIT_CHILD_PID=''
EXPLICIT_CURRENT_STATUS=''
EXPLICIT_BASE_HF_HOME="${HF_HOME:-}"
declare -a PIDS=()

export NEUROSIGVIA_EEG_ROOT="${EEG_DATA_DIR:-${NEUROSIGVIA_EEG_ROOT:-$PROJECT_DIR/data/eeg/processed}}"
export NEUROSIGVIA_WEARABLE_ROOT="${WEARABLE_DATA_ROOT:-${NEUROSIGVIA_WEARABLE_ROOT:-$PROJECT_DIR/data/wearable}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

_explicit_write_status() {
  local state="$1" code="${2:-}" temporary
  [[ -n "${EXPLICIT_CURRENT_STATUS:-}" ]] || return 0
  temporary="$EXPLICIT_CURRENT_STATUS.tmp.$BASHPID"
  {
    printf 'state=%s\nmethod=%s\ndataset=%s\nseed=%s\ngpu=%s\n' \
      "$state" "$METHOD" "$EXPLICIT_DATASET" "$EXPLICIT_SEED" "$EXPLICIT_GPU"
    printf 'worker_pid=%s\nchild_pid=%s\nexit_code=%s\nupdated_at=%s\n' \
      "$BASHPID" "${EXPLICIT_CHILD_PID:-}" "$code" "$(date -Is)"
  } > "$temporary"
  mv -- "$temporary" "$EXPLICIT_CURRENT_STATUS"
}

_explicit_stop_group() {
  local child="${1:-}" attempt
  [[ "$child" =~ ^[1-9][0-9]*$ ]] || return 0
  # 每次训练由 setsid 创建自己的进程组；绝不对调用者或其他实验发信号。
  if kill -0 -- "-$child" 2>/dev/null; then
    kill -TERM -- "-$child" 2>/dev/null || true
    for attempt in {1..30}; do
      kill -0 -- "-$child" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 -- "-$child" 2>/dev/null; then
      kill -KILL -- "-$child" 2>/dev/null || true
    fi
  elif kill -0 "$child" 2>/dev/null; then
    # 处理后台进程刚创建、尚未进入 setsid 的极短时间窗口。
    kill -TERM "$child" 2>/dev/null || true
  fi
  wait "$child" 2>/dev/null || true
}

_explicit_signal() {
  local code="$1" pid
  trap '' TERM INT
  if [[ "$BASHPID" == "$EXPLICIT_OWNER_PID" ]]; then
    for pid in "${PIDS[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  else
    _explicit_stop_group "${EXPLICIT_CHILD_PID:-}"
    _explicit_write_status INTERRUPTED "$code"
  fi
  exit "$code"
}

trap '_explicit_signal 143' TERM
trap '_explicit_signal 130' INT

_explicit_gpu_is_free() {
  local used
  used="$(command timeout 15s nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" || return 1
  used="${used//[[:space:]]/}"
  [[ "$used" =~ ^[0-9]+$ && "$used" -lt 500 ]]
}

_explicit_record_job() {
  local result="$1" cache="$2" log="$3" claim="$4" manifest_fd printed code=0
  shift 4
  # 命令保持原始 Python argv；工作目录另存，便于复核与复现。
  printf -v printed '%q ' env "CUDA_VISIBLE_DEVICES=$EXPLICIT_GPU" "$PYTHON_BIN" "$@"
  printf '%s\n' "$printed" > "$claim/command.sh"
  printf '%s\n' "$PROJECT_DIR" > "$claim/cwd.txt"
  exec {manifest_fd}>>"$STATUS_ROOT/manifest.lock"
  if flock -x "$manifest_fd"; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$METHOD" "$EXPLICIT_DATASET" "$EXPLICIT_SEED" "$EXPLICIT_GPU" \
      "$EXPLICIT_CURRENT_STATUS" "$log" "$result" "$cache" >> "$STATUS_ROOT/manifest.tsv" || code=$?
  else
    code=$?
  fi
  exec {manifest_fd}>&-
  return "$code"
}

_explicit_save_provenance() (
  local path git_root='' commit=''
  local -a source_roots=()
  cd -- "$PROJECT_DIR" || return
  if command -v git >/dev/null 2>&1; then
    git_root="$(git rev-parse --show-toplevel 2>/dev/null)" || git_root=''
  fi
  if [[ "$git_root" == "$PROJECT_DIR" ]]; then
    if commit="$(git rev-parse --verify HEAD 2>/dev/null)"; then
      printf '%s\n' "$commit" > "$STATUS_ROOT/git_commit.txt"
    else
      printf 'UNAVAILABLE: Git worktree has no HEAD commit\n' > "$STATUS_ROOT/git_commit.txt"
    fi
    git diff --binary > "$STATUS_ROOT/tracked_changes.patch"
  else
    printf 'UNAVAILABLE: project is not a Git worktree\n' > "$STATUS_ROOT/git_commit.txt"
    : > "$STATUS_ROOT/tracked_changes.patch"
  fi
  # 与 Git diff 互补：未跟踪的入口和源码也会登记，缓存字节码不计入。
  for path in scripts runners src third_party data_loading; do
    if [[ -d "$path" ]]; then source_roots+=("$path"); fi
  done
  if [[ -f main.py ]]; then source_roots+=(main.py); fi
  if ((${#source_roots[@]})); then
    find "${source_roots[@]}" -type d -name __pycache__ -prune -o -type f -print0 \
      | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$STATUS_ROOT/source_sha256.txt"
  else
    : > "$STATUS_ROOT/source_sha256.txt"
  fi
)

python() {
  local result='' seed='' cache='-' gpu="${CUDA_VISIBLE_DEVICES:-}" option value category
  local i=0 count="$#" log status claim gpu_fd attempt ready=0 code=0 printed
  local -a supplied=("$@")
  local -A reserved_seen=()

  # 只读取运行元信息，不添加、删去、重排任何模型参数。
  # 末尾 "$@" 可以覆盖普通超参数，但不能重复覆盖数据集、种子或输出路径。
  while ((i < count)); do
    option="${supplied[i]%%=*}"
    case "$option" in
      --random_seed|--seed) category=seed ;;
      --result_dir|--output) category=result ;;
      --feature_cache_dir|--dataset|--datasets|--dataset_names|--model|--data_dir|--wearable_label_mode|--eeg_protocol|--eeg_normalization|--reference-run)
        category="$option" ;;
      *) i=$((i + 1)); continue ;;
    esac
    if [[ -n "${reserved_seen[$category]:-}" ]]; then
      printf 'Duplicate reserved argument: %s\n' "$option" >&2
      return 2
    fi
    reserved_seen["$category"]=1
    if [[ "${supplied[i]}" == *=* ]]; then
      value="${supplied[i]#*=}"
    else
      i=$((i + 1))
      if ((i >= count)) || [[ "${supplied[i]}" == --* ]]; then
        printf 'Missing value for %s\n' "$option" >&2
        return 2
      fi
      value="${supplied[i]}"
    fi
    case "$category" in
      seed) seed="$value" ;;
      result) result="$value" ;;
      --feature_cache_dir) cache="$value" ;;
    esac
    i=$((i + 1))
  done

  [[ "$seed" =~ ^[0-9]+$ && "$gpu" =~ ^[0-9]+$ ]] || {
    printf 'Each training command needs one numeric seed and CUDA_VISIBLE_DEVICES.\n' >&2
    return 2
  }
  [[ -n "$result" && "$result" == "$RESULT_ROOT/"* && "$result" != */../* && "$result" != */.. && "$result" != */./* ]] || {
    printf 'Training output must be inside RESULT_ROOT: %s\n' "$result" >&2
    return 2
  }
  EXPLICIT_DATASET="${result##*/}"
  [[ "$EXPLICIT_DATASET" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || return 2

  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    # 先拼成完整字符串，再用一次 printf 输出，避免并发数据集逐词交错。
    printf -v printed '%q ' env "CUDA_VISIBLE_DEVICES=$gpu" "$PYTHON_BIN" "$@"
    printf '%s\n' "$printed"
    return 0
  fi

  EXPLICIT_SEED="$seed"
  EXPLICIT_GPU="$gpu"
  EXPLICIT_CHILD_PID=''
  EXPLICIT_CURRENT_STATUS="$STATUS_ROOT/${EXPLICIT_DATASET}_seed${seed}.status"
  log="$LOG_ROOT/${EXPLICIT_DATASET}_seed${seed}.log"
  claim="$STATUS_ROOT/jobs/${EXPLICIT_DATASET}_seed${seed}"
  [[ ! -e "$result" && ! -e "$log" ]] || {
    printf 'Refusing to overwrite an existing job: %s\n' "$result" >&2
    return 2
  }
  mkdir -- "$claim" || {
    printf 'This dataset/seed was already claimed: %s seed %s\n' "$EXPLICIT_DATASET" "$seed" >&2
    return 2
  }
  # 后台子 shell 重新安装信号处理，确保父脚本取消时能收回自己的训练进程。
  trap '_explicit_signal 143' TERM
  trap '_explicit_signal 130' INT
  _explicit_write_status QUEUED
  _explicit_record_job "$result" "$cache" "$log" "$claim" "$@"
  exec {gpu_fd}>>"$PROJECT_DIR/.aris/gpu_locks/gpu_${gpu}.lock"

  while ! flock -n "$gpu_fd"; do
    if [[ "${WAIT_FOR_GPUS:-0}" != 1 ]]; then
      printf 'GPU %s is reserved by another experiment.\n' "$gpu" >&2
      _explicit_write_status FAILED 75
      exec {gpu_fd}>&-
      return 75
    fi
    _explicit_write_status WAITING_FOR_GPU
    sleep 5
  done
  # 锁内重新检查显存；前一个种子退出后的 CUDA 释放允许短暂延迟。
  if [[ "${WAIT_FOR_GPUS:-0}" == 1 ]]; then
    until _explicit_gpu_is_free "$gpu"; do
      _explicit_write_status WAITING_FOR_GPU
      sleep 5
    done
    ready=1
  else
    for attempt in {1..15}; do
      if _explicit_gpu_is_free "$gpu"; then ready=1; break; fi
      sleep 2
    done
  fi
  if [[ "$ready" != 1 ]]; then
    printf 'GPU %s is occupied; job was not started.\n' "$gpu" >&2
    _explicit_write_status FAILED 75
    exec {gpu_fd}>&-
    return 75
  fi

  export MPLCONFIGDIR="$STATUS_ROOT/runtime/${EXPLICIT_DATASET}_seed${seed}/mpl"
  export HF_HOME="${EXPLICIT_BASE_HF_HOME:-$STATUS_ROOT/runtime/${EXPLICIT_DATASET}_seed${seed}/hf}"
  printf '[RUNNING] %s %s seed=%s GPU=%s log=%s\n' "$METHOD" "$EXPLICIT_DATASET" "$seed" "$gpu" "$log"
  # 不用 exec 替换负责下一种子的 shell。独立进程组也会继承 GPU 锁 fd。
  command setsid "$PYTHON_BIN" "$@" < /dev/null > "$log" 2>&1 &
  EXPLICIT_CHILD_PID=$!
  _explicit_write_status RUNNING
  if wait "$EXPLICIT_CHILD_PID"; then code=0; else code=$?; fi
  # 若训练主进程提前退出，也不遗留本次运行启动的后台子进程。
  _explicit_stop_group "$EXPLICIT_CHILD_PID"
  if [[ "$code" == 0 ]]; then
    _explicit_write_status COMPLETED 0
    printf '[COMPLETED] %s %s seed=%s\n' "$METHOD" "$EXPLICIT_DATASET" "$seed"
  else
    _explicit_write_status FAILED "$code"
    printf '[FAILED exit=%s] %s %s seed=%s log=%s\n' "$code" "$METHOD" "$EXPLICIT_DATASET" "$seed" "$log" >&2
  fi
  EXPLICIT_CHILD_PID=''
  EXPLICIT_CURRENT_STATUS=''
  # 只关闭本 shell 的 fd，避免显式解锁仍由子进程继承的共享锁。
  exec {gpu_fd}>&-
  return "$code"
}

wait_for_jobs() {
  local pid code=0 current
  for pid in "$@"; do
    if wait "$pid"; then
      :
    else
      current=$?
      if [[ "$code" == 0 ]]; then code="$current"; fi
    fi
  done
  return "$code"
}

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  for explicit_dependency in "$PYTHON_BIN" flock setsid timeout nvidia-smi find sort xargs sha256sum; do
    command -v "$explicit_dependency" >/dev/null || {
      printf 'Missing runtime command: %s\n' "$explicit_dependency" >&2
      return 127
    }
  done
  for explicit_path in "$RESULT_ROOT" "$LOG_ROOT" "$STATUS_ROOT" "$CACHE_ROOT"; do
    [[ ! -e "$explicit_path" ]] || {
      printf 'Refusing to overwrite existing run: %s\n' "$explicit_path" >&2
      return 2
    }
  done
  mkdir -p -- "$PROJECT_DIR/status"
  mkdir -- "$STATUS_ROOT" || {
    printf 'RUN_TAG is already claimed: %s\n' "$RUN_TAG" >&2
    return 2
  }
  mkdir -p -- "$RESULT_ROOT" "$LOG_ROOT" "$CACHE_ROOT" "$STATUS_ROOT/jobs" "$PROJECT_DIR/.aris/gpu_locks"
  printf 'method\tdataset\tseed\tgpu\tstatus\tlog\tresult\tcache\n' > "$STATUS_ROOT/manifest.tsv"
  _explicit_save_provenance
  {
    printf 'method=%s\nrun_tag=%s\npython=%s\nstarted_at=%s\n' "$METHOD" "$RUN_TAG" "$PYTHON_BIN" "$(date -Is)"
    printf 'wait_for_gpus=%s\n' "${WAIT_FOR_GPUS:-0}"
  } > "$STATUS_ROOT/run_config.txt"
  printf 'Run: %s\nManifest: %s/manifest.tsv\nLogs: %s\nStatus: %s\n' "$RUN_TAG" "$STATUS_ROOT" "$LOG_ROOT" "$STATUS_ROOT"
fi
