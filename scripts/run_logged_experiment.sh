#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 LOG_PATH EXIT_PATH COMMAND [ARG ...]" >&2
  exit 2
fi

LOG_PATH="$1"
EXIT_PATH="$2"
shift 2

mkdir -p "$(dirname -- "$LOG_PATH")" "$(dirname -- "$EXIT_PATH")"
if [[ -e "$LOG_PATH" || -e "$EXIT_PATH" || -e "${EXIT_PATH}.tmp" ]]; then
  echo "Refusing to overwrite an existing run artifact: $LOG_PATH or $EXIT_PATH" >&2
  exit 73
fi

set +e
{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'command='
  printf ' %q' "$@"
  printf '\n'
  "$@"
} 2>&1 | tee "$LOG_PATH"
PIPE_STATUSES=("${PIPESTATUS[@]}")
COMMAND_STATUS="${PIPE_STATUSES[0]}"
TEE_STATUS="${PIPE_STATUSES[1]}"
STATUS="$COMMAND_STATUS"
if [[ "$STATUS" -eq 0 && "$TEE_STATUS" -ne 0 ]]; then
  STATUS="$TEE_STATUS"
fi

{
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'command_exit=%s tee_exit=%s pre_footer_exit=%s\n' \
    "$COMMAND_STATUS" "$TEE_STATUS" "$STATUS"
} | tee -a "$LOG_PATH"
FOOTER_STATUSES=("${PIPESTATUS[@]}")
FOOTER_COMMAND_STATUS="${FOOTER_STATUSES[0]}"
FOOTER_TEE_STATUS="${FOOTER_STATUSES[1]}"
if [[ "$STATUS" -eq 0 && "$FOOTER_COMMAND_STATUS" -ne 0 ]]; then
  STATUS="$FOOTER_COMMAND_STATUS"
fi
if [[ "$STATUS" -eq 0 && "$FOOTER_TEE_STATUS" -ne 0 ]]; then
  STATUS="$FOOTER_TEE_STATUS"
fi

printf '%s\n' "$STATUS" > "${EXIT_PATH}.tmp"
mv -- "${EXIT_PATH}.tmp" "$EXIT_PATH"
exit "$STATUS"
