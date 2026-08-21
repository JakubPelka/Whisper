#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVED_ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$RESOLVED_ROOT_DIR"

CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/whisper-recordings-watcher.env"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/whisper-recordings-watcher"
STATE_FILE="$STATE_DIR/processed.tsv"
LOG_FILE="$STATE_DIR/watcher.log"
LOCK_FILE="${XDG_RUNTIME_DIR:-/tmp}/whisper-recordings-watcher-${UID}.lock"

WATCH_DIR="/home/jakub-pelka/MobileTransfer/Recordings"
LANGUAGE="sv"
KB_MODEL="large"
KB_REVISION="standard"
WHISPER_MODEL="large-v3-turbo"
WHISPER_PRESET="fast"
TRANSCRIPT_DIR_NAME="output_transkrypcja"
FFPROBE_TIMEOUT_SECONDS="120"
FFMPEG_TIMEOUT_SECONDS="1800"
STABILITY_SECONDS="90"
SCAN_INTERVAL_SECONDS="120"
RECURSIVE_SCAN="false"
RETRY_FAILED="true"
RETRY_DELAY_SECONDS="3600"

if [[ -f "$CONFIG_FILE" ]]; then
  # This is a user-owned local shell configuration file.
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
fi

# Repository location always comes from this script, never from local config.
ROOT_DIR="$RESOLVED_ROOT_DIR"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer (got: $value)"
}

normalize_boolean() {
  case "${1,,}" in
    true|yes|1) printf 'true' ;;
    false|no|0) printf 'false' ;;
    *) die "invalid boolean value: $1" ;;
  esac
}

require_nonnegative_integer "STABILITY_SECONDS" "$STABILITY_SECONDS"
require_nonnegative_integer "SCAN_INTERVAL_SECONDS" "$SCAN_INTERVAL_SECONDS"
require_nonnegative_integer "RETRY_DELAY_SECONDS" "$RETRY_DELAY_SECONDS"
require_nonnegative_integer "FFPROBE_TIMEOUT_SECONDS" "$FFPROBE_TIMEOUT_SECONDS"
require_nonnegative_integer "FFMPEG_TIMEOUT_SECONDS" "$FFMPEG_TIMEOUT_SECONDS"
RECURSIVE_SCAN="$(normalize_boolean "$RECURSIVE_SCAN")"
RETRY_FAILED="$(normalize_boolean "$RETRY_FAILED")"
[[ -n "$TRANSCRIPT_DIR_NAME" && "$TRANSCRIPT_DIR_NAME" != */* && \
   "$TRANSCRIPT_DIR_NAME" != "." && "$TRANSCRIPT_DIR_NAME" != ".." ]] \
  || die "TRANSCRIPT_DIR_NAME must be one safe folder name (got: $TRANSCRIPT_DIR_NAME)"

[[ -d "$WATCH_DIR" ]] || die "watch directory does not exist: $WATCH_DIR"
[[ -x "$ROOT_DIR/scripts/start.sh" ]] || die "launcher is missing or not executable: $ROOT_DIR/scripts/start.sh"
command -v flock >/dev/null 2>&1 || die "flock is required (normally provided by util-linux)"

mkdir -p "$STATE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  exit 0
fi

event() {
  local event_name="$1"
  local subject="$2"
  local detail="${3:-}"
  local timestamp
  timestamp="$(date --iso-8601=seconds)"
  if [[ -n "$detail" ]]; then
    printf '%s\t%s\t%s\t%s\n' "$timestamp" "$event_name" "$subject" "$detail" >> "$LOG_FILE"
  else
    printf '%s\t%s\t%s\n' "$timestamp" "$event_name" "$subject" >> "$LOG_FILE"
  fi
}

initialize_state_file() {
  if [[ ! -e "$STATE_FILE" ]]; then
    printf '# path\tsize_bytes\tmtime_epoch\tprocessed_epoch\tstatus\n' > "$STATE_FILE"
  fi
}

append_state() {
  local file="$1"
  local size="$2"
  local mtime="$3"
  local processed_at="$4"
  local status="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "$file" "$size" "$mtime" "$processed_at" "$status" >> "$STATE_FILE"
}

latest_state() {
  local file="$1"
  awk -F '\t' -v target="$file" '
    $1 == target { size=$2; mtime=$3; processed=$4; status=$5; found=1 }
    END { if (found) printf "%s\t%s\t%s\t%s\n", size, mtime, processed, status }
  ' "$STATE_FILE"
}

find_candidates() {
  local maxdepth_args=()
  if [[ "$RECURSIVE_SCAN" == "false" ]]; then
    maxdepth_args=(-maxdepth 1)
  fi

  find "$WATCH_DIR" "${maxdepth_args[@]}" -type f \
    -not -path '*/output_transkrypcja/*' \
    -not -path "*/$TRANSCRIPT_DIR_NAME/*" \
    -not -path '*/.*' \
    -not -name '*~' \
    -not -iname '*.part' \
    -not -iname '*.part.*' \
    -not -iname '*.partial' \
    -not -iname '*.partial.*' \
    -not -iname '*.tmp' \
    -not -iname '*.tmp.*' \
    -not -iname '*.temp' \
    -not -iname '*.temp.*' \
    -not -iname '*.crdownload' \
    \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.m4a' -o \
       -iname '*.aac' -o -iname '*.flac' -o -iname '*.ogg' -o \
       -iname '*.opus' -o -iname '*.qta' -o -iname '*.mp4' -o \
       -iname '*.mov' -o -iname '*.mkv' -o -iname '*.webm' -o \
       -iname '*.avi' \) \
    -print0 | sort -z
}

file_metadata() {
  stat --printf='%s\t%Y' -- "$1"
}

safe_component() {
  printf '%s' "$1" | sed -E 's/[^[:alnum:]_.-]/_/g; s/^_+//; s/_+$//'
}

resolve_kb_model() {
  case "$KB_MODEL" in
    large|medium) printf 'KBLab/kb-whisper-%s' "$KB_MODEL" ;;
    *) printf '%s' "$KB_MODEL" ;;
  esac
}

expected_outputs_exist() {
  local file="$1"
  local out_dir="$2"
  local stem
  stem="$(basename -- "$file")"
  stem="${stem%.*}"

  if [[ "${LANGUAGE,,}" =~ ^(sv|se|swe|swedish|szwedzki)$ ]]; then
    local kb_model model_part base
    kb_model="$(resolve_kb_model)"
    model_part="$(safe_component "${kb_model//\//_}")"
    base="${stem}_${model_part}_${KB_REVISION}"
    [[ -s "$out_dir/$base.txt" && -s "$out_dir/$base.json" ]]
  else
    local model_part lang_part base
    model_part="$(safe_component "$WHISPER_MODEL")"
    lang_part="$LANGUAGE"
    [[ "${lang_part,,}" == "auto" ]] && lang_part="auto"
    base="${stem}_openai-whisper_${model_part}_${lang_part}_${WHISPER_PRESET}"
    [[ -s "$out_dir/$base.txt" && -s "$out_dir/$base.json" ]]
  fi
}

should_process() {
  local file="$1"
  local size="$2"
  local mtime="$3"
  local state state_size state_mtime processed_at status now

  state="$(latest_state "$file")"
  [[ -n "$state" ]] || return 0
  IFS=$'\t' read -r state_size state_mtime processed_at status <<< "$state"

  if [[ "$state_size" != "$size" || "$state_mtime" != "$mtime" ]]; then
    return 0
  fi

  case "$status" in
    completed|completed-existing|unsupported)
      return 1
      ;;
    failed|processing|transcription-failed|note-failed)
      [[ "$RETRY_FAILED" == "true" ]] || return 1
      now="$(date +%s)"
      (( now - processed_at >= RETRY_DELAY_SECONDS ))
      return
      ;;
    *)
      return 0
      ;;
  esac
}

register_existing() {
  local count=0 file absolute metadata size mtime
  while IFS= read -r -d '' file; do
    absolute="$(readlink -f -- "$file")"
    if [[ "$absolute" == *$'\t'* || "$absolute" == *$'\n'* ]]; then
      continue
    fi
    metadata="$(file_metadata "$absolute")"
    IFS=$'\t' read -r size mtime <<< "$metadata"
    if [[ -z "$(latest_state "$absolute")" ]]; then
      append_state "$absolute" "$size" "$mtime" "$(date +%s)" "completed-existing"
      ((count += 1))
    fi
  done < <(find_candidates)
  printf '%s' "$count"
}

initialize_existing() {
  local count
  initialize_state_file
  count="$(register_existing)"
  event "BASELINE" "$WATCH_DIR" "registered=$count"
}

repair_missing_state() {
  local count
  initialize_state_file
  count="$(register_existing)"
  event "STATE_REPAIR" "$STATE_FILE" "created missing state; baseline registered=$count"
}

failure_reason() {
  local diagnostics="$1"
  local reason
  reason="$(grep -m1 '^AUDIO_PREPARATION_ERROR:' <<< "$diagnostics" 2>/dev/null || true)"
  if [[ -z "$reason" ]]; then
    reason="$(grep -E '^(ERROR|ERROR while processing)' <<< "$diagnostics" 2>/dev/null | tail -1 || true)"
  fi
  [[ -n "$reason" ]] || reason="transcription exited unsuccessfully or expected TXT/JSON output is missing"
  reason="$(printf '%s' "$reason" | tr '\t\r\n' ' ' | cut -c1-500)"
  printf '%s' "$reason"
}

run_scan() {
  local failures=0
  local file absolute first_metadata second_metadata size mtime transcript_dir kb_model
  local started_at elapsed reason diagnostics

  if [[ ! -e "$STATE_FILE" ]]; then
    repair_missing_state
    return 0
  fi

  while IFS= read -r -d '' file; do
    absolute="$(readlink -f -- "$file")"
    if [[ "$absolute" == *$'\t'* || "$absolute" == *$'\n'* ]]; then
      continue
    fi

    first_metadata="$(file_metadata "$absolute")" || continue
    IFS=$'\t' read -r size mtime <<< "$first_metadata"
    if ! should_process "$absolute" "$size" "$mtime"; then
      continue
    fi

    sleep "$STABILITY_SECONDS"
    if [[ ! -f "$absolute" ]]; then
      continue
    fi
    second_metadata="$(file_metadata "$absolute")"
    if [[ "$first_metadata" != "$second_metadata" ]]; then
      continue
    fi

    transcript_dir="$(dirname -- "$absolute")/$TRANSCRIPT_DIR_NAME"
    append_state "$absolute" "$size" "$mtime" "$(date +%s)" "processing"
    event "PROCESSING" "$absolute"
    started_at="$(date +%s)"
    kb_model="$(resolve_kb_model)"
    diagnostics=""

    if diagnostics="$(
         INPUT_FILE="$absolute" \
         LANGUAGE="$LANGUAGE" \
         ENGINE="auto" \
         OUT_DIR="$transcript_dir" \
         KB_WHISPER_MODEL="$kb_model" \
         KB_WHISPER_REVISION="$KB_REVISION" \
         WHISPER_MODEL="$WHISPER_MODEL" \
         WHISPER_PRESET="$WHISPER_PRESET" \
         FFPROBE_TIMEOUT_SECONDS="$FFPROBE_TIMEOUT_SECONDS" \
         FFMPEG_TIMEOUT_SECONDS="$FFMPEG_TIMEOUT_SECONDS" \
         "$ROOT_DIR/scripts/start.sh" 2>&1
       )" \
       && expected_outputs_exist "$absolute" "$transcript_dir"; then
      elapsed=$(( $(date +%s) - started_at ))
      append_state "$absolute" "$size" "$mtime" "$(date +%s)" "completed"
      event "COMPLETED" "$absolute" "${elapsed}s"
    else
      reason="$(failure_reason "$diagnostics")"
      if grep -q '^AUDIO_PREPARATION_ERROR:' <<< "$diagnostics" 2>/dev/null; then
        append_state "$absolute" "$size" "$mtime" "$(date +%s)" "unsupported"
        event "SKIPPED_UNSUPPORTED" "$absolute" "$reason"
      else
        append_state "$absolute" "$size" "$mtime" "$(date +%s)" "transcription-failed"
        event "FAILED" "$absolute" "$reason"
        ((failures += 1))
      fi
    fi
  done < <(find_candidates)

  (( failures == 0 ))
}

case "${1:-}" in
  "") run_scan ;;
  --initialize-existing) initialize_existing ;;
  *) die "usage: $0 [--initialize-existing]" ;;
esac
