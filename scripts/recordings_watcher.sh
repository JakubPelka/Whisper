#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVED_ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$RESOLVED_ROOT_DIR"

CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/whisper-recordings-watcher.env"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/whisper-recordings-watcher"
STATE_FILE="$STATE_DIR/processed.tsv"
LOG_FILE="$STATE_DIR/watcher.log"
LOCK_FILE="$STATE_DIR/watcher.lock"

WATCH_DIR="/home/jakub-pelka/MobileTransfer/Recordings"
LANGUAGE="sv"
KB_MODEL="large"
KB_REVISION="standard"
WHISPER_MODEL="large-v3-turbo"
WHISPER_PRESET="fast"
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

mkdir -p "$STATE_DIR"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"
}

die() {
  log "ERROR: $*"
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
RECURSIVE_SCAN="$(normalize_boolean "$RECURSIVE_SCAN")"
RETRY_FAILED="$(normalize_boolean "$RETRY_FAILED")"

[[ -d "$WATCH_DIR" ]] || die "watch directory does not exist: $WATCH_DIR"
[[ -x "$ROOT_DIR/scripts/start.sh" ]] || die "launcher is missing or not executable: $ROOT_DIR/scripts/start.sh"
command -v flock >/dev/null 2>&1 || die "flock is required (normally provided by util-linux)"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "Another watcher scan is already running; exiting."
  exit 0
fi

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
       -iname '*.opus' -o -iname '*.mp4' -o -iname '*.mov' -o \
       -iname '*.mkv' -o -iname '*.webm' -o -iname '*.avi' \) \
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
    completed|completed-existing)
      return 1
      ;;
    failed|processing)
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

initialize_existing() {
  local count=0 file metadata size mtime absolute
  initialize_state_file

  while IFS= read -r -d '' file; do
    absolute="$(readlink -f -- "$file")"
    if [[ "$absolute" == *$'\t'* || "$absolute" == *$'\n'* ]]; then
      log "Skipping path that cannot be represented safely in TSV state: $absolute"
      continue
    fi
    metadata="$(file_metadata "$absolute")"
    IFS=$'\t' read -r size mtime <<< "$metadata"
    if [[ -z "$(latest_state "$absolute")" ]]; then
      append_state "$absolute" "$size" "$mtime" "$(date +%s)" "completed-existing"
      ((count += 1))
    fi
  done < <(find_candidates)

  log "Initialization complete: registered $count existing supported recording(s); no transcription was run."
}

run_scan() {
  local found=0 processed=0 skipped=0 unstable=0 failed=0
  local file absolute first_metadata second_metadata size mtime out_dir kb_model
  initialize_state_file
  log "Starting scan of $WATCH_DIR (recursive=$RECURSIVE_SCAN)."

  while IFS= read -r -d '' file; do
    ((found += 1))
    absolute="$(readlink -f -- "$file")"
    if [[ "$absolute" == *$'\t'* || "$absolute" == *$'\n'* ]]; then
      log "Skipping path that cannot be represented safely in TSV state: $absolute"
      ((skipped += 1))
      continue
    fi

    first_metadata="$(file_metadata "$absolute")" || continue
    IFS=$'\t' read -r size mtime <<< "$first_metadata"
    if ! should_process "$absolute" "$size" "$mtime"; then
      ((skipped += 1))
      continue
    fi

    log "Checking file stability for $STABILITY_SECONDS second(s): $absolute"
    sleep "$STABILITY_SECONDS"
    if [[ ! -f "$absolute" ]]; then
      log "File disappeared during stability check; deferring: $absolute"
      ((unstable += 1))
      continue
    fi
    second_metadata="$(file_metadata "$absolute")"
    if [[ "$first_metadata" != "$second_metadata" ]]; then
      log "File is still changing; deferring until a later scan: $absolute"
      ((unstable += 1))
      continue
    fi

    out_dir="$(dirname -- "$absolute")/output_transkrypcja"
    append_state "$absolute" "$size" "$mtime" "$(date +%s)" "processing"
    log "Transcribing: $absolute"

    kb_model="$(resolve_kb_model)"
    if INPUT_FILE="$absolute" \
       LANGUAGE="$LANGUAGE" \
       ENGINE="auto" \
       OUT_DIR="$out_dir" \
       KB_WHISPER_MODEL="$kb_model" \
       KB_WHISPER_REVISION="$KB_REVISION" \
       WHISPER_MODEL="$WHISPER_MODEL" \
       WHISPER_PRESET="$WHISPER_PRESET" \
       "$ROOT_DIR/scripts/start.sh" && expected_outputs_exist "$absolute" "$out_dir"; then
      append_state "$absolute" "$size" "$mtime" "$(date +%s)" "completed"
      log "Completed: $absolute"
      ((processed += 1))
    else
      append_state "$absolute" "$size" "$mtime" "$(date +%s)" "failed"
      log "FAILED: transcription exited unsuccessfully or expected TXT/JSON output is missing: $absolute"
      ((failed += 1))
    fi
  done < <(find_candidates)

  log "Scan complete: found=$found processed=$processed skipped=$skipped unstable=$unstable failed=$failed."
  (( failed == 0 ))
}

case "${1:-}" in
  "") run_scan ;;
  --initialize-existing) initialize_existing ;;
  *) die "usage: $0 [--initialize-existing]" ;;
esac
