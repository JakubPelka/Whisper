#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVED_ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$RESOLVED_ROOT_DIR"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG_FILE="$CONFIG_DIR/whisper-recordings-watcher.env"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/whisper-recordings-watcher"
STATE_FILE="$STATE_DIR/processed.tsv"
SYSTEMD_USER_DIR="$CONFIG_DIR/systemd/user"

[[ -x "$ROOT_DIR/scripts/start.sh" ]] || { echo "ERROR: missing executable $ROOT_DIR/scripts/start.sh" >&2; exit 1; }
[[ -x "$ROOT_DIR/scripts/recordings_watcher.sh" ]] || { echo "ERROR: missing executable $ROOT_DIR/scripts/recordings_watcher.sh" >&2; exit 1; }
[[ -f "$ROOT_DIR/config/whisper-recordings-watcher.env.example" ]] || { echo "ERROR: missing watcher configuration template" >&2; exit 1; }
command -v systemctl >/dev/null 2>&1 || { echo "ERROR: systemctl is required" >&2; exit 1; }

mkdir -p "$CONFIG_DIR" "$STATE_DIR" "$SYSTEMD_USER_DIR"
if [[ ! -e "$CONFIG_FILE" ]]; then
  install -m 600 "$ROOT_DIR/config/whisper-recordings-watcher.env.example" "$CONFIG_FILE"
  echo "Created configuration: $CONFIG_FILE"
else
  echo "Preserving existing configuration: $CONFIG_FILE"
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"
ROOT_DIR="$RESOLVED_ROOT_DIR"
WATCH_DIR="${WATCH_DIR:-/home/jakub-pelka/MobileTransfer/Recordings}"
SCAN_INTERVAL_SECONDS="${SCAN_INTERVAL_SECONDS:-120}"
[[ "$SCAN_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: SCAN_INTERVAL_SECONDS must be a positive integer" >&2; exit 1; }
[[ -d "$WATCH_DIR" ]] || { echo "ERROR: recordings directory does not exist: $WATCH_DIR" >&2; exit 1; }

if [[ ! -e "$STATE_FILE" ]]; then
  "$ROOT_DIR/scripts/recordings_watcher.sh" --initialize-existing
else
  echo "Preserving existing processing state: $STATE_FILE"
fi

cat > "$SYSTEMD_USER_DIR/whisper-recordings-watcher.service" <<UNIT
[Unit]
Description=Transcribe new recordings with local Whisper
After=default.target

[Service]
Type=oneshot
WorkingDirectory=$ROOT_DIR
ExecStart=$ROOT_DIR/scripts/recordings_watcher.sh
TimeoutStartSec=infinity
UNIT

cat > "$SYSTEMD_USER_DIR/whisper-recordings-watcher.path" <<UNIT
[Unit]
Description=Watch recordings folder for new files

[Path]
PathChanged=$WATCH_DIR
Unit=whisper-recordings-watcher.service

[Install]
WantedBy=default.target
UNIT

cat > "$SYSTEMD_USER_DIR/whisper-recordings-watcher.timer" <<UNIT
[Unit]
Description=Periodic recordings transcription scan

[Timer]
OnBootSec=${SCAN_INTERVAL_SECONDS}s
OnUnitActiveSec=${SCAN_INTERVAL_SECONDS}s
Unit=whisper-recordings-watcher.service
Persistent=true

[Install]
WantedBy=timers.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now whisper-recordings-watcher.path whisper-recordings-watcher.timer

echo
echo "Watcher installed from: $ROOT_DIR"
echo "Watched directory:       $WATCH_DIR"
echo "Configuration:           $CONFIG_FILE"
echo "State and log directory: $STATE_DIR"
echo
echo "Verify with:"
echo "  systemctl --user status whisper-recordings-watcher.path --no-pager"
echo "  systemctl --user status whisper-recordings-watcher.timer --no-pager"
echo "  journalctl --user -u whisper-recordings-watcher.service -n 100 --no-pager"
echo "  tail -f $STATE_DIR/watcher.log"
