#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${NOTES_VENV_DIR:-$ROOT_DIR/.venv_notes}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating meeting-note environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip wheel
  "$VENV_DIR/bin/python" -m pip install --upgrade -r "$ROOT_DIR/requirements-notes.txt"
elif ! "$VENV_DIR/bin/python" -c 'import openai, pydantic, docx' >/dev/null 2>&1; then
  echo "Installing missing meeting-note dependencies."
  "$VENV_DIR/bin/python" -m pip install --upgrade -r "$ROOT_DIR/requirements-notes.txt"
fi

exec "$VENV_DIR/bin/python" "$ROOT_DIR/src/generate_meeting_note.py" "$@"
