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
elif ! "$VENV_DIR/bin/python" -c 'import fastapi, openai, uvicorn' >/dev/null 2>&1; then
  echo "Installing missing Antek API dependencies."
  "$VENV_DIR/bin/python" -m pip install --upgrade -r "$ROOT_DIR/requirements-notes.txt"
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_DIR/bin/python" -m uvicorn antek_note_api:app \
  --host "${ANTEK_API_HOST:-127.0.0.1}" \
  --port "${ANTEK_API_PORT:-8080}"
