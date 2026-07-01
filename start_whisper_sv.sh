#!/usr/bin/env bash
set -Eeuo pipefail

# Start script for JakubPelka/Whisper
# Swedish-friendly launcher for transcribe_diarize.py
#
# In script menu choose:
#   preset:   3 = Default
#   language: 3 = sv
#   model:    6 = large-v3
#             7 = large-v3-turbo

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
UPDATE_DEPS="${UPDATE_DEPS:-1}"

if [[ ! -f "transcribe_diarize.py" ]]; then
  echo "ERROR: transcribe_diarize.py not found in: $REPO_DIR" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is not installed or not available on PATH." >&2
  echo "Install with:"
  echo "  sudo apt update && sudo apt install -y ffmpeg python3-venv python3-tk git"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating Python virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "$UPDATE_DEPS" == "1" ]]; then
  echo "Installing/updating Whisper stack..."
  python -m pip install -U torch torchaudio pydub pyannote.audio openai-whisper
  python -m pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
else
  echo "Skipping package update because UPDATE_DEPS=0"
fi

echo ""
echo "Checking Whisper model aliases..."
python - <<'PY'
import whisper
models = set(whisper.available_models())
for name in ["large-v3", "large-v3-turbo", "turbo"]:
    print(f"{name}: {'OK' if name in models else 'MISSING'}")
PY

echo ""
echo "Starting transcribe_diarize.py"
echo ""
echo "Recommended choices:"
echo "  preset:   3 = Default"
echo "  language: 3 = sv"
echo "  model:    6 = large-v3"
echo "            7 = large-v3-turbo"
echo ""

python transcribe_diarize.py
