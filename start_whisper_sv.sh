#!/usr/bin/env bash
set -Eeuo pipefail

# Start script for old OpenAI Whisper + pyannote pipeline.
# Reads HF token from secrets/token.txt if HF_TOKEN/HUGGINGFACE_TOKEN is not already set.

cd /home/jakub-pelka/GitHub/Whisper || exit 1

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
UPDATE_DEPS="${UPDATE_DEPS:-0}"
TOKEN_FILE="${TOKEN_FILE:-/home/jakub-pelka/GitHub/Whisper/secrets/token.txt}"

if [[ ! -f "transcribe_diarize.py" ]]; then
  echo "ERROR: transcribe_diarize.py not found."
  exit 1
fi

# --- Load Hugging Face token ---
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_TOKEN:-}" ]]; then
  if [[ -f "$TOKEN_FILE" ]]; then
    HF_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
    export HF_TOKEN
    export HUGGINGFACE_TOKEN="$HF_TOKEN"
    echo "HF token loaded from: $TOKEN_FILE"
  else
    echo "WARNING: token file not found and HF_TOKEN/HUGGINGFACE_TOKEN is not set."
    echo "Pyannote diarization may fail."
    echo "Expected token file:"
    echo "  $TOKEN_FILE"
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is not installed or not available on PATH."
  echo "Install with:"
  echo "  sudo apt update && sudo apt install -y ffmpeg python3-venv python3-tk git"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating Python virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install --upgrade "setuptools<82"

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
echo "  preset:   1 = Faster"
echo "  language: 3 = sv"
echo "  model:    7 = large-v3-turbo"
echo ""
echo "File/folder selection is handled by transcribe_diarize.py."
echo ""

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python transcribe_diarize.py
