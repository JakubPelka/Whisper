#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/jakub-pelka/GitHub/Whisper || exit 1

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv_kb_whisper}"

INPUT_FILE="${INPUT_FILE:-/home/jakub-pelka/MobileTransfer/Recordings/Violett guldvinge.m4a}"
OUT_DIR="${OUT_DIR:-/home/jakub-pelka/MobileTransfer/Recordings/kb_whisper_tests}"

KB_WHISPER_MODEL="${KB_WHISPER_MODEL:-KBLab/kb-whisper-large}"
KB_WHISPER_REVISION="${KB_WHISPER_REVISION:-standard}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg missing."
  echo "Install:"
  echo "  sudo apt update && sudo apt install -y ffmpeg python3-venv git"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade torch torchaudio transformers accelerate safetensors soundfile librosa sentencepiece

echo ""
echo "KB-Whisper test"
echo "Input:    $INPUT_FILE"
echo "Out dir:  $OUT_DIR"
echo "Model:    $KB_WHISPER_MODEL"
echo "Revision: $KB_WHISPER_REVISION"
echo ""

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python transcribe_kb_whisper.py \
  --input "$INPUT_FILE" \
  --outdir "$OUT_DIR" \
  --model "$KB_WHISPER_MODEL" \
  --revision "$KB_WHISPER_REVISION" \
  --chunk-length 30 \
  --batch-size 1
