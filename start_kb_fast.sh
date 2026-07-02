#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/jakub-pelka/GitHub/Whisper || exit 1

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv_kb_whisper}"

DEFAULT_INPUT="/home/jakub-pelka/MobileTransfer/Recordings/Violett guldvinge.m4a"
DEFAULT_OUT_DIR="/home/jakub-pelka/MobileTransfer/Recordings/kb_whisper_tests"

KB_WHISPER_MODEL="${KB_WHISPER_MODEL:-KBLab/kb-whisper-large}"
KB_WHISPER_REVISION="${KB_WHISPER_REVISION:-standard}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

if [[ -z "${INPUT_FILE:-}" ]]; then
  echo ""
  echo "Podaj ścieżkę pliku audio/wideo do transkrypcji."
  echo "ENTER = domyślny test:"
  echo "  $DEFAULT_INPUT"
  echo ""
  read -e -p "Plik wejściowy: " INPUT_FILE
  INPUT_FILE="${INPUT_FILE:-$DEFAULT_INPUT}"
fi

INPUT_FILE="${INPUT_FILE%\"}"
INPUT_FILE="${INPUT_FILE#\"}"
INPUT_FILE="${INPUT_FILE%\'}"
INPUT_FILE="${INPUT_FILE#\'}"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: input file not found:"
  echo "  $INPUT_FILE"
  exit 2
fi

if [[ -z "${OUT_DIR:-}" ]]; then
  echo ""
  echo "Podaj folder wynikowy."
  echo "ENTER = domyślnie:"
  echo "  $DEFAULT_OUT_DIR"
  echo ""
  read -e -p "Folder wynikowy: " OUT_DIR
  OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
fi

OUT_DIR="${OUT_DIR%\"}"
OUT_DIR="${OUT_DIR#\"}"
OUT_DIR="${OUT_DIR%\'}"
OUT_DIR="${OUT_DIR#\'}"

mkdir -p "$OUT_DIR"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg missing."
  echo "Install:"
  echo "  sudo apt update && sudo apt install -y ffmpeg python3-venv git"
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  INSTALL_DEPS=1
fi

source "$VENV_DIR/bin/activate"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  python -m pip install --upgrade pip wheel
  python -m pip install --upgrade "setuptools<82"
  python -m pip install --upgrade torch torchaudio transformers accelerate safetensors soundfile librosa sentencepiece
else
  echo "Skipping dependency install/update. Set INSTALL_DEPS=1 if needed."
fi

echo ""
echo "KB-Whisper fast transcription — no diarization"
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
