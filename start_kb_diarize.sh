#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/jakub-pelka/GitHub/Whisper || exit 1

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv_kb_whisper}"

INPUT_FILE="${INPUT_FILE:-/home/jakub-pelka/MobileTransfer/Recordings/Violett guldvinge.m4a}"
OUT_DIR="${OUT_DIR:-/home/jakub-pelka/MobileTransfer/Recordings/kb_whisper_tests}"

KB_WHISPER_MODEL="${KB_WHISPER_MODEL:-KBLab/kb-whisper-large}"
KB_WHISPER_REVISION="${KB_WHISPER_REVISION:-standard}"

PYANNOTE_MODEL="${PYANNOTE_MODEL:-pyannote/speaker-diarization-3.1}"

# Default: pyannote on CPU, KB-Whisper on GPU.
# This saves VRAM on RTX 2070 8 GB.
DIAR_DEVICE="${DIAR_DEVICE:-cpu}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN/HUGGINGFACE_TOKEN is not set."
  echo ""
  echo "Run first:"
  echo "  read -s -p \"HF token: \" HF_TOKEN; echo; export HF_TOKEN; export HUGGINGFACE_TOKEN=\"\$HF_TOKEN\""
  exit 1
fi

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

python -m pip install --upgrade pip wheel
python -m pip install --upgrade "setuptools<82"
python -m pip install --upgrade torch torchaudio transformers accelerate safetensors soundfile librosa sentencepiece pyannote.audio

echo ""
echo "KB-Whisper + pyannote diarization"
echo "Input:        $INPUT_FILE"
echo "Out dir:      $OUT_DIR"
echo "ASR model:    $KB_WHISPER_MODEL"
echo "ASR revision: $KB_WHISPER_REVISION"
echo "Diar model:   $PYANNOTE_MODEL"
echo "Diar device:  $DIAR_DEVICE"
echo ""

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python transcribe_kb_diarize.py \
  --input "$INPUT_FILE" \
  --outdir "$OUT_DIR" \
  --model "$KB_WHISPER_MODEL" \
  --revision "$KB_WHISPER_REVISION" \
  --diar-model "$PYANNOTE_MODEL" \
  --diar-device "$DIAR_DEVICE"
