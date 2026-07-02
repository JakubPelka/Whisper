#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python3}"
TOKEN_FILE="${TOKEN_FILE:-$ROOT_DIR/secrets/token.txt}"
INSTALL_DEPS="${INSTALL_DEPS:-auto}"
AUDIO_FIND_MAXDEPTH="${AUDIO_FIND_MAXDEPTH:-1}"

SUPPORTED_FIND_EXPR=(
  -iname '*.wav' -o -iname '*.mp3' -o -iname '*.m4a' -o -iname '*.aac' -o
  -iname '*.wma' -o -iname '*.ogg' -o -iname '*.flac' -o -iname '*.mkv' -o
  -iname '*.mp4' -o -iname '*.m4b'
)

INPUT_PATHS=()
ENGINE="${ENGINE:-auto}"
LANGUAGE="${LANGUAGE:-}"
OUT_DIR="${OUT_DIR:-}"

clean_prompt_path() {
  local value="$1"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s' "$value"
}

load_hf_token() {
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    if [[ -f "$TOKEN_FILE" ]]; then
      HF_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
      export HF_TOKEN
      export HUGGINGFACE_TOKEN="$HF_TOKEN"
      export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
      echo "HF token loaded from: $TOKEN_FILE"
    else
      echo "HF token file not found. Continuing without token:"
      echo "  $TOKEN_FILE"
      echo "Note: public KB/OpenAI Whisper models may still work without it."
    fi
  fi
}

add_input_file() {
  local file_path
  file_path="$(clean_prompt_path "$1")"
  if [[ -z "$file_path" ]]; then
    return 0
  fi
  if [[ ! -f "$file_path" ]]; then
    echo "ERROR: input file not found:"
    echo "  $file_path"
    exit 2
  fi
  INPUT_PATHS+=("$file_path")
}

add_input_files_from_semicolon_list() {
  local raw="$1"
  local item
  IFS=';' read -ra items <<< "$raw"
  for item in "${items[@]}"; do
    item="$(clean_prompt_path "$item")"
    if [[ -n "$item" ]]; then
      add_input_file "$item"
    fi
  done
}

add_input_dir() {
  local dir_path
  dir_path="$(clean_prompt_path "$1")"
  if [[ ! -d "$dir_path" ]]; then
    echo "ERROR: input folder not found:"
    echo "  $dir_path"
    exit 2
  fi

  mapfile -d '' found_files < <(
    find "$dir_path" -maxdepth "$AUDIO_FIND_MAXDEPTH" -type f \
      \( "${SUPPORTED_FIND_EXPR[@]}" \) \
      -print0 | sort -z
  )

  if [[ "${#found_files[@]}" -eq 0 ]]; then
    echo "ERROR: no supported audio/video files found in:"
    echo "  $dir_path"
    exit 2
  fi

  local file_path
  for file_path in "${found_files[@]}"; do
    INPUT_PATHS+=("$file_path")
  done
}

ask_input_scope() {
  if [[ -n "${INPUT_FILE:-}" ]]; then
    add_input_file "$INPUT_FILE"
    return 0
  fi

  if [[ -n "${INPUT_FILES:-}" ]]; then
    add_input_files_from_semicolon_list "$INPUT_FILES"
    return 0
  fi

  if [[ -n "${INPUT_DIR:-}" ]]; then
    add_input_dir "$INPUT_DIR"
    return 0
  fi

  echo ""
  echo "Wybierz zakres analizy:"
  echo "  1) jeden plik"
  echo "  2) kilka plików, ścieżki oddzielone średnikiem ;"
  echo "  3) cały folder audio/wideo"
  echo ""
  echo "Tryb bez pytań też jest możliwy:"
  echo "  INPUT_FILE=/path/file.m4a ./scripts/start.sh"
  echo "  INPUT_FILES='/path/a.m4a;/path/b.m4a' ./scripts/start.sh"
  echo "  INPUT_DIR=/path/folder ./scripts/start.sh"
  echo ""
  read -p "Tryb [1-3] (ENTER=1): " INPUT_MODE
  INPUT_MODE="${INPUT_MODE:-1}"

  case "$INPUT_MODE" in
    1)
      read -e -p "Plik wejściowy: " INPUT_FILE
      if [[ -z "$INPUT_FILE" ]]; then
        echo "ERROR: nie podano pliku wejściowego."
        exit 2
      fi
      add_input_file "$INPUT_FILE"
      ;;
    2)
      echo "Podaj ścieżki oddzielone średnikiem ;"
      echo "Przykład: /dane/a.m4a;/dane/b.m4a"
      read -e -p "Pliki wejściowe: " INPUT_FILES
      if [[ -z "$INPUT_FILES" ]]; then
        echo "ERROR: nie podano plików wejściowych."
        exit 2
      fi
      add_input_files_from_semicolon_list "$INPUT_FILES"
      ;;
    3)
      read -e -p "Folder wejściowy: " INPUT_DIR
      if [[ -z "$INPUT_DIR" ]]; then
        echo "ERROR: nie podano folderu wejściowego."
        exit 2
      fi
      add_input_dir "$INPUT_DIR"
      ;;
    *)
      echo "ERROR: nieznany tryb: $INPUT_MODE"
      exit 2
      ;;
  esac

  if [[ "${#INPUT_PATHS[@]}" -eq 0 ]]; then
    echo "ERROR: brak plików do przetworzenia."
    exit 2
  fi
}

ask_language_and_engine() {
  if [[ -z "$LANGUAGE" ]]; then
    echo ""
    echo "Wybierz język nagrania:"
    echo "  1) sv  — szwedzki, użyj KB-Whisper"
    echo "  2) pl  — polski, użyj OpenAI Whisper"
    echo "  3) en  — angielski, użyj OpenAI Whisper"
    echo "  4) auto — automatyczne wykrywanie, użyj OpenAI Whisper"
    echo "  5) other — inny kod języka"
    read -p "Twój wybór [1-5] (ENTER=sv): " LANG_CHOICE
    case "${LANG_CHOICE:-1}" in
      1|"") LANGUAGE="sv" ;;
      2) LANGUAGE="pl" ;;
      3) LANGUAGE="en" ;;
      4) LANGUAGE="auto" ;;
      5)
        read -p "Podaj kod języka, np. de/fr/es/uk: " LANGUAGE
        LANGUAGE="${LANGUAGE:-auto}"
        ;;
      *) LANGUAGE="sv" ;;
    esac
  fi

  if [[ "$ENGINE" == "auto" ]]; then
    case "${LANGUAGE,,}" in
      sv|se|swe|swedish|szwedzki) ENGINE="kb" ;;
      *) ENGINE="whisper" ;;
    esac
  fi

  if [[ "$ENGINE" != "kb" && "$ENGINE" != "whisper" ]]; then
    echo "ERROR: ENGINE must be 'auto', 'kb' or 'whisper'."
    exit 2
  fi
}

ask_output_dir() {
  local default_out
  if [[ "$ENGINE" == "kb" ]]; then
    default_out="$ROOT_DIR/outputs/kb"
  else
    default_out="$ROOT_DIR/outputs/whisper"
  fi

  if [[ -z "$OUT_DIR" ]]; then
    echo ""
    echo "Podaj folder wynikowy."
    echo "ENTER = lokalny folder wynikowy w repo, ignorowany przez Git:"
    echo "  $default_out"
    echo ""
    read -e -p "Folder wynikowy: " OUT_DIR
    OUT_DIR="${OUT_DIR:-$default_out}"
  fi

  OUT_DIR="$(clean_prompt_path "$OUT_DIR")"
  mkdir -p "$OUT_DIR"
}

ask_kb_params() {
  if [[ -z "${KB_SIZE:-}" && -z "${KB_WHISPER_MODEL:-}" ]]; then
    echo ""
    echo "Wybierz model KB-Whisper dla szwedzkiego:"
    echo "  1) large  — lepsza jakość, domyślnie"
    echo "  2) medium — szybszy/lżejszy fallback"
    read -p "Twój wybór [1-2] (ENTER=large): " KB_CHOICE
    case "${KB_CHOICE:-1}" in
      1|"") KB_SIZE="large" ;;
      2) KB_SIZE="medium" ;;
      *) KB_SIZE="large" ;;
    esac
  fi

  if [[ -z "${KB_WHISPER_MODEL:-}" ]]; then
    case "${KB_SIZE:-large}" in
      medium) KB_WHISPER_MODEL="KBLab/kb-whisper-medium" ;;
      large|*) KB_WHISPER_MODEL="KBLab/kb-whisper-large" ;;
    esac
  fi

  if [[ -z "${KB_WHISPER_REVISION:-}" ]]; then
    echo ""
    echo "Wybierz styl transkrypcji KB-Whisper:"
    echo "  1) standard — domyślny"
    echo "  2) strict   — bardziej dosłowny"
    echo "  3) subtitle — bardziej skrócony"
    read -p "Twój wybór [1-3] (ENTER=standard): " REV_CHOICE
    case "${REV_CHOICE:-1}" in
      1|"") KB_WHISPER_REVISION="standard" ;;
      2) KB_WHISPER_REVISION="strict" ;;
      3) KB_WHISPER_REVISION="subtitle" ;;
      *) KB_WHISPER_REVISION="standard" ;;
    esac
  fi
}

ask_whisper_params() {
  if [[ -z "${WHISPER_MODEL:-}" ]]; then
    echo ""
    echo "Wybierz model OpenAI Whisper:"
    echo "  1) tiny"
    echo "  2) base"
    echo "  3) small"
    echo "  4) medium"
    echo "  5) large-v3-turbo — domyślnie"
    echo "  6) large-v3"
    read -p "Twój wybór [1-6] (ENTER=large-v3-turbo): " MODEL_CHOICE
    case "${MODEL_CHOICE:-5}" in
      1) WHISPER_MODEL="tiny" ;;
      2) WHISPER_MODEL="base" ;;
      3) WHISPER_MODEL="small" ;;
      4) WHISPER_MODEL="medium" ;;
      5|"") WHISPER_MODEL="large-v3-turbo" ;;
      6) WHISPER_MODEL="large-v3" ;;
      *) WHISPER_MODEL="large-v3-turbo" ;;
    esac
  fi

  if [[ -z "${WHISPER_PRESET:-}" ]]; then
    echo ""
    echo "Wybierz preset OpenAI Whisper:"
    echo "  1) fast      — szybki, polecany na RTX 2070"
    echo "  2) default   — wolniejszy"
    echo "  3) accurate  — najwolniejszy"
    read -p "Twój wybór [1-3] (ENTER=fast): " PRESET_CHOICE
    case "${PRESET_CHOICE:-1}" in
      1|"") WHISPER_PRESET="fast" ;;
      2) WHISPER_PRESET="default" ;;
      3) WHISPER_PRESET="accurate" ;;
      *) WHISPER_PRESET="fast" ;;
    esac
  fi
}

ensure_base_tools() {
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg missing."
    echo "Install:"
    echo "  sudo apt update && sudo apt install -y ffmpeg python3-venv git"
    exit 1
  fi
}

ensure_kb_deps() {
  local venv_dir="${VENV_DIR:-$ROOT_DIR/.venv_kb}"
  if [[ ! -d "$venv_dir" ]]; then
    echo "Creating venv: $venv_dir"
    "$PYTHON_BIN" -m venv "$venv_dir"
    INSTALL_DEPS=1
  fi

  source "$venv_dir/bin/activate"

  if [[ "$INSTALL_DEPS" == "1" || "$INSTALL_DEPS" == "true" ]]; then
    python -m pip install --upgrade pip wheel
    python -m pip install --upgrade "setuptools<82"
    python -m pip install --upgrade -r requirements-kb.txt
  else
    echo "Skipping dependency install/update. Set INSTALL_DEPS=1 if needed."
  fi
}

ensure_whisper_deps() {
  local venv_dir="${VENV_DIR:-$ROOT_DIR/.venv_whisper}"
  if [[ ! -d "$venv_dir" ]]; then
    echo "Creating venv: $venv_dir"
    "$PYTHON_BIN" -m venv "$venv_dir"
    INSTALL_DEPS=1
  fi

  source "$venv_dir/bin/activate"

  if [[ "$INSTALL_DEPS" == "1" || "$INSTALL_DEPS" == "true" ]]; then
    python -m pip install --upgrade pip wheel
    python -m pip install --upgrade "setuptools<82"
    python -m pip install --upgrade -r requirements-whisper.txt
  else
    echo "Skipping dependency install/update. Set INSTALL_DEPS=1 if needed."
  fi
}

run_kb() {
  ensure_kb_deps

  echo ""
  echo "KB-Whisper Swedish transcription — no diarization"
  echo "Files:    ${#INPUT_PATHS[@]}"
  echo "Out dir:  $OUT_DIR"
  echo "Model:    $KB_WHISPER_MODEL"
  echo "Revision: $KB_WHISPER_REVISION"
  echo ""
  printf 'Input files:\n'
  printf '  %s\n' "${INPUT_PATHS[@]}"
  echo ""

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python src/transcribe_kb.py \
    --input "${INPUT_PATHS[@]}" \
    --outdir "$OUT_DIR" \
    --model "$KB_WHISPER_MODEL" \
    --revision "$KB_WHISPER_REVISION" \
    --chunk-length "${CHUNK_LENGTH:-30}" \
    --batch-size "${BATCH_SIZE:-1}"
}

run_whisper() {
  ensure_whisper_deps

  echo ""
  echo "OpenAI Whisper transcription — clean, no diarization"
  echo "Files:    ${#INPUT_PATHS[@]}"
  echo "Out dir:  $OUT_DIR"
  echo "Language: $LANGUAGE"
  echo "Model:    $WHISPER_MODEL"
  echo "Preset:   $WHISPER_PRESET"
  echo ""
  printf 'Input files:\n'
  printf '  %s\n' "${INPUT_PATHS[@]}"
  echo ""

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python src/transcribe_whisper.py \
    --input "${INPUT_PATHS[@]}" \
    --outdir "$OUT_DIR" \
    --language "$LANGUAGE" \
    --model "$WHISPER_MODEL" \
    --preset "$WHISPER_PRESET"
}

main() {
  echo ""
  echo "Whisper local transcription"
  echo "- sv  → KB-Whisper"
  echo "- inne języki / auto → OpenAI Whisper"
  echo "- diarization/pyannote: wyłączone"

  load_hf_token
  ensure_base_tools
  ask_input_scope
  ask_language_and_engine
  ask_output_dir

  if [[ "$ENGINE" == "kb" ]]; then
    ask_kb_params
    run_kb
  else
    ask_whisper_params
    run_whisper
  fi
}

main "$@"
