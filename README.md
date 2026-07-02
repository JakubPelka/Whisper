# Whisper

**Status:** ACTIVE / LOCAL TOOLING  
**Purpose:** local audio transcription with one launcher and two clean backends.

This repository is a small local transcription toolkit. It is not a backup folder and should not contain audio recordings, generated outputs, model caches or secrets.

## Recommended use

Use one main launcher:

```bash
./scripts/start.sh
```

The launcher asks for:

1. what to analyse:
   - one file,
   - several files,
   - a whole folder,
2. recording language,
3. output folder,
4. model options.

Routing is automatic:

```text
sv / Swedish      -> KB-Whisper
other languages   -> OpenAI Whisper
auto detection    -> OpenAI Whisper
```

Diarization / speaker separation is intentionally removed from the normal workflow.

## Repository structure

```text
Whisper/
├── README.md
├── .gitignore
├── requirements-kb.txt
├── requirements-whisper.txt
├── scripts/
│   └── start.sh
└── src/
    ├── transcribe_kb.py
    └── transcribe_whisper.py
```

Local-only files and folders, not committed:

```text
outputs/
cache/
.venv_kb/
.venv_whisper/
secrets/token.txt   # optional old/local file; not used by normal workflow
```

## Important v6 change: no token in normal workflow

The normal workflow does **not** load or use Hugging Face tokens.

```text
KB-Whisper     -> no token
OpenAI Whisper -> no token
pyannote       -> removed
```

This avoids failures caused by stale, malformed or accidentally exported local tokens.

## Backend choice

### Swedish

For Swedish recordings, use KB-Whisper:

```text
KBLab/kb-whisper-large
KBLab/kb-whisper-medium
```

Default recommendation:

```text
KBLab/kb-whisper-large
revision=standard
```

Use `medium` only if large is too slow or too heavy.

### Other languages

For Polish, English, auto-detection or other languages, use OpenAI Whisper:

```text
tiny
base
small
medium
large-v3-turbo
large-v3
```

Default recommendation:

```text
large-v3-turbo
preset=fast
```

## Interactive examples

Start the menu:

```bash
cd /home/jakub-pelka/GitHub/Whisper
./scripts/start.sh
```

Then choose:

```text
1) one file
2) several files
3) whole folder
```

For several files you can use either:

```text
/path/a.m4a;/path/b.m4a
```

or paste paths copied one-per-line with a universal Copy Path tool.

## Non-interactive examples

One Swedish file:

```bash
INPUT_FILE="/path/to/audio.m4a" LANGUAGE=sv ./scripts/start.sh
```

Several Swedish files, semicolon format:

```bash
INPUT_FILES="/path/a.m4a;/path/b.m4a" LANGUAGE=sv ./scripts/start.sh
```

Several files from clipboard, one path per line:

```bash
INPUT_FILES="$(wl-paste)" LANGUAGE=sv ./scripts/start.sh
```

Whole folder, Swedish, with medium KB model:

```bash
INPUT_DIR="/path/to/folder" LANGUAGE=sv KB_SIZE=medium ./scripts/start.sh
```

Polish file with OpenAI Whisper:

```bash
INPUT_FILE="/path/to/audio.m4a" LANGUAGE=pl WHISPER_MODEL=large-v3-turbo WHISPER_PRESET=fast ./scripts/start.sh
```

Auto language detection with OpenAI Whisper:

```bash
INPUT_FILE="/path/to/audio.m4a" LANGUAGE=auto ./scripts/start.sh
```

Search subfolders when using folder mode:

```bash
AUDIO_FIND_MAXDEPTH=2 INPUT_DIR="/path/to/folder" LANGUAGE=sv ./scripts/start.sh
```

## Outputs

Default output folders:

```text
outputs/kb/
outputs/whisper/
```

Each processed file produces:

```text
.txt
.json
```

The `.txt` file contains the transcription text. The `.json` file keeps metadata and raw model output for later debugging.

## Requirements

System packages:

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv git
```

Python dependencies are split:

```text
requirements-kb.txt
requirements-whisper.txt
```

The launcher creates local virtual environments as needed:

```text
.venv_kb/
.venv_whisper/
```

They are ignored by Git.

First run can download large Python/CUDA packages and the KB model file. That is expected. Later runs reuse the local virtual environment and model cache.

## GPU notes

The practical default for the local RTX 2070 8 GB setup is:

```text
Swedish:         KB-Whisper large
Other languages: OpenAI Whisper large-v3-turbo, fast preset
```

If GPU memory is tight:

```bash
LANGUAGE=sv KB_SIZE=medium ./scripts/start.sh
```

or for OpenAI Whisper:

```bash
LANGUAGE=pl WHISPER_MODEL=medium ./scripts/start.sh
```

## Removed from normal workflow

The following are intentionally not part of the clean workflow:

```text
pyannote diarization
speaker separation
per-speaker segment slicing
moving source audio to processed_files/
HF token loading
```

Reason: these paths were slow, GPU-heavy, fragile or unnecessary compared with plain transcription.

## Recommended cleanup after applying this structure

Remove old root-level scripts and old diarization files:

```bash
rm -f transcribe_diarize.py
rm -f transcribe_kb_whisper.py
rm -f transcribe_kb_diarize.py
rm -f start_kb_whisper_test.sh
rm -f start_kb_fast.sh
rm -f start_kb_diarize.sh
rm -f start_whisper_sv.sh
```

If `secrets/README.md` was tracked earlier, remove it from Git, but do not delete your local token file unless you really want to:

```bash
git rm -f secrets/README.md 2>/dev/null || true
```

Then commit only clean source/config files:

```bash
git add README.md .gitignore requirements-kb.txt requirements-whisper.txt scripts src docs/REPLACE_INSTRUCTIONS.md
git add -u
git commit -m "Remove token loading from transcription workflow"
```

Do not run `git add .` unless you checked `git status --short` carefully.
