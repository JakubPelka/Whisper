# Whisper

**Status:** ACTIVE / LOCAL TOOLING  
**Purpose:** local audio transcription with a clean split between Swedish KB-Whisper and general OpenAI Whisper.

This repository is a small local transcription toolkit. It is not a backup folder and should not contain audio recordings, generated outputs, model caches or secrets.

## Recommended use

Use one main launcher:

```bash
./scripts/start.sh
```

The launcher asks for:

1. what to analyse:
   - one file,
   - several files separated with `;`,
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

Diarization / speaker separation is intentionally removed from the normal workflow. It was too slow and too GPU-heavy for this repo's current practical use.

## Repository structure

```text
Whisper/
├── README.md
├── .gitignore
├── requirements-kb.txt
├── requirements-whisper.txt
├── scripts/
│   └── start.sh
├── src/
│   ├── transcribe_kb.py
│   └── transcribe_whisper.py
└── secrets/
    └── README.md
```

Local-only file, not committed:

```text
secrets/token.txt
```

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
2) several files separated with ;
3) whole folder
```

If you choose Swedish, the launcher uses KB-Whisper. If you choose another language, it uses OpenAI Whisper.

## Non-interactive examples

One Swedish file:

```bash
INPUT_FILE="/path/to/audio.m4a" LANGUAGE=sv ./scripts/start.sh
```

Several Swedish files:

```bash
INPUT_FILES="/path/a.m4a;/path/b.m4a" LANGUAGE=sv ./scripts/start.sh
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

## Secrets and token file

The repo expects a local token file here if a Hugging Face token is needed:

```text
secrets/token.txt
```

The file is ignored by Git and must not be committed.

Set permissions locally:

```bash
chmod 600 secrets/token.txt
```

Check that Git ignores it:

```bash
git check-ignore -v secrets/token.txt
```

Current clean workflows do not use pyannote. KB-Whisper may still benefit from a Hugging Face token in some situations, so the launcher loads the token into the environment if the file exists.

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

## GPU notes

The practical default for the local RTX 2070 8 GB setup is:

```text
Swedish:        KB-Whisper large
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
```

Reason: these paths were slow, GPU-heavy and fragile compared with plain transcription.

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

Then commit only clean source/config files:

```bash
git add README.md .gitignore requirements-kb.txt requirements-whisper.txt scripts src secrets/README.md docs/REPLACE_INSTRUCTIONS.md
git commit -m "Simplify transcription launcher"
```

Do not run `git add .` unless you checked `git status --short` carefully.
