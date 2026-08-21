# Whisper

**Status:** ACTIVE / LOCAL TOOLING  
**Purpose:** local audio transcription plus automatic grounded service-note DOCX files.

This repository is a small local transcription toolkit. It is not a backup folder and should not contain audio recordings, generated outputs, model caches or secrets.

Audio preparation and transcription remain local. After transcription, the
recordings watcher sends only timestamped transcript text to the OpenAI
Responses API for a structured draft and an evidence-verification pass. Audio
is never uploaded by the note stage.

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
3. output folder (ENTER = `output_transkrypcja` next to the source recording/folder),
4. model options.

Routing is automatic:

```text
sv / Swedish      -> KB-Whisper
other languages   -> OpenAI Whisper
auto detection    -> OpenAI Whisper
```

Diarization / speaker separation is intentionally removed from the normal workflow.

Supported input extensions are case-insensitive:

```text
wav mp3 m4a aac flac ogg opus qta mp4 mov mkv webm avi
```

Inputs are probed by content rather than trusted by extension. A compatible
audio stream is selected and converted locally to a temporary mono 16 kHz PCM
WAV before transcription. For QTA, a conventional AAC/PCM/ALAC compatibility
stream is preferred over spatial APAC audio. Source QTA files are never changed,
and the intermediate WAV is deleted after transcription.

## Repository structure

```text
Whisper/
├── README.md
├── .gitignore
├── requirements-kb.txt
├── requirements-whisper.txt
├── config/
│   └── whisper-recordings-watcher.env.example
├── docs/
│   └── RECORDINGS_WATCHER.md
├── scripts/
│   ├── install_watcher.sh
│   ├── recordings_watcher.sh
│   └── start.sh
└── src/
    ├── audio_prepare.py
    ├── transcribe_kb.py
    └── transcribe_whisper.py
```

Local-only files and folders, not committed:

```text
output_transkrypcja/ # default output folder next to input recordings/folders
cache/               # local model cache, ignored by Git
.venv_kb/
.venv_whisper/
secrets/token.txt    # optional old/local file; not used by normal workflow
```

## Important v9 change: default output next to source recordings

By default, transcription results are no longer stored in the repository. Pressing ENTER at the output-folder prompt creates:

```text
/path/to/recording-folder/output_transkrypcja/
```

Examples:

```text
/home/jakub-pelka/MobileTransfer/Recordings/Violett guldvinge.m4a
-> /home/jakub-pelka/MobileTransfer/Recordings/output_transkrypcja/
```

```text
INPUT_DIR=/home/jakub-pelka/MobileTransfer/Recordings
-> /home/jakub-pelka/MobileTransfer/Recordings/output_transkrypcja/
```

This keeps generated work products with the source material and avoids filling the Git repository with outputs.

## Important v8 change: stable local model cache

Large model files are cached inside the repository working folder:

```text
cache/                   # KB-Whisper / Hugging Face models
cache/openai-whisper/    # OpenAI Whisper models
```

The `cache/` folder is ignored by Git. Do not commit it. Keep it locally if you want models to stay available between runs, also weeks later. If you delete `cache/`, models will be downloaded again.

For compatibility with v7, Hugging Face cache stays directly under `cache/`, so the model you already downloaded should be reused. The launcher exports stable cache paths automatically:

```text
HF_HOME
HF_HUB_CACHE
TRANSFORMERS_CACHE
WHISPER_CACHE_DIR
```

You can override the cache location if needed:

```bash
CACHE_ROOT="/mnt/lacie/WhisperCache" ./scripts/start.sh
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
cd /home/jakub-pelka/GitHub/ProductivityMe/Whisper
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

Default output folder:

```text
output_transkrypcja/
```

The folder is created next to the source recording or selected input folder. You can still override it with `OUT_DIR=/custom/path` or by entering a custom folder in the prompt.

Each processed file produces:

```text
.txt
.json
```

The `.txt` file contains the transcription text. The `.json` file keeps metadata and raw model output for later debugging.

With the default watcher profile, a completed transcription also produces:

```text
Notatki/<recording>_tjansteanteckning.docx
Notatki/<recording>_tjansteanteckning.note.json
```

The DOCX is the service note. The audit JSON records transcript hash, source
segment references, API response IDs, verification warnings, and model usage.

## Optional automatic recordings watcher

The optional local watcher monitors
`/home/jakub-pelka/MobileTransfer/Recordings` and transcribes only supported
files added after installation. Existing recordings are registered as a
baseline without transcription. Transcripts go into `output_transkrypcja/` and
grounded service notes go into `Notatki/`; source recordings are never modified.
New `.qta` files follow the same stable copy, local conversion, transcription,
API verification, and DOCX path as `.m4a` files.

Install and verify it with:

```bash
./scripts/install_watcher.sh
systemctl --user status whisper-recordings-watcher.path
systemctl --user status whisper-recordings-watcher.timer
```

Run one scan or view logs:

```bash
./scripts/recordings_watcher.sh
journalctl --user -u whisper-recordings-watcher.service -n 100 --no-pager
tail -f ~/.local/state/whisper-recordings-watcher/watcher.log
```

No-op scans and files that are still being copied are intentionally silent.
`watcher.log` contains only meaningful `BASELINE`, `PROCESSING`, `COMPLETED`,
`FAILED`, `SKIPPED_UNSUPPORTED`, or `STATE_REPAIR` events. The watcher does not
create per-run log files.

Stop the automation with:

```bash
systemctl --user disable --now whisper-recordings-watcher.path
systemctl --user disable --now whisper-recordings-watcher.timer
```

Change the language/model profile in
`~/.config/whisper-recordings-watcher.env`. See
[`docs/RECORDINGS_WATCHER.md`](docs/RECORDINGS_WATCHER.md) for state format,
retry behavior, supported configuration, and operating details.

### Meeting-note API configuration

The watcher defaults to the complete flow:

```text
GENERATE_MEETING_NOTE=true
NOTE_LANGUAGE=sv
NOTE_DRAFT_MODEL=gpt-5.6-luna
NOTE_VERIFICATION_MODEL=gpt-5.6-terra
NOTE_REASONING_EFFORT=medium
OPENAI_ENV_FILE=/path/to/private/openai.env
NOTE_DIR_NAME=Notatki
```

Keep `OPENAI_API_KEY` outside this repository. When a transcript already exists
but its matching note is absent or stale, the watcher reuses the transcript and
runs only the API/DOCX stage. The note tool can also be invoked manually:

```bash
./scripts/generate_meeting_note.sh \
  --transcript-json /path/output_transkrypcja/recording_model_standard.json \
  --outdir /path/Notatki \
  --language sv \
  --draft-model gpt-5.6-luna \
  --verification-model gpt-5.6-terra \
  --env-file /path/to/private/openai.env
```

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
requirements-notes.txt
```

The launcher creates local virtual environments as needed:

```text
.venv_kb/
.venv_whisper/
.venv_notes/
```

They are ignored by Git.

First run can download large Python/CUDA packages and the KB/OpenAI Whisper model files. That is expected. Later runs reuse the local virtual environment and the stable local model cache in `cache/`, as long as that folder is not deleted.

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
