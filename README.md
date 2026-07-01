# Whisper

**Status:** EXPERIMENT / ACTIVE  
**Primary purpose:** local speech-to-text experiments, mainly for Swedish audio transcription.

This repository contains local transcription tools for audio recordings. The current preferred path for Swedish is **KB-Whisper** from KBLab / Kungliga biblioteket.

The older OpenAI Whisper + pyannote script is kept for comparison and for diarization experiments.

---

## Current direction

For Swedish transcription we use:

```text
KBLab/kb-whisper-large
```

This is the default model for the KB-Whisper launcher.

Fallback / lower VRAM option:

```text
KBLab/kb-whisper-medium
```

The older OpenAI Whisper + pyannote pipeline remains useful for diarization tests, but it should not be the default Swedish ASR path.

---

## Why KB-Whisper for Swedish?

KB-Whisper is a family of Swedish Whisper models from KBLab / Kungliga biblioteket.

Links:

- KB-Whisper large: <https://huggingface.co/KBLab/kb-whisper-large>
- KB-Whisper medium: <https://huggingface.co/KBLab/kb-whisper-medium>
- KBLab on Hugging Face: <https://huggingface.co/KBLab>

In this repo:

- `KBLab/kb-whisper-large` is the default model for Swedish.
- `KBLab/kb-whisper-medium` is the practical fallback.
- `revision=standard` is the default transcription style.
- `revision=strict` can be tested for more verbatim-like output.
- `revision=subtitle` can be tested for more condensed output.

---

## Repository contents

Main files:

```text
transcribe_kb_whisper.py       # KB-Whisper Swedish ASR test script
start_kb_whisper_test.sh       # launcher for KB-Whisper tests
transcribe_diarize.py          # older Whisper + pyannote diarization pipeline
start_whisper_sv.sh            # launcher for older OpenAI Whisper path
README.md
.gitignore
```

Recommended current entry point:

```bash
./start_kb_whisper_test.sh
```

---

## Default local test input

The launcher currently expects the local test audio here:

```text
/home/jakub-pelka/MobileTransfer/Recordings/Violett guldvinge.m4a
```

Default output folder:

```text
/home/jakub-pelka/MobileTransfer/Recordings/kb_whisper_tests/
```

These paths are local only. Audio files, temporary files, cache folders and transcription outputs should not be committed to GitHub.

---

## Supported audio formats

The older diarization script supports common audio/video formats such as:

```text
.wav .mp3 .m4a .aac .wma .ogg .flac .mkv .mp4 .m4b
```

The KB-Whisper test script uses `ffmpeg` to convert input audio to mono 16 kHz WAV before transcription. This means `.m4a` recordings from a phone should work as long as `ffmpeg` is installed.

---

## Requirements

System packages:

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv git
```

The KB-Whisper launcher installs Python dependencies into a local virtual environment:

```text
.venv_kb_whisper/
```

The launcher installs or updates:

```text
torch
torchaudio
transformers
accelerate
safetensors
soundfile
librosa
sentencepiece
```

---

## Run KB-Whisper large

```bash
cd /home/jakub-pelka/GitHub/Whisper
./start_kb_whisper_test.sh
```

This uses:

```text
model:    KBLab/kb-whisper-large
revision: standard
language: sv
```

---

## Run KB-Whisper medium fallback

```bash
cd /home/jakub-pelka/GitHub/Whisper
KB_WHISPER_MODEL=KBLab/kb-whisper-medium ./start_kb_whisper_test.sh
```

---

## Run strict transcription style

```bash
cd /home/jakub-pelka/GitHub/Whisper
KB_WHISPER_MODEL=KBLab/kb-whisper-large KB_WHISPER_REVISION=strict ./start_kb_whisper_test.sh
```

---

## Run subtitle-style transcription

```bash
cd /home/jakub-pelka/GitHub/Whisper
KB_WHISPER_MODEL=KBLab/kb-whisper-large KB_WHISPER_REVISION=subtitle ./start_kb_whisper_test.sh
```

---

## GPU notes

Tested locally on RTX 2070 8 GB:

```text
KBLab/kb-whisper-medium  OK
KBLab/kb-whisper-large   OK, acceptable runtime
```

If CUDA memory errors appear, try medium:

```bash
KB_WHISPER_MODEL=KBLab/kb-whisper-medium ./start_kb_whisper_test.sh
```

or force CPU:

```bash
python transcribe_kb_whisper.py --cpu --model KBLab/kb-whisper-large
```

---

## Diarization status

Diarization means separating speakers, for example:

```text
SPEAKER_00: ...
SPEAKER_01: ...
```

Current status:

```text
OpenAI Whisper + pyannote pipeline: exists
KB-Whisper Swedish ASR: works
KB-Whisper + diarization: planned
```

The older `transcribe_diarize.py` script already follows this general flow:

```text
audio
  -> WAV 16 kHz mono
  -> pyannote diarization
  -> segment slicing
  -> OpenAI Whisper transcription
  -> TXT/JSON output
```

The next logical step is a new script that keeps the diarization part but replaces the transcription backend with KB-Whisper:

```text
audio
  -> pyannote speaker diarization
  -> split audio by speaker/time segment
  -> transcribe each segment with KBLab/kb-whisper-large
  -> export speaker-labelled TXT and JSON
```

Recommended future file name:

```text
transcribe_kb_diarize.py
```

---

## Diarization options

### Option A — pyannote + KB-Whisper

Best practical next step.

Pros:

- reuses the current diarization idea,
- should fit the current local repository structure,
- gives speaker-labelled Swedish transcription,
- lets KB-Whisper remain the Swedish ASR backend.

Cons:

- requires a Hugging Face token,
- requires accepting gated pyannote model terms,
- adds GPU/VRAM pressure,
- adds another failure point compared with plain transcription.

Relevant model:

- <https://huggingface.co/pyannote/speaker-diarization-3.1>

### Option B — WhisperX + KB-Whisper

Possible later experiment.

Pros:

- can provide better timestamps and alignment,
- can be useful for subtitle-style workflows,
- KBLab documents WhisperX-style usage for Swedish workflows.

Cons:

- more dependencies,
- likely still needs pyannote for diarization,
- larger change than Option A,
- not ideal as the first stable implementation.

### Option C — no diarization by default

Safe default for quick transcription.

Pros:

- no pyannote token,
- fewer moving parts,
- easier to run from SSH,
- best default for routine Swedish transcription.

Cons:

- no speaker labels.

---

## Recommended default workflow

For normal Swedish audio:

```text
.m4a / .mp3 / .wav
  -> start_kb_whisper_test.sh
  -> KBLab/kb-whisper-large
  -> TXT + JSON output
```

For speaker-labelled transcription, use the future diarization path:

```text
.m4a / .mp3 / .wav
  -> pyannote diarization
  -> KB-Whisper large per segment
  -> speaker-labelled TXT + JSON
```

---

## Security and repository hygiene

Do not commit:

```text
HF tokens
.env files
audio recordings
transcription outputs
cache directories
model files
temporary WAV files
processed_files/
kb_whisper_tests/
transkrypcje/
```

Hugging Face tokens should be passed through environment variables, not written into Python files:

```bash
export HF_TOKEN="hf_..."
export HUGGINGFACE_TOKEN="$HF_TOKEN"
```

If a token was ever committed to a public repo, rotate it.

---

## Recommended `.gitignore` coverage

The repo should ignore at least:

```gitignore
# Local environments
.venv/
.venv_kb_whisper/
__pycache__/
*.pyc

# Local model/cache files
cache/
.cache/
models/

# Audio/input/output data
*.m4a
*.mp3
*.wav
*.flac
*.ogg
*.aac
*.wma
*.m4b
processed_files/
kb_whisper_tests/
transcriptions/
transkrypcje/

# Backups
*.bak*
```

---

## Recommended git workflow

Check what changed:

```bash
git status
git diff
```

Add only source/config files:

```bash
git add README.md transcribe_kb_whisper.py start_kb_whisper_test.sh .gitignore
```

Avoid:

```bash
git add .
```

because that can accidentally add audio, cache files or generated outputs.

Commit:

```bash
git commit -m "Document KB-Whisper Swedish ASR workflow"
```

---

## Roadmap

Short term:

- keep `KBLab/kb-whisper-large` as default Swedish ASR model,
- compare `standard`, `strict`, and `subtitle` revisions,
- clean hardcoded Hugging Face token from the old diarization script,
- keep audio, model cache and outputs out of Git.

Next:

- add `transcribe_kb_diarize.py`,
- reuse pyannote diarization,
- use KB-Whisper large for segment transcription,
- export speaker-labelled TXT and JSON.

Later:

- optional WhisperX experiment,
- optional SRT/VTT export,
- optional batch folder mode,
- optional post-processing with a Swedish LLM for summaries and cleaned meeting notes.
