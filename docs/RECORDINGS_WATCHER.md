# Automatic recordings watcher

The optional watcher transcribes new supported audio and video files added to
`/home/jakub-pelka/MobileTransfer/Recordings`. It runs entirely locally and uses
the repository's existing `scripts/start.sh` workflow. Transcript TXT/JSON files
are written to `output_transkrypcja/`; the note DOCX and audit JSON are written
to `Notatki/`. Per-run diagnostic logs are available in the adjacent `Logi/`
folder; source files are never changed or removed.

After a successful local transcription, the default configuration creates a
Swedish professional service-note DOCX through the OpenAI Responses API. Only
timestamped transcript text is sent to the API; audio remains local. The draft
uses structured output with mandatory source segment IDs and is checked in a
second API pass before the DOCX is rendered locally.

## Install

```bash
./scripts/install_watcher.sh
```

On first installation, every supported recording already present is registered
with status `completed-existing`. Existing recordings are not transcribed. The
installer preserves an existing configuration and processing state.

The installer creates and enables these user units:

- `whisper-recordings-watcher.path` for quick reaction to folder changes;
- `whisper-recordings-watcher.timer` as a periodic fallback;
- `whisper-recordings-watcher.service` for one locked scan at a time.

Check them with:

```bash
systemctl --user status whisper-recordings-watcher.path
systemctl --user status whisper-recordings-watcher.timer

journalctl --user \
  -u whisper-recordings-watcher.service \
  -n 100 \
  --no-pager

tail -f ~/.local/state/whisper-recordings-watcher/watcher.log
```

Run one manual scan with:

```bash
./scripts/recordings_watcher.sh
```

## Configuration

Edit `~/.config/whisper-recordings-watcher.env`. The default profile is Swedish,
KB-Whisper large, revision standard. `KB_MODEL` accepts `large`, `medium`, or a
full Hugging Face model identifier. For another language, set `LANGUAGE` and the
watcher will route through OpenAI Whisper using `WHISPER_MODEL` and
`WHISPER_PRESET`.

`RECURSIVE_SCAN=false` only watches files directly in `WATCH_DIR`. Set it to
`true` to include subfolders; hidden paths, temporary files, and every
configured processed-output directory remain excluded. Legacy
`output_transkrypcja/` directories are also excluded.

`STABILITY_SECONDS` controls the unchanged size/mtime wait before processing.
`SCAN_INTERVAL_SECONDS` is embedded into the timer when the installer runs, so
run `./scripts/install_watcher.sh` again after changing that value. Failed files
are retried only after `RETRY_DELAY_SECONDS` when `RETRY_FAILED=true`; changing a
failed source file also makes it immediately eligible.

Meeting-note settings:

```text
GENERATE_MEETING_NOTE=true
NOTE_LANGUAGE=sv
NOTE_DRAFT_MODEL=gpt-5.6-luna
NOTE_VERIFICATION_MODEL=gpt-5.6-terra
NOTE_REASONING_EFFORT=medium
OPENAI_ENV_FILE=/path/to/private/openai.env
TRANSCRIPT_DIR_NAME=output_transkrypcja
NOTE_DIR_NAME=Notatki
LOG_DIR_NAME=Logi
FFMPEG_TIMEOUT_SECONDS=1800
```

Set `GENERATE_MEETING_NOTE=false` to keep transcription-only behavior. The note
language is independent from the recording/transcription language and defaults
to Swedish. Both directory settings must be single folder names. API keys must
stay outside the repository.

The default hybrid route uses Luna for the lower-cost structured draft and
Terra for the quality-critical evidence verification. The reasoning effort and
strict source-segment validation remain the same for both stages.

`FFMPEG_TIMEOUT_SECONDS` prevents one damaged or stalled media conversion from
blocking every later recording. A timed-out file is marked
`transcription-failed` and remains eligible for retry.

## Processing state

State is appended to
`~/.local/state/whisper-recordings-watcher/processed.tsv`. Its tab-separated
columns are:

```text
absolute path    size_bytes    mtime_epoch    processed_epoch    status
```

Statuses are `processing`, `completed`, `completed-existing`,
`transcription-failed`, and `note-failed` (legacy `failed` is still understood).
Only the expected non-empty TXT/JSON transcript and, when enabled, DOCX/audit
JSON record `completed`. A `note-failed` retry reuses the existing local
transcript and does not run Whisper again. Completed files with unchanged size
and modification time are not processed again. The adjacent `watcher.log` is
the readable local log, and `watcher.lock` prevents overlapping
path/timer/manual runs.

## Stop or disable

```bash
systemctl --user disable --now whisper-recordings-watcher.path
systemctl --user disable --now whisper-recordings-watcher.timer
```

The units, local configuration, log, and state remain in place.
