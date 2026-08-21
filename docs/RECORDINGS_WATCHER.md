# Automatic recordings watcher

The optional watcher transcribes new supported audio and video files added to
`/home/jakub-pelka/MobileTransfer/Recordings`. It uses the same local
`scripts/start.sh` workflow as an interactive transcription. Timestamped TXT and
JSON results are written to `output_transkrypcja/` next to the recording. Source
recordings are never changed or removed.

The watcher performs transcription only. It does not call GPT/APIs, summarize,
generate notes, or run other post-processing.

Supported input extensions are case-insensitive:

```text
wav mp3 m4a aac flac ogg opus qta mp4 mov mkv webm avi
```

Every input is inspected with `ffprobe`; the extension alone is not trusted. A
decodable audio stream is selected and converted locally to a temporary mono
16 kHz PCM WAV. QTA files prefer a conventional AAC/PCM/ALAC compatibility
stream when present. APAC is considered only when it is the usable local option.
The temporary WAV is removed after transcription and is never placed in the
recordings folder.

## Install

```bash
./scripts/install_watcher.sh
```

On first installation, every supported recording already present, including
QTA, is registered with status `completed-existing`. Existing recordings are
not transcribed. The installer preserves existing configuration and processing
state.

The installer creates and enables:

- `whisper-recordings-watcher.path` for quick reaction to folder changes;
- `whisper-recordings-watcher.timer` as a periodic fallback;
- `whisper-recordings-watcher.service` for one locked scan at a time.

Check them with:

```bash
systemctl --user status whisper-recordings-watcher.path
systemctl --user status whisper-recordings-watcher.timer
journalctl --user -u whisper-recordings-watcher.service -n 100 --no-pager
tail -f ~/.local/state/whisper-recordings-watcher/watcher.log
```

Run one manual scan with:

```bash
./scripts/recordings_watcher.sh
```

## Quiet event logging

`~/.local/state/whisper-recordings-watcher/watcher.log` is the only application
event log. It contains only:

```text
BASELINE
PROCESSING
COMPLETED
FAILED
SKIPPED_UNSUPPORTED
STATE_REPAIR
```

No-op scans write nothing and create no per-run files. A recording whose size or
modification time is still changing is silently deferred. Normal lock contention
is also silent. systemd journal activation metadata is not duplicated into the
application log.

`SKIPPED_UNSUPPORTED` is written only for a supported recording candidate that
has no locally decodable audio stream. Unrelated files in the directory are not
logged.

## Configuration

Edit `~/.config/whisper-recordings-watcher.env`. The default profile is Swedish,
KB-Whisper large, revision standard. `KB_MODEL` accepts `large`, `medium`, or a
full Hugging Face model identifier. For another language, set `LANGUAGE`; the
watcher routes through OpenAI Whisper using `WHISPER_MODEL` and
`WHISPER_PRESET`.

`RECURSIVE_SCAN=false` limits scanning to files directly in `WATCH_DIR`.
Temporary, hidden, and transcript-output paths are excluded.

`STABILITY_SECONDS` controls the unchanged size/mtime wait. An unstable file is
deferred without a log entry. `SCAN_INTERVAL_SECONDS` is embedded in the timer
when the installer runs, so rerun `./scripts/install_watcher.sh` after changing
it. Failed transcriptions are retried after `RETRY_DELAY_SECONDS` when
`RETRY_FAILED=true`; changing the source makes it immediately eligible.

`FFPROBE_TIMEOUT_SECONDS` and `FFMPEG_TIMEOUT_SECONDS` prevent damaged or stalled
media from blocking later recordings.

## Processing state

State is kept in one append-only TSV file:

```text
~/.local/state/whisper-recordings-watcher/processed.tsv
```

Its columns are:

```text
absolute path    size_bytes    mtime_epoch    processed_epoch    status
```

Current statuses are `processing`, `completed`, `completed-existing`,
`transcription-failed`, and `unsupported`; legacy failure statuses remain
understood. Completed or unsupported files with unchanged size and modification
time are not attempted again. The runtime lock prevents overlapping path, timer,
and manual runs.

## Stop or disable

```bash
systemctl --user disable --now whisper-recordings-watcher.path
systemctl --user disable --now whisper-recordings-watcher.timer
```

The units, local configuration, event log, and processing state remain in place.
