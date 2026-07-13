# Automatic recordings watcher

The optional watcher transcribes new supported audio and video files added to
`/home/jakub-pelka/MobileTransfer/Recordings`. It runs entirely locally and uses
the repository's existing `scripts/start.sh` workflow. Transcripts are written
to `output_transkrypcja/` next to each source recording; source files are never
changed or removed.

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
`output_transkrypcja/` directory remain excluded.

`STABILITY_SECONDS` controls the unchanged size/mtime wait before processing.
`SCAN_INTERVAL_SECONDS` is embedded into the timer when the installer runs, so
run `./scripts/install_watcher.sh` again after changing that value. Failed files
are retried only after `RETRY_DELAY_SECONDS` when `RETRY_FAILED=true`; changing a
failed source file also makes it immediately eligible.

## Processing state

State is appended to
`~/.local/state/whisper-recordings-watcher/processed.tsv`. Its tab-separated
columns are:

```text
absolute path    size_bytes    mtime_epoch    processed_epoch    status
```

Statuses are `processing`, `completed`, `failed`, and `completed-existing`.
Only a zero launcher exit plus the expected non-empty TXT and JSON files records
`completed`. Completed files with unchanged size and modification time are not
processed again. The adjacent `watcher.log` is the readable local log, and
`watcher.lock` prevents overlapping path/timer/manual runs.

## Stop or disable

```bash
systemctl --user disable --now whisper-recordings-watcher.path
systemctl --user disable --now whisper-recordings-watcher.timer
```

The units, local configuration, log, and state remain in place.
