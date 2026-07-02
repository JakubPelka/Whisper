# Replace instructions for v9

Run from the local repository root:

```bash
cd /home/jakub-pelka/GitHub/Whisper || exit 1

BACKUP_DIR="_backup_before_v9_output_next_to_source_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -a *.py *.sh README.md .gitignore requirements*.txt scripts src docs secrets/README.md "$BACKUP_DIR" 2>/dev/null || true
```

Unpack the ZIP somewhere temporary and sync it into the repo:

```bash
rm -rf /tmp/whisper_v9
mkdir -p /tmp/whisper_v9
unzip -o ~/Downloads/Whisper_clean_restructure_v9.zip -d /tmp/whisper_v9
rsync -av /tmp/whisper_v9/Whisper_clean_restructure_v9/ ./
```

Make launcher executable:

```bash
chmod +x scripts/start.sh
```

Remove old files:

```bash
rm -f transcribe_diarize.py
rm -f transcribe_kb_whisper.py
rm -f transcribe_kb_diarize.py
rm -f start_kb_whisper_test.sh
rm -f start_kb_fast.sh
rm -f start_kb_diarize.sh
rm -f start_whisper_sv.sh
```

Remove tracked secrets documentation if it exists. This does **not** delete your local `secrets/token.txt` unless you ask Git to track it, which it should not.

```bash
git rm -f secrets/README.md 2>/dev/null || true
```

Verify that generated output folders and the local token are ignored. v9 does not use tokens in the normal workflow.

```bash
git check-ignore -v secrets/token.txt 2>/dev/null || true
git check-ignore -v output_transkrypcja 2>/dev/null || true
```

Test:

```bash
unset HF_TOKEN HUGGINGFACE_TOKEN HUGGINGFACE_HUB_TOKEN
./scripts/start.sh
```

Commit:

```bash
git status --short
git add README.md .gitignore requirements-kb.txt requirements-whisper.txt scripts src docs/REPLACE_INSTRUCTIONS.md
git add -u
git commit -m "Write transcription outputs next to source files"
git push
```
