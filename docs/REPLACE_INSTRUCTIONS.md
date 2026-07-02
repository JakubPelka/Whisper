# Replace instructions for v6

Run from the local repository root:

```bash
cd /home/jakub-pelka/GitHub/Whisper || exit 1

BACKUP_DIR="_backup_before_v6_no_token_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -a *.py *.sh README.md .gitignore requirements*.txt scripts src docs secrets/README.md "$BACKUP_DIR" 2>/dev/null || true
```

Unpack the ZIP somewhere temporary and sync it into the repo:

```bash
rm -rf /tmp/whisper_v6
mkdir -p /tmp/whisper_v6
unzip -o ~/Downloads/Whisper_clean_restructure_v6.zip -d /tmp/whisper_v6
rsync -av /tmp/whisper_v6/Whisper_clean_restructure_v6/ ./
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

Verify that the local token is ignored, but remember: v6 does not use it.

```bash
git check-ignore -v secrets/token.txt 2>/dev/null || true
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
git commit -m "Remove token loading from transcription workflow"
git push
```
