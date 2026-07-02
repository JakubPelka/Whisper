# Replace instructions

Run from the local repository root:

```bash
cd /home/jakub-pelka/GitHub/Whisper || exit 1

BACKUP_DIR="_backup_before_start_menu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -a *.py *.sh README.md .gitignore requirements*.txt scripts src secrets/README.md docs "$BACKUP_DIR" 2>/dev/null || true
```

Unpack the ZIP somewhere temporary and sync it into the repo:

```bash
unzip -o ~/Downloads/Whisper_clean_restructure_v5.zip -d /tmp/whisper_v5
rsync -av /tmp/whisper_v5/Whisper_clean_restructure_v5/ ./
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

Verify secrets:

```bash
chmod 600 secrets/token.txt
git check-ignore -v secrets/token.txt
```

Test:

```bash
./scripts/start.sh
```

Commit:

```bash
git status --short
git add README.md .gitignore requirements-kb.txt requirements-whisper.txt scripts src secrets/README.md docs/REPLACE_INSTRUCTIONS.md
git commit -m "Simplify transcription launcher"
git push
```
