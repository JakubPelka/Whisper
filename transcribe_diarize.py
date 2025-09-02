# -*- coding: utf-8 -*-
"""
transcribe_diarize.py
- Okna (tkinter): wybór pliku audio i folderu docelowego
- Terminal: wybór języka (pl/en/sv/auto/other) i modelu Whisper
- Token HF (pyannote 3.1) zapisany w skrypcie
- Diarization: pyannote 3.1 (gated) z fallbackiem do legacy
- Transkrypcja: Whisper (domyślnie large-v3)
- Wyniki: TXT + JSON w folderze docelowym
- Tworzy `arkiv` obok pliku źródłowego i przenosi tam .mp3/.m4a

Wymagania:
  pip install -U openai-whisper pyannote.audio pydub torch torchaudio
  (ffmpeg w PATH dla pydub)
"""

import os
import sys
import math
import json
import shutil
import logging
import warnings
from pathlib import Path
from datetime import timedelta

# --- TOKEN HF w skrypcie (pyannote 3.1) ---
TOKEN_IN_SCRIPT = "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"
os.environ["HUGGINGFACE_TOKEN"] = TOKEN_IN_SCRIPT

# --- wyciszenie zbędnych logów/ostrzeżeń ---
logging.getLogger().setLevel(logging.WARNING)
for name in ["speechbrain", "pytorch_lightning", "huggingface_hub", "urllib3", "torch._dynamo"]:
    logging.getLogger(name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

import torch
import whisper
from pydub import AudioSegment

# (opcjonalnie) lekkie przyspieszenie na NVIDIA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".wma", ".ogg", ".flac", ".mkv", ".mp4", ".m4b"}

# ---------------- GUI: wybór pliku i folderu ----------------
def pick_file_and_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Wybór pliku", "Wybierz plik audio do przetworzenia.")
        fpath = filedialog.askopenfilename(
            title="Wybierz plik audio",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.m4a *.aac *.wma *.ogg *.flac *.mkv *.mp4 *.m4b"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not fpath:
            raise RuntimeError("Anulowano wybór pliku.")
        messagebox.showinfo("Folder wyjściowy", "Wybierz folder, gdzie zapisać transkrypcję.")
        outdir = filedialog.askdirectory(title="Wybierz folder docelowy")
        if not outdir:
            outdir = str(Path(fpath).parent)
        return Path(fpath), Path(outdir)
    except Exception:
        # awaryjnie terminal (gdyby tkinter nie działał)
        print("Nie udało się użyć okienek (tkinter). Używam konsoli.")
        fpath = input("Ścieżka do pliku audio: ").strip('" ')
        outdir = input("Folder docelowy (ENTER = folder pliku): ").strip('" ')
        if not outdir:
            outdir = str(Path(fpath).parent)
        return Path(fpath), Path(outdir)

# ---------------- helpers ----------------
def hhmmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def ensure_wav_mono16k(input_path: Path, work_dir: Path) -> Path:
    audio = AudioSegment.from_file(str(input_path))
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    if audio.channels != 1:
        audio = audio.set_channels(1)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_wav = work_dir / f"{input_path.stem}_mono16k.wav"
    audio.export(str(out_wav), format="wav")
    return out_wav

def slice_segment(audio: AudioSegment, start_s: float, end_s: float) -> AudioSegment:
    a = max(0, int(start_s * 1000.0))
    b = max(0, int(end_s * 1000.0))
    if b <= a:
        b = a + 1
    return audio[a:b]

def choose_language_term():
    print("\nWybierz język transkrypcji:")
    print("  1) pl  2) en  3) sv  4) auto-wykrywanie  5) other (podaj kod, np. 'de')")
    ch = input("Twój wybór [1-5] (ENTER=pl): ").strip()
    mapping = {"1":"pl","2":"en","3":"sv","4":None}
    if ch in mapping:
        return mapping[ch]
    if ch == "5":
        code = input("Podaj ISO 639-1 (np. 'de','fr','it'): ").strip().lower()
        return code or None
    return "pl"

def choose_model_term():
    models = ["tiny","base","small","medium","large-v2","large-v3","large-v3-turbo"]
    default = "large-v3"
    print("\nWybierz model Whisper:")
    for i, m in enumerate(models, start=1):
        print(f"  {i}) {m}")
    ch = input(f"Twój wybór [1-{len(models)}] (ENTER={default}): ").strip()
    if not ch:
        return default
    try:
        idx = int(ch)
        if 1 <= idx <= len(models):
            return models[idx-1]
    except Exception:
        pass
    return default

def load_whisper(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tried = []
    for name in [model_name, "large-v3", "large-v2", "large"]:
        try:
            print(f"Ładowanie Whisper: {name} ...")
            m = whisper.load_model(name, device=device)
            print(f"OK: Whisper={name}")
            return m, name
        except Exception as e:
            tried.append((name, str(e)))
    raise RuntimeError(f"Nie udało się załadować Whisper. Próby: {tried}")

def load_pyannote():
    from pyannote.audio import Pipeline
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    def _try(model_id: str):
        if hf_token:
            try:
                return Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
            except TypeError:
                return Pipeline.from_pretrained(model_id, token=hf_token)
        else:
            return Pipeline.from_pretrained(model_id)

    model_id = "pyannote/speaker-diarization-3.1"
    try:
        print(f"Ładowanie diarization: {model_id} ...")
        pipe = _try(model_id)
        if pipe is None:
            raise RuntimeError("Brak dostępu do gated modelu (pipeline==None).")
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
        print("OK: pyannote 3.1")
        return pipe, model_id
    except Exception as e:
        print(f"UWAGA: 3.1 niedostępny ({e}). Fallback → legacy 'pyannote/speaker-diarization'.")
        legacy = "pyannote/speaker-diarization"
        pipe = _try(legacy)
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
        print("OK: legacy diarization")
        return pipe, legacy

def transcribe_segment(whisper_model, seg_path: Path, language):
    kwargs = dict(
        temperature=0.0, beam_size=5, best_of=5,
        condition_on_previous_text=False, fp16=torch.cuda.is_available()
    )
    if language:
        result = whisper_model.transcribe(str(seg_path), language=language, **kwargs)
    else:
        result = whisper_model.transcribe(str(seg_path), **kwargs)  # auto
    return result.get("text", "").strip()

def process_one_file(input_path: Path, out_dir: Path, language, whisper_model_name):
    assert input_path.exists(), f"Brak pliku: {input_path}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) przygotowanie audio
    work_dir = input_path.parent / "_tmp_diar"
    wav_path = ensure_wav_mono16k(input_path, work_dir)
    whole = AudioSegment.from_file(str(wav_path))

    # 2) diarization
    pipeline, diar_id = load_pyannote()
    diar = pipeline(str(wav_path))

    # 3) segmenty
    MIN_SEG, MAX_SEG = 0.35, 120.0
    segs = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        s = float(turn.start); e = float(turn.end)
        if e - s < MIN_SEG:
            continue
        if e - s > MAX_SEG:
            n = math.ceil((e - s) / MAX_SEG)
            for i in range(n):
                ss = s + i * MAX_SEG
                ee = min(e, ss + MAX_SEG)
                segs.append({"start": ss, "end": ee, "speaker": str(speaker)})
        else:
            segs.append({"start": s, "end": e, "speaker": str(speaker)})
    segs.sort(key=lambda x: x["start"])

    # 4) whisper
    wmodel, used_name = load_whisper(whisper_model_name)

    # 5) transkrypcja per segment
    seg_dir = work_dir / f"{input_path.stem}_segs"
    seg_dir.mkdir(exist_ok=True)
    txt_lines, json_items = [], []

    for i, seg in enumerate(segs, start=1):
        s, e, spk = seg["start"], seg["end"], seg["speaker"]
        clip = slice_segment(whole, s, e)
        seg_path = seg_dir / f"{input_path.stem}_seg{i:04d}.wav"
        clip.export(str(seg_path), format="wav")

        text = transcribe_segment(wmodel, seg_path, language)
        line = f"[{hhmmss(s)}–{hhmmss(e)}] {spk}: {text}"
        print(line)
        txt_lines.append(line)
        json_items.append({
            "start": s, "end": e,
            "start_hhmmss": hhmmss(s), "end_hhmmss": hhmmss(e),
            "speaker": spk,
            "text": text
        })

    # 6) zapisy
    base = input_path.stem
    out_txt = out_dir / f"{base}_transcription.txt"
    out_json = out_dir / f"{base}_transcription.json"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "file": str(input_path),
            "output_dir": str(out_dir),
            "language": language or "auto",
            "whisper_model": used_name,
            "diarization_model": diar_id,
            "segments": json_items
        }, f, ensure_ascii=False, indent=2)

    # 7) arkiv + sprzątanie
    arkiv = input_path.parent / "arkiv"
    arkiv.mkdir(exist_ok=True)
    if input_path.suffix.lower() in {".mp3", ".m4a"}:
        try:
            shutil.move(str(input_path), str(arkiv / input_path.name))
            print(f"Przeniesiono do arkiv: {input_path.name}")
        except Exception as e:
            print(f"UWAGA: nie przeniosłem do arkiv ({e}).")

    try:
        for p in seg_dir.glob("*.wav"):
            p.unlink(missing_ok=True)
        # (work_dir).rmdir()  # odkomentuj, jeśli chcesz usuwać katalog tymczasowy
    except Exception:
        pass

    print(f"\nZapisano:\n - {out_txt}\n - {out_json}\n")

# ---------------- main ----------------
def main():
    print("=" * 70)
    print(" DIARIZATION + WHISPER TRANSCRIPTION")
    print("=" * 70)

    # plik + folder przez okna
    in_path, out_dir = pick_file_and_folder()

    if not in_path.exists():
        print(f"ERROR: Nie ma pliku: {in_path}")
        sys.exit(2)
    if in_path.suffix.lower() not in AUDIO_EXTS:
        print(f"ERROR: Nieobsługiwane rozszerzenie: {in_path.suffix}")
        sys.exit(2)
    out_dir.mkdir(parents=True, exist_ok=True)

    # język + model w terminalu
    language = choose_language_term()
    wmodel = choose_model_term()

    print("\nStart...")
    print("-" * 70)
    try:
        process_one_file(in_path, out_dir, language, wmodel)
    except Exception as e:
        print(f"!! Błąd: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
