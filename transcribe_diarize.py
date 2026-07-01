# -*- coding: utf-8 -*-
"""
transcribe_diarize.py
- Okna (tkinter): wybór jednego lub wielu plików audio + wybór wspólnego folderu docelowego
- Terminal: pytanie o tryb (Faster/More accurate/Default), wybór języka (pl/en/sv/auto/other) i modelu Whisper
- Token HF (pyannote 3.1) zapisany w skrypcie
- Diarization: pyannote 3.1 (gated) z fallbackiem do legacy
- Transkrypcja: Whisper (domyślnie large-v3)
- Wyniki dla każdego pliku: RAW i MERGED → 2×TXT + 2×JSON
- Po przetworzeniu:
    * czyści cały tymczasowy folder per plik (_tmp_diar_{stem}),
    * przenosi oryginalny plik do subfolderu processed_files/ obok źródła
    * zapisuje transkrypcje w: <OUTDIR>/transkrypcje/raw oraz <OUTDIR>/transkrypcje/merged
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

# --- TOKEN HF przez zmienną środowiskową ---
# Nie zapisuj tokenu Hugging Face w repo ani bezpośrednio w kodzie.
# Przed uruchomieniem ustaw:
#   export HF_TOKEN="hf_..."
# albo:
#   export HUGGINGFACE_TOKEN="hf_..."

TOKEN_IN_SCRIPT = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""

if TOKEN_IN_SCRIPT:
    os.environ["HUGGINGFACE_TOKEN"] = TOKEN_IN_SCRIPT
else:
    print(
        "UWAGA: Brak tokenu Hugging Face. "
        "Diarization pyannote może nie zadziałać. "
        "Ustaw HF_TOKEN albo HUGGINGFACE_TOKEN przed uruchomieniem."
    )

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

# --- domyślne parametry (modyfikowane przez preset Fast/Accurate/Default) ---
WHISPER_TEMPERATURE = 0.0
WHISPER_BEAM_SIZE = 4
WHISPER_BEST_OF = 4
WHISPER_COND_PREV = False

# --- parametry scalania (po transkrypcji) ---
MERGE_ENABLED = True
GAP_THRESHOLD = 0.8        # sekundy – maks przerwa między segmentami tego samego mówcy
MAX_BLOCK_LEN = 180.0      # sekundy – maks długość jednego bloku po scaleniu
MERGE_SHORTER_THAN = 0.5   # sekundy – mikro-wstawki łączymy zawsze z sąsiadem (gdy ten sam mówca)

# ---------------- GUI: wybór wielu plików i folderu ----------------
def pick_files_and_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Wybór plików", "Wybierz JEDEN lub WIELE plików audio do przetworzenia.")
        fpaths = filedialog.askopenfilenames(
            title="Wybierz plik(i) audio",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.m4a *.aac *.wma *.ogg *.flac *.mkv *.mp4 *.m4b"),
                ("Wszystkie pliki", "*.*"),
            ],
        )
        if not fpaths:
            raise RuntimeError("Anulowano wybór plików.")
        messagebox.showinfo("Folder wyjściowy", "Wybierz folder, gdzie zapisać transkrypcje.")
        outdir = filedialog.askdirectory(title="Wybierz folder docelowy")
        if not outdir:
            # jeśli nie wybrano, użyj folderu pierwszego pliku
            outdir = str(Path(fpaths[0]).parent)
        return [Path(p) for p in fpaths], Path(outdir)
    except Exception:
        # awaryjnie terminal (gdyby tkinter nie działał)
        print("Nie udało się użyć okienek (tkinter). Używam konsoli.")
        raw = input("Podaj ścieżki do plików audio (oddzielone średnikiem ';'): ").strip()
        paths = [Path(p.strip().strip('\"')) for p in raw.split(';') if p.strip()]
        outdir = input("Folder docelowy (ENTER = folder pierwszego pliku): ").strip('\" ')
        if not outdir and paths:
            outdir = str(paths[0].parent)
        return paths, Path(outdir)

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


def choose_preset_term():
    global WHISPER_BEAM_SIZE, WHISPER_BEST_OF
    print("\nTryb pracy:")
    print("  1) Faster         (beam_size=1, best_of=1)")
    print("  2) More accurate  (beam_size=8, best_of=8)")
    print("  3) Default        (beam_size=4, best_of=4)")
    ch = input("Twój wybór [1-3] (ENTER=Default): ").strip()
    if ch == "1":
        WHISPER_BEAM_SIZE = 1
        WHISPER_BEST_OF = 1
        print("Preset: Faster (1/1)")
    elif ch == "2":
        WHISPER_BEAM_SIZE = 8
        WHISPER_BEST_OF = 8
        print("Preset: More accurate (8/8)")
    else:
        WHISPER_BEAM_SIZE = 4
        WHISPER_BEST_OF = 4
        print("Preset: Default (4/4)")

def choose_language_term():
    print("\nWybierz język transkrykcji:")
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

def safe_unique_move(src: Path, dst_dir: Path):
    """Przenieś src do dst_dir, unikając nadpisania: jeśli istnieje, dodaj sufiks (_1, _2, ...)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    if not target.exists():
        shutil.move(str(src), str(target))
        return target
    stem, suf = src.stem, src.suffix
    i = 1
    while True:
        cand = dst_dir / f"{stem}_{i}{suf}"
        if not cand.exists():
            shutil.move(str(src), str(cand))
            return cand
        i += 1

def cleanup_dir(path: Path):
    """Usuń CAŁY katalog (zawartość + folder), nawet na Windows (readonly/uchwyty)."""
    import stat, time, gc

    def _on_rm_error(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    for attempt in range(3):
        try:
            if path.exists():
                shutil.rmtree(path, onerror=_on_rm_error)
            break
        except Exception:
            gc.collect()
            time.sleep(0.2)

    if path.exists():
        try:
            os.rmdir(path)
        except Exception:
            pass

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
        temperature=WHISPER_TEMPERATURE,
        beam_size=WHISPER_BEAM_SIZE,
        best_of=WHISPER_BEST_OF,
        condition_on_previous_text=WHISPER_COND_PREV,
        fp16=torch.cuda.is_available()
    )
    if language:
        result = whisper_model.transcribe(str(seg_path), language=language, **kwargs)
    else:
        result = whisper_model.transcribe(str(seg_path), **kwargs)  # auto
    return result.get("text", "").strip()

# --------- scalanie po transkrypcji ---------
def _join_text(a: str, b: str) -> str:
    a = (a or "").rstrip()
    b = (b or "").lstrip()
    if not a:
        return b
    if a.endswith((" ", "\n", "\t")):
        return a + b
    return a + " " + b

def merge_segments(raw_segments, gap_thr=GAP_THRESHOLD, max_block=MAX_BLOCK_LEN, short_thr=MERGE_SHORTER_THAN):
    """Łączy sąsiadujące segmenty tego samego mówcy, jeśli przerwa mała i nie przekraczamy limitu długości bloku."""
    if not raw_segments:
        return []

    merged = []
    cur = dict(raw_segments[0])  # copy
    cur["text"] = cur.get("text", "")

    def seg_len(seg): return float(seg["end"] - seg["start"])

    for nxt in raw_segments[1:]:
        same_spk = (str(nxt["speaker"]) == str(cur["speaker"]))
        gap = float(nxt["start"] - cur["end"])
        nxt_len = seg_len(nxt)
        cur_len = seg_len(cur)
        can_merge = same_spk and (gap <= gap_thr or cur_len < short_thr or nxt_len < short_thr) \
                    and ((nxt["end"] - cur["start"]) <= max_block)
        if can_merge:
            cur["end"] = nxt["end"]
            cur["text"] = _join_text(cur.get("text",""), nxt.get("text",""))
        else:
            merged.append(cur)
            cur = dict(nxt)
            cur["text"] = cur.get("text","")
    merged.append(cur)
    return merged


def iter_diarization_tracks(diar_result):
    """
    Compatibility helper for pyannote outputs.

    Older pyannote versions return an Annotation directly.
    Newer pyannote pipelines may return a DiarizeOutput object
    with the actual Annotation stored in .speaker_diarization.
    """
    ann = None

    if hasattr(diar_result, "itertracks"):
        ann = diar_result
    elif hasattr(diar_result, "speaker_diarization"):
        ann = diar_result.speaker_diarization
    elif isinstance(diar_result, dict):
        for key in ("speaker_diarization", "diarization", "annotation"):
            if key in diar_result:
                ann = diar_result[key]
                break

    if ann is None or not hasattr(ann, "itertracks"):
        attrs = [a for a in dir(diar_result) if not a.startswith("_")]
        raise RuntimeError(
            f"Unsupported pyannote diarization output: {type(diar_result)}; "
            f"available attributes={attrs}"
        )

    return ann.itertracks(yield_label=True)

# --------- przetwarzanie jednego pliku ---------
def process_one_file(input_path: Path, out_dir: Path, language, whisper_model, whisper_name, pipeline, diar_id):
    assert input_path.exists(), f"Brak pliku: {input_path}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) przygotuj strukturę docelową: transkrypcje/raw i transkrypcje/merged
    trans_root = out_dir / "transkrypcje"
    raw_dir = trans_root / "raw"
    merged_dir = trans_root / "merged"
    raw_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # 1) przygotowanie audio — unikalny katalog tymczasowy per plik
    work_dir = input_path.parent / f"_tmp_diar_{input_path.stem}"
    wav_path = ensure_wav_mono16k(input_path, work_dir)
    whole = AudioSegment.from_file(str(wav_path))

    # 2) diarization
    diar = pipeline(str(wav_path))

    # 3) segmenty (RAW time)
    MIN_SEG, MAX_SEG = 0.35, 120.0
    segs_time = []
    for turn, _, speaker in iter_diarization_tracks(diar):
        s = float(turn.start); e = float(turn.end)
        if e - s < MIN_SEG:
            continue
        if e - s > MAX_SEG:
            n = math.ceil((e - s) / MAX_SEG)
            for i in range(n):
                ss = s + i * MAX_SEG
                ee = min(e, ss + MAX_SEG)
                segs_time.append({"start": ss, "end": ee, "speaker": str(speaker)})
        else:
            segs_time.append({"start": s, "end": e, "speaker": str(speaker)})
    segs_time.sort(key=lambda x: x["start"])

    # 4) transkrypcja per segment (RAW tekst)
    seg_dir = work_dir / f"{input_path.stem}_segs"
    seg_dir.mkdir(exist_ok=True)
    raw_segments = []

    for i, seg in enumerate(segs_time, start=1):
        s, e, spk = seg["start"], seg["end"], seg["speaker"]
        clip = slice_segment(whole, s, e)
        seg_path = seg_dir / f"{input_path.stem}_seg{i:04d}.wav"
        clip.export(str(seg_path), format="wav")
        # zwolnij pamięć klipu
        del clip

        text = transcribe_segment(whisper_model, seg_path, language)
        raw_segments.append({
            "start": s, "end": e,
            "start_hhmmss": hhmmss(s), "end_hhmmss": hhmmss(e),
            "speaker": spk, "text": text
        })
        print(f"[{hhmmss(s)}–{hhmmss(e)}] {spk}: {text}")

    # 5) scalanie (MERGED)
    merged_segments = merge_segments(raw_segments) if MERGE_ENABLED else list(raw_segments)

    # 6) zapisy RAW & MERGED (do transkrypcje/raw i transkrypcje/merged)
    base = input_path.stem
    raw_txt    = raw_dir / f"{base}_raw.txt"
    raw_json   = raw_dir / f"{base}_raw.json"
    merged_txt = merged_dir / f"{base}_merged.txt"
    merged_json= merged_dir / f"{base}_merged.json"

    # TXT RAW
    with open(raw_txt, "w", encoding="utf-8") as f:
        for seg in raw_segments:
            f.write(f"[{seg['start_hhmmss']}–{seg['end_hhmmss']}] {seg['speaker']}: {seg['text']}\n")

    # TXT MERGED
    with open(merged_txt, "w", encoding="utf-8") as f:
        for seg in merged_segments:
            f.write(f"[{hhmmss(seg['start'])}–{hhmmss(seg['end'])}] {seg['speaker']}: {seg['text']}\n")

    # JSON RAW
    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump({
            "file": str(input_path),
            "output_dir": str(out_dir),
            "language": language or "auto",
            "whisper_model": whisper_name,
            "diarization_model": diar_id,
            "segments": raw_segments
        }, f, ensure_ascii=False, indent=2)

    # JSON MERGED
    with open(merged_json, "w", encoding="utf-8") as f:
        json.dump({
            "file": str(input_path),
            "output_dir": str(out_dir),
            "language": language or "auto",
            "whisper_model": whisper_name,
            "diarization_model": diar_id,
            "merge_params": {
                "gap_threshold_sec": GAP_THRESHOLD,
                "max_block_len_sec": MAX_BLOCK_LEN,
                "merge_shorter_than_sec": MERGE_SHORTER_THAN
            },
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "start_hhmmss": hhmmss(seg["start"]),
                    "end_hhmmss": hhmmss(seg["end"]),
                    "speaker": seg["speaker"],
                    "text": seg["text"],
                } for seg in merged_segments
            ]
        }, f, ensure_ascii=False, indent=2)

    # 7) przenieś oryginalny plik do processed_files
    processed = input_path.parent / "processed_files"
    try:
        moved_to = safe_unique_move(input_path, processed)
        print(f"Przeniesiono oryginał do: {moved_to}")
    except Exception as e:
        print(f"UWAGA: nie przeniosłem oryginału do processed_files ({e}).")

    # 8) ZWOLNIJ UCHWYTY, usuń WAV i WYCZYŚĆ CAŁY folder tymczasowy
    try:
        # zwolnij ewentualne referencje, żeby na Windows nie trzymały uchwytów
        whole = None
        import gc
        gc.collect()
        if Path(wav_path).exists():
            Path(wav_path).unlink()
    except Exception:
        pass

    cleanup_dir(work_dir)
    print(f"[OK] Wyczyszczono katalog tymczasowy: {work_dir} (istnieje? {work_dir.exists()})")

    print(f"\nZapisano:\n - {raw_txt}\n - {merged_txt}\n - {raw_json}\n - {merged_json}\n")

# ---------------- main ----------------
def main():
    print("=" * 70)
    print(" DIARIZATION + WHISPER TRANSCRIPTION — batch")
    print("=" * 70)

    # pliki + wspólny folder przez okna
    in_paths, out_dir = pick_files_and_folder()
    in_paths = [p for p in in_paths if p.exists() and p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    if not in_paths:
        print("ERROR: Brak poprawnych plików wejściowych.")
        sys.exit(2)
    out_dir.mkdir(parents=True, exist_ok=True)

    # preset, język, model
    choose_preset_term()
    language = choose_language_term()
    whisper_model_name = choose_model_term()

    # Załaduj modele (raz na batch)
    try:
        whisper_model, whisper_name = load_whisper(whisper_model_name)
        pipeline, diar_id = load_pyannote()
    except Exception as e:
        print(f"!! Błąd podczas ładowania modeli: {e}")
        sys.exit(1)

    # Przetwarzaj po kolei
    for idx, in_path in enumerate(in_paths, start=1):
        print("\n" + "-" * 70)
        print(f"Plik {idx}/{len(in_paths)}: {in_path.name}")
        print("-" * 70)
        try:
            process_one_file(in_path, out_dir, language, whisper_model, whisper_name, pipeline, diar_id)
        except Exception as e:
            print(f"!! Błąd dla {in_path.name}: {e}")

    print("\n*** Zakończono batch ***")

if __name__ == "__main__":
    main()
