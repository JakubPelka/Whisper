#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testowy skrypt ASR dla KBLab KB-Whisper / Swedish Whispers.
Bez diarization. Cel: szybkie porównanie jakości transkrypcji szwedzkiego.

Domyślny model:
  KBLab/kb-whisper-large

Przykłady:
  python transcribe_kb_whisper.py
  python transcribe_kb_whisper.py --model KBLab/kb-whisper-large
  python transcribe_kb_whisper.py --revision strict
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


DEFAULT_INPUT = "/home/jakub-pelka/MobileTransfer/Recordings/Violett guldvinge.m4a"
DEFAULT_MODEL = os.environ.get("KB_WHISPER_MODEL", "KBLab/kb-whisper-large")
DEFAULT_REVISION = os.environ.get("KB_WHISPER_REVISION", "standard")


def run(cmd):
    subprocess.run(cmd, check=True)


def hhmmss(seconds):
    if seconds is None:
        return "??:??:??.???"
    seconds = max(0.0, float(seconds))
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def convert_to_wav16k_mono(input_path, tmp_dir):
    out_wav = tmp_dir / f"{input_path.stem}_kb_whisper_16k_mono.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(out_wav),
    ]
    run(cmd)
    return out_wav


def resolve_device(force_cpu=False):
    if force_cpu:
        return "cpu", torch.float32
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    return "cpu", torch.float32


def load_asr(model_id, revision, device, torch_dtype):
    print(f"Loading model: {model_id}")
    print(f"Revision:      {revision}")
    print(f"Device:        {device}")
    print(f"Torch dtype:   {torch_dtype}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        cache_dir="cache",
        low_cpu_mem_usage=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        cache_dir="cache",
    )

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )


def save_outputs(result, input_path, out_dir, model_id, revision, elapsed_s):
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_model = model_id.replace("/", "_")
    base = input_path.stem
    txt_path = out_dir / f"{base}_{safe_model}_{revision}.txt"
    json_path = out_dir / f"{base}_{safe_model}_{revision}.json"

    text = result.get("text", "").strip()
    chunks = result.get("chunks") or []

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# Input: {input_path}\n")
        f.write(f"# Model: {model_id}\n")
        f.write(f"# Revision: {revision}\n")
        f.write(f"# Elapsed seconds: {elapsed_s:.1f}\n")
        f.write("\n")

        if chunks:
            for ch in chunks:
                ts = ch.get("timestamp") or (None, None)
                start, end = ts if isinstance(ts, (list, tuple)) and len(ts) == 2 else (None, None)
                f.write(f"[{hhmmss(start)}–{hhmmss(end)}] {ch.get('text', '').strip()}\n")
        else:
            f.write(text + "\n")

    payload = {
        "input": str(input_path),
        "model": model_id,
        "revision": revision,
        "elapsed_seconds": elapsed_s,
        "result": result,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("")
    print("Saved:")
    print(f"  {txt_path}")
    print(f"  {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input audio/video file")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="HF revision: standard, strict, subtitle")
    parser.add_argument("--chunk-length", type=int, default=30, help="Chunk length in seconds")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size; keep 1 for low VRAM")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.outdir).expanduser() if args.outdir else input_path.parent / "kb_whisper_tests"

    device, torch_dtype = resolve_device(force_cpu=args.cpu)

    with tempfile.TemporaryDirectory(prefix="kb_whisper_") as td:
        tmp_dir = Path(td)
        print(f"Converting to WAV 16 kHz mono: {input_path}")
        wav_path = convert_to_wav16k_mono(input_path, tmp_dir)

        try:
            asr = load_asr(args.model, args.revision, device, torch_dtype)
        except torch.cuda.OutOfMemoryError:
            print("")
            print("CUDA OOM while loading model.")
            print("Try one of:")
            print("  1) KB_WHISPER_MODEL=KBLab/kb-whisper-large ./start_kb_whisper_test.sh")
            print("  2) KB_WHISPER_MODEL=KBLab/kb-whisper-small ./start_kb_whisper_test.sh")
            print("  3) python transcribe_kb_whisper.py --cpu --model KBLab/kb-whisper-large")
            raise

        print("")
        print("Transcribing...")
        t0 = time.time()

        result = asr(
            str(wav_path),
            chunk_length_s=args.chunk_length,
            batch_size=args.batch_size,
            return_timestamps=True,
            generate_kwargs={
                "task": "transcribe",
                "language": "sv",
                "condition_on_prev_tokens": False,
            },
        )

        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f} seconds.")
        save_outputs(result, input_path, out_dir, args.model, args.revision, elapsed)


if __name__ == "__main__":
    main()
