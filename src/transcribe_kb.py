#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KB-Whisper transcription for Swedish audio.

Purpose:
  - fast local transcription of Swedish recordings
  - no diarization / no speaker separation
  - TXT + JSON output

Default model:
  KBLab/kb-whisper-large

Typical use:
  python src/transcribe_kb.py --input audio1.m4a audio2.m4a --outdir results
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


DEFAULT_MODEL = os.environ.get("KB_WHISPER_MODEL", "KBLab/kb-whisper-large")
DEFAULT_REVISION = os.environ.get("KB_WHISPER_REVISION", "standard")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def clean_path(value: str) -> Path:
    return Path(value.strip().strip('"').strip("'")).expanduser()


def safe_name(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def ffmpeg_to_wav16k_mono(input_path: Path, tmp_dir: Path) -> Path:
    out_wav = tmp_dir / f"{input_path.stem}_16k_mono.wav"
    run([
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        str(out_wav),
    ])
    return out_wav


def resolve_device(force_cpu: bool) -> tuple[str, torch.dtype]:
    if force_cpu:
        return "cpu", torch.float32
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    return "cpu", torch.float32


def load_asr(model_id: str, revision: str, force_cpu: bool):
    device, torch_dtype = resolve_device(force_cpu)

    print(f"Loading KB-Whisper model: {model_id}")
    print(f"Revision:                {revision}")
    print(f"Device:                  {device}")
    print(f"Torch dtype:             {torch_dtype}")

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or None
    )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        cache_dir="cache",
        low_cpu_mem_usage=True,
        token=token,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        cache_dir="cache",
        token=token,
    )

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    return asr, device


def save_outputs(
    *,
    input_path: Path,
    outdir: Path,
    model_id: str,
    revision: str,
    result: dict[str, Any],
    elapsed_seconds: float,
) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    model_part = safe_name(model_id.replace("/", "_"))
    base_name = f"{stem}_{model_part}_{revision}"

    txt_path = outdir / f"{base_name}.txt"
    json_path = outdir / f"{base_name}.json"

    text = (result.get("text") or "").strip()

    txt_path.write_text(text + "\n", encoding="utf-8")

    payload = {
        "input": str(input_path),
        "model": model_id,
        "revision": revision,
        "language": "sv",
        "task": "transcribe",
        "elapsed_seconds": round(elapsed_seconds, 3),
        "text": text,
        "raw_result": result,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return txt_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast KB-Whisper transcription for Swedish audio.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more input audio/video files")
    parser.add_argument("--outdir", required=True, help="Output folder")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Model revision: standard, strict, subtitle")
    parser.add_argument("--chunk-length", type=int, default=30, help="Chunk length in seconds")
    parser.add_argument("--batch-size", type=int, default=1, help="Pipeline batch size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    input_paths = [clean_path(value) for value in args.input]
    outdir = clean_path(args.outdir)

    missing = [path for path in input_paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
        return 2

    total_start = time.time()
    asr, _device = load_asr(args.model, args.revision, args.cpu)

    failures = 0
    for index, input_path in enumerate(input_paths, start=1):
        print("")
        print("=" * 72)
        print(f"File {index}/{len(input_paths)}: {input_path}")
        print("=" * 72)

        start = time.time()

        try:
            with tempfile.TemporaryDirectory(prefix="kb_whisper_") as td:
                tmp_dir = Path(td)
                wav_path = ffmpeg_to_wav16k_mono(input_path, tmp_dir)

                print("")
                print("Transcribing...")
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

            elapsed = time.time() - start
            txt_path, json_path = save_outputs(
                input_path=input_path,
                outdir=outdir,
                model_id=args.model,
                revision=args.revision,
                result=result,
                elapsed_seconds=elapsed,
            )

            print("")
            print("Saved:")
            print(f"  {txt_path}")
            print(f"  {json_path}")
            print(f"Elapsed: {elapsed:.1f} s")
        except Exception as exc:
            failures += 1
            print(f"ERROR while processing {input_path}: {exc}", file=sys.stderr)

    total_elapsed = time.time() - total_start
    print("")
    print("Batch finished")
    print(f"Files:    {len(input_paths)}")
    print(f"Failures: {failures}")
    print(f"Elapsed:  {total_elapsed:.1f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
