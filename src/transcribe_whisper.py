#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean OpenAI Whisper transcription.

Purpose:
  - transcription for languages other than Swedish, or general fallback
  - no pyannote
  - no diarization / no speaker separation
  - TXT + JSON output

Typical use:
  python src/transcribe_whisper.py --input audio1.m4a audio2.m4a --outdir results --language pl --model large-v3-turbo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import whisper


PRESETS = {
    "fast": {
        "beam_size": 1,
        "best_of": 1,
        "condition_on_previous_text": False,
        "temperature": 0.0,
    },
    "default": {
        "beam_size": 4,
        "best_of": 4,
        "condition_on_previous_text": False,
        "temperature": 0.0,
    },
    "accurate": {
        "beam_size": 8,
        "best_of": 8,
        "condition_on_previous_text": True,
        "temperature": 0.0,
    },
}


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


def resolve_device(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize_language(language: str) -> str | None:
    language = (language or "").strip()
    if language.lower() in {"", "auto", "none", "-"}:
        return None
    return language


def save_outputs(
    *,
    input_path: Path,
    outdir: Path,
    model_name: str,
    language: str | None,
    preset: str,
    result: dict[str, Any],
    elapsed_seconds: float,
) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    lang_part = language or "auto"
    base_name = f"{input_path.stem}_openai-whisper_{safe_name(model_name)}_{lang_part}_{preset}"

    txt_path = outdir / f"{base_name}.txt"
    json_path = outdir / f"{base_name}.json"

    text = (result.get("text") or "").strip()
    txt_path.write_text(text + "\n", encoding="utf-8")

    payload = {
        "input": str(input_path),
        "backend": "openai-whisper",
        "model": model_name,
        "language": language or "auto",
        "preset": preset,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "text": text,
        "segments": result.get("segments", []),
        "raw_result": result,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return txt_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean OpenAI Whisper transcription without diarization.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more input audio/video files")
    parser.add_argument("--outdir", required=True, help="Output folder")
    parser.add_argument("--language", default="auto", help="Language code, e.g. pl/en/sv/de, or auto")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model name")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="fast", help="Transcription preset")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    input_paths = [clean_path(value) for value in args.input]
    outdir = clean_path(args.outdir)

    missing = [path for path in input_paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
        return 2

    language = normalize_language(args.language)
    device = resolve_device(args.cpu)
    options = dict(PRESETS[args.preset])

    print(f"Loading OpenAI Whisper model: {args.model}")
    print(f"Device:                     {device}")
    print(f"Language:                   {language or 'auto'}")
    print(f"Preset:                     {args.preset}")

    total_start = time.time()
    model = whisper.load_model(args.model, device=device)

    failures = 0
    for index, input_path in enumerate(input_paths, start=1):
        print("")
        print("=" * 72)
        print(f"File {index}/{len(input_paths)}: {input_path}")
        print("=" * 72)

        start = time.time()
        print("")
        print("Transcribing...")

        try:
            result = model.transcribe(
                str(input_path),
                language=language,
                task="transcribe",
                fp16=(device == "cuda"),
                **options,
            )

            elapsed = time.time() - start
            txt_path, json_path = save_outputs(
                input_path=input_path,
                outdir=outdir,
                model_name=args.model,
                language=language,
                preset=args.preset,
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
