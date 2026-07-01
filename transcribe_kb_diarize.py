#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KB-Whisper + pyannote diarization pipeline.

Flow:
  input audio/video
  -> ffmpeg WAV 16 kHz mono
  -> pyannote speaker diarization
  -> split speaker/time segments
  -> transcribe each segment with KBLab/kb-whisper-large
  -> export RAW and MERGED TXT/JSON

No token is stored in this file.
Use:
  export HF_TOKEN="hf_..."
"""

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from pyannote.audio import Pipeline as PyannotePipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline


DEFAULT_INPUT = "/home/jakub-pelka/MobileTransfer/Recordings/Violett guldvinge.m4a"
DEFAULT_MODEL = os.environ.get("KB_WHISPER_MODEL", "KBLab/kb-whisper-large")
DEFAULT_REVISION = os.environ.get("KB_WHISPER_REVISION", "standard")
DEFAULT_DIAR_MODEL = os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")

MIN_SEG = float(os.environ.get("MIN_SEG", "0.35"))
MAX_SEG = float(os.environ.get("MAX_SEG", "30.0"))

MERGE_ENABLED = True
GAP_THRESHOLD = float(os.environ.get("GAP_THRESHOLD", "0.8"))
MAX_BLOCK_LEN = float(os.environ.get("MAX_BLOCK_LEN", "180.0"))
MERGE_SHORTER_THAN = float(os.environ.get("MERGE_SHORTER_THAN", "0.5"))


def run(cmd):
    subprocess.run(cmd, check=True)


def hhmmss(seconds):
    seconds = max(0.0, float(seconds))
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def convert_to_wav16k_mono(input_path, tmp_dir):
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


def cut_wav_segment(wav_path, out_path, start_s, end_s):
    run([
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i", str(wav_path),
        "-ac", "1",
        "-ar", "16000",
        str(out_path),
    ])



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


def split_long_segment(start, end, speaker, max_len):
    length = end - start
    if length <= max_len:
        return [{"start": start, "end": end, "speaker": speaker}]

    parts = []
    n = math.ceil(length / max_len)
    for i in range(n):
        s = start + i * max_len
        e = min(end, s + max_len)
        if e - s >= MIN_SEG:
            parts.append({"start": s, "end": e, "speaker": speaker})
    return parts


def resolve_asr_device(force_cpu=False):
    if force_cpu:
        return "cpu", torch.float32
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    return "cpu", torch.float32


def load_pyannote(model_id, diar_device):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if not token:
        raise RuntimeError(
            "Brak HF_TOKEN/HUGGINGFACE_TOKEN. "
            "Ustaw token Hugging Face przed uruchomieniem."
        )

    print(f"Loading pyannote diarization: {model_id}")
    print(f"Pyannote device: {diar_device}")

    pipe = PyannotePipeline.from_pretrained(model_id, token=token)

    if diar_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DIAR_DEVICE=cuda, ale CUDA nie jest dostępna.")
        pipe.to(torch.device("cuda"))
    elif diar_device == "cpu":
        pipe.to(torch.device("cpu"))
    else:
        raise RuntimeError("DIAR_DEVICE musi być 'cpu' albo 'cuda'.")

    return pipe


def load_kb_whisper(model_id, revision, force_cpu=False):
    device, torch_dtype = resolve_asr_device(force_cpu=force_cpu)

    print(f"Loading KB-Whisper model: {model_id}")
    print(f"Revision: {revision}")
    print(f"ASR device: {device}")
    print(f"Torch dtype: {torch_dtype}")

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

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    return asr, device


def transcribe_segment(asr, seg_path):
    result = asr(
        str(seg_path),
        generate_kwargs={
            "task": "transcribe",
            "language": "sv",
        },
    )
    return (result.get("text") or "").strip()


def join_text(a, b):
    a = (a or "").rstrip()
    b = (b or "").lstrip()
    if not a:
        return b
    if not b:
        return a
    return a + " " + b


def merge_segments(raw_segments):
    if not raw_segments:
        return []

    merged = []
    cur = dict(raw_segments[0])

    def seg_len(seg):
        return float(seg["end"] - seg["start"])

    for nxt in raw_segments[1:]:
        same_speaker = str(nxt["speaker"]) == str(cur["speaker"])
        gap = float(nxt["start"] - cur["end"])
        cur_len_after = float(nxt["end"] - cur["start"])
        short_next = seg_len(nxt) < MERGE_SHORTER_THAN

        if same_speaker and (gap <= GAP_THRESHOLD or short_next) and cur_len_after <= MAX_BLOCK_LEN:
            cur["end"] = nxt["end"]
            cur["end_hhmmss"] = hhmmss(cur["end"])
            cur["text"] = join_text(cur.get("text", ""), nxt.get("text", ""))
        else:
            merged.append(cur)
            cur = dict(nxt)

    merged.append(cur)
    return merged


def save_outputs(input_path, out_dir, raw_segments, merged_segments, model_id, revision, diar_model, elapsed_s):
    out_root = out_dir / "kb_diarization"
    raw_dir = out_root / "raw"
    merged_dir = out_root / "merged"
    raw_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    base = input_path.stem
    safe_model = model_id.replace("/", "_")

    raw_txt = raw_dir / f"{base}_{safe_model}_{revision}_raw.txt"
    raw_json = raw_dir / f"{base}_{safe_model}_{revision}_raw.json"
    merged_txt = merged_dir / f"{base}_{safe_model}_{revision}_merged.txt"
    merged_json = merged_dir / f"{base}_{safe_model}_{revision}_merged.json"

    with open(raw_txt, "w", encoding="utf-8") as f:
        f.write(f"# Input: {input_path}\n")
        f.write(f"# ASR model: {model_id}\n")
        f.write(f"# ASR revision: {revision}\n")
        f.write(f"# Diarization model: {diar_model}\n")
        f.write(f"# Elapsed seconds: {elapsed_s:.1f}\n\n")

        for seg in raw_segments:
            f.write(f"[{seg['start_hhmmss']}–{seg['end_hhmmss']}] {seg['speaker']}: {seg['text']}\n")

    with open(merged_txt, "w", encoding="utf-8") as f:
        f.write(f"# Input: {input_path}\n")
        f.write(f"# ASR model: {model_id}\n")
        f.write(f"# ASR revision: {revision}\n")
        f.write(f"# Diarization model: {diar_model}\n")
        f.write(f"# Elapsed seconds: {elapsed_s:.1f}\n\n")

        for seg in merged_segments:
            f.write(f"[{seg['start_hhmmss']}–{seg['end_hhmmss']}] {seg['speaker']}: {seg['text']}\n")

    payload_common = {
        "input": str(input_path),
        "asr_model": model_id,
        "asr_revision": revision,
        "diarization_model": diar_model,
        "elapsed_seconds": elapsed_s,
        "merge_params": {
            "enabled": MERGE_ENABLED,
            "gap_threshold_sec": GAP_THRESHOLD,
            "max_block_len_sec": MAX_BLOCK_LEN,
            "merge_shorter_than_sec": MERGE_SHORTER_THAN,
        },
    }

    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump({**payload_common, "segments": raw_segments}, f, ensure_ascii=False, indent=2)

    with open(merged_json, "w", encoding="utf-8") as f:
        json.dump({**payload_common, "segments": merged_segments}, f, ensure_ascii=False, indent=2)

    print("")
    print("Saved:")
    print(f"  {raw_txt}")
    print(f"  {merged_txt}")
    print(f"  {raw_json}")
    print(f"  {merged_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input audio/video file")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF ASR model id")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="HF ASR revision")
    parser.add_argument("--diar-model", default=DEFAULT_DIAR_MODEL, help="pyannote diarization model")
    parser.add_argument("--diar-device", default=os.environ.get("DIAR_DEVICE", "cpu"), choices=["cpu", "cuda"])
    parser.add_argument("--cpu-asr", action="store_true", help="Force KB-Whisper ASR on CPU")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.outdir).expanduser() if args.outdir else input_path.parent / "kb_whisper_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    with tempfile.TemporaryDirectory(prefix="kb_diar_") as td:
        tmp_dir = Path(td)
        seg_dir = tmp_dir / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)

        print(f"Converting to WAV 16 kHz mono: {input_path}")
        wav_path = convert_to_wav16k_mono(input_path, tmp_dir)

        diar_pipe = load_pyannote(args.diar_model, args.diar_device)

        print("")
        print("Running diarization...")
        diarization = diar_pipe(str(wav_path))

        # Zwolnij GPU/CPU pamięć po pyannote przed ładowaniem KB-Whisper.
        del diar_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        diar_segments = []
        for turn, _, speaker in iter_diarization_tracks(diarization):
            s = float(turn.start)
            e = float(turn.end)
            if e - s < MIN_SEG:
                continue
            diar_segments.extend(split_long_segment(s, e, str(speaker), MAX_SEG))

        diar_segments.sort(key=lambda x: x["start"])

        print(f"Speaker/time segments: {len(diar_segments)}")

        asr, asr_device = load_kb_whisper(args.model, args.revision, force_cpu=args.cpu_asr)

        raw_segments = []

        print("")
        print("Transcribing speaker segments...")
        for i, seg in enumerate(diar_segments, start=1):
            s = seg["start"]
            e = seg["end"]
            spk = seg["speaker"]

            seg_path = seg_dir / f"seg_{i:04d}_{spk}.wav"
            cut_wav_segment(wav_path, seg_path, s, e)

            text = transcribe_segment(asr, seg_path)

            item = {
                "start": s,
                "end": e,
                "start_hhmmss": hhmmss(s),
                "end_hhmmss": hhmmss(e),
                "speaker": spk,
                "text": text,
            }
            raw_segments.append(item)

            print(f"[{item['start_hhmmss']}–{item['end_hhmmss']}] {spk}: {text}")

        merged_segments = merge_segments(raw_segments) if MERGE_ENABLED else list(raw_segments)

        elapsed = time.time() - start_time
        save_outputs(
            input_path=input_path,
            out_dir=out_dir,
            raw_segments=raw_segments,
            merged_segments=merged_segments,
            model_id=args.model,
            revision=args.revision,
            diar_model=args.diar_model,
            elapsed_s=elapsed,
        )

    print("")
    print("*** Done ***")


if __name__ == "__main__":
    main()
