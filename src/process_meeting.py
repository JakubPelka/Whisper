#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Headless shared meeting processing entrypoint for JakubPelka/Whisper.

Combines:
1. Audio preparation (audio_prepare.py)
2. Local whisper.cpp CUDA parity transcription (whisper_cpp_runtime.py)
3. Shared Luna/Terra note generation core (generate_meeting_note.py)
4. Deterministic multi-format rendering (DOCX, PDF, TXT, MD)
5. Content-free provenance / audit metadata (provenance.json)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

# Add src to pythonpath if needed
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audio_prepare import prepared_audio
from generate_meeting_note import (
    create_note_with_api,
    ensure_api_key,
    render_docx,
    render_note_markdown,
    render_pdf,
)
from whisper_cpp_runtime import transcribe_with_whisper_cpp

LOGGER = logging.getLogger("process_meeting")


def get_git_commit_sha(repo_path: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def report_progress(percent: int, message: str, callback: Callable[[int, str], None] | None = None) -> None:
    LOGGER.info("PROGRESS [%d%%]: %s", percent, message)
    print(f"PROGRESS: {percent} {message}", flush=True)
    if callback:
        try:
            callback(percent, message)
        except Exception:
            pass


def process_meeting(
    input_file: Path | str,
    recording_language: str = "auto",
    note_type: str = "serviceNote",
    note_language: str = "pl",
    context: str | None = None,
    vocabulary: str | None = None,
    output_dir: Path | str | None = None,
    openai_env_file: Path | str | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Path]:
    """Headless entrypoint for full audio-to-note pipeline."""

    input_path = Path(input_file).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        out_dir = Path("output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = SRC_DIR.parent
    if openai_env_file:
        env_f = Path(openai_env_file)
    else:
        env_f = repo_root / "secrets" / "openai.env"
    ensure_api_key(env_f if env_f.is_file() else None)

    report_progress(5, "Preparing audio...", progress_callback)
    if cancel_checker and cancel_checker():
        raise RuntimeError("Cancelled before audio preparation.")

    with prepared_audio(input_path) as prep:
        report_progress(20, "Transcribing audio with whisper.cpp CUDA...", progress_callback)
        if cancel_checker and cancel_checker():
            raise RuntimeError("Cancelled before transcription.")

        transcription_result = transcribe_with_whisper_cpp(
            audio_path=prep.path,
            language=recording_language,
            initial_prompt=vocabulary,
            work_dir=out_dir,
            cancel_checker=cancel_checker,
        )

    segments = transcription_result["segments"]
    detected_lang = transcription_result["language"]
    model_info = transcription_result["model_info"]
    runtime_info = transcription_result["runtime_info"]

    report_progress(50, "Saving raw transcription output...", progress_callback)
    transcript_json_path = out_dir / "transcript.json"
    transcript_txt_path = out_dir / "transcript.txt"

    with open(transcript_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": detected_lang,
                "segments": segments,
                "model_info": model_info,
                "runtime_info": runtime_info,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(transcript_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Language: {detected_lang}\n\n")
        for seg in segments:
            f.write(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}\n")

    report_progress(60, "Generating meeting note via Luna/Terra shared core...", progress_callback)
    if cancel_checker and cancel_checker():
        raise RuntimeError("Cancelled before note generation.")

    api_segments = [
        {
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": str(s["text"]),
        }
        for s in segments
    ]

    generation_result = create_note_with_api(
        segments=api_segments,
        language=note_language,
        note_preset=note_type,
        meeting_context=context,
    )

    final_note = generation_result.final_note

    report_progress(85, "Rendering output documents (DOCX, PDF, TXT, MD)...", progress_callback)

    docx_path = out_dir / "note.docx"
    pdf_path = out_dir / "note.pdf"
    txt_path = out_dir / "note.txt"
    md_path = out_dir / "note.md"
    provenance_path = out_dir / "provenance.json"

    render_docx(
        note=final_note,
        output_path=docx_path,
        source_name=input_path.name,
        language=note_language,
        note_preset=note_type,
    )
    render_pdf(
        note=final_note,
        output_path=pdf_path,
        source_name=input_path.name,
        language=note_language,
        note_preset=note_type,
    )

    md_content = render_note_markdown(final_note, language=note_language, note_preset=note_type)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    commit_sha = get_git_commit_sha(repo_root)

    provenance_data = {
        "whisper_repo_commit": commit_sha,
        "whisper_cpp_version": runtime_info["version"],
        "backend": runtime_info["backend"],
        "model_id": model_info["id"],
        "model_file": model_info["file"],
        "model_sha256": model_info["sha256"],
        "quantization": model_info["quantization"],
        "recording_language_requested": recording_language,
        "recording_language_detected": detected_lang,
        "note_preset": note_type,
        "note_language": note_language,
        "has_context": bool(context and context.strip()),
        "has_vocabulary": bool(vocabulary and vocabulary.strip()),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance_data, f, indent=2)

    report_progress(100, "Meeting note processing completed.", progress_callback)

    return {
        "docx": docx_path,
        "pdf": pdf_path,
        "txt": txt_path,
        "md": md_path,
        "transcript_json": transcript_json_path,
        "transcript_txt": transcript_txt_path,
        "provenance": provenance_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Headless meeting transcription and note generation."
    )
    parser.add_argument("--input", required=True, help="Input audio or video file path")
    parser.add_argument(
        "--recording-language",
        default="auto",
        choices=["sv", "pl", "en", "auto"],
        help="Recording language (default: auto)",
    )
    parser.add_argument(
        "--note-type",
        default="serviceNote",
        choices=["shortSummary", "conversationNote", "serviceNote"],
        help="Note type preset (default: serviceNote)",
    )
    parser.add_argument(
        "--note-language",
        default="pl",
        choices=["sv", "pl", "en"],
        help="Target note language (default: pl)",
    )
    parser.add_argument("--context", default=None, help="Optional background context")
    parser.add_argument(
        "--vocabulary", default=None, help="Optional transcription vocabulary / terminology hint"
    )
    parser.add_argument(
        "--outdir", required=True, help="Output directory for generated artifacts"
    )
    parser.add_argument("--openai-env", default=None, help="Path to custom openai.env file")

    args = parser.parse_args()

    try:
        process_meeting(
            input_file=args.input,
            recording_language=args.recording_language,
            note_type=args.note_type,
            note_language=args.note_language,
            context=args.context,
            vocabulary=args.vocabulary,
            output_dir=args.outdir,
            openai_env_file=args.openai_env,
        )
        return 0
    except Exception as e:
        LOGGER.exception("Failed to process meeting: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
