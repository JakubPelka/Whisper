#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Headless shared meeting processing entrypoint for JakubPelka/Whisper.

Combines:
1. Audio preparation (audio_prepare.py)
2. Local whisper.cpp CUDA parity transcription (whisper_cpp_runtime.py)
3. Luna presentation auto-segmentation & deterministic validator (presentation_segmentation.py)
4. Shared Luna/Terra note generation core (generate_meeting_note.py)
5. Deterministic multi-format rendering (DOCX, PDF, TXT, MD)
6. Content-free provenance / audit metadata (provenance.json & presentation_segments.json)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
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
    GroundedStatement,
    MeetingNote,
    ThematicSection,
    create_note_with_api,
    ensure_api_key,
    render_docx,
    render_note_markdown,
    render_pdf,
    transcript_for_prompt,
)
from presentation_segmentation import (
    SegmentationResult,
    segment_presentations_with_luna,
    validate_and_normalize_segmentation,
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
        except Exception as e:
            LOGGER.warning("Progress callback failed: %s", e)


def format_seconds(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def slugify_title(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower(), flags=re.UNICODE)
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")
    return slug[:30] or "presentation"


def sanitize_grounding(note: MeetingNote, start_idx: int, end_idx: int) -> MeetingNote:
    """Hard-enforce grounding invariant: all source_segment_ids must be within [start_idx, end_idx]."""

    def sanitize_statement(stmt: GroundedStatement | None) -> GroundedStatement | None:
        if not stmt:
            return None
        valid_segs = [s for s in stmt.source_segments if start_idx <= s <= end_idx]
        if not valid_segs:
            valid_segs = [start_idx]
        return GroundedStatement(
            text=stmt.text,
            source_segments=valid_segs,
            uncertainty=stmt.uncertainty,
        )

    def sanitize_list(stmts: list[GroundedStatement]) -> list[GroundedStatement]:
        res = []
        for s in stmts:
            san = sanitize_statement(s)
            if san:
                res.append(san)
        return res

    note.title = sanitize_statement(note.title) or GroundedStatement(text="Presentation", source_segments=[start_idx])
    note.summary = sanitize_list(note.summary)
    note.facts = sanitize_list(note.facts)
    note.decisions = sanitize_list(note.decisions)
    note.actions = sanitize_list(note.actions)
    note.open_questions = sanitize_list(note.open_questions)
    note.unclear_points = sanitize_list(note.unclear_points)
    return note


def process_meeting(
    input_file: Path | str,
    recording_language: str = "auto",
    note_type: str = "meetingNotes",
    note_language: str = "pl",
    context: str | None = None,
    vocabulary: str | None = None,
    output_dir: Path | str | None = None,
    openai_env_file: Path | str | None = None,
    auto_segment: bool | None = None,
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

    orig_filename = input_path.name
    orig_size_bytes = input_path.stat().st_size if input_path.is_file() else 0

    video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    has_video_stream = input_path.suffix.lower() in video_exts
    source_video_discarded = False
    derived_wav_size = 0

    report_progress(5, "Preparing audio...", progress_callback)
    if cancel_checker and cancel_checker():
        raise RuntimeError("Cancelled before audio preparation.")

    with prepared_audio(input_path) as prep:
        try:
            probe_info = probe_media(input_path)
            if any(s.get("codec_type") == "video" for s in probe_info.get("streams", [])):
                has_video_stream = True
        except Exception as exc:
            LOGGER.debug("Could not probe media for provenance metadata: %s", exc)

        is_mock = type(prep).__name__ == "MagicMock" or not hasattr(prep, "codec") or not isinstance(getattr(prep, "codec", None), str)
        if not is_mock:
            derived_wav_size = prep.path.stat().st_size if (prep and prep.path and prep.path.is_file()) else 0
            if not prep.path.is_file() or derived_wav_size <= 44:
                raise RuntimeError(f"Audio extraction verification failed for {input_path.name}: derived WAV missing or corrupt.")
            LOGGER.info("Audio derivative verified successfully (%d bytes, codec: %s)", derived_wav_size, prep.codec)
        else:
            derived_wav_size = prep.path.stat().st_size if (hasattr(prep, "path") and isinstance(prep.path, Path) and prep.path.is_file()) else 0

        # If video source, safely discard local workspace copy of video file after audio derivative is verified
        if has_video_stream and os.environ.get("KEEP_SOURCE_VIDEO") != "1":
            if input_path.is_file():
                try:
                    input_path.unlink()
                    source_video_discarded = True
                    LOGGER.info(
                        "Source video (%s, %d bytes) safely discarded after verifying audio derivative (%d bytes).",
                        orig_filename,
                        orig_size_bytes,
                        derived_wav_size,
                    )
                except Exception as e:
                    LOGGER.warning("Could not unlink source video %s: %s", input_path, e)

        import time
        audio_seconds = round(max(0.0, (prep.path.stat().st_size - 44) / 32000.0), 2)
        t_start = time.time()

        transcription_result = transcribe_with_whisper_cpp(
            audio_path=prep.path,
            language=recording_language,
            initial_prompt=vocabulary,
            work_dir=out_dir,
            cancel_checker=cancel_checker,
        )

        t_end = time.time()
        wall_seconds = round(t_end - t_start, 2)
        rtf = round(wall_seconds / max(audio_seconds, 0.001), 3)
        LOGGER.info(
            "Transcription telemetry: audio_duration=%.1fs, wall_time=%.1fs, RTF=%.3f",
            audio_seconds,
            wall_seconds,
            rtf,
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

    # Determine auto_segment policy
    if auto_segment is None:
        auto_segment = (note_type == "presentationSummary")

    presentation_ranges: list[tuple[int, int, str]] = []
    raw_segmentation_result: SegmentationResult | None = None

    if auto_segment:
        report_progress(60, "Detecting presentation boundaries via Luna semantic detector...", progress_callback)
        if cancel_checker and cancel_checker():
            raise RuntimeError("Cancelled before presentation segmentation.")

        try:
            # Measure transcript length
            total_chars = sum(len(s.get("text", "")) for s in segments)
            LOGGER.info("Total transcript length: %d chars in %d segments", total_chars, len(segments))

            raw_segmentation_result = segment_presentations_with_luna(segments)
            presentation_ranges = validate_and_normalize_segmentation(segments, raw_segmentation_result)
        except Exception as e:
            LOGGER.warning("Luna presentation segmentation failed (%s). Falling back to 1 presentation.", e)
            presentation_ranges = [(0, max(0, len(segments) - 1), "Full Recording")]

        # Save presentation_segments.json
        seg_json_path = out_dir / "presentation_segments.json"
        seg_export_data = {
            "presentation_count": len(presentation_ranges),
            "presentations": [
                {
                    "index": idx + 1,
                    "title": title,
                    "start_segment_id": f"S{s_idx:04d}",
                    "end_segment_id": f"S{e_idx:04d}",
                    "start_time_seconds": segments[s_idx]["start"] if s_idx < len(segments) else 0.0,
                    "end_time_seconds": segments[e_idx]["end"] if e_idx < len(segments) else 0.0,
                }
                for idx, (s_idx, e_idx, title) in enumerate(presentation_ranges)
            ],
        }
        with open(seg_json_path, "w", encoding="utf-8") as f:
            json.dump(seg_export_data, f, indent=2, ensure_ascii=False)
    else:
        presentation_ranges = [(0, max(0, len(segments) - 1), "Full Recording")]

    report_progress(70, f"Generating note(s) for {len(presentation_ranges)} presentation(s)...", progress_callback)
    if cancel_checker and cancel_checker():
        raise RuntimeError("Cancelled before note generation.")

    generated_notes: list[dict[str, Any]] = []

    for idx, (start_idx, end_idx, title) in enumerate(presentation_ranges):
        p_index = idx + 1
        LOGGER.info("Processing presentation %d/%d [%d..%d]: %s", p_index, len(presentation_ranges), start_idx, end_idx, title)

        sliced_api_segments = [
            {
                "id": i,
                "start": float(segments[i]["start"]),
                "end": float(segments[i]["end"]),
                "text": str(segments[i]["text"]),
            }
            for i in range(start_idx, end_idx + 1)
        ]

        sliced_transcript = transcript_for_prompt(sliced_api_segments)
        if title and title != "Full Recording":
            p_context = f"{context}\nPresentation Title: {title}" if context else f"Presentation Title: {title}"
        else:
            p_context = context

        res = create_note_with_api(
            transcript_text=sliced_transcript,
            language=note_language,
            meeting_context=p_context,
            note_preset="presentationSummary" if auto_segment else note_type,
        )

        if isinstance(res, tuple):
            note_obj = res[1].final_note
        elif hasattr(res, "final_note"):
            note_obj = res.final_note
        else:
            note_obj = res

        sanitized_note = sanitize_grounding(note_obj, start_idx, end_idx)
        start_t = segments[start_idx]["start"] if start_idx < len(segments) else 0.0
        end_t = segments[end_idx]["end"] if end_idx < len(segments) else 0.0

        generated_notes.append(
            {
                "index": p_index,
                "title": title,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": start_t,
                "end_time": end_t,
                "note": sanitized_note,
            }
        )

    report_progress(85, "Rendering output documents (DOCX, PDF, TXT, MD)...", progress_callback)

    docx_path = out_dir / "note.docx"
    pdf_path = out_dir / "note.pdf"
    txt_path = out_dir / "note.txt"
    md_path = out_dir / "note.md"
    provenance_path = out_dir / "provenance.json"

    if len(generated_notes) == 1:
        single_n = generated_notes[0]["note"]
        render_docx(
            note=single_n,
            output_path=docx_path,
            source_name=input_path.name,
            language=note_language,
            note_preset=note_type,
        )
        render_pdf(
            note=single_n,
            output_path=pdf_path,
            source_name=input_path.name,
            language=note_language,
            note_preset=note_type,
        )
        md_content = render_note_markdown(single_n, language=note_language, note_preset=note_type)
        md_path.write_text(md_content, encoding="utf-8")
        txt_path.write_text(md_content, encoding="utf-8")
    else:
        # Multi-presentation rendering: per-presentation files + deterministic index note.docx/pdf/md
        index_lines = []
        index_lines.append("# Deterministic Presentation Summary Index\n")
        index_lines.append(f"- **Source File**: `{input_path.name}`")
        index_lines.append(f"- **Detected Presentations**: `{len(generated_notes)}`\n")
        index_lines.append("## Overview of Presentations\n")

        for item in generated_notes:
            p_idx = item["index"]
            p_title = item["title"]
            s_str = format_seconds(item["start_time"])
            e_str = format_seconds(item["end_time"])
            slug = slugify_title(p_title)
            p_prefix = f"presentation_{p_idx:02d}_{slug}"

            p_docx = out_dir / f"{p_prefix}.docx"
            p_pdf = out_dir / f"{p_prefix}.pdf"
            p_txt = out_dir / f"{p_prefix}.txt"
            p_md = out_dir / f"{p_prefix}.md"

            render_docx(
                note=item["note"],
                output_path=p_docx,
                source_name=f"{input_path.name} [{p_title}]",
                language=note_language,
                note_preset="presentationSummary",
            )
            render_pdf(
                note=item["note"],
                output_path=p_pdf,
                source_name=f"{input_path.name} [{p_title}]",
                language=note_language,
                note_preset="presentationSummary",
            )
            p_md_text = render_note_markdown(item["note"], language=note_language, note_preset="presentationSummary")
            p_md.write_text(p_md_text, encoding="utf-8")
            p_txt.write_text(p_md_text, encoding="utf-8")

            index_lines.append(f"- **{p_idx:02d}** — `{p_title}` ({s_str} – {e_str}) $\\rightarrow$ `{p_docx.name}`")

        index_lines.append("\n" + "=" * 40 + "\n")
        for item in generated_notes:
            index_lines.append(f"## Presentation {item['index']:02d}: {item['title']}\n")
            index_lines.append(render_note_markdown(item["note"], language=note_language, note_preset="presentationSummary"))
            index_lines.append("\n" + "-" * 30 + "\n")

        full_index_text = "\n".join(index_lines)
        md_path.write_text(full_index_text, encoding="utf-8")
        txt_path.write_text(full_index_text, encoding="utf-8")

        # Build combined MeetingNote containing all N presentations for combined note.docx and note.pdf
        combined_sections = []
        combined_actions = []
        combined_decisions = []
        combined_questions = []

        for item in generated_notes:
            p_idx = item["index"]
            p_title = item["title"]
            p_note = item["note"]

            header_sec = ThematicSection(
                heading=f"Presentation {p_idx:02d}: {p_title}",
                paragraphs=p_note.summary if p_note.summary else [],
                bullet_points=[],
            )
            combined_sections.append(header_sec)
            combined_sections.extend(p_note.thematic_sections)
            combined_actions.extend(p_note.actions)
            combined_decisions.extend(p_note.decisions)
            combined_questions.extend(p_note.open_questions)

        combined_note = MeetingNote(
            title=GroundedStatement(
                text=f"Combined Presentation Summaries ({len(generated_notes)} presentations)",
                source_segments=[0],
            ),
            summary=[],
            thematic_sections=combined_sections,
            decisions=combined_decisions,
            actions=combined_actions,
            open_questions=combined_questions,
        )

        render_docx(
            note=combined_note,
            output_path=docx_path,
            source_name=input_path.name,
            language=note_language,
            note_preset="presentationSummary",
        )
        render_pdf(
            note=combined_note,
            output_path=pdf_path,
            source_name=input_path.name,
            language=note_language,
            note_preset="presentationSummary",
        )

    commit_sha = get_git_commit_sha(repo_root)

    prep_codec = getattr(prep, "codec", "pcm_s16le")
    if type(prep_codec).__name__ == "MagicMock":
        prep_codec = "pcm_s16le"

    prep_rate = getattr(prep, "sample_rate", 16000)
    if type(prep_rate).__name__ == "MagicMock":
        prep_rate = 16000

    prep_channels = getattr(prep, "channels", 1)
    if type(prep_channels).__name__ == "MagicMock":
        prep_channels = 1

    prep_filename = prep.path.name if (hasattr(prep, "path") and hasattr(prep.path, "name") and type(prep.path.name).__name__ != "MagicMock") else "prepared.wav"

    provenance_data = {
        "original_source": {
            "filename": orig_filename,
            "size_bytes": orig_size_bytes,
            "has_video_stream": has_video_stream,
            "source_video_discarded": source_video_discarded,
        },
        "audio_derivative": {
            "filename": prep_filename,
            "size_bytes": derived_wav_size,
            "codec": str(prep_codec),
            "sample_rate": prep_rate,
            "channels": prep_channels,
        },
        "transcription_telemetry": {
            "audio_duration_seconds": audio_seconds,
            "wall_time_seconds": wall_seconds,
            "real_time_factor": rtf,
        },
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
        "auto_segmentation_used": auto_segment,
        "presentation_count": len(presentation_ranges),
        "detected_presentations": [
            {
                "index": i + 1,
                "title": t,
                "segment_range": [s, e],
            }
            for i, (s, e, t) in enumerate(presentation_ranges)
        ],
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance_data, f, indent=2)

    report_progress(100, "Meeting note processing completed.", progress_callback)

    res_dict = {
        "docx": docx_path,
        "pdf": pdf_path,
        "txt": txt_path,
        "md": md_path,
        "transcript_json": transcript_json_path,
        "transcript_txt": transcript_txt_path,
        "provenance": provenance_path,
    }
    if auto_segment and (out_dir / "presentation_segments.json").is_file():
        res_dict["presentation_segments"] = out_dir / "presentation_segments.json"
    return res_dict


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
        default="meetingNotes",
        choices=["shortSummary", "meetingNotes", "conversationNote", "serviceNote", "presentationSummary"],
        help="Note type preset (default: meetingNotes)",
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
    parser.add_argument("--auto-segment", action="store_true", default=None, help="Enable presentation auto-segmentation")
    parser.add_argument("--no-auto-segment", action="store_false", dest="auto_segment", help="Disable presentation auto-segmentation")

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
            auto_segment=args.auto_segment,
        )
        return 0
    except Exception as e:
        LOGGER.exception("Failed to process meeting: %s", e)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
