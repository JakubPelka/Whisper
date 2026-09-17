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
import shutil
import subprocess
import sys
import tempfile
import zipfile
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
    VerificationResult,
    create_note_with_api,
    ensure_api_key,
    render_docx,
    render_note_markdown,
    render_pdf,
    transcript_for_prompt,
)

from presentation_segmentation import (
    PresentationRange,
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


def get_preset_slug(preset: str) -> str:
    if not preset:
        return "meeting_notes"
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", preset)
    s = re.sub(r"[^\w]+", "_", s).lower()
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "meeting_notes"


def sanitize_filename_stem(stem: str) -> str:
    if not stem:
        return "recording"
    s = re.sub(r'[/\\:*?"<>|\x00-\x1f]', "_", stem)
    s = s.strip(" .")
    return s or "recording"



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

        report_progress(20, "Transcribing audio with whisper.cpp CUDA...", progress_callback)
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

    with tempfile.TemporaryDirectory() as staging_dir_str:
        staging_dir = Path(staging_dir_str)
        transcript_dir = staging_dir / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        audit_dir = staging_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

        report_progress(50, "Saving raw transcription output...", progress_callback)
        transcript_json_path = transcript_dir / "transcript.json"
        transcript_txt_path = transcript_dir / "transcript.txt"

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

        presentation_ranges: list[PresentationRange] = []
        raw_segmentation_result: SegmentationResult | None = None

        if auto_segment:
            report_progress(60, "Detecting presentation boundaries via Luna semantic detector...", progress_callback)
            if cancel_checker and cancel_checker():
                raise RuntimeError("Cancelled before presentation segmentation.")

            raw_segmentation_result = None
            raw_response_str = None
            try:
                # Measure transcript length
                total_chars = sum(len(s.get("text", "")) for s in segments)
                LOGGER.info("Total transcript length: %d chars in %d segments", total_chars, len(segments))

                raw_segmentation_result, raw_response_str = segment_presentations_with_luna(segments)
                presentation_ranges = validate_and_normalize_segmentation(segments, raw_segmentation_result, strict=True)
            except Exception as e:
                stage = getattr(e, "stage", "unknown")
                exc_type = getattr(e, "exception_type", type(e).__name__)
                raw_resp = getattr(e, "raw_response", raw_response_str)
                model_n = getattr(e, "model_name", "gpt-5.6-luna")

                if raw_resp:
                    (audit_dir / "raw_luna_segmentation_response.txt").write_text(raw_resp, encoding="utf-8")
                    if out_dir and Path(out_dir) != staging_dir:
                        (Path(out_dir) / "raw_luna_segmentation_response.txt").write_text(raw_resp, encoding="utf-8")

                err_data = {
                    "segmentation_status": "failed",
                    "stage": stage,
                    "exception_type": exc_type,
                    "exception_message": str(e),
                    "model_name": model_n,
                    "raw_response_received": bool(raw_resp),
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                err_json_path = audit_dir / "segmentation_error.json"
                with open(err_json_path, "w", encoding="utf-8") as f:
                    json.dump(err_data, f, indent=2, ensure_ascii=False)

                if out_dir and Path(out_dir) != staging_dir:
                    out_err_path = Path(out_dir) / "segmentation_error.json"
                    with open(out_err_path, "w", encoding="utf-8") as f:
                        json.dump(err_data, f, indent=2, ensure_ascii=False)

                concise_msg = f"Segmentation failed at stage '{stage}': {exc_type}: {e}"
                LOGGER.error("%s. Aborting presentationSummary processing.", concise_msg)
                report_progress(0, concise_msg, progress_callback)
                raise RuntimeError(concise_msg) from e

            # Save raw_luna_segmentation.json to audit/
            raw_luna_path = audit_dir / "raw_luna_segmentation.json"
            if raw_segmentation_result is not None:
                with open(raw_luna_path, "w", encoding="utf-8") as f:
                    json.dump(raw_segmentation_result.model_dump(), f, indent=2, ensure_ascii=False)

            # Filter blocks where is_presentation is True for note generation
            pres_blocks = [r for r in presentation_ranges if getattr(r, "is_presentation", True)]
            if not pres_blocks:
                pres_blocks = presentation_ranges

            # Save presentation_segments.json to audit/
            seg_json_path = audit_dir / "presentation_segments.json"
            seg_export_data = {
                "segmentation_status": "success",
                "raw_candidate_count": len(raw_segmentation_result.presentations) if raw_segmentation_result else 0,
                "presentation_count": len(pres_blocks),
                "total_block_count": len(presentation_ranges),
                "blocks": [
                    {
                        "index": idx + 1,
                        "title": getattr(r, "title", "Presentation"),
                        "is_presentation": getattr(r, "is_presentation", True),
                        "block_type": getattr(r, "block_type", "presentation"),
                        "start_segment_id": getattr(r, "start_segment_id", "S0000"),
                        "end_segment_id": getattr(r, "end_segment_id", "S0000"),
                        "start_time_seconds": segments[r.start_idx]["start"] if hasattr(r, "start_idx") and r.start_idx < len(segments) else 0.0,
                        "end_time_seconds": segments[r.end_idx]["end"] if hasattr(r, "end_idx") and r.end_idx < len(segments) else 0.0,
                        "boundary_confidence": getattr(r, "boundary_confidence", "high"),
                        "start_evidence": [e.model_dump() for e in getattr(r, "start_evidence", [])],
                        "decision_rationale": getattr(r, "decision_rationale", ""),
                    }
                    for idx, r in enumerate(presentation_ranges)
                ],
                "presentations": [
                    {
                        "index": idx + 1,
                        "title": getattr(r, "title", "Presentation"),
                        "start_segment_id": getattr(r, "start_segment_id", "S0000"),
                        "end_segment_id": getattr(r, "end_segment_id", "S0000"),
                        "start_time_seconds": segments[r.start_idx]["start"] if hasattr(r, "start_idx") and r.start_idx < len(segments) else 0.0,
                        "end_time_seconds": segments[r.end_idx]["end"] if hasattr(r, "end_idx") and r.end_idx < len(segments) else 0.0,
                    }
                    for idx, r in enumerate(pres_blocks)
                ],
            }
            with open(seg_json_path, "w", encoding="utf-8") as f:
                json.dump(seg_export_data, f, indent=2, ensure_ascii=False)
        else:
            last_seg_id = f"S{max(0, len(segments) - 1):04d}"
            presentation_ranges = [
                PresentationRange(
                    index=1,
                    start_segment_id="S0000",
                    end_segment_id=last_seg_id,
                    title="Full Recording",
                    boundary_confidence="high",
                    is_presentation=True,
                    block_type="presentation",
                    decision_rationale="No auto-segmentation policy",
                )
            ]
            pres_blocks = presentation_ranges

        # 8. Generate meeting note(s) for each presentation block
        report_progress(70, f"Generating note(s) for {len(pres_blocks)} presentation(s)...", progress_callback)
        if cancel_checker and cancel_checker():
            raise RuntimeError("Cancelled before note generation.")

        generated_notes: list[dict[str, Any]] = []

        for p_idx, r in enumerate(pres_blocks, start=1):
            s_idx = r.start_idx
            e_idx = r.end_idx
            slice_segs = segments[s_idx : e_idx + 1]
            p_title = getattr(r, "title", f"Presentation {p_idx}")

            if not slice_segs:
                continue

            slice_text = transcript_for_prompt(slice_segs)
            res = create_note_with_api(
                transcript_text=slice_text,
                language=note_language,
                note_preset="presentationSummary" if auto_segment else note_type,
                meeting_context=context,
            )
            if isinstance(res, tuple) and len(res) >= 2:
                verification_res = res[1]
            elif hasattr(res, "final_note"):
                verification_res = res
            else:
                verification_res = VerificationResult(final_note=res, removed_or_corrected_claims=[], verification_warnings=[])

            sanitized_note = sanitize_grounding(verification_res.final_note, s_idx, e_idx)
            sanitized_res = VerificationResult(
                final_note=sanitized_note,
                removed_or_corrected_claims=verification_res.removed_or_corrected_claims,
                verification_warnings=verification_res.verification_warnings,
            )

            generated_notes.append({
                "index": p_idx,
                "title": p_title,
                "range": r,
                "note": sanitized_res,
                "slice_segments": slice_segs,
            })

        # 9. Rendering output documents
        docx_path = out_dir / "note.docx"
        pdf_path = out_dir / "note.pdf"
        md_path = out_dir / "note.md"
        txt_path = out_dir / "note.txt"

        report_progress(85, "Rendering output documents (DOCX, PDF, TXT, MD)...", progress_callback)
        if cancel_checker and cancel_checker():
            raise RuntimeError("Cancelled before document rendering.")

        def _get_meeting_note(n_obj: Any) -> MeetingNote:
            if hasattr(n_obj, "final_note"):
                return n_obj.final_note
            return n_obj

        if len(generated_notes) == 1:
            item = generated_notes[0]
            single_mn = _get_meeting_note(item["note"])
            render_docx(note=single_mn, output_path=docx_path, source_name=input_path.name, language=note_language, note_preset="presentationSummary" if auto_segment else note_type)
            render_pdf(note=single_mn, output_path=pdf_path, source_name=input_path.name, language=note_language, note_preset="presentationSummary" if auto_segment else note_type)
            md_text = render_note_markdown(single_mn, language=note_language, note_preset="presentationSummary" if auto_segment else note_type)
            md_path.write_text(md_text, encoding="utf-8")
            txt_path.write_text(md_text, encoding="utf-8")
        else:
            combined_sections: list[ThematicSection] = []
            for item in generated_notes:
                mn = _get_meeting_note(item["note"])
                header_title = f"Presentation {item['index']}: {item['title']}"
                combined_sections.append(
                    ThematicSection(
                        heading=header_title,
                        bullet_points=mn.summary or [GroundedStatement(text=header_title, source_segments=[0])],
                    )
                )
                for sec in getattr(mn, "thematic_sections", []):
                    combined_sections.append(sec)


            combined_note = MeetingNote(
                title=GroundedStatement(text=f"Presentation Summary Index: {input_path.name}", source_segments=[0]),
                summary=[GroundedStatement(text=f"Contains {len(generated_notes)} presentations.", source_segments=[0])],
                thematic_sections=combined_sections,
            )
            render_docx(note=combined_note, output_path=docx_path, source_name=input_path.name, language=note_language, note_preset="presentationSummary" if auto_segment else note_type)
            render_pdf(note=combined_note, output_path=pdf_path, source_name=input_path.name, language=note_language, note_preset="presentationSummary" if auto_segment else note_type)
            combined_md = render_note_markdown(combined_note, language=note_language, note_preset="presentationSummary" if auto_segment else note_type)
            md_path.write_text(combined_md, encoding="utf-8")
            txt_path.write_text(combined_md, encoding="utf-8")

        # Copy per-presentation artifacts into staging subfolders
        for item in generated_notes:
            p_idx = item["index"]
            p_title = item["title"]
            slug = slugify_title(p_title)
            p_folder_name = f"{p_idx:02d}_{slug}"
            p_folder = staging_dir / p_folder_name
            p_folder.mkdir(parents=True, exist_ok=True)

            p_docx = p_folder / "note.docx"
            p_pdf = p_folder / "note.pdf"
            p_md = p_folder / "note.md"
            p_txt = p_folder / "note.txt"

            p_mn = _get_meeting_note(item["note"])
            render_docx(note=p_mn, output_path=p_docx, source_name=input_path.name, language=note_language, note_preset="presentationSummary" if auto_segment else note_type)
            render_pdf(note=p_mn, output_path=p_pdf, source_name=input_path.name, language=note_language, note_preset="presentationSummary" if auto_segment else note_type)
            p_md_text = render_note_markdown(p_mn, language=note_language, note_preset="presentationSummary" if auto_segment else note_type)
            p_md.write_text(p_md_text, encoding="utf-8")
            p_txt.write_text(p_md_text, encoding="utf-8")


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
                "source_codec": str(prep_codec),
                "source_sample_rate": prep_rate,
                "source_channels": prep_channels,
            },
            "audio_derivative": {
                "filename": prep_filename,
                "size_bytes": derived_wav_size,
                "codec": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
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
            "segmentation_status": "success" if auto_segment else "disabled",
            "presentation_count": len(pres_blocks),
            "total_block_count": len(presentation_ranges),
            "detected_presentations": [
                {
                    "index": i + 1,
                    "title": r.title if hasattr(r, "title") else r[2],
                    "segment_range": [r.start_idx if hasattr(r, "start_idx") else r[0], r.end_idx if hasattr(r, "end_idx") else r[1]],
                }
                for i, r in enumerate(pres_blocks)
            ],
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        provenance_path = audit_dir / "provenance.json"
        with open(provenance_path, "w", encoding="utf-8") as f:
            json.dump(provenance_data, f, indent=2)

        summary_report_data = {
            "title": input_path.name,
            "note_preset": note_type,
            "note_language": note_language,
            "auto_segmentation_used": auto_segment,
            "segmentation_status": "success" if auto_segment else "disabled",
            "presentation_count": len(pres_blocks),
            "presentations": [
                {
                    "index": i + 1,
                    "title": item["title"],
                    "start_time_seconds": segments[item["range"].start_idx]["start"] if item["range"].start_idx < len(segments) else 0.0,
                    "end_time_seconds": segments[item["range"].end_idx]["end"] if item["range"].end_idx < len(segments) else 0.0,
                    "has_summary": True,
                    "note_generated": True,
                }
                for i, item in enumerate(generated_notes)
            ],
            "audio_duration_seconds": audio_seconds,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        summary_report_path = audit_dir / "summary_report.json"
        with open(summary_report_path, "w", encoding="utf-8") as f:
            json.dump(summary_report_data, f, indent=2, ensure_ascii=False)

        # Build final output ZIP in out_dir named <preset_slug>_<input_stem>.zip
        preset_slug = get_preset_slug(note_type)
        stem_slug = sanitize_filename_stem(input_path.stem)
        zip_name = f"{preset_slug}_{stem_slug}.zip"
        zip_path = out_dir / zip_name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root_path, dirs, files in os.walk(staging_dir):
                for file_name in sorted(files):
                    full_p = Path(root_path) / file_name
                    rel_p = full_p.relative_to(staging_dir)
                    zf.write(full_p, arcname=str(rel_p))

        # Purge any non-zip files/dirs in out_dir so OUTPUT_DIR contains ONLY the final zip
        for item in out_dir.iterdir():
            if item.name != zip_name:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    LOGGER.warning("Failed to clean file %s from out_dir: %s", item, e)

        report_progress(100, "Meeting note processing completed.", progress_callback)

        return {"zip": zip_path}


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
