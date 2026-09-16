#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from generate_meeting_note import GroundedStatement, MeetingNote, VerificationResult
from presentation_segmentation import PresentationRange, SegmentationResult
from process_meeting import process_meeting, sanitize_grounding


def test_sanitize_grounding():
    note = MeetingNote(
        title=GroundedStatement(text="Test Title", source_segments=[2]),
        summary=[
            GroundedStatement(text="Valid inside range", source_segments=[2, 3]),
            GroundedStatement(text="Outside range", source_segments=[99]),
        ],
        facts=[GroundedStatement(text="Mixed range", source_segments=[1, 100])],
    )

    sanitized = sanitize_grounding(note, start_idx=2, end_idx=5)
    assert sanitized.title.source_segments == [2]
    assert sanitized.summary[0].source_segments == [2, 3]
    # Outside range segment 99 falls back to start_idx 2
    assert sanitized.summary[1].source_segments == [2]
    # Mixed range segment 100 is filtered out, leaving 2
    assert sanitized.facts[0].source_segments == [2]


def test_process_meeting_multi_presentation_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")

    audio_path = tmp_path / "multi_test.wav"
    audio_path.write_bytes(b"dummy wav content")

    out_dir = tmp_path / "multi_output"

    mock_transcription = {
        "language": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": 5.0, "text": "First talk segment."},
            {"id": 1, "start": 5.0, "end": 10.0, "text": "Second talk segment."},
        ],
        "model_info": {"id": "test-model", "file": "test.bin", "sha256": "dummy", "quantization": "q5_0"},
        "runtime_info": {"version": "1.8.6", "backend": "CUDA"},
    }

    mock_segmentation = SegmentationResult(
        version=1,
        presentation_count=2,
        presentations=[
            PresentationRange(
                index=1,
                start_segment_id="S0000",
                end_segment_id="S0000",
                title="First Talk",
                boundary_confidence="high",
            ),
            PresentationRange(
                index=2,
                start_segment_id="S0001",
                end_segment_id="S0001",
                title="Second Talk",
                boundary_confidence="high",
            ),
        ],
    )

    mock_note1 = VerificationResult(
        final_note=MeetingNote(
            title=GroundedStatement(text="First Talk", source_segments=[0]),
            summary=[GroundedStatement(text="Summary 1", source_segments=[0])],
        ),
        removed_or_corrected_claims=[],
        verification_warnings=[],
    )

    mock_note2 = VerificationResult(
        final_note=MeetingNote(
            title=GroundedStatement(text="Second Talk", source_segments=[1]),
            summary=[GroundedStatement(text="Summary 2", source_segments=[1])],
        ),
        removed_or_corrected_claims=[],
        verification_warnings=[],
    )

    raw_mock_json = json.dumps(mock_segmentation.model_dump())
    with patch("process_meeting.prepared_audio") as mock_prep, \
         patch("process_meeting.transcribe_with_whisper_cpp", return_value=mock_transcription), \
         patch("process_meeting.segment_presentations_with_luna", return_value=(mock_segmentation, raw_mock_json)), \
         patch("process_meeting.create_note_with_api", side_effect=[mock_note1, mock_note2]):

        mock_prep.return_value.__enter__.return_value.path = audio_path

        artifacts = process_meeting(
            input_file=audio_path,
            recording_language="en",
            note_type="presentationSummary",
            note_language="pl",
            auto_segment=True,
            output_dir=out_dir,
        )

        assert [p.name for p in out_dir.iterdir()] == ["meeting_notes.zip"]

        import zipfile
        with zipfile.ZipFile(artifacts["zip"], "r") as zf:
            names = zf.namelist()
            assert "01_first_talk/note.docx" in names
            assert "01_first_talk/note.pdf" in names
            assert "01_first_talk/note.md" in names
            assert "02_second_talk/note.docx" in names
            assert "02_second_talk/note.pdf" in names
            assert "02_second_talk/note.md" in names
            assert "audit/provenance.json" in names
            assert "audit/presentation_segments.json" in names
            assert "audit/summary_report.json" in names

            prov_data = json.loads(zf.read("audit/provenance.json").decode("utf-8"))
            assert prov_data["presentation_count"] == 2
            assert prov_data["auto_segmentation_used"] is True
            assert prov_data["segmentation_status"] == "success"

            sum_data = json.loads(zf.read("audit/summary_report.json").decode("utf-8"))
            assert sum_data["presentation_count"] == 2
            assert sum_data["segmentation_status"] == "success"


def test_process_meeting_presentation_summary_segmentation_failure_aborts_without_note_generation(tmp_path, monkeypatch):
    from presentation_segmentation import SegmentationError

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")

    audio_path = tmp_path / "fail_test.wav"
    audio_path.write_bytes(b"dummy wav content")

    out_dir = tmp_path / "fail_output"

    mock_transcription = {
        "language": "en",
        "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "Segment text."}],
        "model_info": {"id": "test-model", "file": "test.bin", "sha256": "dummy", "quantization": "q5_0"},
        "runtime_info": {"version": "1.8.6", "backend": "CUDA"},
    }

    seg_error = SegmentationError(
        message="JSON decode failed: Invalid char",
        stage="json_decode",
        exception_type="JSONDecodeError",
        raw_response="INVALID JSON RESPONSE TEXT",
        model_name="gpt-5.6-luna",
    )

    with patch("process_meeting.prepared_audio") as mock_prep, \
         patch("process_meeting.transcribe_with_whisper_cpp", return_value=mock_transcription), \
         patch("process_meeting.segment_presentations_with_luna", side_effect=seg_error), \
         patch("process_meeting.create_note_with_api") as mock_create_note:

        mock_prep.return_value.__enter__.return_value.path = audio_path

        with pytest.raises(RuntimeError) as exc_info:
            process_meeting(
                input_file=audio_path,
                recording_language="en",
                note_type="presentationSummary",
                note_language="pl",
                auto_segment=True,
                output_dir=out_dir,
            )

        assert "Segmentation failed at stage 'json_decode'" in str(exc_info.value)
        # Verify no note creation API calls were made!
        mock_create_note.assert_not_called()


def test_process_meeting_single_presentation_success_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")

    audio_path = tmp_path / "single_test.wav"
    audio_path.write_bytes(b"dummy wav content")

    out_dir = tmp_path / "single_output"

    mock_transcription = {
        "language": "en",
        "segments": [{"id": 0, "start": 0.0, "end": 5.0, "text": "Single keynote segment."}],
        "model_info": {"id": "test-model", "file": "test.bin", "sha256": "dummy", "quantization": "q5_0"},
        "runtime_info": {"version": "1.8.6", "backend": "CUDA"},
    }

    mock_segmentation = SegmentationResult(
        version=1,
        presentation_count=1,
        presentations=[
            PresentationRange(
                index=1,
                start_segment_id="S0000",
                end_segment_id="S0000",
                title="Single Keynote Talk",
                boundary_confidence="high",
            )
        ],
    )

    mock_note = VerificationResult(
        final_note=MeetingNote(
            title=GroundedStatement(text="Single Keynote Talk", source_segments=[0]),
            summary=[GroundedStatement(text="Summary text", source_segments=[0])],
        ),
        removed_or_corrected_claims=[],
        verification_warnings=[],
    )

    raw_json = json.dumps(mock_segmentation.model_dump())
    with patch("process_meeting.prepared_audio") as mock_prep, \
         patch("process_meeting.transcribe_with_whisper_cpp", return_value=mock_transcription), \
         patch("process_meeting.segment_presentations_with_luna", return_value=(mock_segmentation, raw_json)), \
         patch("process_meeting.create_note_with_api", return_value=mock_note) as mock_create_note:

        mock_prep.return_value.__enter__.return_value.path = audio_path

        artifacts = process_meeting(
            input_file=audio_path,
            recording_language="en",
            note_type="presentationSummary",
            note_language="pl",
            auto_segment=True,
            output_dir=out_dir,
        )

        mock_create_note.assert_called_once()
        assert artifacts["zip"].is_file()

        import zipfile
        with zipfile.ZipFile(artifacts["zip"], "r") as zf:
            prov_data = json.loads(zf.read("audit/provenance.json").decode("utf-8"))
            assert prov_data["presentation_count"] == 1
            assert prov_data["segmentation_status"] == "success"



