#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for process_meeting.py headless entrypoint in JakubPelka/Whisper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from generate_meeting_note import GroundedStatement, MeetingNote, VerificationResult
from process_meeting import process_meeting


@pytest.fixture
def mock_meeting_note():
    note = MeetingNote(
        title=GroundedStatement(text="Test Meeting", source_segments=[0]),
        summary=[GroundedStatement(text="This is a test summary.", source_segments=[0])],
    )
    return VerificationResult(
        final_note=note,
        removed_or_corrected_claims=[],
        verification_warnings=[],
    )


def test_process_meeting_flow(tmp_path, mock_meeting_note, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")

    audio_path = tmp_path / "test_input.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    out_dir = tmp_path / "output_artifacts"

    with patch("process_meeting.create_note_with_api", return_value=mock_meeting_note) as mock_api:
        artifacts = process_meeting(
            input_file=audio_path,
            recording_language="en",
            note_type="serviceNote",
            note_language="pl",
            context="Test context",
            vocabulary="Test terminology",
            output_dir=out_dir,
        )

        mock_api.assert_called_once()
        call_kwargs = mock_api.call_args.kwargs
        assert call_kwargs["language"] == "pl"
        assert call_kwargs["note_preset"] == "serviceNote"
        assert call_kwargs["meeting_context"] == "Test context"

    assert artifacts["zip"].is_file()
    # Verify ZIP-Only output contract in output_dir
    assert [p.name for p in out_dir.iterdir()] == ["meeting_notes.zip"]

    with zipfile.ZipFile(artifacts["zip"], "r") as zf:
        names = zf.namelist()
        assert "01_full_recording/note.docx" in names
        assert "01_full_recording/note.pdf" in names
        assert "01_full_recording/note.txt" in names
        assert "01_full_recording/note.md" in names
        assert "audit/provenance.json" in names
        assert "transcript/transcript.json" in names
        assert "transcript/transcript.txt" in names

        prov_content = zf.read("audit/provenance.json").decode("utf-8")
        prov = json.loads(prov_content)

    assert prov["whisper_cpp_version"] == "1.8.6"
    assert prov["backend"] == "CUDA"
    assert prov["note_preset"] == "serviceNote"
    assert prov["note_language"] == "pl"
    assert prov["has_context"] is True
    assert prov["has_vocabulary"] is True

    # Ensure no transcript or note content leaks into provenance.json
    assert "This is a test summary." not in prov_content


def test_process_meeting_video_cleanup_and_provenance(tmp_path, mock_meeting_note, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")

    video_path = tmp_path / "video_input.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert video_path.is_file()
    orig_video_size = video_path.stat().st_size

    out_dir = tmp_path / "output_video_test"

    with patch("process_meeting.create_note_with_api", return_value=mock_meeting_note):
        artifacts = process_meeting(
            input_file=video_path,
            recording_language="en",
            note_type="meetingNotes",
            note_language="pl",
            output_dir=out_dir,
        )

    # Assert source video file unlinked from workspace after audio extraction verification
    assert not video_path.exists()
    assert [p.name for p in out_dir.iterdir()] == ["meeting_notes.zip"]

    # Assert provenance records original source metadata inside zip
    with zipfile.ZipFile(artifacts["zip"], "r") as zf:
        prov = json.loads(zf.read("audit/provenance.json").decode("utf-8"))

    assert "original_source" in prov
    assert prov["original_source"]["filename"] == "video_input.mp4"
    assert prov["original_source"]["size_bytes"] == orig_video_size
    assert prov["original_source"]["has_video_stream"] is True
    assert prov["original_source"]["source_video_discarded"] is True
    assert "audio_derivative" in prov
