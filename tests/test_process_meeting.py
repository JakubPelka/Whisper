#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for process_meeting.py headless entrypoint in JakubPelka/Whisper."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

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

    assert artifacts["docx"].is_file()
    assert artifacts["pdf"].is_file()
    assert artifacts["txt"].is_file()
    assert artifacts["md"].is_file()
    assert artifacts["transcript_json"].is_file()
    assert artifacts["transcript_txt"].is_file()
    assert artifacts["provenance"].is_file()

    with open(artifacts["provenance"], "r", encoding="utf-8") as f:
        prov = json.load(f)

    assert prov["whisper_cpp_version"] == "1.8.6"
    assert prov["backend"] == "CUDA"
    assert prov["note_preset"] == "serviceNote"
    assert prov["note_language"] == "pl"
    assert prov["has_context"] is True
    assert prov["has_vocabulary"] is True

    # Ensure no transcript or note content leaks into provenance.json
    prov_str = json.dumps(prov)
    assert "This is a test summary." not in prov_str


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

    # Assert provenance records original source metadata
    with open(artifacts["provenance"], "r", encoding="utf-8") as f:
        prov = json.load(f)

    assert "original_source" in prov
    assert prov["original_source"]["filename"] == "video_input.mp4"
    assert prov["original_source"]["size_bytes"] == orig_video_size
    assert prov["original_source"]["has_video_stream"] is True
    assert prov["original_source"]["source_video_discarded"] is True
    assert "audio_derivative" in prov

