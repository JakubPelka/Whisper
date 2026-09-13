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
