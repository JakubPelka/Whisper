import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from pydantic import ValidationError
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antek_note_api import (  # noqa: E402
    GenerateNoteRequest,
    app,
    canonical_segments,
    generate_note,
)
from generate_meeting_note import (  # noqa: E402
    GroundedStatement,
    MeetingNote,
    VerificationResult,
)


def request_payload(context="Dev HAJK", preset="meetingNotes"):
    return {
        "request_id": str(uuid4()),
        "meeting_id": str(uuid4()),
        "language": "sv",
        "note_preset": preset,
        "meeting_context": context,
        "transcript": {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Första segmentet."},
                {"start": 2.0, "end": 4.0, "text": "Andra segmentet."},
            ]
        },
    }


class RequestSchemaTests(unittest.TestCase):
    def test_health_and_bearer_auth_boundary(self):
        client = TestClient(app)
        self.assertEqual(client.get("/healthz").json(), {"status": "ok"})
        with patch.dict(os.environ, {"ANTEK_API_TOKEN": "private-test-token"}, clear=False):
            response = client.post("/v1/notes/generate", json=request_payload())
        self.assertEqual(response.status_code, 401)

    def test_request_contains_no_audio_field(self):
        schema = str(GenerateNoteRequest.model_json_schema()).lower()
        self.assertNotIn("audio", schema)

    def test_rejects_empty_transcript(self):
        payload = request_payload()
        payload["transcript"]["segments"] = []
        with self.assertRaises(ValidationError):
            GenerateNoteRequest.model_validate(payload)

    def test_backend_numbers_every_complete_segment(self):
        request = GenerateNoteRequest.model_validate(request_payload())
        segments = canonical_segments(request)
        self.assertEqual([item["id"] for item in segments], [1, 2])
        self.assertEqual([item["text"] for item in segments], ["Första segmentet.", "Andra segmentet."])


class EngineIntegrationTests(unittest.TestCase):
    def test_api_calls_shared_engine_and_records_separate_usage(self):
        request = GenerateNoteRequest.model_validate(request_payload(context=None, preset="shortSummary"))
        calls = []

        def fake_create_note_with_api(**kwargs):
            calls.append(kwargs)
            note = MeetingNote(summary=[GroundedStatement(text="Styrkt", source_segments=[1])])
            return (
                note,
                VerificationResult(final_note=note),
                {
                    "draft_response_id": "draft-id",
                    "verification_response_id": "verification-id",
                    "api_usage": {
                        "draft": {"input_tokens": 100, "output_tokens": 20},
                        "verification": {"input_tokens": 140, "output_tokens": 25},
                    },
                },
            )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False), patch(
            "antek_note_api.create_note_with_api", side_effect=fake_create_note_with_api
        ):
            response = generate_note(request)

        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["meeting_context"])
        self.assertEqual(calls[0]["note_preset"], "shortSummary")
        self.assertIn("[S0002", calls[0]["transcript_text"])
        self.assertEqual(response.usage["draft"]["input_tokens"], 100)
        self.assertEqual(response.usage["verification"]["input_tokens"], 140)
        self.assertEqual(response.status, "verified")

    def test_audit_uses_transcript_hash_without_full_transcript(self):
        request = GenerateNoteRequest.model_validate(request_payload())
        note = MeetingNote(summary=[GroundedStatement(text="Styrkt", source_segments=[1])])
        metadata = {
            "draft_response_id": "draft-id",
            "verification_response_id": "verification-id",
            "api_usage": {"draft": None, "verification": None},
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "ANTEK_AUDIT_DIR": directory},
            clear=False,
        ), patch(
            "antek_note_api.create_note_with_api",
            return_value=(note, VerificationResult(final_note=note), metadata),
        ):
            generate_note(request)
            audit = next(Path(directory).glob("*.note.json")).read_text(encoding="utf-8")

        self.assertIn("transcript_sha256", audit)
        self.assertNotIn("Första segmentet", audit)


if __name__ == "__main__":
    unittest.main()
