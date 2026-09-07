import os
import sqlite3
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
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database_path = str(Path(self.directory.name) / "ledger.sqlite3")
        self.environment = patch.dict(
            os.environ,
            {
                "ANTEK_LEDGER_DB": self.database_path,
                "ANTEK_BOOTSTRAP_TOKEN": "bootstrap-test-token",
                "ANTEK_PRIVATE_ALPHA_INCLUDED_CREDITS": "1000",
            },
            clear=False,
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def bootstrap(self, client):
        response = client.post(
            "/v1/auth/bootstrap",
            headers={"X-Antek-Bootstrap": "bootstrap-test-token"},
            json={"installation_id": str(uuid4()), "app_version": "1.0-test"},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_health_and_bearer_auth_boundary(self):
        client = TestClient(app)
        self.assertEqual(client.get("/healthz").json(), {"status": "ok"})
        response = client.post("/v1/notes/generate", json=request_payload())
        self.assertEqual(response.status_code, 401)

    def test_bootstrap_issues_subject_credential_and_real_balance(self):
        client = TestClient(app)
        bootstrap = self.bootstrap(client)
        self.assertIn("access_credential", bootstrap)
        self.assertEqual(bootstrap["available_credits"], 1000)
        balance = client.get(
            "/v1/me/credits",
            headers={"Authorization": f"Bearer {bootstrap['access_credential']}"},
        )
        self.assertEqual(balance.status_code, 200)
        self.assertEqual(balance.json()["included_credits"], 1000)

    def test_request_contains_no_audio_field(self):
        schema = str(GenerateNoteRequest.model_json_schema()).lower()
        self.assertNotIn("audio", schema)
        self.assertNotIn("meeting_id", schema)

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
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database_path = str(Path(self.directory.name) / "ledger.sqlite3")
        self.environment = patch.dict(
            os.environ,
            {"ANTEK_LEDGER_DB": self.database_path, "ANTEK_BOOTSTRAP_TOKEN": "bootstrap-test-token"},
            clear=False,
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

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

    def test_generate_persists_only_non_content_processing_statistics(self):
        client = TestClient(app)
        bootstrap = client.post(
            "/v1/auth/bootstrap",
            headers={"X-Antek-Bootstrap": "bootstrap-test-token"},
            json={"installation_id": str(uuid4()), "app_version": "1.0-test"},
        ).json()
        payload = request_payload(context="Private context")
        payload["recording_duration_seconds"] = 42
        request_id = payload["request_id"]
        note = MeetingNote(summary=[GroundedStatement(text="Styrkt", source_segments=[1])])
        metadata = {
            "draft_response_id": "draft-id",
            "verification_response_id": "verification-id",
            "api_usage": {
                "draft": {"input_tokens": 100, "output_tokens": 20},
                "verification": {"input_tokens": 140, "output_tokens": 25},
            },
        }
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False), patch(
            "antek_note_api.create_note_with_api",
            return_value=(note, VerificationResult(final_note=note), metadata),
        ):
            response = client.post(
                "/v1/notes/generate",
                headers={
                    "Authorization": f"Bearer {bootstrap['access_credential']}",
                    "Idempotency-Key": request_id,
                },
                json=payload,
            )

        self.assertEqual(response.status_code, 200)
        connection = sqlite3.connect(self.database_path)
        row = connection.execute(
            """SELECT recording_duration_seconds, transcript_utf8_bytes, context_utf8_bytes,
                      credits_charged, pipeline_version FROM processing_operations"""
        ).fetchone()
        self.assertEqual(row[0], 42)
        self.assertGreater(row[1], 0)
        self.assertEqual(row[2], len("Private context".encode("utf-8")))
        self.assertGreater(row[3], 0)
        self.assertEqual(row[4], "luna-terra-v2-natural-notes")
        serialized_rows = "\n".join(str(value) for value in connection.execute(
            """SELECT operation_id, subject_id, status, recording_duration_seconds, transcript_utf8_bytes,
                      transcript_character_count, context_present, context_utf8_bytes, note_preset,
                      transcript_language, estimated_credits, credits_charged
               FROM processing_operations"""
        ).fetchone())
        self.assertNotIn("Första segmentet", serialized_rows)
        self.assertNotIn("Private context", serialized_rows)

    def test_retry_replays_cached_result_without_a_second_ledger_charge(self):
        client = TestClient(app)
        bootstrap = client.post(
            "/v1/auth/bootstrap",
            headers={"X-Antek-Bootstrap": "bootstrap-test-token"},
            json={"installation_id": str(uuid4()), "app_version": "1.0-test"},
        ).json()
        payload = request_payload(context=None)
        note = MeetingNote(summary=[GroundedStatement(text="Styrkt", source_segments=[1])])
        metadata = {
            "draft_response_id": "draft-id", "verification_response_id": "verification-id",
            "api_usage": {"draft": {"input_tokens": 100}, "verification": {"input_tokens": 100}},
        }
        headers = {
            "Authorization": f"Bearer {bootstrap['access_credential']}",
            "Idempotency-Key": payload["request_id"],
        }
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False), patch(
            "antek_note_api.create_note_with_api",
            return_value=(note, VerificationResult(final_note=note), metadata),
        ) as engine:
            first = client.post("/v1/notes/generate", headers=headers, json=payload)
            second = client.post("/v1/notes/generate", headers=headers, json=payload)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(engine.call_count, 1)
        connection = sqlite3.connect(self.database_path)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM credit_ledger WHERE entry_type = 'charge'").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
