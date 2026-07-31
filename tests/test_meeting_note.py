import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_meeting_note import (  # noqa: E402
    GroundedStatement,
    MeetingNote,
    UnclearPoint,
    VerificationResult,
    create_note_with_api,
    load_transcript,
    read_named_secret,
    render_docx,
    transcript_for_prompt,
    validate_evidence,
)


class TranscriptTests(unittest.TestCase):
    def test_loads_numbered_timestamped_segments(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "transcript.json"
            path.write_text(
                json.dumps(
                    {
                        "input": "/recordings/meeting.m4a",
                        "segments": [
                            {"start": 0.0, "end": 4.2, "text": "Första punkten."},
                            {"start": 4.2, "end": 8.0, "text": "Beslut saknas."},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            _, segments = load_transcript(path)

        self.assertEqual([segment["id"] for segment in segments], [1, 2])
        prompt = transcript_for_prompt(segments)
        self.assertIn("[S0001 00:00:00-00:00:04]", prompt)
        self.assertIn("[S0002 00:00:04-00:00:08]", prompt)

    def test_falls_back_to_full_text_when_segments_are_missing(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "transcript.json"
            path.write_text(json.dumps({"text": "Endast löpande text."}), encoding="utf-8")
            _, segments = load_transcript(path)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "Endast löpande text.")


class EvidenceTests(unittest.TestCase):
    def test_accepts_existing_source_segments(self):
        note = MeetingNote(summary=[GroundedStatement(text="Styrkt uppgift", source_segments=[1])])
        validate_evidence(note, [{"id": 1}])

    def test_rejects_missing_source_segments(self):
        note = MeetingNote(summary=[GroundedStatement(text="Ostyrkt uppgift", source_segments=[9])])
        with self.assertRaisesRegex(ValueError, "missing segment"):
            validate_evidence(note, [{"id": 1}])


class ApiRoutingTests(unittest.TestCase):
    def test_uses_luna_for_draft_and_terra_for_verification(self):
        calls = []

        class FakeResponses:
            def parse(self, **kwargs):
                calls.append(kwargs["model"])
                if kwargs["text_format"] is MeetingNote:
                    parsed = MeetingNote(summary=[GroundedStatement(text="Styrkt", source_segments=[1])])
                else:
                    parsed = VerificationResult(
                        final_note=MeetingNote(
                            summary=[GroundedStatement(text="Styrkt", source_segments=[1])]
                        )
                    )
                return SimpleNamespace(id=f"response-{len(calls)}", output_parsed=parsed, usage=None)

        fake_openai = ModuleType("openai")
        fake_openai.OpenAI = lambda **_kwargs: SimpleNamespace(responses=FakeResponses())

        with patch.dict(sys.modules, {"openai": fake_openai}):
            create_note_with_api(
                transcript_text="[S0001 00:00:00-00:00:01] Styrkt",
                language="sv",
                draft_model="gpt-5.6-luna",
                verification_model="gpt-5.6-terra",
                reasoning_effort="medium",
            )

        self.assertEqual(calls, ["gpt-5.6-luna", "gpt-5.6-terra"])


class SecretTests(unittest.TestCase):
    def test_reads_only_requested_quoted_secret(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "secrets.env"
            path.write_text('OTHER="ignore"\nexport OPENAI_API_KEY="sk-test-value"\n', encoding="utf-8")
            self.assertEqual(read_named_secret(path, "OPENAI_API_KEY"), "sk-test-value")


class DocxTests(unittest.TestCase):
    def test_renders_swedish_docx_without_transcript_appendix(self):
        note = MeetingNote(
            title=GroundedStatement(text="Provärende", source_segments=[1]),
            summary=[GroundedStatement(text="En styrkt sammanfattning.", source_segments=[1])],
            unclear_points=[UnclearPoint(description="Ett namn kunde inte uppfattas.", source_segments=[2])],
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "note.docx"
            render_docx(
                note=note,
                output_path=path,
                source_name="meeting.m4a",
                language="sv",
            )
            payload = path.read_bytes()

        self.assertTrue(payload.startswith(b"PK"))
        self.assertGreater(len(payload), 1000)


if __name__ == "__main__":
    unittest.main()
