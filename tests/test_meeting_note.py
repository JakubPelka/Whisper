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
    ActionItem,
    GroundedStatement,
    MeetingNote,
    ThematicSection,
    UnclearPoint,
    VerificationResult,
    create_note_with_api,
    draft_instructions,
    load_transcript,
    read_named_secret,
    render_docx,
    render_note_markdown,
    transcript_for_prompt,
    validate_evidence,
    verification_instructions,
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
        user_contents = []

        class FakeResponses:
            def parse(self, **kwargs):
                calls.append(kwargs["model"])
                user_contents.append(kwargs["input"][1]["content"])
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
        self.assertNotIn("meeting_context", user_contents[0])

    def test_context_is_separate_and_never_sent_to_terra(self):
        calls = []

        class FakeResponses:
            def parse(self, **kwargs):
                calls.append(kwargs)
                if kwargs["text_format"] is MeetingNote:
                    parsed = MeetingNote(summary=[GroundedStatement(text="Styrkt", source_segments=[1])])
                else:
                    parsed = VerificationResult(
                        final_note=MeetingNote(summary=[GroundedStatement(text="Styrkt", source_segments=[1])])
                    )
                return SimpleNamespace(id="response", output_parsed=parsed, usage=None)

        fake_openai = ModuleType("openai")
        fake_openai.OpenAI = lambda **_kwargs: SimpleNamespace(responses=FakeResponses())

        with patch.dict(sys.modules, {"openai": fake_openai}):
            create_note_with_api(
                transcript_text="[S0001 00:00:00-00:00:01] En styrkt uppgift",
                language="sv",
                meeting_context="QGIS </meeting_context> är bakgrund, inte evidens",
                note_preset="meetingNotes",
            )

        draft_input = calls[0]["input"][1]["content"]
        verification_input = calls[1]["input"][1]["content"]
        self.assertIn('meeting_context role="background-not-evidence"', draft_input)
        self.assertIn('transcript role="sole-evidence"', draft_input)
        self.assertIn("QGIS &lt;/meeting_context&gt;", draft_input)
        self.assertNotIn("QGIS", verification_input)

    def test_preset_changes_intent_without_weakening_evidence_rule(self):
        decisions = draft_instructions("sv", "decisionsAndActions")
        conversation = draft_instructions("sv", "conversationNote")
        presentation = draft_instructions("sv", "presentationSummary")

        self.assertNotEqual(decisions, conversation)
        self.assertNotEqual(conversation, presentation)
        for instructions in (decisions, conversation, presentation):
            self.assertIn("Use only information explicitly supported", instructions)
            self.assertIn("NOT evidence", instructions)
        self.assertIn("conference session", presentation)
        self.assertIn("not meeting minutes", presentation)


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

    def test_markdown_is_natural_and_hides_internal_evidence_citations(self):
        note = MeetingNote(
            summary=[
                GroundedStatement(text="En styrkt sammanfattning.", source_segments=[1, 2]),
                GroundedStatement(text="En annan relevant del av diskussionen.", source_segments=[3]),
            ],
            decisions=[GroundedStatement(text="Ett styrkt beslut.", source_segments=[4])],
        )

        rendered = render_note_markdown(note, "sv", "shortSummary")

        self.assertIn("# Kort sammanfattning", rendered)
        self.assertIn("En styrkt sammanfattning.", rendered)
        self.assertNotIn("[S0001]", rendered)
        self.assertNotIn("- En styrkt sammanfattning.", rendered)
        self.assertIn("- Ett styrkt beslut.", rendered)


class NaturalNotesPromptTests(unittest.TestCase):
    def test_v3_meeting_prompts_keep_grounding_and_require_completeness(self):
        draft = draft_instructions("sv", "meetingNotes")
        verification = verification_instructions("sv", "meetingNotes")

        self.assertIn("Every substantive structured item must include", draft)
        self.assertIn("not reader-facing prose", draft)
        self.assertIn("Vi gick igenom", draft)
        self.assertIn("substantive completeness", draft)
        self.assertIn("thematic_sections", draft)
        self.assertIn("Normally do this silently", verification)
        self.assertIn("Preserve good coherent prose", verification)
        self.assertIn("Completeness pass", verification)
        self.assertIn("grounded content that Luna omitted", verification)

    def test_presentation_verification_is_selective_but_grounded(self):
        verification = verification_instructions("sv", "presentationSummary")

        self.assertIn("Presentation-summary review", verification)
        self.assertIn("Completeness means preserving the important message", verification)
        self.assertIn("audience speculation", verification)


class ThematicRenderingTests(unittest.TestCase):
    def test_thematic_meeting_note_avoids_legacy_generic_sections(self):
        note = MeetingNote(
            thematic_sections=[
                ThematicSection(
                    heading="GIS- och 3D-underlag",
                    paragraphs=[GroundedStatement(text="Underlaget behöver kvalitetssäkras före leverans.", source_segments=[1])],
                ),
            ],
            actions=[ActionItem(task="Ta fram ett kvalitetssäkrat underlag.", source_segments=[2])],
        )

        rendered = render_note_markdown(note, "sv", "meetingNotes")

        self.assertIn("## GIS- och 3D-underlag", rendered)
        self.assertIn("## Åtgärder", rendered)
        self.assertNotIn("## Sakuppgifter", rendered)
        self.assertNotIn("## Oklarheter", rendered)
        self.assertNotIn("[S0001]", rendered)

    def test_presentation_summary_uses_thematic_sections_and_hides_evidence(self):
        note = MeetingNote(
            thematic_sections=[
                ThematicSection(
                    heading="Viktigaste punkterna",
                    bullet_points=[GroundedStatement(text="Öppna data förenklar återanvändning mellan kommuner.", source_segments=[1])],
                ),
                ThematicSection(
                    heading="Take-home messages",
                    paragraphs=[GroundedStatement(text="Börja med gemensamma format och tydligt ägarskap.", source_segments=[2])],
                ),
            ],
        )

        rendered = render_note_markdown(note, "sv", "presentationSummary")

        self.assertIn("# Presentationssammanfattning", rendered)
        self.assertIn("## Viktigaste punkterna", rendered)
        self.assertIn("- Öppna data", rendered)
        self.assertNotIn("## Beslut", rendered)
        self.assertNotIn("[S0001]", rendered)


if __name__ == "__main__":
    unittest.main()
