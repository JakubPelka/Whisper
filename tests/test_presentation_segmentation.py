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

from presentation_segmentation import (
    BoundaryEvidence,
    PresentationRange,
    SegmentationResult,
    format_segment_id,
    parse_segment_id,
    prepare_prompt_transcript,
    segment_presentations_with_luna,
    validate_and_normalize_segmentation,
)


def make_dummy_segments(n: int) -> list[dict]:
    return [
        {
            "id": i,
            "start": float(i * 5),
            "end": float((i + 1) * 5),
            "text": f"This is segment text line {i}.",
        }
        for i in range(n)
    ]


def test_segment_id_formatting_and_parsing():
    assert format_segment_id(0) == "S0000"
    assert format_segment_id(12) == "S0012"
    assert parse_segment_id("S0042") == 42
    assert parse_segment_id("s0005") == 5
    assert parse_segment_id("15") == 15
    assert parse_segment_id("invalid") is None


def test_prepare_prompt_transcript():
    segments = make_dummy_segments(2)
    prompt_text = prepare_prompt_transcript(segments)
    assert "[S0000] (0.00s -> 5.00s)" in prompt_text
    assert "[S0001] (5.00s -> 10.00s)" in prompt_text


# --- Synthetic Fixtures A-F Tests ---

def test_fixture_a_single_presentation_no_split():
    segments = make_dummy_segments(10)
    result = SegmentationResult(
        version=1,
        presentation_count=1,
        presentations=[
            PresentationRange(
                index=1,
                start_segment_id="S0000",
                end_segment_id="S0009",
                title="Single Keynote Talk",
                boundary_confidence="high",
            )
        ],
    )
    ranges = validate_and_normalize_segmentation(segments, result)
    assert len(ranges) == 1
    assert ranges[0] == (0, 9, "Single Keynote Talk")


def test_fixture_b_two_clear_presentations():
    segments = make_dummy_segments(20)
    result = SegmentationResult(
        version=1,
        presentation_count=2,
        presentations=[
            PresentationRange(
                index=1,
                start_segment_id="S0000",
                end_segment_id="S0009",
                title="First Presentation on AI",
                boundary_confidence="high",
            ),
            PresentationRange(
                index=2,
                start_segment_id="S0010",
                end_segment_id="S0019",
                title="Second Presentation on Robotics",
                boundary_confidence="high",
            ),
        ],
    )
    ranges = validate_and_normalize_segmentation(segments, result)
    assert len(ranges) == 2
    assert ranges[0] == (0, 9, "First Presentation on AI")
    assert ranges[1] == (10, 19, "Second Presentation on Robotics")


def test_fixture_c_presentation_with_qa_kept_together():
    segments = make_dummy_segments(15)
    # Model correctly includes Q&A in main presentation (S0000 - S0014)
    result = SegmentationResult(
        version=1,
        presentation_count=1,
        presentations=[
            PresentationRange(
                index=1,
                start_segment_id="S0000",
                end_segment_id="S0014",
                title="Presentation with Q&A Session",
                boundary_confidence="high",
            )
        ],
    )
    ranges = validate_and_normalize_segmentation(segments, result)
    assert len(ranges) == 1
    assert ranges[0] == (0, 14, "Presentation with Q&A Session")


def test_fixture_d_moderator_transition_not_separate_presentation():
    segments = make_dummy_segments(25)
    # Segments 10-11 are moderator transitions, correctly assigned to start of second presentation
    result = SegmentationResult(
        version=1,
        presentation_count=2,
        presentations=[
            PresentationRange(
                index=1,
                start_segment_id="S0000",
                end_segment_id="S0009",
                title="First Keynote",
                boundary_confidence="high",
            ),
            PresentationRange(
                index=2,
                start_segment_id="S0010",
                end_segment_id="S0024",
                title="Second Keynote",
                boundary_confidence="high",
            ),
        ],
    )
    ranges = validate_and_normalize_segmentation(segments, result)
    assert len(ranges) == 2
    assert ranges[0] == (0, 9, "First Keynote")
    assert ranges[1] == (10, 24, "Second Keynote")


def test_fixture_e_ambiguous_topic_change_low_confidence_merged():
    segments = make_dummy_segments(20)
    # Model returned a low-confidence boundary at S0010 -> Validator MUST merge it into preceding range
    result = SegmentationResult(
        version=1,
        presentation_count=2,
        presentations=[
            PresentationRange(
                index=1,
                start_segment_id="S0000",
                end_segment_id="S0009",
                title="Main Topic Discussion",
                boundary_confidence="high",
            ),
            PresentationRange(
                index=2,
                start_segment_id="S0010",
                end_segment_id="S0019",
                title="Subtopic Shift",
                boundary_confidence="low",
            ),
        ],
    )
    ranges = validate_and_normalize_segmentation(segments, result)
    assert len(ranges) == 1
    assert ranges[0] == (0, 19, "Main Topic Discussion")


def test_fixture_f_invalid_model_output_fallback_one_presentation():
    segments = make_dummy_segments(10)

    # 1. Invalid segment ID
    bad_id_result = SegmentationResult(
        version=1,
        presentation_count=1,
        presentations=[
            PresentationRange(
                index=1,
                start_segment_id="INVALID_ID",
                end_segment_id="S0005",
                title="Bad ID",
                boundary_confidence="high",
            )
        ],
    )
    ranges = validate_and_normalize_segmentation(segments, bad_id_result)
    assert len(ranges) == 1
    assert ranges[0] == (0, 9, "Full Recording")

    # 2. Out of bounds index
    oob_result = SegmentationResult(
        version=1,
        presentation_count=1,
        presentations=[
            PresentationRange(
                index=1,
                start_segment_id="S0000",
                end_segment_id="S0999",
                title="Out of bounds",
                boundary_confidence="high",
            )
        ],
    )
    ranges = validate_and_normalize_segmentation(segments, oob_result)
    assert len(ranges) == 1
    assert ranges[0] == (0, 9, "Full Recording")

    # 3. None / Empty result
    ranges = validate_and_normalize_segmentation(segments, None)
    assert len(ranges) == 1
    assert ranges[0] == (0, 9, "Full Recording")


def test_segment_presentations_with_luna_mocked():
    segments = make_dummy_segments(5)
    mock_json_response = json.dumps(
        {
            "version": 1,
            "presentation_count": 1,
            "presentations": [
                {
                    "index": 1,
                    "start_segment_id": "S0000",
                    "end_segment_id": "S0004",
                    "title": "Mocked Presentation",
                    "boundary_confidence": "high",
                    "start_evidence": [{"segment_id": "S0000", "signal": "formal_opening"}],
                }
            ],
        }
    )

    with patch("openai.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_response = mock_client.chat.completions.create.return_value
        mock_response.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": mock_json_response})()})()
        ]

        res, raw_str = segment_presentations_with_luna(segments, api_key="sk-test-mock")
        assert res.presentation_count == 1
        assert res.presentations[0].title == "Mocked Presentation"
        assert raw_str == mock_json_response


def test_segment_presentations_with_luna_test_guard_no_api_calls(monkeypatch):
    segments = make_dummy_segments(5)
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-plausible-test-key-12345")
    monkeypatch.delenv("ALLOW_REAL_AI_API", raising=False)

    res, raw_str = segment_presentations_with_luna(segments)
    assert isinstance(res, SegmentationResult)
    assert res.presentation_count == 1
    assert len(res.presentations) == 1
    assert res.presentations[0].start_segment_id == "S0000"
    assert res.presentations[0].end_segment_id == "S0004"
    assert res.presentations[0].title == "Full Recording"
    assert res.presentations[0].is_presentation is True
    assert raw_str is not None


def test_segmentation_error_stages_on_invalid_responses():
    from presentation_segmentation import SegmentationError

    segments = make_dummy_segments(5)

    # 1. JSON Decode Error
    with patch("openai.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_response = mock_client.chat.completions.create.return_value
        mock_response.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": "NOT VALID JSON {{{ "})()})()
        ]

        with pytest.raises(SegmentationError) as exc_info:
            segment_presentations_with_luna(segments, api_key="sk-test-mock")
        assert exc_info.value.stage == "json_decode"
        assert exc_info.value.raw_response == "NOT VALID JSON {{{ "

    # 2. Schema Validation Error
    with patch("openai.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_response = mock_client.chat.completions.create.return_value
        mock_response.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": json.dumps({"wrong_key": "data"})})()})()
        ]

        with pytest.raises(SegmentationError) as exc_info:
            segment_presentations_with_luna(segments, api_key="sk-test-mock")
        assert exc_info.value.stage == "schema_validation"
        assert exc_info.value.raw_response == json.dumps({"wrong_key": "data"})


def test_schema_validation_fails_on_invalid_enum_and_types():
    from presentation_segmentation import SegmentationError

    segments = make_dummy_segments(5)

    invalid_confidence = {
        "version": 1,
        "presentation_count": 1,
        "presentations": [
            {
                "index": 1,
                "start_segment_id": "S0000",
                "end_segment_id": "S0004",
                "title": "Talk",
                "boundary_confidence": "super_high",
                "is_presentation": True,
                "block_type": "presentation",
            }
        ],
    }

    with patch("openai.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": json.dumps(invalid_confidence)})()})()
        ]
        with pytest.raises(SegmentationError) as exc_info:
            segment_presentations_with_luna(segments, api_key="sk-test-mock")
        assert exc_info.value.stage == "schema_validation"

    invalid_block_type = {
        "version": 1,
        "presentation_count": 1,
        "presentations": [
            {
                "index": 1,
                "start_segment_id": "S0000",
                "end_segment_id": "S0004",
                "title": "Talk",
                "boundary_confidence": "high",
                "is_presentation": True,
                "block_type": "keynote_speech",
            }
        ],
    }

    with patch("openai.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": json.dumps(invalid_block_type)})()})()
        ]
        with pytest.raises(SegmentationError) as exc_info:
            segment_presentations_with_luna(segments, api_key="sk-test-mock")
        assert exc_info.value.stage == "schema_validation"


def test_strict_coverage_validation_raises_normalization_error():
    from presentation_segmentation import PresentationRange, SegmentationError, SegmentationResult, validate_and_normalize_segmentation

    segments = make_dummy_segments(10)

    # 1. Non-zero first start
    res1 = SegmentationResult(
        version=1,
        presentation_count=1,
        presentations=[
            PresentationRange(index=1, start_segment_id="S0002", end_segment_id="S0009", title="Talk", boundary_confidence="high")
        ],
    )
    with pytest.raises(SegmentationError) as exc_info:
        validate_and_normalize_segmentation(segments, res1, strict=True)
    assert exc_info.value.stage == "normalization"
    assert "S0000" in str(exc_info.value)

    # 2. Coverage gap
    res2 = SegmentationResult(
        version=1,
        presentation_count=2,
        presentations=[
            PresentationRange(index=1, start_segment_id="S0000", end_segment_id="S0003", title="Talk 1", boundary_confidence="high"),
            PresentationRange(index=2, start_segment_id="S0005", end_segment_id="S0009", title="Talk 2", boundary_confidence="high"),
        ],
    )
    with pytest.raises(SegmentationError) as exc_info:
        validate_and_normalize_segmentation(segments, res2, strict=True)
    assert exc_info.value.stage == "normalization"
    assert "gap or overlap" in str(exc_info.value)

    # 3. Incomplete tail coverage
    res3 = SegmentationResult(
        version=1,
        presentation_count=1,
        presentations=[
            PresentationRange(index=1, start_segment_id="S0000", end_segment_id="S0005", title="Talk 1", boundary_confidence="high")
        ],
    )
    with pytest.raises(SegmentationError) as exc_info:
        validate_and_normalize_segmentation(segments, res3, strict=True)
    assert exc_info.value.stage == "normalization"
    assert "Incomplete tail coverage" in str(exc_info.value)



def test_conference_housekeeping_and_multi_keynote_segmentation():
    segments = make_dummy_segments(30)
    result = SegmentationResult(
        version=1,
        presentation_count=2,
        presentations=[
            PresentationRange(
                index=1,
                start_segment_id="S0000",
                end_segment_id="S0004",
                title="Welcome and Housekeeping",
                boundary_confidence="high",
                is_presentation=False,
                block_type="intro",
                start_evidence=[BoundaryEvidence(segment_id="S0000", signal="housekeeping")],
                decision_rationale="Moderator introduction and logistics.",
            ),
            PresentationRange(
                index=2,
                start_segment_id="S0005",
                end_segment_id="S0014",
                title="Carol Williams Keynote",
                boundary_confidence="high",
                is_presentation=True,
                block_type="presentation",
                start_evidence=[BoundaryEvidence(segment_id="S0005", signal="moderator_intro")],
                decision_rationale="Keynote speech by Carol Williams.",
            ),
            PresentationRange(
                index=3,
                start_segment_id="S0015",
                end_segment_id="S0019",
                title="Coffee Break",
                boundary_confidence="high",
                is_presentation=False,
                block_type="break",
                start_evidence=[BoundaryEvidence(segment_id="S0015", signal="pause_transition")],
                decision_rationale="Intermission audio gap.",
            ),
            PresentationRange(
                index=4,
                start_segment_id="S0020",
                end_segment_id="S0029",
                title="Kit Stoner BCT Update",
                boundary_confidence="high",
                is_presentation=True,
                block_type="presentation",
                start_evidence=[BoundaryEvidence(segment_id="S0020", signal="speaker_change")],
                decision_rationale="Presentation by Kit Stoner on BCT updates.",
            ),
        ],
    )
    ranges = validate_and_normalize_segmentation(segments, result)
    assert len(ranges) == 4
    assert ranges[0].is_presentation is False
    assert ranges[0].block_type == "intro"
    assert ranges[1].is_presentation is True
    assert ranges[1].title == "Carol Williams Keynote"
    assert ranges[2].is_presentation is False
    assert ranges[2].block_type == "break"
    assert ranges[3].is_presentation is True
    assert ranges[3].title == "Kit Stoner BCT Update"

    pres_only = [r for r in ranges if r.is_presentation]
    assert len(pres_only) == 2
    assert pres_only[0].title == "Carol Williams Keynote"
    assert pres_only[1].title == "Kit Stoner BCT Update"


