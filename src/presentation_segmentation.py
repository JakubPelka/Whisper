#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Luna Semantic Presentation Boundary Detector & Deterministic Validator.

Analyzes timestamped Whisper transcripts and identifies logical presentation / talk boundaries.
Enforces strict JSON schema, under-split policy, evidence tracking, and single-presentation fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("presentation_segmentation")


class BoundaryEvidence(BaseModel):
    segment_id: str
    signal: str


class PresentationRange(BaseModel):
    index: int
    start_segment_id: str
    end_segment_id: str
    title: str
    boundary_confidence: Literal["high", "medium", "low"]
    start_evidence: list[BoundaryEvidence] = Field(default_factory=list)


class SegmentationResult(BaseModel):
    version: int = 1
    presentation_count: int
    presentations: list[PresentationRange]


def format_segment_id(idx: int) -> str:
    """Format segment index as S0000 style identifier."""
    return f"S{idx:04d}"


def parse_segment_id(seg_id_str: str) -> int | None:
    """Parse integer index from S0000 string or plain integer string."""
    s = str(seg_id_str).strip()
    if s.startswith("S") or s.startswith("s"):
        try:
            return int(s[1:])
        except ValueError:
            return None
    try:
        return int(s)
    except ValueError:
        return None


def prepare_prompt_transcript(segments: list[dict[str, Any]]) -> str:
    """Format transcript segments with strict segment IDs and timestamps for Luna."""
    lines = []
    for i, seg in enumerate(segments):
        seg_id = format_segment_id(i)
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = str(seg.get("text", "")).strip()
        lines.append(f"[{seg_id}] ({start:.2f}s -> {end:.2f}s) {text}")
    return "\n".join(lines)


SEGMENTATION_SYSTEM_PROMPT = """You are Luna, an expert semantic presentation boundary detector for conference and meeting recordings.

Your task is to analyze the complete timestamped transcript and identify logical presentation / talk boundaries.

### KEY PRINCIPLES:
1. WHEN UNCERTAIN, KEEP MATERIAL TOGETHER. Prefer under-splitting over over-splitting.
2. Q&A, audience questions, speaker answers, and short moderator interactions belong to the presentation they follow.
3. Low-confidence boundaries alone MUST NOT cause a split.
4. Do NOT split for: slide changes, case studies, examples, panel discussions within the same session, or technical pauses.
5. Strong signals of a new presentation include a combination of:
   - Previous speaker formally closes / receives applause / moderator closes previous topic.
   - Moderator introduces a new speaker / new topic.
   - New speaker introduces themselves / formal opening ("jag ska prata om...", "dzisiaj opowiem o...", "my presentation today...").
   - Clear thematic reset + new talk context.

### OUTPUT CONTRACT:
Return STRICT JSON only matching this schema:
{
  "version": 1,
  "presentation_count": <number>,
  "presentations": [
    {
      "index": 1,
      "start_segment_id": "S0000",
      "end_segment_id": "S0120",
      "title": "<Short grounded working title>",
      "boundary_confidence": "high",
      "start_evidence": [
        {"segment_id": "S0000", "signal": "formal_opening"}
      ]
    }
  ]
}

DO NOT generate meeting notes or summaries. Return ONLY the JSON object.
"""


def segment_presentations_with_luna(
    segments: list[dict[str, Any]],
    api_key: str | None = None,
    model_name: str = "gpt-5.6-luna",
) -> SegmentationResult:
    """Invoke OpenAI API to detect logical presentation boundaries."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        env_file = Path(__file__).resolve().parent.parent / "secrets" / "openai.env"
        if env_file.is_file():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("OPENAI_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
            except Exception:
                pass

    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for Luna presentation segmentation.")

    from openai import OpenAI

    client = OpenAI(api_key=key)
    if type(client).__module__.startswith("openai"):
        if ("PYTEST_CURRENT_TEST" in os.environ or os.environ.get("APP_ENV") == "testing") and os.environ.get("ALLOW_REAL_AI_API") != "1":
            total_segs = len(segments)
            end_seg_id = f"S{total_segs - 1:04d}" if total_segs > 0 else "S0000"
            return SegmentationResult(
                version=1,
                presentation_count=1,
                presentations=[
                    PresentationRange(
                        index=1,
                        start_segment_id="S0000",
                        end_segment_id=end_seg_id,
                        title="Full Recording",
                        boundary_confidence="high",
                        start_evidence=[BoundaryEvidence(segment_id="S0000", signal="test_stub")],
                    )
                ],
            )

    prompt_transcript = prepare_prompt_transcript(segments)

    messages = [
        {"role": "system", "content": SEGMENTATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze the following transcript and return presentation boundaries:\n\n{prompt_transcript}"},
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    return SegmentationResult.model_validate(data)


def validate_and_normalize_segmentation(
    segments: list[dict[str, Any]],
    result: SegmentationResult | None,
) -> list[tuple[int, int, str]]:
    """Validate segmentation result for structural correctness and full coverage.

    Invariants:
    1. Returns non-empty list of ranges (start_idx, end_idx, title).
    2. Chronologically ordered, contiguous, non-overlapping.
    3. Covers all segments from 0 to len(segments) - 1.
    4. If any rule is violated, falls back to single presentation [(0, len(segments) - 1, "Full Recording")].
    """
    total_segments = len(segments)
    fallback = [(0, max(0, total_segments - 1), "Full Recording")]

    if not segments or not result or not result.presentations:
        logger.warning("Segmentation result empty or missing presentations. Falling back to single presentation.")
        return fallback

    parsed_ranges: list[tuple[int, int, str, str]] = []

    for item in result.presentations:
        s_idx = parse_segment_id(item.start_segment_id)
        e_idx = parse_segment_id(item.end_segment_id)
        conf = item.boundary_confidence

        if s_idx is None or e_idx is None:
            logger.warning("Invalid segment ID format in result (%s, %s). Falling back.", item.start_segment_id, item.end_segment_id)
            return fallback

        if s_idx < 0 or e_idx >= total_segments or s_idx > e_idx:
            logger.warning("Out of bounds segment range [%d, %d] for total %d. Falling back.", s_idx, e_idx, total_segments)
            return fallback

        title = item.title.strip() or f"Presentation {item.index}"
        parsed_ranges.append((s_idx, e_idx, title, conf))

    # Sort ranges by start_idx
    parsed_ranges.sort(key=lambda x: x[0])

    # Rule: Low-confidence boundary alone must not cause a split.
    # Merge low confidence splits into preceding range
    merged_ranges: list[tuple[int, int, str]] = []
    for s_idx, e_idx, title, conf in parsed_ranges:
        if not merged_ranges:
            merged_ranges.append((s_idx, e_idx, title))
            continue

        prev_s, prev_e, prev_title = merged_ranges[-1]
        if conf == "low":
            # Low confidence split -> merge into previous presentation
            logger.info("Low confidence boundary detected at segment %d. Merging into preceding presentation '%s'.", s_idx, prev_title)
            merged_ranges[-1] = (prev_s, max(prev_e, e_idx), prev_title)
        else:
            merged_ranges.append((s_idx, e_idx, title))

    # Check contiguous coverage without gaps or overlaps
    curr_start = 0
    final_ranges: list[tuple[int, int, str]] = []

    for i, (s_idx, e_idx, title) in enumerate(merged_ranges):
        if i == 0 and s_idx != 0:
            # Force start at 0
            s_idx = 0

        if s_idx != curr_start:
            # Adjust gap or overlap deterministically to maintain continuity
            if s_idx > curr_start:
                # Gap -> adjust start_idx to cover gap
                s_idx = curr_start
            elif s_idx < curr_start:
                # Overlap -> adjust start_idx to after previous end
                s_idx = curr_start

        if s_idx > e_idx:
            continue

        final_ranges.append((s_idx, e_idx, title))
        curr_start = e_idx + 1

    # Ensure last range extends to total_segments - 1
    if final_ranges:
        last_s, last_e, last_title = final_ranges[-1]
        if last_e < total_segments - 1:
            final_ranges[-1] = (last_s, total_segments - 1, last_title)
    else:
        return fallback

    return final_ranges
