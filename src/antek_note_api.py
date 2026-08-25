#!/usr/bin/env python3
"""Thin authenticated text-only HTTP API for Antek note generation."""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, model_validator

from generate_meeting_note import (
    MeetingNote,
    atomic_write_json,
    create_note_with_api,
    ensure_api_key,
    render_note_markdown,
    transcript_for_prompt,
    validate_evidence,
)


LOGGER = logging.getLogger("antek_note_api")
NotePreset = Literal[
    "serviceNote",
    "meetingNotes",
    "decisionsAndActions",
    "conversationNote",
    "shortSummary",
]


class TranscriptSegmentRequest(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_segment(self) -> "TranscriptSegmentRequest":
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("segment text must not be blank")
        if self.end < self.start:
            raise ValueError("segment end must not be before start")
        return self


class TranscriptRequest(BaseModel):
    segments: list[TranscriptSegmentRequest] = Field(min_length=1)


class GenerateNoteRequest(BaseModel):
    request_id: UUID
    meeting_id: UUID
    language: str = Field(min_length=1, max_length=32)
    note_preset: NotePreset = "meetingNotes"
    meeting_context: str | None = Field(default=None, max_length=20_000)
    transcript: TranscriptRequest


class VerificationResponse(BaseModel):
    removed_or_corrected_claims: list[str]
    warnings: list[str]


class ModelsResponse(BaseModel):
    draft: str
    verification: str


class ResponseIDs(BaseModel):
    draft: str | None
    verification: str | None


class GenerateNoteResponse(BaseModel):
    request_id: UUID
    status: Literal["verified"]
    note_text: str
    final_note: MeetingNote
    verification: VerificationResponse
    models: ModelsResponse
    response_ids: ResponseIDs
    usage: dict[str, dict[str, Any] | None]


security = HTTPBearer(auto_error=False)
app = FastAPI(title="Antek private note API", version="1.0.0")


def require_private_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    expected = os.environ.get("ANTEK_API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="API authentication is not configured")
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not secrets.compare_digest(credentials.credentials, expected)
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")


def canonical_segments(request: GenerateNoteRequest) -> list[dict[str, Any]]:
    return [
        {"id": index, "start": segment.start, "end": segment.end, "text": segment.text}
        for index, segment in enumerate(request.transcript.segments, start=1)
    ]


def generate_note(request: GenerateNoteRequest) -> GenerateNoteResponse:
    ensure_api_key(None)
    segments = canonical_segments(request)
    transcript_text = transcript_for_prompt(segments)
    draft_model = os.environ.get("ANTEK_DRAFT_MODEL", "gpt-5.6-luna")
    verification_model = os.environ.get("ANTEK_VERIFICATION_MODEL", "gpt-5.6-terra")
    reasoning_effort = os.environ.get("ANTEK_REASONING_EFFORT", "medium")
    draft, verification, metadata = create_note_with_api(
        transcript_text=transcript_text,
        language=request.language,
        meeting_context=request.meeting_context,
        note_preset=request.note_preset,
        draft_model=draft_model,
        verification_model=verification_model,
        reasoning_effort=reasoning_effort,
    )
    validate_evidence(verification.final_note, segments)
    note_text = render_note_markdown(verification.final_note, request.language, request.note_preset)
    usage = metadata["api_usage"]
    response = GenerateNoteResponse(
        request_id=request.request_id,
        status="verified",
        note_text=note_text,
        final_note=verification.final_note,
        verification=VerificationResponse(
            removed_or_corrected_claims=verification.removed_or_corrected_claims,
            warnings=verification.verification_warnings,
        ),
        models=ModelsResponse(draft=draft_model, verification=verification_model),
        response_ids=ResponseIDs(
            draft=metadata["draft_response_id"],
            verification=metadata["verification_response_id"],
        ),
        usage=usage,
    )
    save_audit(
        request=request,
        response=response,
        transcript_text=transcript_text,
        reasoning_effort=reasoning_effort,
        draft_note=draft,
    )
    return response


def save_audit(
    *,
    request: GenerateNoteRequest,
    response: GenerateNoteResponse,
    transcript_text: str,
    reasoning_effort: str,
    draft_note: MeetingNote,
) -> None:
    configured_directory = os.environ.get("ANTEK_AUDIT_DIR")
    if not configured_directory:
        return
    audit_directory = Path(configured_directory).expanduser().resolve()
    atomic_write_json(
        audit_directory / f"{request.request_id}.note.json",
        {
            "request_id": str(request.request_id),
            "meeting_id": str(request.meeting_id),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "transcript_sha256": hashlib.sha256(transcript_text.encode("utf-8")).hexdigest(),
            "language": request.language,
            "note_preset": request.note_preset,
            "models": response.models.model_dump(mode="json"),
            "reasoning_effort": reasoning_effort,
            "response_ids": response.response_ids.model_dump(mode="json"),
            "usage": response.usage,
            "draft_note": draft_note.model_dump(mode="json"),
            "removed_or_corrected_claims": response.verification.removed_or_corrected_claims,
            "verification_warnings": response.verification.warnings,
            "final_note": response.final_note.model_dump(mode="json"),
        },
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/notes/generate",
    response_model=GenerateNoteResponse,
    dependencies=[Depends(require_private_token)],
)
async def generate_note_endpoint(request: GenerateNoteRequest) -> GenerateNoteResponse:
    try:
        return await run_in_threadpool(generate_note, request)
    except HTTPException:
        raise
    except Exception:
        LOGGER.exception("note generation failed request_id=%s", request.request_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Note generation failed for request {request.request_id}",
        ) from None
