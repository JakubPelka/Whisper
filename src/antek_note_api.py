#!/usr/bin/env python3
"""Thin authenticated text-only HTTP API for Antek note generation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
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
    "presentationSummary",
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
    language: str = Field(min_length=1, max_length=32)
    note_preset: NotePreset = "meetingNotes"
    meeting_context: str | None = Field(default=None, max_length=20_000)
    recording_duration_seconds: float | None = Field(default=None, ge=0, le=86_400)
    transcript: TranscriptRequest


class AnonymousBootstrapRequest(BaseModel):
    installation_id: UUID
    app_version: str = Field(min_length=1, max_length=64)


class CreditsResponse(BaseModel):
    subject_id: UUID
    included_credits: int
    purchased_credits: int
    available_credits: int


class AnonymousBootstrapResponse(CreditsResponse):
    access_credential: str


class ProcessingEstimate(BaseModel):
    calibration_version: str
    transcript_credits: int
    context_credits: int
    reference_credits: int = 0
    total_credits: int


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
_SCHEMA_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ledger_database_path() -> str:
    # The test suite deliberately injects a temporary path. Production must set a
    # durable path explicitly rather than silently using a database in /tmp.
    return os.environ.get("ANTEK_LEDGER_DB", "antek-processing-ledger.sqlite3")


def database() -> sqlite3.Connection:
    connection = sqlite3.connect(ledger_database_path(), timeout=15, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    initialize_schema(connection)
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    # SQLite DDL is idempotent. The lock avoids a first-request race under the
    # development server while BEGIN IMMEDIATE protects balances/operations.
    with _SCHEMA_LOCK:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS subjects (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                future_account_type TEXT,
                future_account_external_id TEXT
            );
            CREATE TABLE IF NOT EXISTS installations (
                id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL REFERENCES subjects(id),
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                app_version TEXT NOT NULL,
                revoked_at TEXT
            );
            CREATE TABLE IF NOT EXISTS access_credentials (
                credential_hash TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL REFERENCES subjects(id),
                installation_id TEXT NOT NULL REFERENCES installations(id),
                created_at TEXT NOT NULL,
                revoked_at TEXT
            );
            CREATE TABLE IF NOT EXISTS credit_accounts (
                subject_id TEXT PRIMARY KEY REFERENCES subjects(id),
                included_balance INTEGER NOT NULL DEFAULT 0,
                purchased_balance INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS credit_ledger (
                transaction_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL REFERENCES subjects(id),
                operation_id TEXT,
                entry_type TEXT NOT NULL CHECK(entry_type IN ('grant','charge','refund','adjustment')),
                bucket TEXT NOT NULL CHECK(bucket IN ('included','purchased')),
                credit_delta INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                pricing_version TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(subject_id, operation_id, entry_type, bucket)
            );
            CREATE TABLE IF NOT EXISTS processing_operations (
                operation_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL REFERENCES subjects(id),
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                recording_duration_seconds REAL,
                transcript_utf8_bytes INTEGER NOT NULL,
                transcript_character_count INTEGER NOT NULL,
                context_present INTEGER NOT NULL,
                context_utf8_bytes INTEGER NOT NULL,
                reference_material_present INTEGER NOT NULL DEFAULT 0,
                reference_material_utf8_bytes INTEGER NOT NULL DEFAULT 0,
                note_preset TEXT NOT NULL,
                transcript_language TEXT NOT NULL,
                privacy_masking_enabled INTEGER NOT NULL DEFAULT 0,
                pipeline_version TEXT NOT NULL,
                pricing_version TEXT NOT NULL,
                estimated_credits INTEGER NOT NULL,
                credits_charged INTEGER NOT NULL DEFAULT 0,
                supplier_cost_micros INTEGER NOT NULL DEFAULT 0,
                processing_elapsed_ms INTEGER
            );
            CREATE TABLE IF NOT EXISTS operation_response_cache (
                operation_id TEXT PRIMARY KEY REFERENCES processing_operations(operation_id),
                response_json TEXT NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS provider_usage_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT NOT NULL REFERENCES processing_operations(operation_id),
                attempt_number INTEGER NOT NULL,
                provider_stage TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                supplier_cost_micros INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS processing_operations_created_at
                ON processing_operations(created_at);
            """
        )


def credential_hash(credential: str) -> str:
    return hashlib.sha256(credential.encode("utf-8")).hexdigest()


def bootstrap_secret() -> str:
    # ANTEK_API_TOKEN is accepted only as a short private-alpha migration bridge.
    # It is never accepted as a Generate credential.
    return os.environ.get("ANTEK_BOOTSTRAP_TOKEN") or os.environ.get("ANTEK_API_TOKEN", "")


def private_alpha_grant() -> int:
    return max(0, int(os.environ.get("ANTEK_PRIVATE_ALPHA_INCLUDED_CREDITS", "100000")))


def balances(connection: sqlite3.Connection, subject_id: str) -> CreditsResponse:
    row = connection.execute(
        "SELECT included_balance, purchased_balance FROM credit_accounts WHERE subject_id = ?",
        (subject_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown subject")
    included, purchased = max(0, row["included_balance"]), max(0, row["purchased_balance"])
    return CreditsResponse(
        subject_id=UUID(subject_id),
        included_credits=included,
        purchased_credits=purchased,
        available_credits=included + purchased,
    )


def append_ledger_entry(
    connection: sqlite3.Connection,
    *,
    subject_id: str,
    entry_type: Literal["grant", "charge", "refund", "adjustment"],
    bucket: Literal["included", "purchased"],
    credit_delta: int,
    operation_id: str | None = None,
    pricing_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        """INSERT INTO credit_ledger
           (transaction_id, subject_id, operation_id, entry_type, bucket, credit_delta, created_at, pricing_version, metadata_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (str(uuid4()), subject_id, operation_id, entry_type, bucket, credit_delta, utc_now(), pricing_version, json.dumps(metadata or {})),
    )
    column = "included_balance" if bucket == "included" else "purchased_balance"
    connection.execute(
        f"UPDATE credit_accounts SET {column} = {column} + ?, updated_at = ? WHERE subject_id = ?",
        (credit_delta, utc_now(), subject_id),
    )


def subject_for_credentials(credentials: HTTPAuthorizationCredentials | None) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Antek access credential")
    connection = database()
    try:
        row = connection.execute(
            """SELECT subject_id FROM access_credentials
               WHERE credential_hash = ? AND revoked_at IS NULL""",
            (credential_hash(credentials.credentials),),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Antek access credential")
        return str(row["subject_id"])
    finally:
        connection.close()


def require_subject(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    return subject_for_credentials(credentials)


def byte_count(value: str | None) -> int:
    return len((value or "").encode("utf-8"))


def transcript_text_for_measurement(request: GenerateNoteRequest) -> str:
    return "\n".join(segment.text for segment in request.transcript.segments)


def pipeline_version_for(note_preset: NotePreset) -> str:
    """Keep independently tuned note presets distinguishable in usage records."""
    if note_preset == "meetingNotes":
        return os.environ.get(
            "ANTEK_MEETING_PIPELINE_VERSION",
            "luna-terra-v4-natural-notes",
        )
    if note_preset == "shortSummary":
        return os.environ.get(
            "ANTEK_SHORT_SUMMARY_PIPELINE_VERSION",
            "luna-terra-v2-short-summary",
        )
    if note_preset == "presentationSummary":
        return os.environ.get(
            "ANTEK_PRESENTATION_PIPELINE_VERSION",
            "luna-terra-v2-presentation-summary",
        )
    return os.environ.get("ANTEK_PIPELINE_VERSION", "luna-terra-v3-natural-notes")


def estimate_for(request: GenerateNoteRequest) -> ProcessingEstimate:
    # The client has the same versioned, deliberately conservative approximation.
    # Actual customer settlement remains based on provider usage after Generate.
    transcript_credits = max(1, (byte_count(transcript_text_for_measurement(request)) + 999) // 1000)
    context_credits = (byte_count(request.meeting_context) + 999) // 1000
    preset_margin = 2 if request.note_preset in {"meetingNotes", "decisionsAndActions"} else 1
    return ProcessingEstimate(
        calibration_version=os.environ.get("ANTEK_CALIBRATION_VERSION", "private-alpha-v1"),
        transcript_credits=transcript_credits,
        context_credits=context_credits,
        total_credits=transcript_credits + context_credits + preset_margin,
    )


def canonical_segments(request: GenerateNoteRequest) -> list[dict[str, Any]]:
    return [
        {"id": index, "start": segment.start, "end": segment.end, "text": segment.text}
        for index, segment in enumerate(request.transcript.segments, start=1)
    ]


def integer_usage(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def usage_value(usage: dict[str, Any] | None, *names: str) -> int:
    if not usage:
        return 0
    for name in names:
        if name in usage:
            return integer_usage(usage[name])
    return 0


def configured_cost_micros(stage: str, *, input_tokens: int, cached_tokens: int, output_tokens: int, reasoning_tokens: int) -> int:
    """Cost is configured server-side so provider pricing never reaches iOS."""
    prefix = f"ANTEK_{stage.upper()}"
    def rate(name: str) -> int:
        return max(0, int(os.environ.get(f"{prefix}_{name}_COST_MICROS_PER_MILLION", "0")))
    return (
        input_tokens * rate("INPUT")
        + cached_tokens * rate("CACHED_INPUT")
        + output_tokens * rate("OUTPUT")
        + reasoning_tokens * rate("REASONING")
    ) // 1_000_000


def record_provider_usage(
    connection: sqlite3.Connection,
    *,
    operation_id: str,
    metadata_usage: dict[str, dict[str, Any] | None],
    draft_model: str,
    verification_model: str,
) -> tuple[int, int]:
    total_tokens = 0
    total_cost_micros = 0
    for stage, model in (("luna", draft_model), ("terra", verification_model)):
        raw = metadata_usage.get("draft" if stage == "luna" else "verification")
        input_tokens = usage_value(raw, "input_tokens", "prompt_tokens")
        cached_tokens = usage_value(raw, "cached_input_tokens")
        output_tokens = usage_value(raw, "output_tokens", "completion_tokens")
        details = raw.get("output_tokens_details", {}) if raw else {}
        reasoning_tokens = usage_value(details, "reasoning_tokens")
        total_tokens += input_tokens + cached_tokens + output_tokens + reasoning_tokens
        supplier_cost_micros = configured_cost_micros(
            stage, input_tokens=input_tokens, cached_tokens=cached_tokens,
            output_tokens=output_tokens, reasoning_tokens=reasoning_tokens,
        )
        total_cost_micros += supplier_cost_micros
        connection.execute(
            """INSERT INTO provider_usage_attempts
               (operation_id, attempt_number, provider_stage, model, input_tokens, cached_input_tokens,
                output_tokens, reasoning_tokens, supplier_cost_micros, created_at)
               VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (operation_id, stage, model, input_tokens, cached_tokens, output_tokens, reasoning_tokens, supplier_cost_micros, utc_now()),
        )
    # Keep the operation summary as a sum: one customer operation can involve
    # several provider calls/attempt rows in a future retry-capable engine.
    connection.execute(
        "UPDATE processing_operations SET supplier_cost_micros = supplier_cost_micros + ? WHERE operation_id = ?",
        (total_cost_micros, operation_id),
    )
    return total_tokens, total_cost_micros


def actual_charge_for(total_provider_tokens: int, fallback_estimate: int) -> int:
    # This is deliberately simple for the private alpha. The pricing/calibration
    # version makes the rule auditable and replaceable without rewriting history.
    if total_provider_tokens <= 0:
        return fallback_estimate
    return max(1, (total_provider_tokens + 999) // 1000)


def settle_charge(
    connection: sqlite3.Connection,
    *,
    subject_id: str,
    operation_id: str,
    credits: int,
    pricing_version: str,
) -> None:
    account = balances(connection, subject_id)
    if account.available_credits < credits:
        # A generous preflight estimate should make this exceptional. Do not create
        # a negative balance or charge a different operation as a side effect.
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="insufficient_credits")
    remaining = credits
    included = min(account.included_credits, remaining)
    if included:
        append_ledger_entry(
            connection, subject_id=subject_id, operation_id=operation_id,
            entry_type="charge", bucket="included", credit_delta=-included,
            pricing_version=pricing_version,
        )
        remaining -= included
    if remaining:
        append_ledger_entry(
            connection, subject_id=subject_id, operation_id=operation_id,
            entry_type="charge", bucket="purchased", credit_delta=-remaining,
            pricing_version=pricing_version,
        )


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


@app.post("/v1/auth/bootstrap", response_model=AnonymousBootstrapResponse)
def bootstrap_anonymous_subject(
    request: AnonymousBootstrapRequest,
    x_antek_bootstrap: str | None = Header(default=None),
) -> AnonymousBootstrapResponse:
    expected = bootstrap_secret()
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Bootstrap authentication is not configured")
    if not x_antek_bootstrap or not secrets.compare_digest(x_antek_bootstrap, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bootstrap credential")

    connection = database()
    try:
        connection.execute("BEGIN IMMEDIATE")
        installation_id = str(request.installation_id)
        existing = connection.execute(
            "SELECT subject_id FROM installations WHERE id = ? AND revoked_at IS NULL", (installation_id,)
        ).fetchone()
        if existing:
            subject_id = str(existing["subject_id"])
            connection.execute(
                "UPDATE installations SET last_seen_at = ?, app_version = ? WHERE id = ?",
                (utc_now(), request.app_version, installation_id),
            )
        else:
            subject_id = str(uuid4())
            now = utc_now()
            connection.execute("INSERT INTO subjects (id, created_at, status) VALUES (?, ?, 'anonymous')", (subject_id, now))
            connection.execute(
                """INSERT INTO installations (id, subject_id, created_at, last_seen_at, app_version)
                   VALUES (?, ?, ?, ?, ?)""",
                (installation_id, subject_id, now, now, request.app_version),
            )
            connection.execute(
                """INSERT INTO credit_accounts (subject_id, included_balance, purchased_balance, updated_at)
                   VALUES (?, 0, 0, ?)""",
                (subject_id, now),
            )
            grant = private_alpha_grant()
            if grant:
                append_ledger_entry(
                    connection, subject_id=subject_id, entry_type="grant", bucket="included",
                    credit_delta=grant, pricing_version="private-alpha-grant-v1",
                    metadata={"source": "private_alpha_bootstrap"},
                )
        credential = secrets.token_urlsafe(48)
        connection.execute(
            """INSERT INTO access_credentials (credential_hash, subject_id, installation_id, created_at)
               VALUES (?, ?, ?, ?)""",
            (credential_hash(credential), subject_id, installation_id, utc_now()),
        )
        balance = balances(connection, subject_id)
        connection.execute("COMMIT")
        return AnonymousBootstrapResponse(**balance.model_dump(), access_credential=credential)
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


@app.get("/v1/me/credits", response_model=CreditsResponse)
def credits_endpoint(subject_id: str = Depends(require_subject)) -> CreditsResponse:
    connection = database()
    try:
        return balances(connection, subject_id)
    finally:
        connection.close()


@app.post("/v1/processing/estimate", response_model=ProcessingEstimate)
def estimate_endpoint(request: GenerateNoteRequest, subject_id: str = Depends(require_subject)) -> ProcessingEstimate:
    # Deliberately not used by the app before Generate: this exists for contract
    # parity/testing only and accepts text only after authenticated submission.
    _ = subject_id
    return estimate_for(request)


@app.post(
    "/v1/notes/generate",
    response_model=GenerateNoteResponse,
)
async def generate_note_endpoint(
    request: GenerateNoteRequest,
    subject_id: str = Depends(require_subject),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> GenerateNoteResponse:
    try:
        idempotency_request_id = UUID(idempotency_key) if idempotency_key else None
    except ValueError:
        idempotency_request_id = None
    if idempotency_request_id != request.request_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Idempotency-Key must equal request_id")
    connection = database()
    started = time.monotonic()
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """SELECT operation.subject_id, operation.status, cache.response_json, cache.expires_at
               FROM processing_operations operation
               LEFT JOIN operation_response_cache cache ON cache.operation_id = operation.operation_id
               WHERE operation.operation_id = ?""",
            (str(request.request_id),),
        ).fetchone()
        if existing:
            if existing["subject_id"] != subject_id:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="operation_id belongs to another subject")
            if existing["status"] == "succeeded" and existing["response_json"] and (existing["expires_at"] or 0) > time.time():
                connection.execute("COMMIT")
                return GenerateNoteResponse.model_validate_json(existing["response_json"])
            connection.execute("COMMIT")
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="operation_already_completed")

        estimate = estimate_for(request)
        if balances(connection, subject_id).available_credits < estimate.total_credits:
            connection.execute("COMMIT")
            raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="insufficient_credits")
        transcript_text = transcript_text_for_measurement(request)
        context_bytes = byte_count(request.meeting_context)
        connection.execute(
            """INSERT INTO processing_operations
               (operation_id, subject_id, status, created_at, recording_duration_seconds, transcript_utf8_bytes,
                transcript_character_count, context_present, context_utf8_bytes, note_preset, transcript_language,
                pipeline_version, pricing_version, estimated_credits)
               VALUES (?, ?, 'processing', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(request.request_id), subject_id, utc_now(), request.recording_duration_seconds,
                byte_count(transcript_text), len(transcript_text), int(context_bytes > 0), context_bytes,
                request.note_preset, request.language,
                pipeline_version_for(request.note_preset),
                os.environ.get("ANTEK_PRICING_VERSION", "private-alpha-v1"), estimate.total_credits,
            ),
        )
        connection.execute("COMMIT")

        response = await run_in_threadpool(generate_note, request)
        connection.execute("BEGIN IMMEDIATE")
        models = response.models
        total_tokens, _supplier_cost_micros = record_provider_usage(
            connection, operation_id=str(request.request_id), metadata_usage=response.usage,
            draft_model=models.draft, verification_model=models.verification,
        )
        pricing_version = os.environ.get("ANTEK_PRICING_VERSION", "private-alpha-v1")
        charge = actual_charge_for(total_tokens, estimate.total_credits)
        settle_charge(
            connection, subject_id=subject_id, operation_id=str(request.request_id),
            credits=charge, pricing_version=pricing_version,
        )
        elapsed_ms = round((time.monotonic() - started) * 1000)
        connection.execute(
            """UPDATE processing_operations SET status = 'succeeded', completed_at = ?, credits_charged = ?,
               processing_elapsed_ms = ? WHERE operation_id = ?""",
            (utc_now(), charge, elapsed_ms, str(request.request_id)),
        )
        # Short-lived retry cache; deliberately separate from the ledger and the
        # non-content processing-statistics record.
        connection.execute(
            "INSERT INTO operation_response_cache (operation_id, response_json, expires_at) VALUES (?, ?, ?)",
            (str(request.request_id), response.model_dump_json(), time.time() + 3600),
        )
        connection.execute("COMMIT")
        return response
    except HTTPException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        # Do not log transcript/context or credentials. request_id is safe.
        LOGGER.exception("note generation failed request_id=%s", request.request_id)
        failure_connection = database()
        try:
            failure_connection.execute(
                "UPDATE processing_operations SET status = 'failed', completed_at = ?, processing_elapsed_ms = ? WHERE operation_id = ? AND subject_id = ?",
                (utc_now(), round((time.monotonic() - started) * 1000), str(request.request_id), subject_id),
            )
        finally:
            failure_connection.close()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Note generation failed for request {request.request_id}",
        ) from None
    finally:
        connection.close()


@app.get("/v1/internal/processing-stats")
def processing_statistics(
    x_antek_admin: str | None = Header(default=None),
) -> dict[str, Any]:
    expected = os.environ.get("ANTEK_ADMIN_TOKEN", "")
    if not expected or not x_antek_admin or not secrets.compare_digest(expected, x_antek_admin):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin access required")
    connection = database()
    try:
        row = connection.execute(
            """SELECT COUNT(*) AS operation_count,
                      COALESCE(SUM(recording_duration_seconds), 0) AS total_recording_seconds,
                      COALESCE(SUM(transcript_utf8_bytes), 0) AS total_transcript_bytes,
                      COALESCE(SUM(context_utf8_bytes), 0) AS total_context_bytes,
                      COALESCE(SUM(credits_charged), 0) AS total_credits_charged,
                      COALESCE(SUM(supplier_cost_micros), 0) AS total_supplier_cost_micros,
                      COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) AS failed_operations
               FROM processing_operations"""
        ).fetchone()
        return dict(row)
    finally:
        connection.close()
