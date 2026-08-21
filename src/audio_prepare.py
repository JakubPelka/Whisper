#!/usr/bin/env python3
"""Probe media and prepare a decodable mono 16 kHz PCM WAV for Whisper."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class AudioPreparationError(RuntimeError):
    """Raised when no audio stream can be converted locally."""

    def __init__(self, message: str, *, concise_reason: str) -> None:
        super().__init__(message)
        self.concise_reason = concise_reason


@dataclass(frozen=True)
class PreparedAudio:
    path: Path
    stream_index: int
    codec: str
    sample_rate: int | None
    channels: int | None


def _run(
    command: list[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def probe_media(source: Path) -> dict[str, Any]:
    """Return ffprobe JSON without printing container metadata."""
    try:
        result = _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(source),
            ],
            timeout=int(os.environ.get("FFPROBE_TIMEOUT_SECONDS", "120")),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioPreparationError(
            f"Could not inspect audio locally: {exc}\n\nSource file was not modified.",
            concise_reason="ffprobe could not inspect the recording",
        ) from exc

    if result.returncode != 0:
        detail = _single_line(result.stderr) or "ffprobe rejected the input"
        raise AudioPreparationError(
            f"Could not inspect audio locally: {detail}\n\nSource file was not modified.",
            concise_reason="ffprobe could not inspect the recording",
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AudioPreparationError(
            "ffprobe returned invalid stream information.\n\nSource file was not modified.",
            concise_reason="ffprobe returned invalid stream information",
        ) from exc


def _codec_label(stream: dict[str, Any]) -> str:
    codec_name = str(stream.get("codec_name") or "").lower()
    codec_tag = str(stream.get("codec_tag_string") or "").lower()
    if codec_name:
        return codec_name
    if codec_tag and codec_tag != "[0][0][0][0]":
        return codec_tag
    return "unknown"


def _stream_priority(stream: dict[str, Any]) -> tuple[int, int, int]:
    codec = _codec_label(stream)
    compatibility_rank = 0 if (
        codec == "aac" or codec == "alac" or codec.startswith("pcm_")
    ) else 2 if codec == "apac" else 1
    default_rank = 0 if int((stream.get("disposition") or {}).get("default", 0)) else 1
    return compatibility_rank, default_rank, int(stream.get("index", 0))


def audio_streams(probe: dict[str, Any]) -> list[dict[str, Any]]:
    streams = [
        stream
        for stream in probe.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]
    return sorted(streams, key=_stream_priority)


def describe_stream(stream: dict[str, Any]) -> str:
    codec = _codec_label(stream).upper()
    sample_rate = stream.get("sample_rate") or "unknown rate"
    channels = stream.get("channels") or "unknown channels"
    return f"stream {stream.get('index', '?')}: {codec}, {sample_rate} Hz, {channels} channel(s)"


def _single_line(value: str, limit: int = 400) -> str:
    compact = " ".join(value.split())
    return compact[:limit]


def _failure_message(source: Path, streams: list[dict[str, Any]], detail: str = "") -> str:
    qta = source.suffix.lower() == ".qta"
    heading = (
        "QTA audio was detected, but no locally decodable audio stream was found."
        if qta
        else "No locally decodable audio stream was found."
    )
    detected = "\n".join(f"- {describe_stream(stream)}" for stream in streams)
    if not detected:
        detected = "- no audio streams"
    message = f"{heading}\n\nDetected streams:\n{detected}"
    if detail:
        message += f"\n\nFFmpeg detail: {_single_line(detail)}"
    return f"{message}\n\nSource file was not modified."


def prepare_audio(source: Path, output_wav: Path) -> PreparedAudio:
    """Select a usable stream and convert it to mono 16 kHz PCM WAV."""
    source = source.expanduser().resolve()
    output_wav = output_wav.expanduser().resolve()
    probe = probe_media(source)
    streams = audio_streams(probe)
    if not streams:
        raise AudioPreparationError(
            _failure_message(source, streams),
            concise_reason="no audio streams were detected",
        )

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    last_detail = ""
    timeout = int(os.environ.get("FFMPEG_TIMEOUT_SECONDS", "1800"))

    for stream in streams:
        output_wav.unlink(missing_ok=True)
        stream_index = int(stream["index"])
        try:
            result = _run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-map",
                    f"0:{stream_index}",
                    "-vn",
                    "-sn",
                    "-dn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(output_wav),
                ],
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_detail = str(exc)
            continue

        if result.returncode == 0 and output_wav.is_file() and output_wav.stat().st_size > 44:
            return PreparedAudio(
                path=output_wav,
                stream_index=stream_index,
                codec=_codec_label(stream),
                sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
                channels=int(stream["channels"]) if stream.get("channels") else None,
            )
        last_detail = result.stderr

    output_wav.unlink(missing_ok=True)
    raise AudioPreparationError(
        _failure_message(source, streams, last_detail),
        concise_reason="ffmpeg could not decode any detected audio stream",
    )


@contextmanager
def prepared_audio(source: Path) -> Iterator[PreparedAudio]:
    """Yield a temporary prepared WAV and remove it after use."""
    with tempfile.TemporaryDirectory(prefix="whisper_audio_") as temp_dir:
        output_wav = Path(temp_dir) / f"{source.stem}_16k_mono.wav"
        yield prepare_audio(source, output_wav)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source audio/video recording")
    parser.add_argument("--output", required=True, help="Destination WAV used for an explicit conversion test")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    try:
        prepared = prepare_audio(source, output)
    except AudioPreparationError as exc:
        print(f"AUDIO_PREPARATION_ERROR: {exc.concise_reason}", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 3

    print(
        f"selected_stream={prepared.stream_index} codec={prepared.codec} "
        f"source_rate={prepared.sample_rate} source_channels={prepared.channels}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
