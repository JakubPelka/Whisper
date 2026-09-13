#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""whisper.cpp v1.8.6 CUDA Parity Transcription Manager for JakubPelka/Whisper.

Manages whisper.cpp v1.8.6 CUDA runtime binary, model downloading, checksum
verification, and transcription execution with parameter parity matching Antek
iOS.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger("whisper_cpp_runtime")

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "sv": {
        "id": "kb-whisper-large-q5_0",
        "file": "kb-whisper-large-q5_0.bin",
        "quantization": "q5_0",
        "url": "https://huggingface.co/KBLab/kb-whisper-large/resolve/main/ggml-model-q5_0.bin",
        "sha256": "6d2863812d7410322bb7d8647a5c7260761300fa946714c9ed66d22bb30bcb19",
        "expected_size": 1_081_140_203,
    },
    "default": {
        "id": "whisper-large-v3-turbo-q5_0",
        "file": "ggml-large-v3-turbo-q5_0.bin",
        "quantization": "q5_0",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        "sha256": "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
        "expected_size": 574_041_195,
    },
}


def get_whisper_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_models_dir(custom_dir: Path | str | None = None) -> Path:
    if custom_dir:
        path = Path(custom_dir).expanduser().resolve()
    elif os.environ.get("WHISPER_MODELS_DIR"):
        path = Path(os.environ["WHISPER_MODELS_DIR"]).expanduser().resolve()
    else:
        path = get_whisper_repo_root() / "cache" / "whisper.cpp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_whisper_cli_path(custom_path: Path | str | None = None) -> Path:
    if custom_path:
        path = Path(custom_path).expanduser().resolve()
    elif os.environ.get("WHISPER_CPP_PATH"):
        path = Path(os.environ["WHISPER_CPP_PATH"]).expanduser().resolve()
    else:
        path = (
            get_whisper_repo_root()
            / "runtime"
            / "whisper.cpp-v1.8.6"
            / "build"
            / "bin"
            / "whisper-cli"
        )
    return path


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_model(language: str, models_dir: Path | str | None = None) -> tuple[Path, dict[str, Any]]:
    target_dir = resolve_models_dir(models_dir)
    lang_key = (language or "").strip().lower()
    spec = MODEL_REGISTRY.get(lang_key, MODEL_REGISTRY["default"])

    model_path = target_dir / spec["file"]
    expected_sha = spec["sha256"]

    if model_path.is_file():
        actual_sha = compute_sha256(model_path)
        if actual_sha == expected_sha:
            LOGGER.info("Model %s verified (SHA256: %s)", spec["file"], actual_sha)
            return model_path, spec
        else:
            LOGGER.warning(
                "Model %s corrupted or hash mismatch (got %s, expected %s). Redownloading...",
                spec["file"],
                actual_sha,
                expected_sha,
            )
            model_path.unlink(missing_ok=True)

    LOGGER.info("Downloading %s from %s...", spec["file"], spec["url"])
    temp_path = model_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(spec["url"], temp_path)
        downloaded_sha = compute_sha256(temp_path)
        if downloaded_sha != expected_sha:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 checksum mismatch for downloaded model {spec['file']}. "
                f"Expected {expected_sha}, got {downloaded_sha}"
            )
        temp_path.rename(model_path)
        LOGGER.info("Successfully downloaded and verified %s", spec["file"])
        return model_path, spec
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download/verify whisper model {spec['file']}: {e}") from e


def ensure_whisper_cli(custom_path: Path | str | None = None) -> Path:
    cli_path = resolve_whisper_cli_path(custom_path)
    if cli_path.is_file() and os.access(cli_path, os.X_OK):
        return cli_path

    raise RuntimeError(
        f"whisper-cli executable not found or not executable at '{cli_path}'. "
        "Ensure whisper.cpp v1.8.6 with CUDA is built under runtime/ or WHISPER_CPP_PATH is set."
    )


def transcribe_with_whisper_cpp(
    audio_path: Path,
    language: str = "auto",
    initial_prompt: str | None = None,
    work_dir: Path | None = None,
    cli_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    num_threads: int = 4,
    cancel_checker: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Runs whisper.cpp CLI with CUDA parity parameters and returns standardized dictionary.

    Returns dict with structure:
    {
        "language": str,
        "segments": [{"start": float, "end": float, "text": str}],
        "model_info": dict,
        "runtime_info": dict,
    }
    """
    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {audio_path}")

    executable = ensure_whisper_cli(cli_path)
    model_path, model_spec = ensure_model(language, models_dir)

    target_lang = (language or "auto").strip().lower()
    if target_lang in ("none", "-", ""):
        target_lang = "auto"

    if work_dir:
        out_dir = Path(work_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = out_dir / f"whisper_cpp_{audio_path.stem}"
    else:
        temp_d = tempfile.TemporaryDirectory(prefix="whisper_cpp_")
        out_dir = Path(temp_d.name)
        out_prefix = out_dir / "output"

    cmd = [
        str(executable),
        "-m",
        str(model_path),
        "-f",
        str(audio_path),
        "-l",
        target_lang,
        "-bs",
        "1",
        "-bo",
        "1",
        "-mc",
        "0",
        "-t",
        str(num_threads),
        "-oj",
        "-of",
        str(out_prefix),
    ]

    if initial_prompt and initial_prompt.strip():
        cmd.extend(["--prompt", initial_prompt.strip()])

    LOGGER.info("Executing whisper-cli: %s", " ".join(cmd))

    if cancel_checker and cancel_checker():
        raise RuntimeError("Transcription cancelled before subprocess launch.")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        while proc.poll() is None:
            if cancel_checker and cancel_checker():
                LOGGER.warning("Cancellation signal detected. Terminating whisper-cli process group...")
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
                if proc.poll() is None:
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        pass
                raise RuntimeError("Whisper.cpp transcription job was cancelled.")
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass

        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"whisper-cli failed with return code {proc.returncode}:\n{stderr}\n{stdout}"
            )

    except Exception:
        if proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
        raise

    json_file = out_prefix.with_suffix(".json")
    if not json_file.is_file():
        raise FileNotFoundError(f"whisper-cli completed but json output file is missing: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    detected_lang = raw_data.get("result", {}).get("language", target_lang)
    raw_segments = raw_data.get("transcription", [])

    parsed_segments: list[dict[str, Any]] = []
    for i, seg in enumerate(raw_segments):
        text = (seg.get("text") or "").strip()
        offsets = seg.get("offsets", {})
        start_sec = round(offsets.get("from", 0) / 1000.0, 3)
        end_sec = round(offsets.get("to", 0) / 1000.0, 3)
        if text:
            parsed_segments.append(
                {
                    "segment_id": i,
                    "start": start_sec,
                    "end": end_sec,
                    "text": text,
                }
            )

    runtime_info = {
        "runtime": "whisper.cpp",
        "version": "1.8.6",
        "backend": "CUDA",
        "executable": str(executable),
        "sampling": "greedy",
        "no_context": True,
        "carry_initial_prompt": False,
        "has_initial_prompt": bool(initial_prompt and initial_prompt.strip()),
    }

    return {
        "language": detected_lang,
        "segments": parsed_segments,
        "model_info": model_spec,
        "runtime_info": runtime_info,
        "json_output_path": str(json_file),
    }
