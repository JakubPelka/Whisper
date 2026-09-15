#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for whisper_cpp_runtime.py in JakubPelka/Whisper."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import pytest

from whisper_cpp_runtime import (
    MODEL_REGISTRY,
    compute_sha256,
    ensure_model,
    ensure_whisper_cli,
    resolve_models_dir,
    resolve_whisper_cli_path,
    transcribe_with_whisper_cpp,
)


def test_model_registry_specs():
    assert "sv" in MODEL_REGISTRY
    assert "default" in MODEL_REGISTRY
    assert MODEL_REGISTRY["sv"]["file"] == "kb-whisper-large-q5_0.bin"
    assert MODEL_REGISTRY["sv"]["sha256"] == "6d2863812d7410322bb7d8647a5c7260761300fa946714c9ed66d22bb30bcb19"
    assert MODEL_REGISTRY["default"]["file"] == "ggml-large-v3-turbo-q5_0.bin"
    assert MODEL_REGISTRY["default"]["sha256"] == "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"


def test_ensure_whisper_cli_exists():
    cli_path = ensure_whisper_cli()
    assert cli_path.is_file()
    assert os.access(cli_path, os.X_OK)


def test_ensure_model_cached(tmp_path):
    # Verify model resolution for Swedish and Default
    sv_path, sv_spec = ensure_model("sv")
    assert sv_path.is_file()
    assert sv_spec["id"] == "kb-whisper-large-q5_0"

    default_path, default_spec = ensure_model("pl")
    assert default_path.is_file()
    assert default_spec["id"] == "whisper-large-v3-turbo-q5_0"


def test_transcribe_with_whisper_cpp_execution(tmp_path):
    audio_path = tmp_path / "test_audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    result = transcribe_with_whisper_cpp(
        audio_path=audio_path,
        language="en",
        work_dir=tmp_path,
    )

    assert "language" in result
    assert "segments" in result
    assert result["runtime_info"]["runtime"] == "whisper.cpp"
    assert result["runtime_info"]["version"] == "1.8.6"
    assert result["runtime_info"]["backend"] == "CUDA"
    assert Path(result["json_output_path"]).is_file()


def test_transcribe_cancellation_safety(tmp_path):
    audio_path = tmp_path / "test_audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=5",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    cancelled = False
    def cancel_checker():
        nonlocal cancelled
        cancelled = True
        return True

    with pytest.raises(RuntimeError, match="cancelled"):
        transcribe_with_whisper_cpp(
            audio_path=audio_path,
            language="en",
            work_dir=tmp_path,
            cancel_checker=cancel_checker,
        )


def test_pipe_deadlock_prevention(tmp_path, monkeypatch):
    """Proves wrapper drains pipes continuously so child writing >1 MiB output never deadlocks."""
    audio_path = tmp_path / "dummy_audio.wav"
    audio_path.write_bytes(b"RIFF" + b"\x00" * 40)

    fake_cli = tmp_path / "fake_whisper_cli.py"
    fake_cli.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        "# Write >1.5 MiB to stdout and stderr to exceed default 64KiB OS pipe buffer\n"
        "sys.stdout.write('A' * 1_500_000 + '\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stderr.write('B' * 1_500_000 + '\\n')\n"
        "sys.stderr.flush()\n"
        "# Generate expected output JSON\n"
        "out_prefix = sys.argv[sys.argv.index('-of') + 1]\n"
        "out_json = out_prefix + '.json'\n"
        "with open(out_json, 'w') as f:\n"
        "    json.dump({'result': {'language': 'en'}, 'transcription': [{'text': 'Hello', 'offsets': {'from': 0, 'to': 1000}}]}, f)\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)

    monkeypatch.setattr("whisper_cpp_runtime.ensure_whisper_cli", lambda custom_path=None: fake_cli)
    monkeypatch.setattr("whisper_cpp_runtime.ensure_model", lambda lang, mdir=None: (tmp_path / "model.bin", {"id": "fake"}))

    res = transcribe_with_whisper_cpp(
        audio_path=audio_path,
        language="en",
        work_dir=tmp_path,
    )
    assert res["language"] == "en"
    assert len(res["segments"]) == 1
    assert res["segments"][0]["text"] == "Hello"

