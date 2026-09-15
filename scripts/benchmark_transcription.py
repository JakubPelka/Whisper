#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Bounded local transcription benchmark script for JakubPelka/Whisper.

Generates a 10-minute synthetic 16kHz WAV recording and runs transcribe_with_whisper_cpp
to measure wall clock time, audio duration, and Real-Time Factor (RTF).
"""

import sys
import time
import tempfile
import subprocess
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_cpp_runtime import transcribe_with_whisper_cpp


def main() -> int:
    duration_sec = 600  # 10 minutes
    print(f"Generating synthetic {duration_sec}s (10 min) 16kHz mono WAV benchmark file...")

    with tempfile.TemporaryDirectory(prefix="whisper_bench_") as temp_dir:
        audio_path = Path(temp_dir) / "benchmark_10min.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={duration_sec}",
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

        audio_size = audio_path.stat().st_size
        audio_duration = round((audio_size - 44) / 32000.0, 2)
        print(f"Audio file generated: {audio_path.name} (size: {audio_size} bytes, duration: {audio_duration}s)")

        print("Executing whisper-cli CUDA transcription benchmark...")
        t0 = time.time()
        res = transcribe_with_whisper_cpp(
            audio_path=audio_path,
            language="auto",
            work_dir=Path(temp_dir),
        )
        t1 = time.time()

        wall_time = round(t1 - t0, 2)
        rtf = round(wall_time / max(audio_duration, 0.001), 4)

        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS:")
        print(f"  Model ID:           {res['model_info']['id']}")
        print(f"  Backend:            {res['runtime_info']['backend']}")
        print(f"  Audio Duration:     {audio_duration} seconds")
        print(f"  Wall-Clock Time:    {wall_time} seconds")
        print(f"  Real-Time Factor:   {rtf} (wall / audio)")
        print("=" * 50)

        return 0


if __name__ == "__main__":
    sys.exit(main())
