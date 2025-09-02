# Whisper
Transcribtion and diarisation of audio recordings with whisper algorithm


README — diarization + transcription (Whisper + pyannote)

Purpose
-------
Single-file script `transcribe_diarize.py` that:
- converts source audio to WAV 16 kHz mono,
- performs speaker diarization (pyannote 3.1, falls back to legacy if needed),
- performs transcription (Whisper, default: large-v3),
- saves TXT and JSON,
- creates `arkiv/` next to the source file and moves `.mp3/.m4a` there after processing.

Requirements
------------
- Python 3.10+
- ffmpeg available on PATH (required by pydub)
- Packages:
  pip install -U openai-whisper pyannote.audio pydub torch torchaudio
  (Choose the proper PyTorch build for your CUDA/GPU.)

How to run (interactive)
------------------------
python transcribe_diarize.py

1) File picker windows: choose the input audio file and the output folder.
2) Terminal prompts: choose language (pl/en/sv/auto/other) and Whisper model
   (tiny, base, small, medium, large-v2, large-v3, large-v3-turbo).
3) The script runs and prints progress lines per segment:
   [hh:mm:ss.mmm–hh:mm:ss.mmm] SpeakerX: text

Outputs
-------
In the selected output folder:
- <Name>_transcription.txt  — transcript with timestamps + speakers
- <Name>_transcription.json — full structure (segments, models used, language)

Additionally:
- A folder `arkiv/` is created next to the source file and the original `.mp3/.m4a`
  is moved there after successful processing.

Key settings (inside the script)
--------------------------------
- HF token for pyannote 3.1: variable TOKEN_IN_SCRIPT at the top of the file.
  (pyannote 3.1 is gated — accept the model terms on Hugging Face once.)
- Models are chosen at runtime via terminal prompts; default Whisper: large-v3.
  Faster option with a small quality tradeoff: large-v3-turbo.
- Diarization segment bounds: MIN_SEG = 0.35 s, MAX_SEG = 120.0 s.
- Logging/Warnings are reduced at startup — adjust if you need more verbosity.

Processing flow
---------------
1) Audio is converted to mono 16 kHz WAV (pydub + ffmpeg).
2) Diarization: try pyannote/speaker-diarization-3.1 (using the token) →
   fallback to pyannote/speaker-diarization if 3.1 is unavailable.
3) Whisper: transcribe each diarized segment (beam search, temperature=0.0,
   condition_on_previous_text=False to reduce drift).
4) Save TXT/JSON. Create arkiv/ and move the original .mp3/.m4a. Clean up temp segments.

Tips for quality/speed
----------------------
- Language: explicitly set (e.g., pl) for stability; auto is available if needed.
- Model: large-v3 (best quality) / large-v3-turbo (faster).
- GPU: on CPU it will work but be slower; ensure correct PyTorch/CUDA for your GPU.
- Long monologues: you can raise MAX_SEG (e.g., 180 s) if you prefer fewer splits.

Troubleshooting
---------------
- "Could not download … 3.1 / gated":
  Accept terms on the model page and ensure TOKEN_IN_SCRIPT is a valid Read token.
- pydub/ffmpeg errors:
  Install ffmpeg and add it to the system PATH.
- Legacy pyannote warnings about 0.x/torch 1.x:
  That means fallback to legacy pipeline — it will work, but 3.1 is preferred.
- Noisy logs:
  Logging is already reduced; remove or adjust the logging/warnings section if needed.
- Windows profiles / cache:
  Running as a different Windows user changes cache paths; keep that in mind.

Roadmap (for later)
-------------------
- Export SRT/VTT with speaker labels.
- Merge adjacent segments from the same speaker.
- Batch mode (process many files/folder).
- Option to run without diarization (single-speaker transcription).
- Consider using faster-whisper for performance-sensitive environments.
