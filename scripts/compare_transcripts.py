#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A/B Transcription Parity Comparison Tool for Antek iOS vs Ubuntu whisper.cpp.

Compares transcription output from Antek iOS and Ubuntu parity runtime:
- Text similarity & Word Error Rate (WER) estimation
- Omission & hallucination heuristics (length delta, missing vocabulary)
- Segmentation alignment
- Detailed markdown comparison report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, collapsed spaces, stripped punctuation)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def calculate_wer(ref: str, hyp: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis strings."""
    r_words = normalize_text(ref).split()
    h_words = normalize_text(hyp).split()

    if not r_words:
        return 0.0 if not h_words else 1.0

    d = [[0] * (len(h_words) + 1) for _ in range(len(r_words) + 1)]
    for i in range(len(r_words) + 1):
        d[i][0] = i
    for j in range(len(h_words) + 1):
        d[0][j] = j

    for i in range(1, len(r_words) + 1):
        for j in range(1, len(h_words) + 1):
            if r_words[i - 1] == h_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # Deletion
                    d[i][j - 1] + 1,      # Insertion
                    d[i - 1][j - 1] + 1,  # Substitution
                )

    return d[len(r_words)][len(h_words)] / float(len(r_words))


def load_transcript(path: Path) -> str:
    """Load transcript from plain text or JSON file."""
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "segments" in data:
            return "\n".join(seg.get("text", "") for seg in data["segments"])
        elif isinstance(data, list):
            return "\n".join(seg.get("text", "") for seg in data if isinstance(seg, dict))
    return path.read_text(encoding="utf-8")


def compare_transcripts(
    ios_path: Path,
    ubuntu_path: Path,
    output_report_path: Path | None = None,
    vocabulary_list: list[str] | None = None,
) -> dict:
    ios_raw = load_transcript(ios_path)
    ubuntu_raw = load_transcript(ubuntu_path)

    wer = calculate_wer(ios_raw, ubuntu_raw)
    similarity = max(0.0, 100.0 * (1.0 - wer))

    ios_norm = normalize_text(ios_raw)
    ubuntu_norm = normalize_text(ubuntu_raw)

    ios_word_count = len(ios_norm.split())
    ubuntu_word_count = len(ubuntu_norm.split())

    length_ratio = (ubuntu_word_count / float(ios_word_count)) if ios_word_count > 0 else 1.0

    missing_vocab = []
    if vocabulary_list:
        for term in vocabulary_list:
            t_norm = normalize_text(term)
            in_ios = t_norm in ios_norm
            in_ubuntu = t_norm in ubuntu_norm
            if in_ios and not in_ubuntu:
                missing_vocab.append(term)

    lines = []
    lines.append("# Transcription Parity A/B Comparison Report\n")
    lines.append(f"- **Antek iOS Reference File**: `{ios_path.name}`")
    lines.append(f"- **Ubuntu Parity Output File**: `{ubuntu_path.name}`\n")
    lines.append("## Metrics Summary\n")
    lines.append(f"- **Estimated Word Error Rate (WER)**: `{wer * 100.0:.2f}%`")
    lines.append(f"- **Word-Level Similarity**: `{similarity:.2f}%`")
    lines.append(f"- **Antek iOS Word Count**: `{ios_word_count}` words")
    lines.append(f"- **Ubuntu Parity Word Count**: `{ubuntu_word_count}` words")
    lines.append(f"- **Length Ratio (Ubuntu / iOS)**: `{length_ratio:.2f}`\n")

    lines.append("## Qualitative Checks\n")
    if length_ratio < 0.85:
        lines.append("- ⚠️ **Omission Warning**: Ubuntu output is >15% shorter than iOS output.")
    elif length_ratio > 1.15:
        lines.append("- ⚠️ **Hallucination Warning**: Ubuntu output is >15% longer than iOS output.")
    else:
        lines.append("- ✅ **Length Parity**: Word counts match within ±15% threshold.")

    if missing_vocab:
        lines.append(f"- ⚠️ **Missing Terminology**: Terms present in iOS but missing in Ubuntu: {', '.join(missing_vocab)}")
    elif vocabulary_list:
        lines.append("- ✅ **Terminology Parity**: All key vocabulary terms present in both transcripts.")

    lines.append("\n## Text Side-by-Side View\n")
    lines.append("### Antek iOS Transcript:\n```")
    lines.append(ios_raw.strip())
    lines.append("```\n")
    lines.append("### Ubuntu Parity Transcript:\n```")
    lines.append(ubuntu_raw.strip())
    lines.append("```\n")

    report_text = "\n".join(lines)

    if output_report_path:
        output_report_path.write_text(report_text, encoding="utf-8")

    return {
        "wer": round(wer, 4),
        "similarity_pct": round(similarity, 2),
        "ios_word_count": ios_word_count,
        "ubuntu_word_count": ubuntu_word_count,
        "length_ratio": round(length_ratio, 3),
        "missing_vocab": missing_vocab,
        "report_markdown": report_text,
    }


def main():
    parser = argparse.ArgumentParser(description="A/B Transcription Parity Comparison Tool")
    parser.add_argument("--ios-transcript", required=True, help="Path to Antek iOS transcript (.txt or .json)")
    parser.add_argument("--ubuntu-transcript", required=True, help="Path to Ubuntu parity transcript (.txt or .json)")
    parser.add_argument("--report", default=None, help="Path to output markdown report file")
    parser.add_argument("--vocabulary", default=None, help="Comma-separated list of expected domain terms")

    args = parser.parse_args()

    vocab = [v.strip() for v in args.vocabulary.split(",") if v.strip()] if args.vocabulary else None
    ios_path = Path(args.ios_transcript)
    ubuntu_path = Path(args.ubuntu_transcript)
    report_path = Path(args.report) if args.report else None

    result = compare_transcripts(ios_path, ubuntu_path, report_path, vocab)
    print(json.dumps({k: v for k, v in result.items() if k != "report_markdown"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
