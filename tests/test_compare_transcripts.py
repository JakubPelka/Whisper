#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for compare_transcripts.py A/B comparison script."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import sys

# Add scripts to sys.path
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from compare_transcripts import calculate_wer, compare_transcripts, normalize_text


def test_normalize_text():
    assert normalize_text("  Hej! Det här... Är ETT test.  ") == "hej det här är ett test"


def test_calculate_wer():
    assert calculate_wer("detta är ett test", "detta är ett test") == 0.0
    assert calculate_wer("detta är ett test", "detta är ett annat test") > 0.0


def test_compare_transcripts_report(tmp_path):
    ios_file = tmp_path / "ios.txt"
    ubuntu_file = tmp_path / "ubuntu.txt"
    report_file = tmp_path / "report.md"

    ios_file.write_text("Mötet handlar om Perun Works och Antek iOS.", encoding="utf-8")
    ubuntu_file.write_text("Mötet handlar om Perun Works och Antek iOS.", encoding="utf-8")

    result = compare_transcripts(
        ios_path=ios_file,
        ubuntu_path=ubuntu_file,
        output_report_path=report_file,
        vocabulary_list=["Perun Works", "Antek iOS"],
    )

    assert result["similarity_pct"] == 100.0
    assert result["wer"] == 0.0
    assert len(result["missing_vocab"]) == 0
    assert report_file.is_file()
    assert "Metrics Summary" in report_file.read_text(encoding="utf-8")
