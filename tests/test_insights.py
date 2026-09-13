"""Sentences and summaries the website shows."""

from __future__ import annotations

import json

import pytest

from artefacts import REPORT_PATH, needs_report
from trafficflow.config import load_site
from trafficflow.insights import build_results, classification_summary, live_line

LIMIT = 96.56


def test_free_flow_and_breakdown_read_differently():
    assert live_line(95.0, "A", "light", LIMIT).startswith("Free-flowing")
    jam = live_line(20.0, "F", "heavy", LIMIT)
    assert jam.startswith("Stop-and-go") and "LOS F" in jam and "heavy" in jam


def test_no_speed_yet_says_so():
    assert live_line(None, None, None, LIMIT).startswith("Measuring")


@needs_report
def test_the_classification_card_uses_the_grouped_split():
    summary = classification_summary(json.loads(REPORT_PATH.read_text(encoding="utf-8")))
    assert summary["split"] == "grouped by recording"
    assert summary["accuracy"] == pytest.approx(0.9449, abs=1e-3)
    assert summary["recall"]["heavy"] == pytest.approx(39 / 44)
    assert summary["majority_accuracy"] == pytest.approx(0.6496, abs=1e-3)


def test_results_from_a_few_rows():
    rows = [
        {"clip": f"c{i}", "traffic_class": cls, "hour": h, "density_pc_mi_ln": d,
         "speed_median_kmh": s, "level_of_service": los}
        for i, (cls, h, d, s, los) in enumerate([
            ("light", 9, 8, 100, "A"), ("light", 10, 12, 95, "B"), ("medium", 16, 25, 70, "C"),
            ("heavy", 17, 45, 25, "E"), ("heavy", 17, 60, 12, "F"),
        ])
    ]
    results = build_results(rows, load_site(), None)
    assert results["clips"] == 5
    assert len(results["points"]) == 5
    assert results["los"]["F"] == 1
    assert [h["hour"] for h in results["by_hour"]] == [9, 10, 16, 17]
    assert results["fundamental"]["free_flow_speed_kmh"] > 90
    assert results["classification"] is None
