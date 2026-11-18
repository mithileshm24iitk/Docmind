"""Offline tests for eval/calibrate_gate.py (fake embeddings, no network)."""

import pytest
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.chain import best_distance
from app.retrieval import vectorstore
from eval import calibrate_gate
from eval.calibrate_gate import measure, suggest_threshold


def seed() -> None:
    vectorstore.add_chunks(
        "default",
        [
            Document(page_content="the dot product is a projection", metadata={"source": "g.pdf", "page": 3, "chunk_id": 0}),
            Document(page_content="cross product area of parallelogram", metadata={"source": "g.pdf", "page": 4, "chunk_id": 1}),
        ],
    )


def test_suggest_threshold_midpoint_and_no_overlap():
    r = suggest_threshold([0.20, 0.35, 0.30], [0.55, 0.70])
    assert r == {"max_in": 0.35, "min_out": 0.55, "suggested": 0.45, "overlap": False}


def test_suggest_threshold_detects_overlap():
    assert suggest_threshold([0.2, 0.6], [0.5, 0.9])["overlap"] is True
    assert suggest_threshold([0.5], [0.5])["overlap"] is True  # a tie cannot be separated


def test_measure_uses_the_chain_gate_distance(fake_embeddings):
    seed()
    q = "dot product projection"
    row = measure(q, k=2)
    assert row["gate"] == pytest.approx(best_distance("default", q))
    assert row["hybrid"] == pytest.approx(row["gate"])  # closest chunk is in the hybrid results
    assert row["top_source"] == "g.pdf"


def test_main_prints_summary_and_changes_nothing(fake_embeddings, monkeypatch, capsys):
    seed()
    monkeypatch.setattr(calibrate_gate, "IN_SCOPE", ["dot product projection"])
    monkeypatch.setattr(calibrate_gate, "OUT_OF_SCOPE", ["pasta carbonara recipe"])
    before = get_settings().relevance_max_distance

    calibrate_gate.main()

    out = capsys.readouterr().out
    assert "max in-scope distance" in out
    assert "min out-of-scope distance" in out
    assert ("suggested RELEVANCE_MAX_DISTANCE" in out) or ("RANGES OVERLAP" in out)
    assert get_settings().relevance_max_distance == before


def test_main_on_empty_store_says_so(fake_embeddings, capsys):
    calibrate_gate.main()
    assert "is empty" in capsys.readouterr().out
