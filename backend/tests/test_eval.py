"""Unit tests for Phase 8: structured tracing and the RAG evaluation harness."""

import argparse
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.rag import tracing
from app.rag.chain import answer, stream_answer
from app.retrieval import vectorstore
from eval.run_eval import (
    compute_mrr,
    compute_percentile,
    compute_recall_at_k,
    estimate_cost_per_1k_queries,
    first_hit_rank,
    format_markdown_summary,
    is_chunk_hit,
    is_page_hit,
    main,
    parse_k_list,
    run_evaluation,
)

TRACE_FIELDS = {
    "timestamp",
    "session_id",
    "question",
    "rewritten_question",
    "grounded",
    "best_distance",
    "retrieval_ms",
    "generation_ms",
    "input_tokens",
    "output_tokens",
    "estimated_cost_usd",
    "model",
    "num_sources",
}


def test_recall_percentile_and_cost_math():
    # recall@k calculation
    assert compute_recall_at_k([]) == 0.0
    assert compute_recall_at_k([True, True, False, True]) == 0.75
    assert compute_recall_at_k([False, False]) == 0.0

    # p50 and p95 percentile calculation
    assert compute_percentile([], 50) == 0.0
    assert compute_percentile([42.0], 95) == 42.0
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert compute_percentile(latencies, 50) == 30.0
    assert compute_percentile(latencies, 95) == 48.0

    # cost estimation math
    # 1,000,000 input tokens @ $0.075 + 1,000,000 output tokens @ $0.30 = $0.375 total
    # across 10 queries => $0.0375 per query => $37.50 per 1,000 queries
    cost_1k = estimate_cost_per_1k_queries(
        total_input_tokens=1_000_000,
        total_output_tokens=1_000_000,
        num_queries=10,
        input_price_per_1m=0.075,
        output_price_per_1m=0.30,
    )
    assert cost_1k == 37.5
    assert estimate_cost_per_1k_queries(100, 100, 0) == 0.0
    assert tracing.estimate_cost_usd(1_000_000, 1_000_000, 0.075, 0.30) == 0.375


def test_tracing_writes_valid_jsonl_entries(tmp_path: Path, fake_embeddings, fake_llm, monkeypatch):
    trace_file = tmp_path / "custom_traces.jsonl"
    monkeypatch.setattr(tracing, "DEFAULT_TRACE_PATH", trace_file)

    vectorstore.add_chunks(
        "default",
        [
            Document(
                page_content="photosynthesis lets plants turn sunlight into chemical energy",
                metadata={"source": "biology.pdf", "page": 2, "chunk_id": 0},
            )
        ],
    )

    # 1. Grounded answer() call
    res = answer("how does photosynthesis turn sunlight into energy", session_id="s-trace-1", user_id="u1")
    assert res["grounded"] is True

    # 2. Un-grounded answer() call (blocked by relevance gate)
    res_blocked = answer("quantum chromodynamics gluon confinement", session_id="s-trace-2", user_id="u1")
    assert res_blocked["grounded"] is False

    # 3. Streaming stream_answer() call
    list(stream_answer("how does photosynthesis turn sunlight into energy", session_id="s-trace-3", user_id="u1"))

    lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    entries = [json.loads(line) for line in lines]
    for entry in entries:
        assert set(entry.keys()) == TRACE_FIELDS

    assert entries[0]["session_id"] == "s-trace-1"
    assert entries[0]["grounded"] is True
    assert entries[0]["input_tokens"] == 120
    assert entries[0]["output_tokens"] == 15
    assert entries[0]["estimated_cost_usd"] > 0
    assert entries[0]["num_sources"] == 1

    assert entries[1]["session_id"] == "s-trace-2"
    assert entries[1]["grounded"] is False
    assert entries[1]["num_sources"] == 0
    assert entries[1]["estimated_cost_usd"] == 0.0

    assert entries[2]["session_id"] == "s-trace-3"
    assert entries[2]["grounded"] is True
    assert entries[2]["num_sources"] == 1


def test_run_eval_end_to_end_fake_embeddings_and_llm(tmp_path: Path, fake_embeddings, fake_llm, capsys):
    # Seed fake documents into the test collection
    vectorstore.add_chunks(
        "default",
        [
            Document(
                page_content="Mithilesh studies Materials Science at Indian Institute of Technology Kanpur with CPI 8.2.",
                metadata={"source": "resume.pdf", "page": 1, "chunk_id": 0},
            ),
            Document(
                page_content="Geometrically the dot product A . B is the product of A times the projection of B along A.",
                metadata={"source": "physics.pdf", "page": 3, "chunk_id": 1},
            ),
        ],
    )

    sample_dataset = [
        {
            "id": "q1",
            "question": "Mithilesh Materials Science Indian Institute of Technology Kanpur CPI 8.2",
            "expected_source": "resume.pdf",
            "expected_page": 1,
            "expected_keywords": ["8.2"],
            "should_answer": True,
            "category": "keyword",
        },
        {
            "id": "q2",
            "question": "Geometrically dot product projection of B along A",
            "expected_source": "physics.pdf",
            "expected_page": 3,
            "expected_keywords": ["projection"],
            "should_answer": True,
            "category": "semantic",
        },
        {
            "id": "q3",
            "question": "sourdough bread baking fermentation temperature flour",
            "expected_source": "",
            "expected_page": 0,
            "expected_keywords": [],
            "should_answer": False,
        },
    ]

    dataset_file = tmp_path / "dataset.json"
    results_file = tmp_path / "results.json"
    dataset_file.write_text(json.dumps(sample_dataset), encoding="utf-8")

    # 1. Test retrieval-only mode via CLI main()
    out_retrieval = main(
        [
            "--dataset",
            str(dataset_file),
            "--k",
            "2",
            "--k-list",
            "1,2",
            "--retrieval-only",
            "--output",
            str(results_file),
        ]
    )
    assert fake_llm.call_count == 0
    m = out_retrieval["metrics"]
    assert m["retrieval_only"] is True
    assert m["k_list"] == [1, 2]
    for name in ["vector", "bm25", "hybrid"]:
        assert m["retrieval"][name]["recall@2"] == 1.0
        assert m["retrieval"][name]["mrr"] > 0
        assert m["retrieval"][name]["page_recall@2"] == 1.0
    assert set(m["by_category"]) == {"keyword", "semantic"}
    assert m["by_category"]["keyword"]["num_questions"] == 1
    assert m["refusal_accuracy"] is None

    # 2. Test full mode (retrieval + generation)
    out_full = run_evaluation(
        dataset=sample_dataset,
        k=2,
        retrieval_only=False,
        results_path=results_file,
        k_list=[1, 2],
        sleep=lambda s: None,
    )
    metrics = out_full["metrics"]
    assert metrics["retrieval_only"] is False
    for name in ["vector", "bm25", "hybrid"]:
        assert metrics["retrieval"][name]["recall@2"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["retrieval_p50_ms"] >= 0.0
    assert metrics["generation_p50_ms"] >= 0.0
    assert metrics["estimated_cost_per_1k_usd"] > 0.0

    saved = json.loads(results_file.read_text(encoding="utf-8"))
    assert saved["metrics"]["retrieval"]["hybrid"]["recall@2"] == 1.0
    assert len(saved["questions"]) == 3

    md = format_markdown_summary(metrics)
    assert "| vector | " in md and "Recall@2" in md
    assert "#### By category" in md
    assert "| Refusal Accuracy | 100.00% |" in md
    captured = capsys.readouterr().out
    assert "DocuMind RAG Evaluation Summary" in captured


# ---------- Part 3: chunk-level hits, MRR, k-list, categories, throttle ----------

ITEM = {"expected_source": "a.pdf", "expected_page": 2, "expected_keywords": ["Volume", "parallelepiped"]}


def test_chunk_hit_needs_source_and_keyword():
    meta = {"source": "a.pdf", "page": 2}
    assert is_chunk_hit(meta, "the VOLUME of a box", ITEM)  # case-insensitive
    assert not is_chunk_hit(meta, "right page but no keyword", ITEM)
    assert not is_chunk_hit({"source": "b.pdf", "page": 2}, "volume", ITEM)
    assert not is_chunk_hit(meta, "volume", {**ITEM, "expected_keywords": []})


def test_page_hit_is_secondary_and_ignores_keywords():
    assert is_page_hit({"source": "a.pdf", "page": 2}, ITEM)
    assert not is_page_hit({"source": "a.pdf", "page": 3}, ITEM)
    assert is_page_hit({"source": "a.pdf", "page": 9}, {**ITEM, "expected_page": 0})


def test_first_hit_rank_and_mrr():
    chunks = [
        ("unrelated", {"source": "a.pdf", "page": 2}),
        ("a parallelepiped", {"source": "a.pdf", "page": 2}),
    ]
    assert first_hit_rank(chunks, ITEM) == 2
    assert first_hit_rank(chunks[:1], ITEM) is None
    assert compute_mrr([1, 2, None, 4]) == round((1 + 0.5 + 0 + 0.25) / 4, 4)
    assert compute_mrr([]) == 0.0


def test_parse_k_list():
    assert parse_k_list("5,1,3,3") == [1, 3, 5]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_k_list("0,1")


def _seed_two_chunks():
    vectorstore.add_chunks(
        "default",
        [
            Document(page_content="filler words about nothing", metadata={"source": "a.pdf", "page": 2, "chunk_id": 0}),
            Document(page_content="the box volume formula", metadata={"source": "a.pdf", "page": 2, "chunk_id": 1}),
        ],
    )


def test_page_match_without_keyword_is_not_a_hit(fake_embeddings):
    _seed_two_chunks()
    item = {
        "id": "q", "question": "filler words", "category": "semantic", "should_answer": True,
        "expected_source": "a.pdf", "expected_page": 2, "expected_keywords": ["nonexistent"],
    }
    m = run_evaluation([item], retrieval_only=True, results_path=None, k_list=[1, 2])["metrics"]
    for name in ["vector", "bm25", "hybrid"]:
        assert m["retrieval"][name]["recall@2"] == 0.0  # chunk-level: no keyword -> miss
        assert m["retrieval"][name]["mrr"] == 0.0
        assert m["retrieval"][name]["page_recall@2"] == 1.0  # page-level still matches


def test_uncategorized_items_get_their_own_row(fake_embeddings):
    _seed_two_chunks()
    item = {
        "id": "q", "question": "box volume", "should_answer": True,
        "expected_source": "a.pdf", "expected_page": 2, "expected_keywords": ["volume"],
    }
    m = run_evaluation([item], retrieval_only=True, results_path=None, k_list=[1])["metrics"]
    assert list(m["by_category"]) == ["uncategorized"]


def test_full_mode_throttles_only_between_llm_calls(fake_embeddings, fake_llm):
    _seed_two_chunks()
    grounded = {
        "question": "box volume formula", "should_answer": True,
        "expected_source": "a.pdf", "expected_page": 2, "expected_keywords": ["volume"],
    }
    refusal = {"question": "sourdough baking temperature", "should_answer": False}
    waits: list[float] = []
    run_evaluation(
        [grounded, grounded, refusal, grounded],
        results_path=None, k_list=[1], throttle_seconds=13, sleep=waits.append,
    )
    # LLM calls: q1, q2, q4 (q3 is blocked by the relevance gate, no LLM).
    # Waits: before q2 (after q1), before q3 (after q2). None before q4 (q3 used no LLM).
    assert fake_llm.call_count == 3
    assert waits == [13, 13]


def test_retrieval_only_never_throttles(fake_embeddings, fake_llm):
    _seed_two_chunks()
    item = {"question": "box volume", "should_answer": True, "expected_source": "a.pdf", "expected_keywords": ["volume"]}
    waits: list[float] = []
    run_evaluation([item, item], retrieval_only=True, results_path=None, sleep=waits.append)
    assert waits == []
