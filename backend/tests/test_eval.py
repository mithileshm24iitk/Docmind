"""Unit tests for Phase 8: structured tracing and the RAG evaluation harness."""

import json
from pathlib import Path

from langchain_core.documents import Document

from app.rag import tracing
from app.rag.chain import answer, stream_answer
from app.retrieval import vectorstore
from eval.run_eval import (
    compute_percentile,
    compute_recall_at_k,
    estimate_cost_per_1k_queries,
    format_markdown_summary,
    main,
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
    res = answer("how does photosynthesis turn sunlight into energy", session_id="s-trace-1")
    assert res["grounded"] is True

    # 2. Un-grounded answer() call (blocked by relevance gate)
    res_blocked = answer("quantum chromodynamics gluon confinement", session_id="s-trace-2")
    assert res_blocked["grounded"] is False

    # 3. Streaming stream_answer() call
    list(stream_answer("how does photosynthesis turn sunlight into energy", session_id="s-trace-3"))

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
        },
        {
            "id": "q2",
            "question": "Geometrically dot product projection of B along A",
            "expected_source": "physics.pdf",
            "expected_page": 3,
            "expected_keywords": ["projection"],
            "should_answer": True,
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
            "--retrieval-only",
            "--output",
            str(results_file),
        ]
    )
    assert fake_llm.call_count == 0
    assert out_retrieval["metrics"]["retrieval_only"] is True
    assert out_retrieval["metrics"]["vector_recall_at_k"] == 1.0
    assert out_retrieval["metrics"]["bm25_recall_at_k"] == 1.0
    assert out_retrieval["metrics"]["hybrid_recall_at_k"] == 1.0
    assert out_retrieval["metrics"]["refusal_accuracy"] is None

    # 2. Test full mode (retrieval + generation)
    out_full = run_evaluation(
        dataset=sample_dataset,
        k=2,
        retrieval_only=False,
        results_path=results_file,
    )
    metrics = out_full["metrics"]
    assert metrics["retrieval_only"] is False
    assert metrics["vector_recall_at_k"] == 1.0
    assert metrics["bm25_recall_at_k"] == 1.0
    assert metrics["hybrid_recall_at_k"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["retrieval_p50_ms"] >= 0.0
    assert metrics["generation_p50_ms"] >= 0.0
    assert metrics["estimated_cost_per_1k_usd"] > 0.0

    saved = json.loads(results_file.read_text(encoding="utf-8"))
    assert saved["metrics"]["hybrid_recall_at_k"] == 1.0
    assert len(saved["questions"]) == 3

    md = format_markdown_summary(metrics)
    assert "| Vector Recall@2 | 100.00% |" in md
    assert "| Refusal Accuracy | 100.00% |" in md
    captured = capsys.readouterr().out
    assert "DocuMind RAG Evaluation Summary" in captured
