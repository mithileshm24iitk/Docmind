"""CLI evaluation harness for DocuMind RAG: recall@k, refusal accuracy, latency, and cost.

Usage:
    python -m eval.run_eval --dataset backend/eval/dataset.json --k 5 --retrieval-only
    python -m eval.run_eval --dataset backend/eval/dataset.json --k 5
"""

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

from app.config import BACKEND_DIR
from app.rag.chain import answer
from app.rag.prompts import NOT_FOUND_ANSWER
from app.retrieval import vectorstore
from app.retrieval.bm25 import bm25_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vectorstore import DEFAULT_COLLECTION

# Configurable token pricing constants (USD per 1M tokens, Gemini Flash defaults)
INPUT_PRICE_PER_1M_USD = 0.075
OUTPUT_PRICE_PER_1M_USD = 0.30

DEFAULT_DATASET_PATH = BACKEND_DIR / "eval" / "dataset.json"
DEFAULT_RESULTS_PATH = BACKEND_DIR / "eval" / "results.json"

REFUSAL_PHRASES = (
    NOT_FOUND_ANSWER.lower(),
    "don't contain the answer",
    "do not contain the answer",
    "not mentioned in the provided",
    "couldn't find this in your documents",
)


def resolve_dataset_path(raw_path: str | Path) -> Path:
    """Resolve dataset path whether run from repo root or backend/ directory."""
    path = Path(raw_path)
    if path.exists():
        return path
    alt = BACKEND_DIR.parent / path
    if alt.exists():
        return alt
    alt_in_backend = BACKEND_DIR / path
    if alt_in_backend.exists():
        return alt_in_backend
    return path


def is_chunk_match(
    metadata: dict[str, Any],
    text: str,
    expected_source: str,
    expected_page: int | None,
    expected_keywords: list[str] | None = None,
) -> bool:
    """Return True if a retrieved chunk matches the expected source + page (or keywords)."""
    source = str(metadata.get("source", ""))
    page = metadata.get("page")

    source_matches = bool(expected_source) and (
        source == expected_source or expected_source in source
    )
    if source_matches and (not expected_page or page == expected_page):
        return True

    if source_matches and expected_keywords:
        lower_text = text.lower()
        if any(kw.lower() in lower_text for kw in expected_keywords):
            return True

    return False


def compute_recall_at_k(hits: list[bool]) -> float:
    """Compute recall@k as the fraction of answerable questions where a top-k chunk matched."""
    if not hits:
        return 0.0
    return round(sum(1 for h in hits if h) / len(hits), 4)


def compute_percentile(values: list[float], percentile: float) -> float:
    """Compute the p-th percentile (e.g. 50 or 95) of a list of numbers using nearest-rank/linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return round(float(sorted_vals[0]), 2)
    rank = (percentile / 100.0) * (len(sorted_vals) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    weight = rank - low
    interpolated = sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight
    return round(float(interpolated), 2)


def estimate_cost_per_1k_queries(
    total_input_tokens: int,
    total_output_tokens: int,
    num_queries: int,
    input_price_per_1m: float = INPUT_PRICE_PER_1M_USD,
    output_price_per_1m: float = OUTPUT_PRICE_PER_1M_USD,
) -> float:
    """Calculate the estimated USD cost per 1,000 queries from observed token totals."""
    if num_queries <= 0:
        return 0.0
    total_cost_usd = (
        total_input_tokens * input_price_per_1m
        + total_output_tokens * output_price_per_1m
    ) / 1_000_000.0
    cost_per_1k = (total_cost_usd / num_queries) * 1000.0
    return round(cost_per_1k, 6)


def is_refusal_response(result: dict[str, Any]) -> bool:
    """Check whether an answer() response refused to answer an out-of-scope question."""
    if not result.get("grounded", True):
        return True
    text = str(result.get("answer", "")).lower()
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def _evaluate_retriever_hit(
    chunks: list[tuple[str, dict[str, Any]]],
    item: dict[str, Any],
) -> bool:
    expected_source = item.get("expected_source", "")
    expected_page = item.get("expected_page")
    expected_keywords = item.get("expected_keywords") or []
    return any(
        is_chunk_match(meta, text, expected_source, expected_page, expected_keywords)
        for text, meta in chunks
    )


def run_evaluation(
    dataset: list[dict[str, Any]],
    k: int = 5,
    retrieval_only: bool = False,
    collection: str = DEFAULT_COLLECTION,
    results_path: Path | None = DEFAULT_RESULTS_PATH,
) -> dict[str, Any]:
    """Run retrieval (and optionally full RAG generation) evaluation on the dataset."""
    vector_hits: list[bool] = []
    bm25_hits: list[bool] = []
    hybrid_hits: list[bool] = []
    refusal_hits: list[bool] = []

    retrieval_latencies_ms: list[float] = []
    generation_latencies_ms: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0
    per_question: list[dict[str, Any]] = []

    for item in dataset:
        qid = item.get("id", "")
        question = item["question"]
        should_answer = bool(item.get("should_answer", True))

        q_record: dict[str, Any] = {
            "id": qid,
            "question": question,
            "should_answer": should_answer,
        }

        if should_answer:
            # 1. Vector search
            t0 = time.perf_counter()
            vec_res = vectorstore.similarity_search(collection, question, k=k)
            vec_ms = round((time.perf_counter() - t0) * 1000, 2)
            vec_chunks = [(doc.page_content, doc.metadata) for doc, _ in vec_res]
            v_hit = _evaluate_retriever_hit(vec_chunks, item)

            # 2. BM25 search
            t0 = time.perf_counter()
            bm25_res = bm25_search(collection, question, k=k)
            bm25_ms = round((time.perf_counter() - t0) * 1000, 2)
            bm25_chunks = [(doc.page_content, doc.metadata) for doc, _ in bm25_res]
            b_hit = _evaluate_retriever_hit(bm25_chunks, item)

            # 3. Hybrid search
            t0 = time.perf_counter()
            hyb_res = hybrid_search(collection, question, k=k)
            hyb_ms = round((time.perf_counter() - t0) * 1000, 2)
            hyb_chunks = [(c["text"], c["metadata"]) for c in hyb_res]
            h_hit = _evaluate_retriever_hit(hyb_chunks, item)

            vector_hits.append(v_hit)
            bm25_hits.append(b_hit)
            hybrid_hits.append(h_hit)
            retrieval_latencies_ms.append(hyb_ms)

            q_record.update(
                {
                    "vector_hit": v_hit,
                    "bm25_hit": b_hit,
                    "hybrid_hit": h_hit,
                    "vector_ms": vec_ms,
                    "bm25_ms": bm25_ms,
                    "hybrid_ms": hyb_ms,
                }
            )

        if not retrieval_only:
            res = answer(question, k=k, collection=collection)
            stats = res.get("stats") or {}
            gen_ms = stats.get("generation_ms")
            if gen_ms is not None:
                generation_latencies_ms.append(float(gen_ms))
            in_tok = stats.get("input_tokens") or 0
            out_tok = stats.get("output_tokens") or 0
            total_input_tokens += int(in_tok)
            total_output_tokens += int(out_tok)

            if not should_answer:
                refused = is_refusal_response(res)
                refusal_hits.append(refused)
                q_record["refusal_correct"] = refused
                if stats.get("retrieval_ms") is not None:
                    retrieval_latencies_ms.append(float(stats["retrieval_ms"]))

            q_record.update(
                {
                    "grounded": res.get("grounded"),
                    "best_distance": res.get("best_distance"),
                    "answer": res.get("answer"),
                    "num_sources": len(res.get("sources") or []),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "generation_ms": gen_ms,
                }
            )

        per_question.append(q_record)

    num_queries = len(dataset) if not retrieval_only else 0
    metrics: dict[str, Any] = {
        "k": k,
        "retrieval_only": retrieval_only,
        "num_questions": len(dataset),
        "num_answerable": len(vector_hits),
        "num_refusal": len(dataset) - len(vector_hits),
        "vector_recall_at_k": compute_recall_at_k(vector_hits),
        "bm25_recall_at_k": compute_recall_at_k(bm25_hits),
        "hybrid_recall_at_k": compute_recall_at_k(hybrid_hits),
        "refusal_accuracy": (
            compute_recall_at_k(refusal_hits) if not retrieval_only else None
        ),
        "retrieval_p50_ms": compute_percentile(retrieval_latencies_ms, 50),
        "retrieval_p95_ms": compute_percentile(retrieval_latencies_ms, 95),
        "generation_p50_ms": (
            compute_percentile(generation_latencies_ms, 50)
            if not retrieval_only
            else None
        ),
        "generation_p95_ms": (
            compute_percentile(generation_latencies_ms, 95)
            if not retrieval_only
            else None
        ),
        "total_input_tokens": total_input_tokens if not retrieval_only else None,
        "total_output_tokens": total_output_tokens if not retrieval_only else None,
        "estimated_cost_per_1k_usd": (
            estimate_cost_per_1k_queries(
                total_input_tokens, total_output_tokens, num_queries
            )
            if not retrieval_only
            else None
        ),
    }

    output = {"metrics": metrics, "questions": per_question}
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    return output


def format_markdown_summary(metrics: dict[str, Any]) -> str:
    """Format evaluation metrics as a clean Markdown summary table."""
    k = metrics["k"]
    mode = "Retrieval-Only" if metrics["retrieval_only"] else "Full RAG (Retrieval + LLM)"
    refusal_str = (
        f"{metrics['refusal_accuracy']:.2%}"
        if metrics["refusal_accuracy"] is not None
        else "N/A (retrieval-only)"
    )
    gen_p50_str = (
        f"{metrics['generation_p50_ms']:.1f} ms"
        if metrics["generation_p50_ms"] is not None
        else "N/A"
    )
    gen_p95_str = (
        f"{metrics['generation_p95_ms']:.1f} ms"
        if metrics["generation_p95_ms"] is not None
        else "N/A"
    )
    cost_str = (
        f"${metrics['estimated_cost_per_1k_usd']:.4f}"
        if metrics["estimated_cost_per_1k_usd"] is not None
        else "N/A"
    )

    lines = [
        f"### DocuMind RAG Evaluation Summary ({mode}, k={k})",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Questions Evaluated | {metrics['num_questions']} ({metrics['num_answerable']} answerable, {metrics['num_refusal']} refusal) |",
        f"| Vector Recall@{k} | {metrics['vector_recall_at_k']:.2%} |",
        f"| BM25 Recall@{k} | {metrics['bm25_recall_at_k']:.2%} |",
        f"| Hybrid Recall@{k} | {metrics['hybrid_recall_at_k']:.2%} |",
        f"| Refusal Accuracy | {refusal_str} |",
        f"| Retrieval Latency (p50 / p95) | {metrics['retrieval_p50_ms']:.1f} ms / {metrics['retrieval_p95_ms']:.1f} ms |",
        f"| Generation Latency (p50 / p95) | {gen_p50_str} / {gen_p95_str} |",
        f"| Estimated Cost per 1,000 Queries | {cost_str} |",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse CLI arguments, run the evaluation, print Markdown table, and save results.json."""
    parser = argparse.ArgumentParser(description="Run DocuMind RAG evaluation harness.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k chunks to retrieve for recall@k evaluation.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate only vector, BM25, and hybrid retrieval without calling the generation LLM.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_RESULTS_PATH),
        help="Path to write full JSON results.",
    )
    args = parser.parse_args(argv)

    dataset_path = resolve_dataset_path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    results = run_evaluation(
        dataset=dataset,
        k=args.k,
        retrieval_only=args.retrieval_only,
        results_path=Path(args.output),
    )
    print(format_markdown_summary(results["metrics"]))
    return results


if __name__ == "__main__":
    main()
