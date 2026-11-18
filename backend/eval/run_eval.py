"""CLI evaluation harness for DocuMind RAG: recall@k, MRR, refusal accuracy, latency, cost.

A retrieved chunk is a HIT when its source matches the item's expected_source AND
its text contains at least one expected_keyword (case-insensitive). Page-level
recall (right source + right page) is reported as a secondary metric.

Usage (from backend/):
    python -m eval.run_eval --retrieval-only
    python -m eval.run_eval --retrieval-only --k-list 1,3,5,10
    python -m eval.run_eval --throttle-seconds 13      # full mode, calls the LLM
"""

import argparse
import json
import math
from collections.abc import Callable
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


RETRIEVERS = ("vector", "bm25", "hybrid")
DEFAULT_K_LIST = (1, 3, 5)
DEFAULT_THROTTLE_SECONDS = 13.0  # free tier allows ~5 LLM requests/minute
UNCATEGORIZED = "uncategorized"

Chunk = tuple[str, dict[str, Any]]  # (text, metadata)


def _source_matches(metadata: dict[str, Any], expected_source: str) -> bool:
    source = str(metadata.get("source", ""))
    return bool(expected_source) and (source == expected_source or expected_source in source)


def is_chunk_hit(metadata: dict[str, Any], text: str, item: dict[str, Any]) -> bool:
    """Chunk-level hit: right source AND the text contains an expected keyword.

    An item with no expected_keywords can never produce a chunk-level hit.
    """
    if not _source_matches(metadata, item.get("expected_source", "")):
        return False
    lower_text = text.lower()
    return any(kw.lower() in lower_text for kw in item.get("expected_keywords") or [])


def is_page_hit(metadata: dict[str, Any], item: dict[str, Any]) -> bool:
    """Page-level hit (secondary metric): right source and right page (any page if unset)."""
    if not _source_matches(metadata, item.get("expected_source", "")):
        return False
    expected_page = item.get("expected_page")
    return not expected_page or metadata.get("page") == expected_page


def first_hit_rank(chunks: list[Chunk], item: dict[str, Any]) -> int | None:
    """1-based rank of the first chunk-level hit, or None if no chunk hits."""
    for rank, (text, meta) in enumerate(chunks, start=1):
        if is_chunk_hit(meta, text, item):
            return rank
    return None


def compute_recall_at_k(hits: list[bool]) -> float:
    """Fraction of answerable questions with a hit in the top k."""
    if not hits:
        return 0.0
    return round(sum(1 for h in hits if h) / len(hits), 4)


def compute_mrr(ranks: list[int | None]) -> float:
    """Mean Reciprocal Rank: average of 1/rank of the first hit (0 when there is none)."""
    if not ranks:
        return 0.0
    return round(sum(1.0 / r for r in ranks if r) / len(ranks), 4)


def parse_k_list(raw: str) -> list[int]:
    """Parse "1,3,5" into a sorted list of unique positive ints."""
    values = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("--k-list needs positive integers, e.g. 1,3,5")
    return values


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


def retrieve_all(
    collection: str, question: str, k_list: list[int]
) -> tuple[dict[str, dict[int, list[Chunk]]], dict[str, float]]:
    """Return {retriever: {k: top-k chunks}} plus each retriever's latency in ms.

    Vector and BM25 rankings do not depend on k, so one call at max(k) is sliced.
    Hybrid fetches 3*k candidates per retriever, so its top results can change
    with k: it is called once per k, exactly like the app calls it.
    """
    max_k = max(k_list)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    vec = [(d.page_content, d.metadata) for d, _ in vectorstore.similarity_search(collection, question, k=max_k)]
    timings["vector"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    bm = [(d.page_content, d.metadata) for d, _ in bm25_search(collection, question, k=max_k)]
    timings["bm25"] = round((time.perf_counter() - t0) * 1000, 2)

    hyb: dict[int, list[Chunk]] = {}
    for k in k_list:
        t0 = time.perf_counter()
        hyb[k] = [(c["text"], c["metadata"]) for c in hybrid_search(collection, question, k=k)]
        if k == max_k:
            timings["hybrid"] = round((time.perf_counter() - t0) * 1000, 2)

    results = {
        "vector": {k: vec[:k] for k in k_list},
        "bm25": {k: bm[:k] for k in k_list},
        "hybrid": hyb,
    }
    return results, timings


def score_question(
    results: dict[str, dict[int, list[Chunk]]], item: dict[str, Any], k_list: list[int]
) -> dict[str, Any]:
    """Per-question record: for each retriever, hit@k, first-hit rank, page hit at max(k)."""
    max_k = max(k_list)
    record: dict[str, Any] = {}
    for name in RETRIEVERS:
        by_k = results[name]
        record[f"{name}_hit_at"] = {str(k): first_hit_rank(by_k[k], item) is not None for k in k_list}
        record[f"{name}_rank"] = first_hit_rank(by_k[max_k], item)
        record[f"{name}_page_hit"] = any(is_page_hit(meta, item) for _, meta in by_k[max_k])
    return record


def summarize(records: list[dict[str, Any]], k_list: list[int]) -> dict[str, dict[str, float]]:
    """Aggregate per-question records into recall@k, MRR and page recall per retriever."""
    max_k = max(k_list)
    summary: dict[str, dict[str, float]] = {}
    for name in RETRIEVERS:
        row = {
            f"recall@{k}": compute_recall_at_k([r[f"{name}_hit_at"][str(k)] for r in records])
            for k in k_list
        }
        row["mrr"] = compute_mrr([r[f"{name}_rank"] for r in records])
        row[f"page_recall@{max_k}"] = compute_recall_at_k([r[f"{name}_page_hit"] for r in records])
        summary[name] = row
    return summary


def run_evaluation(
    dataset: list[dict[str, Any]],
    k: int = 5,
    retrieval_only: bool = False,
    collection: str = DEFAULT_COLLECTION,
    results_path: Path | None = DEFAULT_RESULTS_PATH,
    k_list: list[int] | None = None,
    throttle_seconds: float = DEFAULT_THROTTLE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run retrieval (and optionally full RAG generation) evaluation on the dataset.

    k is the top-k passed to answer() in full mode; k_list holds the cut-offs for
    recall@k. In full mode we wait throttle_seconds between LLM calls so the
    free-tier rate limit is not hit.
    """
    k_list = sorted(set(k_list or DEFAULT_K_LIST))
    answerable: list[dict[str, Any]] = []  # scored records of should_answer items
    refusal_hits: list[bool] = []

    retrieval_latencies_ms: list[float] = []
    generation_latencies_ms: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0
    per_question: list[dict[str, Any]] = []
    last_call_used_llm = False

    for item in dataset:
        question = item["question"]
        should_answer = bool(item.get("should_answer", True))
        q_record: dict[str, Any] = {
            "id": item.get("id", ""),
            "question": question,
            "category": item.get("category", UNCATEGORIZED),
            "should_answer": should_answer,
        }

        if should_answer:
            results, timings = retrieve_all(collection, question, k_list)
            scored = score_question(results, item, k_list)
            answerable.append({**scored, "category": q_record["category"]})
            retrieval_latencies_ms.append(timings["hybrid"])
            q_record.update(scored)
            q_record.update({f"{name}_ms": ms for name, ms in timings.items()})

        if not retrieval_only:
            if last_call_used_llm and throttle_seconds > 0:
                sleep(throttle_seconds)
            res = answer(question, k=k, collection=collection)
            stats = res.get("stats") or {}
            gen_ms = stats.get("generation_ms")
            # generation_ms is None when the relevance gate skipped the LLM.
            last_call_used_llm = gen_ms is not None
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

    by_category: dict[str, Any] = {}
    for category in sorted({r["category"] for r in answerable}):
        rows = [r for r in answerable if r["category"] == category]
        by_category[category] = {"num_questions": len(rows), **summarize(rows, k_list)}

    num_queries = len(dataset) if not retrieval_only else 0
    metrics: dict[str, Any] = {
        "k": k,
        "k_list": k_list,
        "retrieval_only": retrieval_only,
        "hit_definition": "chunk: source matches AND text contains an expected_keyword",
        "num_questions": len(dataset),
        "num_answerable": len(answerable),
        "num_refusal": len(dataset) - len(answerable),
        "retrieval": summarize(answerable, k_list),
        "by_category": by_category,
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

    k_list = metrics["k_list"]
    max_k = max(k_list)
    header = ["Retriever", *[f"Recall@{kk}" for kk in k_list], "MRR", f"Page Recall@{max_k}"]

    def retriever_rows(summary: dict[str, Any], prefix: list[str]) -> list[str]:
        rows = []
        for name in RETRIEVERS:
            s = summary[name]
            cells = [*prefix, name, *[f"{s[f'recall@{kk}']:.2%}" for kk in k_list]]
            cells += [f"{s['mrr']:.3f}", f"{s[f'page_recall@{max_k}']:.2%}"]
            rows.append("| " + " | ".join(cells) + " |")
        return rows

    lines = [
        f"### DocuMind RAG Evaluation Summary ({mode})",
        "",
        "Hit = retrieved chunk is from the expected source AND contains an expected keyword.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + " :--- |" * len(header),
        *retriever_rows(metrics["retrieval"], []),
    ]

    if metrics["by_category"]:
        cat_header = ["Category", "N", *header]
        lines += [
            "",
            "#### By category",
            "",
            "| " + " | ".join(cat_header) + " |",
            "|" + " :--- |" * len(cat_header),
        ]
        for category, summary in metrics["by_category"].items():
            lines += retriever_rows(summary, [category, str(summary["num_questions"])])

    lines += [
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Questions Evaluated | {metrics['num_questions']} ({metrics['num_answerable']} answerable, {metrics['num_refusal']} refusal) |",
        f"| Refusal Accuracy | {refusal_str} |",
        f"| Retrieval Latency (hybrid, p50 / p95) | {metrics['retrieval_p50_ms']:.1f} ms / {metrics['retrieval_p95_ms']:.1f} ms |",
        f"| Generation Latency (p50 / p95) | {gen_p50_str} / {gen_p95_str} |",
        f"| Estimated Cost per 1,000 Queries (answer k={k}) | {cost_str} |",
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
        help="Top-k chunks passed to the RAG chain in full mode.",
    )
    parser.add_argument(
        "--k-list",
        type=parse_k_list,
        default=list(DEFAULT_K_LIST),
        help="Comma-separated cut-offs for recall@k (default: 1,3,5).",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=DEFAULT_THROTTLE_SECONDS,
        help="Seconds to wait between LLM calls in full mode (free tier: ~5 requests/min).",
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
        k_list=args.k_list,
        throttle_seconds=args.throttle_seconds,
    )
    print(format_markdown_summary(results["metrics"]))
    return results


if __name__ == "__main__":
    main()
