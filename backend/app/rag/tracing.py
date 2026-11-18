"""Lightweight structured JSONL tracer for RAG chat requests."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from app.config import BACKEND_DIR

DEFAULT_TRACE_PATH = BACKEND_DIR / "data" / "traces.jsonl"

# Default Gemini Flash pricing per 1M tokens (USD)
INPUT_PRICE_PER_1M_USD = 0.075
OUTPUT_PRICE_PER_1M_USD = 0.30


def estimate_cost_usd(
    input_tokens: int | None,
    output_tokens: int | None,
    input_price_per_1m: float = INPUT_PRICE_PER_1M_USD,
    output_price_per_1m: float = OUTPUT_PRICE_PER_1M_USD,
) -> float:
    """Estimate query cost in USD from input and output token counts."""
    in_tok = input_tokens or 0
    out_tok = output_tokens or 0
    cost = (in_tok * input_price_per_1m + out_tok * output_price_per_1m) / 1_000_000
    return round(cost, 8)


def log_trace(
    *,
    session_id: str | None,
    question: str,
    rewritten_question: str,
    grounded: bool,
    best_distance: float | None,
    retrieval_ms: float | None,
    generation_ms: float | None,
    input_tokens: int | None,
    output_tokens: int | None,
    model: str | None,
    num_sources: int,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    """Append one structured JSON trace line to backend/data/traces.jsonl."""
    path = trace_path or DEFAULT_TRACE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "question": question,
        "rewritten_question": rewritten_question,
        "grounded": grounded,
        "best_distance": best_distance,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimate_cost_usd(input_tokens, output_tokens),
        "model": model,
        "num_sources": num_sources,
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry
