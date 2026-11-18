"""Calibrate the relevance gate (RELEVANCE_MAX_DISTANCE) on the real Chroma store.

No HTTP, no auth, no LLM calls: only query embeddings (Gemini) + local search.
Nothing is written and no setting is changed; it only prints a suggestion.

For each question it prints:
  gate      = chain.best_distance(): cosine distance of the single closest chunk.
              This is exactly the number the chat chain compares with
              RELEVANCE_MAX_DISTANCE (before hybrid retrieval runs), so the
              suggested threshold is based on it.
  hybrid    = smallest cosine distance among the chunks hybrid_search returns
              (the chunks the LLM would see). BM25-only chunks have no vector
              distance and are skipped. Usually equal to `gate`.
Query rewriting is skipped (each question is treated as a first turn).

Run from backend/:
    .venv/bin/python -m eval.calibrate_gate
"""

from langchain_core.documents import Document

from app.config import get_settings
from app.rag.chain import best_distance
from app.retrieval import vectorstore
from app.retrieval.hybrid import CANDIDATE_MULTIPLIER, chunk_key, hybrid_search
from app.retrieval.vectorstore import DEFAULT_COLLECTION

IN_SCOPE = [
    # Griffiths, Introduction to Electrodynamics (chapter 1, vector analysis)
    "What is the geometric meaning of the dot product of two vectors?",
    "Is the cross product commutative, and what does its magnitude represent?",
    "What does the scalar triple product represent geometrically?",
    "How many components does a second-rank tensor have in three dimensions?",
    # Resume
    "Which university does Mithilesh study at and what is his CPI?",
    "Which techniques were used in the deep neural network compression project?",
    "What tech stack was used to build the DocuMind RAG assistant?",
    "What rank did Mithilesh get in the Goldman Sachs quant hackathon?",
]

OUT_OF_SCOPE = [
    "Who won the 2022 FIFA football World Cup?",
    "What is the capital of France?",
    "How do I cook a classic spaghetti carbonara?",
    "What is the current price of bitcoin?",
    "How should I train for my first marathon?",
    "What will the weather be in Tokyo tomorrow?",
    "How do I change a flat tire on a car?",
    "Who wrote the Harry Potter books?",
]


def measure(question: str, k: int, collection: str = DEFAULT_COLLECTION) -> dict:
    """Return the gate distance, best hybrid-chunk distance and top hybrid source for one question."""
    gate = best_distance(collection, question)

    # Same candidate pool size hybrid_search uses, to look up each chunk's distance.
    candidates = vectorstore.similarity_search(collection, question, k * CANDIDATE_MULTIPLIER)
    distance_by_key = {chunk_key(doc): float(dist) for doc, dist in candidates}

    chunks = hybrid_search(collection, question, k)
    chunk_distances = []
    for c in chunks:
        key = chunk_key(Document(page_content=c["text"], metadata=c["metadata"]))
        if key in distance_by_key:  # BM25-only chunks have no vector distance
            chunk_distances.append(distance_by_key[key])
    top_source = chunks[0]["metadata"]["source"] if chunks else "-"
    return {
        "question": question,
        "gate": gate,
        "hybrid": min(chunk_distances) if chunk_distances else None,
        "top_source": top_source,
    }


def suggest_threshold(in_scope: list[float], out_of_scope: list[float]) -> dict:
    """Max in-scope, min out-of-scope, their midpoint, and whether the two ranges overlap."""
    max_in = max(in_scope)
    min_out = min(out_of_scope)
    return {
        "max_in": max_in,
        "min_out": min_out,
        "suggested": round((max_in + min_out) / 2, 4),
        "overlap": max_in >= min_out,
    }


def _fmt(value: float | None) -> str:
    return "  none" if value is None else f"{value:.4f}"


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"  {'gate':>6}  {'hybrid':>6}  top hybrid source                question")
    for r in rows:
        source = r["top_source"][:32]
        print(f"  {_fmt(r['gate'])}  {_fmt(r['hybrid'])}  {source:<32} {r['question']}")


def main() -> None:
    settings = get_settings()
    k = settings.top_k
    current = settings.relevance_max_distance

    docs = vectorstore.list_documents(DEFAULT_COLLECTION)
    if not docs:
        print(f"The Chroma collection at {settings.chroma_dir} is empty. Upload the documents first.")
        return
    print(f"Chroma: {settings.chroma_dir}  ({len(docs)} files)  top_k={k}")
    for d in docs:
        print(f"  - {d['filename']} ({d['num_chunks']} chunks)")

    in_rows = [measure(q, k) for q in IN_SCOPE]
    out_rows = [measure(q, k) for q in OUT_OF_SCOPE]
    _print_table("IN-SCOPE (should pass the gate)", in_rows)
    _print_table("OUT-OF-SCOPE (should be blocked)", out_rows)

    result = suggest_threshold([r["gate"] for r in in_rows], [r["gate"] for r in out_rows])
    print("\nBased on the gate distance (what RELEVANCE_MAX_DISTANCE is compared with):")
    print(f"  max in-scope distance     : {result['max_in']:.4f}")
    print(f"  min out-of-scope distance : {result['min_out']:.4f}")

    if result["overlap"]:
        print("  RANGES OVERLAP: no single threshold separates these questions.")
        print("  Questions inside the overlap:")
        for r in in_rows + out_rows:
            if result["min_out"] <= r["gate"] <= result["max_in"]:
                print(f"    {r['gate']:.4f}  {r['question']}")
        print(f"  Midpoint {result['suggested']} is shown for reference only.")
    else:
        margin = result["min_out"] - result["max_in"]
        print(f"  suggested RELEVANCE_MAX_DISTANCE = {result['suggested']}  (gap {margin:.4f})")

    for label, rows, should_pass in (("in-scope", in_rows, True), ("out-of-scope", out_rows, False)):
        wrong = [r for r in rows if (r["gate"] <= current) != should_pass]
        verdict = "all correct" if not wrong else f"{len(wrong)} wrong"
        print(f"  current value {current}: {label} {verdict}")
    print("\nNothing was changed. To use a new value, set RELEVANCE_MAX_DISTANCE yourself.")


if __name__ == "__main__":
    main()
