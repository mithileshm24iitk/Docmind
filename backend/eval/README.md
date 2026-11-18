# DocuMind evaluation

`run_eval.py` measures how well each retriever (vector, BM25, hybrid) finds the
right chunk, and, in full mode, how the whole RAG chain answers or refuses.

## Run it

From `backend/`, after uploading the documents the dataset refers to (through
the app or `POST /documents/upload`) into the same `CHROMA_DIR`:

```bash
# Retrieval only: no LLM calls, only query embeddings
.venv/bin/python -m eval.run_eval --retrieval-only

# Other recall cut-offs
.venv/bin/python -m eval.run_eval --retrieval-only --k-list 1,3,5,10

# Full mode: also calls answer() for every question (uses the LLM).
# Waits --throttle-seconds between LLM calls (default 13) so the free tier
# (~5 requests/minute) is not exceeded. Use 0 on a paid key.
.venv/bin/python -m eval.run_eval --throttle-seconds 13
```

It prints Markdown tables and writes everything (per-question details too) to
`eval/results.json` (change with `--output`).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `eval/dataset.json` | Question set |
| `--k-list` | `1,3,5` | Cut-offs for recall@k |
| `--k` | `5` | Top-k passed to the RAG chain in full mode |
| `--retrieval-only` | off | Skip the LLM; only retrieval metrics |
| `--throttle-seconds` | `13` | Pause between LLM calls in full mode |
| `--output` | `eval/results.json` | Where the JSON results go |

## What counts as a hit

- **Chunk-level hit (main metric):** a retrieved chunk is a hit when its
  `source` matches `expected_source` **and** its text contains at least one of
  `expected_keywords` (case-insensitive). This checks that the retriever found
  the passage with the answer, not just the right file.
- **Page-level hit (secondary):** right `source` and `page == expected_page`
  (any page if `expected_page` is 0 or missing). Reported as `Page Recall@max(k)`.

## Metrics

- **Recall@k:** share of answerable questions with at least one hit in the top k.
- **MRR (Mean Reciprocal Rank):** average of `1 / rank` of the first hit, over
  the top max(k) results (0 if none). Rewards putting the answer first.
- **By category:** the same table split by each item's `category`.
- **Refusal accuracy** (full mode): share of `should_answer: false` questions
  that the chain refused.
- **Latency p50/p95 and estimated cost per 1,000 queries** (full mode).

Vector and BM25 are queried once at max(k) and sliced. Hybrid is queried once
per k because its candidate pool grows with k (3·k from each retriever), so its
top-1 at k=1 can differ from the first item of its k=5 list. This matches what
the app does for a given `TOP_K`.

## Dataset format

`dataset.json` is a list of objects:

```json
{
  "id": "physics_04",
  "question": "Geometrically, what does |A . (B x C)| represent?",
  "expected_source": "Griffiths.pdf",
  "expected_page": 7,
  "expected_keywords": ["volume", "parallelepiped"],
  "should_answer": true,
  "category": "semantic"
}
```

| Field | Required | Notes |
| --- | --- | --- |
| `id` | yes | Unique label shown in `results.json` |
| `question` | yes | Sent to each retriever and, in full mode, to the chain |
| `expected_source` | answerable items | Filename as uploaded (a substring match also counts) |
| `expected_page` | no | Used only for the page-level metric; `0` = any page |
| `expected_keywords` | answerable items | Words/phrases that appear in the answer passage. Pick distinctive ones: a very common keyword (e.g. `"5"`) makes any chunk from that file a hit. With no keywords an item can never hit |
| `should_answer` | yes | `false` = out-of-scope question; scored only by refusal accuracy |
| `category` | no | `"semantic"`, `"keyword"` or `"mixed"`; missing = `"uncategorized"` |

Categories:

- **keyword:** the question relies on exact, rare terms or numbers that appear
  in the text (project names, "Goldman Sachs", "DocuMind"). BM25 should do well.
- **semantic:** a paraphrased, conceptual question that shares few words with
  the passage ("geometric meaning of the cross product"). Embeddings should do well.
- **mixed:** a named term plus a paraphrased ask. Hybrid should do well.

Refusal items (`should_answer: false`) are not given a category because they
are not part of the retrieval metrics.
