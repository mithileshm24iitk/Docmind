# DocuMind

DocuMind is a RAG (retrieval-augmented generation) assistant. You upload your
own documents (PDF, DOCX, TXT), ask questions, and it answers **only** from
those documents, with citations (file + page). If the documents don't support
an answer, it says it doesn't know.

Features: hybrid search (vector + BM25 keyword search fused with Reciprocal
Rank Fusion), a relevance gate, conversation memory, streaming answers
(Server-Sent Events), Google login with Supabase, and an eval/tracing layer.

## Architecture

```text
 Browser ──► Next.js frontend (my-app/, deployed on Vercel)
                 │  fetch + SSE, Bearer JWT
                 ▼
          FastAPI backend (backend/, Docker on Google Cloud Run)
           ├─ api/        /documents, /chat/ask, /chat/stream, /auth
           ├─ ingestion/  load files → split into chunks
           ├─ retrieval/  Gemini embeddings → ChromaDB, BM25, hybrid RRF
           ├─ rag/        prompt + Gemini LLM, memory, tracing (JSONL)
           └─ db.py       Supabase client (users, chats)
                 │
                 ├──► Gemini API (embeddings + chat model)
                 └──► Supabase (Postgres)
```

- **Vector store:** ChromaDB on local disk (`CHROMA_DIR`). The BM25 index is
  rebuilt in memory from it.
- **Traces:** one JSON line per question in `TRACES_PATH` (best effort: a
  failed write logs a warning and never breaks a request).
- **Streaming:** `/chat/stream` sends `Cache-Control: no-cache` and
  `X-Accel-Buffering: no`, and no GZip middleware is used, so tokens are not
  buffered by proxies such as Cloud Run's front end.

> **Note on Cloud Run storage:** `/tmp` in Cloud Run is in-memory and is wiped
> when an instance stops. Each instance also has its own copy. Uploaded
> documents and conversation memory are therefore lost on restart and are not
> shared between instances. For a demo, deploy with `--max-instances=1`.

## Run locally

Backend (Python 3.12):

```bash
cd backend
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env            # then fill in the values
.venv/bin/uvicorn app.main:app --reload --port 8000
.venv/bin/python -m pytest -q   # offline tests (fake LLM/embeddings)
```

Frontend:

```bash
cd my-app
npm install
cp .env.example .env.local      # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                     # http://localhost:3000
```

## Docker

Build and run the backend image locally (env vars passed from your `.env`
file at run time; the file itself is never copied into the image):

```bash
cd backend
docker build -t documind-backend .
docker run --rm -p 8080:8080 --env-file .env \
  -e CORS_ORIGINS=http://localhost:3000 documind-backend
curl http://localhost:8080/health
```

Cloud Run sets `PORT` itself; the container listens on `${PORT:-8080}`.

## Frontend build-time variable

`NEXT_PUBLIC_*` values are **inlined into the JavaScript bundle at
`next build` time**. Changing `NEXT_PUBLIC_API_URL` later has no effect until
you rebuild/redeploy the frontend. On Vercel, set it in the project's
environment variables before the build.

## Environment variables

Backend (`backend/app/config.py`). Real env vars override `backend/.env`;
the `.env` file is optional and missing in the container.

| Variable | Description | Secret? | Default |
| --- | --- | --- | --- |
| `GOOGLE_API_KEY` | Gemini API key (embeddings + LLM) | **Yes** | empty |
| `SUPABASE_URL` | Supabase project URL | No | empty |
| `SUPABASE_KEY` | Supabase API key (also a fallback JWT signing key) | **Yes** | empty |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID | No | empty |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret (also used to sign app JWTs) | **Yes** | empty |
| `CORS_ORIGINS` | Comma-separated allowed browser origins | No | `http://localhost:3000,http://127.0.0.1:3000` |
| `CHROMA_DIR` | ChromaDB folder | No | `backend/data/chroma` (container: `/tmp/chroma`) |
| `TRACES_PATH` | JSONL trace file | No | `backend/data/traces.jsonl` (container: `/tmp/traces.jsonl`) |
| `EMBEDDING_MODEL` | Gemini embedding model | No | `gemini-embedding-001` |
| `LLM_MODEL` | Gemini chat model | No | see `config.py` |
| `LLM_FALLBACK_MODELS` | JSON list of fallback models | No | see `config.py` |
| `LLM_TEMPERATURE` | LLM temperature | No | `0.2` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking sizes (characters) | No | `1000` / `200` |
| `TOP_K` | Chunks sent to the LLM | No | `5` |
| `RELEVANCE_MAX_DISTANCE` | Relevance gate cosine-distance cutoff | No | `0.6` |
| `MEMORY_MAX_TURNS` | Conversation turns kept per session | No | `5` |
| `PORT` | Port uvicorn listens on (set by Cloud Run) | No | `8080` |
| `FORWARDED_ALLOW_IPS` | IPs uvicorn trusts for `X-Forwarded-*` headers | No | `*` in the image |

Frontend (`my-app/`):

| Variable | Description | Secret? | Default |
| --- | --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | Backend base URL (build-time) | No (public in the bundle) | `http://localhost:8000` |
