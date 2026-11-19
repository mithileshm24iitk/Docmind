# DocuMind

DocuMind is a RAG (retrieval-augmented generation) assistant. You upload your
own documents (PDF, DOCX, TXT), ask questions, and it answers **only** from
those documents, with citations (file + page). If the documents don't support
an answer, it says it doesn't know.

Features: hybrid search (vector + BM25 keyword search fused with Reciprocal
Rank Fusion), a relevance gate, conversation memory, streaming answers
(Server-Sent Events), Google login with Supabase, document upload/delete, and
an eval/tracing layer.

> **Deployment status:** DocuMind is run locally (FastAPI on `http://localhost:8000`,
> Next.js on `http://localhost:3000`). The `backend/Dockerfile` and Google Cloud Run
> setup below are provided as ready-to-use configuration, but have **not** been deployed.

## Architecture

```text
 Browser
    │
    ▼
 Next.js frontend (my-app/)          login banner, upload, delete, chat UI
    │  fetch + SSE, "Authorization: Bearer <app JWT>"
    ▼
 FastAPI backend (backend/, local; Dockerfile + Cloud Run config not deployed)
    ├─ api/auth.py       Google OAuth code → user → app JWT (signed with AUTH_SECRET)
    ├─ api/documents.py  upload (20 MB / 300 pages), list, delete, search
    ├─ api/chat.py       /chat/ask (JSON), /chat/stream (SSE), clear session
    ├─ ingestion/        load PDF/DOCX/TXT → split into chunks
    ├─ retrieval/        Gemini embeddings (batches of 50, backoff on 429/503)
    │                     → ChromaDB (vector) + BM25 (keyword) → RRF fusion
    ├─ rag/              relevance gate → prompt → Gemini LLM (+ fallback models),
    │                     memory, tracing (JSONL)
    └─ db.py             Supabase client
    │
    ├──► Gemini API      embeddings + chat model
    ├──► Supabase        Postgres: users, chat_messages (conversation memory)
    └──► local disk      CHROMA_DIR (vectors), TRACES_PATH (traces)
                         On Cloud Run this is /tmp: wiped on restart.
```

How a question is answered:

1. If the session has history, the follow-up is rewritten into a standalone query.
2. **Relevance gate:** the closest chunk's cosine distance must be at most
   `RELEVANCE_MAX_DISTANCE`, otherwise the answer is a fixed "not found" (no LLM call).
3. **Hybrid retrieval:** vector + BM25 results fused with RRF, top `TOP_K` chunks.
4. Gemini answers from those chunks with `[file, p.N]` citations; the turn is
   saved in Supabase. `/chat/stream` sends `token` events, then `done` (or `error`).

Streaming: `/chat/stream` sends `Cache-Control: no-cache` and
`X-Accel-Buffering: no`, and no GZip middleware is used, so proxies such as
Cloud Run's front end do not buffer tokens.

### API

All routes except `/`, `/health` and `/auth/*` need `Authorization: Bearer <token>`.

| Method + path | What it does |
| --- | --- |
| `GET /auth/google` | Google consent-screen URL |
| `POST /auth/google` | Exchange Google `code` → `{access_token, user}` |
| `GET /auth/verify` | Check a token |
| `POST /documents/upload` | Load → chunk → embed → store. 413 if > 20 MB or > 300 pages; 429/503 if embedding is rate-limited/unavailable after 5 tries |
| `GET /documents` | List indexed files with chunk counts |
| `DELETE /documents/{filename}` | Remove that file's chunks (404 if unknown) |
| `POST /documents/search` | Debug search: `vector`, `bm25` or `hybrid` |
| `POST /chat/ask` | Answer as JSON |
| `POST /chat/stream` | Answer as Server-Sent Events |
| `DELETE /chat/sessions/{id}` | Clear a session's memory |

## Run locally

Backend (Python 3.12):

```bash
cd backend
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt   # runtime deps + pytest
cp .env.example .env                            # then fill in the values
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Tests (fully offline: fake LLM, embeddings and Supabase; no keys needed):

```bash
cd backend
.venv/bin/python -m pytest -q
```

`requirements.txt` holds only what the app needs at runtime (it is all the
Docker image installs). `requirements-dev.txt` adds test tools on top.

Frontend:

```bash
cd my-app
npm install
cp .env.example .env.local      # NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                     # http://localhost:3000
npm run lint
```

Generate an `AUTH_SECRET` (at least 32 characters):

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

## Supabase setup

1. Create a Supabase project.
2. Open the SQL editor and run [`supabase/schema.sql`](supabase/schema.sql).
   It creates `users` and `chat_messages` plus an index on
   `chat_messages(user_id, session_id, created_at)` (the history query).
3. Copy the **Project URL** into `SUPABASE_URL` and the **secret** API key into
   `SUPABASE_KEY` (Project Settings → API Keys).

**Row Level Security (RLS):** the browser never talks to Supabase; only the
backend does. You can therefore turn RLS on with **no policies** (see the
commented section at the end of `schema.sql`). That blocks the publishable key
completely. The backend must then use the server **secret** key (`sb_secret_…`,
or the legacy `service_role` key), which bypasses RLS. Never use the
publishable/anon key as `SUPABASE_KEY`, and never put the secret key in the
frontend or in any `NEXT_PUBLIC_` variable.

## Google OAuth setup

In Google Cloud Console → APIs & Services → Credentials, create an **OAuth
client ID** (type: Web application). Under **Authorized redirect URIs**, add
the frontend's origin exactly, because the frontend sends
`redirect_uri = window.location.origin`:

- `http://localhost:3000` for local dev
- your deployed frontend URL, e.g. `https://your-app.vercel.app`

Put the client ID in `GOOGLE_CLIENT_ID` and the client secret in
`GOOGLE_CLIENT_SECRET`. If sign-in fails, the login screen shows the
backend's error message in a red banner.

## Docker (optional, not deployed)

The `backend/Dockerfile` is provided for containerizing the backend, though this
project is currently run locally and has not been deployed. To build and run the
backend image locally (env vars come from your `.env` at run time; the file
itself is never copied into the image):

```bash
cd backend
docker build -t documind-backend .
docker run --rm -p 8080:8080 --env-file .env \
  -e CORS_ORIGINS=http://localhost:3000 documind-backend
curl http://localhost:8080/health
```

The image installs only `requirements.txt` and copies only `app/` (no `.env`,
tests, eval or data). Cloud Run sets `PORT`; the container listens on
`${PORT:-8080}`.

## Deploy the backend to Cloud Run (optional, not deployed)

These Cloud Run + Secret Manager steps are provided as reference configuration
and have **not** been deployed. Secrets go in Secret Manager and are mounted as
env vars. They never go in the image, the repo, or plain `--set-env-vars`.
Replace the `<…>` values.

```bash
PROJECT_ID=<your-project-id>
REGION=<e.g. europe-west1>
gcloud config set project "$PROJECT_ID"

# 1. Enable the APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com

# 2. Create one secret per sensitive value. `read -s` keeps the value out of
#    your shell history and off the screen; paste it when prompted.
for NAME in GOOGLE_API_KEY SUPABASE_KEY GOOGLE_CLIENT_SECRET AUTH_SECRET; do
  read -rs -p "$NAME: " VALUE; echo
  printf '%s' "$VALUE" | gcloud secrets create "$NAME" --data-file=-
done
unset VALUE
# To rotate later: printf '%s' "$NEW" | gcloud secrets versions add NAME --data-file=-

# 3. Let the Cloud Run runtime service account read them
#    (default: the Compute Engine default service account)
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
for NAME in GOOGLE_API_KEY SUPABASE_KEY GOOGLE_CLIENT_SECRET AUTH_SECRET; do
  gcloud secrets add-iam-policy-binding "$NAME" \
    --member="serviceAccount:$SA" --role="roles/secretmanager.secretAccessor"
done

# 4. Non-secret settings in a small YAML file (avoids comma-escaping problems
#    with CORS_ORIGINS). Contains no secrets.
cat > /tmp/documind-env.yaml <<'EOF'
SUPABASE_URL: "https://<project-ref>.supabase.co"
GOOGLE_CLIENT_ID: "<client-id>.apps.googleusercontent.com"
CORS_ORIGINS: "https://<your-frontend>.vercel.app,http://localhost:3000"
EOF

# 5. Build from backend/Dockerfile and deploy
gcloud run deploy documind-backend \
  --source backend \
  --region "$REGION" \
  --allow-unauthenticated \
  --max-instances=1 \
  --memory=1Gi \
  --timeout=300 \
  --env-vars-file=/tmp/documind-env.yaml \
  --set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest,SUPABASE_KEY=SUPABASE_KEY:latest,GOOGLE_CLIENT_SECRET=GOOGLE_CLIENT_SECRET:latest,AUTH_SECRET=AUTH_SECRET:latest

# 6. Check it
curl "$(gcloud run services describe documind-backend --region "$REGION" --format='value(status.url)')/health"
```

Then build the frontend with `NEXT_PUBLIC_API_URL` set to the Cloud Run URL
(on Vercel: project env vars, then redeploy). Add the frontend origin to the
OAuth client's redirect URIs and to `CORS_ORIGINS`.

`--allow-unauthenticated` only means Cloud Run does not demand a Google IAM
identity. The app still requires its own login token on every data route.
`--max-instances=1` keeps one copy of the documents (see limitations).

## Frontend build-time variable

`NEXT_PUBLIC_*` values are **inlined into the JavaScript bundle at
`next build` time**. Changing `NEXT_PUBLIC_API_URL` later has no effect until
you rebuild/redeploy the frontend.

## Environment variables

Backend (`backend/app/config.py`). Real env vars override `backend/.env`;
the `.env` file is optional and missing in the container. Names (no values)
are in [`backend/.env.example`](backend/.env.example); a test checks that it
lists every setting.

| Variable | Description | Secret? | Default |
| --- | --- | --- | --- |
| `GOOGLE_API_KEY` | Gemini API key (embeddings + LLM) | **Yes** | empty |
| `SUPABASE_URL` | Supabase project URL | No | empty |
| `SUPABASE_KEY` | Supabase server **secret** key (never the publishable key) | **Yes** | empty |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID | No | empty |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret | **Yes** | empty |
| `AUTH_SECRET` | Key that signs login JWTs; at least 32 chars, no fallback (auth returns 503 if missing/short) | **Yes** | empty |
| `CORS_ORIGINS` | Comma-separated allowed browser origins | No | `http://localhost:3000,http://127.0.0.1:3000` |
| `CHROMA_DIR` | ChromaDB folder | No | `backend/data/chroma` (container: `/tmp/chroma`) |
| `TRACES_PATH` | JSONL trace file | No | `backend/data/traces.jsonl` (container: `/tmp/traces.jsonl`) |
| `EMBEDDING_MODEL` | Gemini embedding model | No | `gemini-embedding-001` |
| `LLM_MODEL` | Gemini chat model | No | see `config.py` |
| `LLM_FALLBACK_MODELS` | JSON list of models tried on 503/timeout | No | see `config.py` |
| `LLM_TEMPERATURE` | LLM temperature | No | `0.2` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking sizes (characters) | No | `1000` / `200` |
| `TOP_K` | Chunks sent to the LLM | No | `5` |
| `RELEVANCE_MAX_DISTANCE` | Relevance gate cosine-distance cutoff | No | `0.6` |
| `MEMORY_MAX_TURNS` | Conversation turns kept per session | No | `5` |
| `PORT` | Port uvicorn listens on (set by Cloud Run; not read by app code) | No | `8080` |
| `FORWARDED_ALLOW_IPS` | IPs uvicorn trusts for `X-Forwarded-*` (not read by app code) | No | `*` in the image |

Frontend (`my-app/`, names in [`my-app/.env.example`](my-app/.env.example)):

| Variable | Description | Secret? | Default |
| --- | --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | Backend base URL (build-time) | No (public in the bundle) | `http://localhost:8000` |

## Evaluation

Upload the documents named in `backend/eval/dataset.json` first, then from `backend/`:

```bash
.venv/bin/python -m eval.run_eval --retrieval-only              # no LLM calls
.venv/bin/python -m eval.run_eval --retrieval-only --k-list 1,3,5,10
.venv/bin/python -m eval.run_eval --throttle-seconds 13         # full RAG run
```

It reports recall@1/3/5 and MRR for vector, BM25 and hybrid retrieval, a
per-category table (`semantic` / `keyword` / `mixed`), page-level recall,
refusal accuracy, latency p50/p95 and estimated cost. A retrieved chunk counts
as a hit only if it comes from the expected file **and** contains an expected
keyword. Details and the dataset format: [`backend/eval/README.md`](backend/eval/README.md).

## Known limitations

- **Chroma and BM25 are not persistent on Cloud Run.** `CHROMA_DIR` is `/tmp`,
  which is in-memory and wiped whenever the instance restarts or scales to
  zero. Uploaded documents must then be re-uploaded. The BM25 index is rebuilt
  from Chroma in memory, so it resets too. Each instance has its own copy,
  hence `--max-instances=1`. Conversation memory and users live in Supabase
  and do survive restarts.
- **Free-tier rate limits.** The Gemini free tier allows only a few requests
  per minute. Uploads embed in batches of 50 with exponential backoff
  (1s, 2s, 4s, 8s; 5 tries) and then return a clear 429/503. Chat returns 429
  when the quota is hit, and the eval waits 13 s between LLM calls by default.
- **Documents are shared by all users.** There is one Chroma collection: every
  signed-in user sees, searches and can delete every uploaded file. Chat
  history *is* per user.
- Re-uploading a file with the same name overwrites its chunks by ID, but if
  the new version has fewer chunks, the extra old chunks stay. Delete the file
  first to be safe.
- Scanned (image-only) PDFs have no extractable text and are rejected (422).
- The app JWT lives in `localStorage` and cannot be revoked before it expires
  (24 h); rotating `AUTH_SECRET` logs everyone out.
