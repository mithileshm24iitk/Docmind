# DocuMind: architecture plan

Goal: a RAG assistant that answers ONLY from the user's uploaded documents,
with citations, conversation memory, streaming responses, Google login,
and a deployed demo on Google Cloud Run.

## Stack
- Backend: FastAPI (backend/app/...)
- Frontend: Next.js (my-app/)
- LLM + embeddings: Gemini via langchain-google-genai (key in backend/.env as GOOGLE_API_KEY)
- Vector store: ChromaDB (persistent, backend/data/chroma, gitignored)
- Keyword search: BM25 (rank_bm25); hybrid fusion with Reciprocal Rank Fusion
- Auth + chat history: Google OAuth + Supabase (later phase)
- Deploy: Docker + Cloud Run (later phase)

## Backend layout
backend/app/
  main.py          app + routers + CORS
  config.py        settings from env (pydantic-settings)
  api/             HTTP routes (documents.py, chat.py)
  ingestion/       loaders.py, chunking.py            (DONE: phase 1)
  retrieval/       embeddings.py, vectorstore.py, bm25.py, hybrid.py
  rag/             prompts.py, chain.py, memory.py
backend/tests/
backend/eval/      question sets + scripts (phase 8)

## Phases
1. Loaders + chunking (DONE)
2. Embeddings + ChromaDB: ingest endpoint + similarity search endpoint
3. BM25 + hybrid retrieval (RRF)
4. RAG chain: grounded prompt, citations (source + page), "I don't know" when unsupported
5. Conversation memory (session_id) + SSE streaming endpoint
6. Next.js chat UI with file upload + streaming display
7. Google login + Supabase (users, chats)
8. Evals: recall@k, p50/p95 latency, cost per query, a small tracing/log layer
9. Dockerfile + Cloud Run deploy
