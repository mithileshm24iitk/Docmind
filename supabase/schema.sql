-- DocuMind Supabase schema. Run once in the Supabase SQL editor.
-- Matches what the backend writes (backend/app/api/auth.py, backend/app/rag/memory.py).
-- Safe to re-run: every statement uses IF NOT EXISTS.

-- One row per Google account. id is Google's stable user id ("sub" claim).
create table if not exists public.users (
    id         text primary key,
    email      text unique,
    name       text,
    created_at timestamptz not null default now()
);

-- One row per question/answer turn in a chat session.
create table if not exists public.chat_messages (
    id                uuid primary key default gen_random_uuid(),
    user_id           text not null references public.users (id),
    session_id        text not null,
    user_message      text not null,
    assistant_message text not null,
    created_at        timestamptz not null default now()
);

-- The backend loads history with:
--   where user_id = ? and session_id = ? order by created_at
-- so this index covers that query exactly.
create index if not exists chat_messages_user_session_created_idx
    on public.chat_messages (user_id, session_id, created_at);


-- ---------------------------------------------------------------------------
-- Optional: Row Level Security (RLS)
-- ---------------------------------------------------------------------------
-- Why: with RLS off, anyone holding the project's *publishable* key (the one
-- that is safe to ship in a browser) can read and write these tables through
-- the Supabase REST API. DocuMind's browser never talks to Supabase directly;
-- only the backend does, so we can lock the tables down completely.
--
-- How:
--   1. Uncomment and run the two lines below. With RLS on and NO policies,
--      the publishable key can no longer read or write anything.
--   2. The backend must then use the Supabase server *secret* key
--      (Project Settings -> API Keys -> "secret", or the legacy
--      "service_role" key). It bypasses RLS. Put it in SUPABASE_KEY, which is
--      a backend-only secret (backend/.env locally, Secret Manager on Cloud Run).
--   3. Never use the publishable / anon key in SUPABASE_KEY, and never put the
--      secret key in the frontend or in any NEXT_PUBLIC_ variable.
--
-- alter table public.users         enable row level security;
-- alter table public.chat_messages enable row level security;
