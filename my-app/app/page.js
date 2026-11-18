'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

// Backend URL. NEXT_PUBLIC_ values are inlined at `next build` time, so set it
// before building (e.g. in Vercel project settings), not at runtime.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const TOKEN_STORAGE_KEY = 'documind_token';

/**
 * Build request headers containing the Bearer token when authenticated.
 */
function buildAuthHeaders(token, extra = {}) {
  const headers = { ...extra };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

/**
 * Deduplicate sources by filename + page so citation badges stay compact.
 */
function uniqueSources(sources = []) {
  const seen = new Set();
  const result = [];
  for (const s of sources) {
    const key = `${s.filename}:${s.page}`;
    if (!seen.has(key)) {
      seen.add(key);
      result.push(s);
    }
  }
  return result;
}

/**
 * Format latency_ms from backend stats (which provides latency_ms or retrieval_ms + generation_ms).
 */
function formatLatencyMs(stats) {
  if (!stats) return null;
  if (typeof stats.latency_ms === 'number') {
    return Math.round(stats.latency_ms);
  }
  const retrieval = typeof stats.retrieval_ms === 'number' ? stats.retrieval_ms : 0;
  const generation = typeof stats.generation_ms === 'number' ? stats.generation_ms : 0;
  if (retrieval || generation) {
    return Math.round(retrieval + generation);
  }
  return null;
}

/**
 * Extract a readable error message from a failed fetch Response.
 */
async function readErrorMessage(res, fallbackPrefix = 'Request failed') {
  const text = await res.text().catch(() => '');
  if (text) {
    try {
      const parsed = JSON.parse(text);
      if (typeof parsed.detail === 'string') return parsed.detail;
      if (Array.isArray(parsed.detail)) {
        return parsed.detail.map((d) => d.msg || JSON.stringify(d)).join(', ');
      }
      if (typeof parsed.message === 'string') return parsed.message;
      if (typeof parsed.error === 'string') return parsed.error;
    } catch {
      return `${fallbackPrefix} (HTTP ${res.status}): ${text.slice(0, 200)}`;
    }
  }
  return `${fallbackPrefix} (HTTP ${res.status})`;
}

// Short labels for the "busy / overloaded" statuses the backend can return.
const STATUS_LABELS = {
  429: 'Rate limit reached (HTTP 429). Wait about a minute, then try again.',
  502: 'The AI service returned an error (HTTP 502). Please try again.',
  503: 'Service temporarily unavailable (HTTP 503). Please try again shortly.',
  504: 'The AI service timed out (HTTP 504). Please try again.',
};

/**
 * Put a clear status label in front of the backend's own message for 429/502/503/504.
 */
function withStatusLabel(status, message) {
  const label = STATUS_LABELS[status];
  if (!label) return message;
  return message ? `${label} Details: ${message}` : label;
}

/**
 * Read the backend's error message from a failed Response and add a status label.
 */
async function describeHttpError(res, fallbackPrefix) {
  return withStatusLabel(res.status, await readErrorMessage(res, fallbackPrefix));
}

/**
 * Turn a thrown fetch error into a readable message ("Failed to fetch" = backend unreachable).
 */
function networkMessage(err, fallback) {
  if (err?.name === 'TypeError') {
    return `Cannot reach backend at ${API_BASE}. Is FastAPI running?`;
  }
  return err?.message || fallback;
}

/**
 * Resolve who is logged in: finish a Google `?code=` redirect, or verify a saved token.
 * Never throws. Returns { token, user } on success, { error } on failure, {} if logged out.
 */
async function checkAuth() {
  try {
    const params = new URLSearchParams(window.location.search);
    const oauthCode = params.get('code');
    const oauthError = params.get('error');
    if (oauthCode || oauthError) {
      // Remove ?code= / ?error= so a refresh does not reuse a one-time code.
      window.history.replaceState({}, '', window.location.pathname);
    }
    if (oauthError) {
      return { error: `Google sign-in was cancelled or failed (${oauthError}).` };
    }

    if (oauthCode) {
      const res = await fetch(`${API_BASE}/auth/google`, {
        method: 'POST',
        headers: buildAuthHeaders('', { 'Content-Type': 'application/json' }),
        body: JSON.stringify({ code: oauthCode, redirect_uri: window.location.origin }),
      });
      if (!res.ok) {
        return { error: await describeHttpError(res, 'Google sign-in failed') };
      }
      const data = await res.json();
      localStorage.setItem(TOKEN_STORAGE_KEY, data.access_token);
      return { token: data.access_token, user: data.user };
    }

    const storedToken = localStorage.getItem(TOKEN_STORAGE_KEY);
    if (!storedToken) return {};

    const res = await fetch(`${API_BASE}/auth/verify`, {
      headers: buildAuthHeaders(storedToken),
    });
    if (res.status === 401) {
      // The saved token is expired or was signed with an old key: drop it.
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      const msg = await readErrorMessage(res, 'Session check failed');
      return { error: `Your session has expired. Please sign in again. (${msg})` };
    }
    if (!res.ok) {
      // Server-side problem (e.g. 503 auth not configured): keep the token, show why.
      return { error: await describeHttpError(res, 'Could not verify your session') };
    }
    const data = await res.json();
    if (!data.valid) {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      return { error: 'Your session is no longer valid. Please sign in again.' };
    }
    return { token: storedToken, user: data.user };
  } catch (err) {
    return { error: networkMessage(err, 'Authentication failed.') };
  }
}

// React Strict Mode (on by default in dev) runs effects twice. A Google `code`
// can be exchanged only once, so both runs share this one promise; otherwise
// the first run's result was discarded and the user silently landed back on
// the login screen.
let authCheckPromise = null;
function checkAuthOnce() {
  if (!authCheckPromise) authCheckPromise = checkAuth();
  return authCheckPromise;
}

export default function Home() {
  const [token, setToken] = useState('');
  const [user, setUser] = useState(null);
  const [authChecking, setAuthChecking] = useState(true);
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState('');

  const [sessionId, setSessionId] = useState('');
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const [deletingFile, setDeletingFile] = useState(''); // filename being deleted, '' if none

  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [chatError, setChatError] = useState('');

  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Sign out. `reason` (optional) is shown on the login screen, e.g. after a 401.
  const handleLogout = useCallback((reason = '') => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    authCheckPromise = null;
    setToken('');
    setUser(null);
    setMessages([]);
    setDocuments([]);
    setChatError('');
    setAuthError(typeof reason === 'string' ? reason : '');
  }, []);

  // Any 401 from a protected route means the token is stale: clear it and say so.
  const handleExpiredSession = useCallback(() => {
    handleLogout('Your session has expired. Please sign in again.');
  }, [handleLogout]);

  const fetchDocuments = useCallback(
    async (activeToken = token) => {
      if (!activeToken) return;
      try {
        const res = await fetch(`${API_BASE}/documents`, {
          headers: buildAuthHeaders(activeToken),
        });
        if (res.status === 401) {
          handleExpiredSession();
          return;
        }
        if (!res.ok) {
          throw new Error(await describeHttpError(res, 'Failed to load documents'));
        }
        const data = await res.json();
        setDocuments(data.documents || []);
      } catch (err) {
        setChatError(networkMessage(err, 'Failed to load documents.'));
      }
    },
    [token, handleExpiredSession]
  );

  // On mount: finish a Google `?code=` callback or verify the saved token.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSessionId(crypto.randomUUID());

    let ignore = false;
    checkAuthOnce().then((result) => {
      if (ignore) return;
      if (result.token && result.user) {
        setToken(result.token);
        setUser(result.user);
      }
      setAuthError(result.error || '');
      setAuthChecking(false);
    });
    return () => {
      ignore = true;
    };
  }, []);

  // Load documents once authenticated
  useEffect(() => {
    if (!token || !user) return;
    let ignore = false;
    async function loadDocs() {
      if (!ignore) {
        await fetchDocuments(token);
      }
    }
    loadDocs();
    return () => {
      ignore = true;
    };
  }, [token, user, fetchDocuments]);

  // Auto-scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleGoogleLogin() {
    setAuthLoading(true);
    setAuthError('');
    try {
      const redirectUri = window.location.origin;
      const res = await fetch(
        `${API_BASE}/auth/google?redirect_uri=${encodeURIComponent(redirectUri)}`,
        {
          headers: buildAuthHeaders(token),
        }
      );
      if (!res.ok) {
        throw new Error(await describeHttpError(res, 'Could not start Google login'));
      }
      const data = await res.json();
      if (!data.auth_url) {
        throw new Error('Backend did not return a Google authorization URL.');
      }
      window.location.href = data.auth_url;
    } catch (err) {
      setAuthError(networkMessage(err, 'Failed to start Google login.'));
      setAuthLoading(false);
    }
  }

  async function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file || !token) return;

    setUploading(true);
    setUploadError('');
    setUploadSuccess(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/documents/upload`, {
        method: 'POST',
        headers: buildAuthHeaders(token),
        body: formData,
      });

      if (res.status === 401) {
        handleExpiredSession();
        return;
      }

      if (!res.ok) {
        // 413 (too big / too many pages), 429 / 503 (embedding rate limit) etc.
        throw new Error(await describeHttpError(res, 'Upload failed'));
      }

      const data = await res.json();
      setUploadSuccess({
        filename: data.filename,
        pages: data.pages,
        num_chunks: data.num_chunks,
      });
      await fetchDocuments(token);
    } catch (err) {
      setUploadError(networkMessage(err, 'Upload failed.'));
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  }

  async function handleDeleteDocument(filename) {
    if (!token || deletingFile) return;
    if (!window.confirm(`Delete "${filename}" from the index?`)) return;

    setDeletingFile(filename);
    setUploadError('');
    try {
      const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
        headers: buildAuthHeaders(token),
      });
      if (res.status === 401) {
        handleExpiredSession();
        return;
      }
      // 404 = already gone (e.g. deleted in another tab): just refresh the list.
      if (!res.ok && res.status !== 404) {
        throw new Error(await describeHttpError(res, 'Delete failed'));
      }
      if (uploadSuccess?.filename === filename) setUploadSuccess(null);
      await fetchDocuments(token);
    } catch (err) {
      setUploadError(networkMessage(err, 'Delete failed.'));
    } finally {
      setDeletingFile('');
    }
  }

  async function handleClearSession() {
    if (isStreaming || !token) return;
    setChatError('');

    const oldId = sessionId;
    const nextId = crypto.randomUUID();
    setSessionId(nextId);
    setMessages([]);

    if (!oldId) return;
    try {
      const res = await fetch(`${API_BASE}/chat/sessions/${encodeURIComponent(oldId)}`, {
        method: 'DELETE',
        headers: buildAuthHeaders(token),
      });
      if (res.status === 401) {
        handleExpiredSession();
        return;
      }
      if (!res.ok) {
        throw new Error(await describeHttpError(res, 'Failed to clear session'));
      }
    } catch (err) {
      setChatError(
        err.name === 'TypeError'
          ? `Local chat cleared, but backend at ${API_BASE} was unreachable.`
          : err.message
      );
    }
  }

  async function handleSendMessage(e) {
    e?.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || isStreaming || !token) return;

    const activeSessionId = sessionId || crypto.randomUUID();
    if (!sessionId) {
      setSessionId(activeSessionId);
    }

    setChatError('');
    setQuestion('');
    setIsStreaming(true);

    // Append user message and empty assistant placeholder immediately
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: trimmed },
      {
        role: 'assistant',
        content: '',
        sources: [],
        grounded: null,
        rewritten_question: null,
        stats: null,
      },
    ]);

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: buildAuthHeaders(token, { 'Content-Type': 'application/json' }),
        body: JSON.stringify({ session_id: activeSessionId, question: trimmed }),
      });

      if (res.status === 401) {
        handleExpiredSession();
        return;
      }

      // Check !res.ok before touching res.body / TextDecoder
      if (!res.ok) {
        setChatError(await describeHttpError(res, 'Chat request failed'));
        setMessages((prev) => {
          const next = [...prev];
          if (
            next.length > 0 &&
            next[next.length - 1].role === 'assistant' &&
            !next[next.length - 1].content
          ) {
            next.pop();
          }
          return next;
        });
        return;
      }

      if (!res.body) {
        throw new Error('Streaming response body is empty.');
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let receivedAnyEvent = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const rawEvent of events) {
          const line = rawEvent.trim();
          if (!line.startsWith('data:')) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;

          const event = JSON.parse(jsonStr);
          receivedAnyEvent = true;

          if (event.type === 'error') {
            // The backend sends {"type":"error","detail":...,"status_code":...}
            // when the LLM fails after the stream has started (HTTP is already 200).
            throw new Error(
              withStatusLabel(
                event.status_code,
                event.detail || event.message || 'The AI service failed to answer.'
              )
            );
          } else if (event.type === 'token') {
            setMessages((prev) => {
              const next = [...prev];
              const lastIdx = next.length - 1;
              if (lastIdx >= 0 && next[lastIdx].role === 'assistant') {
                next[lastIdx] = {
                  ...next[lastIdx],
                  content: next[lastIdx].content + (event.content || ''),
                };
              }
              return next;
            });
          } else if (event.type === 'done') {
            const latencyMs = formatLatencyMs(event.stats);
            const normalizedStats = event.stats
              ? { ...event.stats, latency_ms: latencyMs }
              : null;

            setMessages((prev) => {
              const next = [...prev];
              const lastIdx = next.length - 1;
              if (lastIdx >= 0 && next[lastIdx].role === 'assistant') {
                next[lastIdx] = {
                  ...next[lastIdx],
                  sources: event.sources || [],
                  grounded: event.grounded ?? false,
                  rewritten_question: event.rewritten_question || null,
                  stats: normalizedStats,
                };
              }
              return next;
            });
          }
        }
      }

      if (!receivedAnyEvent) {
        throw new Error(
          'The server closed the stream without returning an answer (possible LLM or API key error on backend).'
        );
      }
    } catch (err) {
      // TypeError = network failure (backend down, or connection dropped mid-stream).
      const friendly =
        err.name === 'TypeError'
          ? `Lost connection to the backend at ${API_BASE} while streaming. Is FastAPI running?`
          : err.message || 'An error occurred while streaming the answer.';
      setChatError(friendly);

      // Remove empty assistant placeholder if the stream failed to connect or produced no content
      setMessages((prev) => {
        const next = [...prev];
        if (
          next.length > 0 &&
          next[next.length - 1].role === 'assistant' &&
          !next[next.length - 1].content
        ) {
          next.pop();
        }
        return next;
      });
    } finally {
      setIsStreaming(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  }

  // Protected gate: only render the main chat interface when authenticated
  if (!token || !user) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-zinc-950 text-zinc-100 p-4">
        <div className="w-full max-w-sm bg-zinc-900 border border-zinc-800 rounded-2xl p-6 space-y-5 text-center shadow-lg">
          <div className="space-y-1">
            <h1 className="text-xl font-semibold tracking-tight text-white">DocuMind</h1>
            <p className="text-xs text-zinc-400">
              Sign in with Google to upload documents and chat with your files.
            </p>
          </div>

          {authError && (
            <div
              role="alert"
              className="text-xs bg-red-950/80 border border-red-800 text-red-200 rounded-lg p-3 text-left"
            >
              {authError}
            </div>
          )}

          {authChecking ? (
            <div className="text-xs text-zinc-400 py-3 animate-pulse">
              Checking authentication...
            </div>
          ) : (
            <button
              type="button"
              onClick={handleGoogleLogin}
              disabled={authLoading}
              className="w-full py-2.5 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium transition disabled:opacity-50 cursor-pointer"
            >
              {authLoading ? 'Redirecting to Google...' : 'Sign in with Google'}
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col md:flex-row h-screen bg-zinc-950 text-zinc-100">
      {/* Sidebar */}
      <aside className="w-full md:w-80 shrink-0 bg-zinc-900 border-b md:border-b-0 md:border-r border-zinc-800 flex flex-col p-4 gap-5 md:h-full overflow-y-auto">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-white">DocuMind</h1>
            <p className="text-xs text-zinc-400">Grounded RAG Assistant</p>
          </div>
          <button
            type="button"
            onClick={handleClearSession}
            disabled={isStreaming}
            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 transition disabled:opacity-50 cursor-pointer"
          >
            New Chat
          </button>
        </div>

        {/* Upload section */}
        <div className="flex flex-col gap-2.5 bg-zinc-950/60 p-3.5 rounded-xl border border-zinc-800">
          <label
            htmlFor="file-upload"
            className="text-xs font-medium uppercase tracking-wider text-zinc-400"
          >
            Upload Document (.pdf, .docx, .txt)
          </label>
          <input
            ref={fileInputRef}
            id="file-upload"
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={handleFileUpload}
            disabled={uploading}
            className="block w-full text-xs text-zinc-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-emerald-600 file:text-white hover:file:bg-emerald-500 file:cursor-pointer disabled:opacity-50"
          />

          {uploading && (
            <div className="text-xs text-amber-400 animate-pulse">
              Uploading, chunking &amp; embedding...
            </div>
          )}

          {uploadError && (
            <div
              role="alert"
              className="text-xs bg-red-950/80 border border-red-800 text-red-200 rounded-lg p-2.5"
            >
              {uploadError}
            </div>
          )}

          {uploadSuccess && (
            <div className="text-xs bg-emerald-950/70 border border-emerald-800 text-emerald-200 rounded-lg p-2.5 space-y-0.5">
              <div className="font-medium truncate">Uploaded: {uploadSuccess.filename}</div>
              <div className="text-emerald-300/90">
                {uploadSuccess.pages} page(s) &middot; {uploadSuccess.num_chunks} chunk(s)
              </div>
            </div>
          )}
        </div>

        {/* Uploaded Documents List */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-medium uppercase tracking-wider text-zinc-400">
              Indexed Documents ({documents.length})
            </h2>
          </div>

          {documents.length === 0 ? (
            <p className="text-xs text-zinc-500 italic">
              No documents uploaded yet. Upload a file above to start asking questions.
            </p>
          ) : (
            <ul className="space-y-1.5 overflow-y-auto pr-1">
              {documents.map((doc) => (
                <li
                  key={doc.filename}
                  className="flex items-center justify-between text-xs bg-zinc-950/50 border border-zinc-800/80 rounded-lg px-3 py-2"
                >
                  <span className="truncate font-medium text-zinc-200 mr-2" title={doc.filename}>
                    {doc.filename}
                  </span>
                  <span className="shrink-0 flex items-center gap-1.5">
                    <span className="text-[11px] text-zinc-400 bg-zinc-800 px-2 py-0.5 rounded-full">
                      {doc.num_chunks} chunks
                    </span>
                    <button
                      type="button"
                      onClick={() => handleDeleteDocument(doc.filename)}
                      disabled={Boolean(deletingFile) || uploading}
                      aria-label={`Delete ${doc.filename}`}
                      title="Delete document"
                      className="text-[11px] px-2 py-0.5 rounded-md border border-zinc-700 text-zinc-400 hover:text-red-300 hover:border-red-800 transition disabled:opacity-40 cursor-pointer"
                    >
                      {deletingFile === doc.filename ? 'Deleting...' : 'Delete'}
                    </button>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* User & Session Footer */}
        <div className="pt-2 border-t border-zinc-800/80 space-y-1.5 text-[11px] text-zinc-400">
          <div className="flex items-center justify-between gap-2">
            <span className="truncate" title={user.email}>
              {user.name || user.email}
            </span>
            <button
              type="button"
              onClick={() => handleLogout()}
              className="text-zinc-400 hover:text-white underline cursor-pointer shrink-0"
            >
              Sign out
            </button>
          </div>
          <div className="text-zinc-500 truncate">
            Session: <span className="font-mono">{sessionId ? sessionId.slice(0, 8) : '...'}</span>
          </div>
        </div>
      </aside>

      {/* Main Chat Window */}
      <main className="flex-1 flex flex-col min-h-0">
        {/* Error Alert Banner */}
        {chatError && (
          <div
            role="alert"
            className="mx-4 mt-4 p-3 rounded-xl bg-red-950/90 border border-red-800 text-red-200 text-xs flex items-center justify-between"
          >
            <span>{chatError}</span>
            <button
              type="button"
              onClick={() => setChatError('')}
              className="ml-4 text-red-300 hover:text-white font-medium cursor-pointer"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Scrollable Message History */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto text-zinc-400 space-y-2">
              <div className="text-base font-medium text-zinc-200">
                Ask anything about your documents
              </div>
              <p className="text-xs text-zinc-500">
                Upload a PDF, DOCX, or TXT file in the sidebar, then ask a question below. Answers
                are grounded strictly in your uploaded files with page citations.
              </p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex flex-col max-w-3xl mx-auto ${
                  msg.role === 'user' ? 'items-end' : 'items-start'
                }`}
              >
                <div className="text-[11px] uppercase tracking-wider text-zinc-500 mb-1 px-1">
                  {msg.role === 'user' ? 'You' : 'DocuMind'}
                </div>

                <div
                  className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                    msg.role === 'user'
                      ? 'bg-emerald-600 text-white max-w-xl'
                      : 'bg-zinc-900 border border-zinc-800 text-zinc-100 w-full'
                  }`}
                >
                  {msg.content || (
                    <span className="text-zinc-500 italic animate-pulse">Thinking...</span>
                  )}
                </div>

                {/* Assistant metadata & citation badges */}
                {msg.role === 'assistant' && msg.grounded !== null && (
                  <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
                    {uniqueSources(msg.sources).map((src, sIdx) => (
                      <span
                        key={sIdx}
                        title={src.snippet || ''}
                        className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-800/80 text-emerald-300 text-[11px] font-medium"
                      >
                        [{src.filename}, p.{src.page}]
                      </span>
                    ))}

                    {msg.grounded === false && (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full bg-amber-950/70 border border-amber-800/70 text-amber-300 text-[11px]">
                        Unverified / Not in docs
                      </span>
                    )}

                    {msg.stats?.latency_ms != null && (
                      <span className="text-[11px] text-zinc-500 ml-1">
                        {msg.stats.latency_ms} ms
                      </span>
                    )}

                    {msg.rewritten_question && (
                      <span
                        className="text-[11px] text-zinc-500 italic ml-1"
                        title="Standalone query rewritten from conversation history"
                      >
                        (searched: &ldquo;{msg.rewritten_question}&rdquo;)
                      </span>
                    )}
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Bottom Input Form */}
        <div className="border-t border-zinc-800 bg-zinc-950 p-4">
          <form
            onSubmit={handleSendMessage}
            className="max-w-3xl mx-auto flex items-end gap-2 bg-zinc-900 border border-zinc-800 rounded-2xl p-2 focus-within:border-zinc-700"
          >
            <textarea
              rows={1}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
              placeholder={
                isStreaming ? 'Waiting for response...' : 'Ask a question about your documents...'
              }
              className="flex-1 bg-transparent text-sm text-zinc-100 placeholder-zinc-500 resize-none px-2 py-1.5 focus:outline-none disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isStreaming || !question.trim()}
              className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold transition disabled:opacity-40 disabled:hover:bg-emerald-600 cursor-pointer"
            >
              {isStreaming ? 'Streaming...' : 'Send'}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
