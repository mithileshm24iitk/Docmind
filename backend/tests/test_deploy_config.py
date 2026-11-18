"""Phase 9: settings and app startup work the way Cloud Run needs."""

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.chat import SSE_HEADERS
from app.config import Settings
from app.rag import tracing

BACKEND_DIR = Path(__file__).resolve().parent.parent


def test_cors_origins_default(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    s = Settings(_env_file=None)
    assert s.cors_origin_list == ["http://localhost:3000", "http://127.0.0.1:3000"]


def test_cors_origins_parsing_trims_spaces_and_empties(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", " https://a.vercel.app/ , ,http://localhost:3000,")
    s = Settings(_env_file=None)
    assert s.cors_origin_list == ["https://a.vercel.app", "http://localhost:3000"]


def test_app_starts_without_env_file(tmp_path):
    """Import the app in a fresh process where backend/.env does not exist."""
    script = (
        "from app import config\n"
        "from pathlib import Path\n"
        "config.Settings.model_config['env_file'] = Path('/nonexistent/.env')\n"
        "from fastapi.testclient import TestClient\n"
        "from app.main import app\n"
        "print(TestClient(app).get('/health').json()['status'])\n"
    )
    env = {"PATH": os.environ.get("PATH", ""), "CHROMA_DIR": str(tmp_path / "chroma")}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_cors_preflight_allows_configured_origin():
    from app.main import app

    res = TestClient(app).options(
        "/health",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_traces_path_from_env(tmp_path, monkeypatch):
    from app.config import get_settings

    target = tmp_path / "custom" / "traces.jsonl"
    monkeypatch.setenv("TRACES_PATH", str(target))
    get_settings.cache_clear()
    tracing.log_trace(
        session_id="s", question="q", rewritten_question="q", grounded=True,
        best_distance=0.1, retrieval_ms=1.0, generation_ms=2.0,
        input_tokens=10, output_tokens=5, model="m", num_sources=1,
    )
    assert json.loads(target.read_text().strip())["question"] == "q"


def test_trace_write_failure_does_not_crash(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")  # a file, so mkdir of a child path fails
    entry = tracing.log_trace(
        session_id=None, question="q", rewritten_question="q", grounded=False,
        best_distance=None, retrieval_ms=None, generation_ms=None,
        input_tokens=None, output_tokens=None, model=None, num_sources=0,
        trace_path=blocker / "traces.jsonl",
    )
    assert entry["question"] == "q"


def test_stream_response_disables_buffering(fake_embeddings, fake_llm):
    from app.main import app

    client = TestClient(app)
    client.post("/documents/upload", files={"file": ("n.txt", b"Plants use sunlight.", "text/plain")})
    r = client.post("/chat/stream", json={"session_id": "s1", "question": "plants sunlight"})
    assert r.status_code == 200
    for name, value in SSE_HEADERS.items():
        assert r.headers[name] == value
    assert "content-encoding" not in r.headers  # no gzip on the stream


def test_test_tools_are_not_in_the_docker_requirements():
    """The image installs only requirements.txt, so pytest must live in requirements-dev.txt."""
    runtime = (BACKEND_DIR / "requirements.txt").read_text().lower()
    dev = (BACKEND_DIR / "requirements-dev.txt").read_text().lower()
    assert "pytest" not in runtime
    assert "-r requirements.txt" in dev and "pytest==" in dev


def test_env_example_lists_every_setting():
    """Every variable Settings reads must be named in backend/.env.example."""
    import re

    text = (BACKEND_DIR / ".env.example").read_text()
    listed = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, flags=re.MULTILINE))
    expected = {name.upper() for name in Settings.model_fields}
    assert expected <= listed, expected - listed
