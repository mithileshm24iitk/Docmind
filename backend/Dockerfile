# DocuMind backend image for Google Cloud Run (phase 9).
FROM python:3.12-slim

# No .pyc files, unbuffered logs (so Cloud Run shows them immediately).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only the application code goes in the image (no .env, data, tests, eval).
COPY app/ ./app/

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser
USER appuser

# Container defaults. /tmp is the writable (in-memory, ephemeral) disk on
# Cloud Run. Override any of these with real env vars at deploy time.
# FORWARDED_ALLOW_IPS lets uvicorn trust X-Forwarded-* from Cloud Run's proxy.
ENV PORT=8080 \
    CHROMA_DIR=/tmp/chroma \
    TRACES_PATH=/tmp/traces.jsonl \
    FORWARDED_ALLOW_IPS="*"

EXPOSE 8080

# Shell form so ${PORT} is expanded; exec so uvicorn gets SIGTERM directly.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers
