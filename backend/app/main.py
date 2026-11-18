import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, chat, documents
from app.config import get_settings

# Show our app's INFO logs (e.g. RAG latency/token stats) next to uvicorn's.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="DocuMind API", version="0.1.0")

# Origins come from CORS_ORIGINS. Note: no GZip/buffering middleware here, so
# Server-Sent Events from /chat/stream reach the browser token by token.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "DocuMind API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
