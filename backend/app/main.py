import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, chat, documents

# Show our app's INFO logs (e.g. RAG latency/token stats) next to uvicorn's.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="DocuMind API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
