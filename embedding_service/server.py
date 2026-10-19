"""OpenAI-compatible E5 embedding endpoint for an Apple Silicon Mac mini."""

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel


MODEL_ID = "intfloat/multilingual-e5-small"
_lock = threading.Lock()


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float"] = "float"


def _require_key(authorization: str | None = Header(default=None)) -> None:
    key = os.environ.get("READPAPER_EMBEDDING_API_KEY", "")
    if not key or authorization != f"Bearer {key}":
        raise HTTPException(status_code=401, detail="Invalid embedding API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("READPAPER_EMBEDDING_API_KEY"):
        raise RuntimeError("READPAPER_EMBEDDING_API_KEY is required")
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS GPU is unavailable")
    app.state.model = SentenceTransformer(MODEL_ID, device="mps")
    app.state.model.max_seq_length = 512
    app.state.batch_size = max(1, int(os.environ.get("READPAPER_EMBEDDING_BATCH_SIZE", "32")))
    yield
    del app.state.model


app = FastAPI(title="Readpaper embedding service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID, "device": "mps"}


@app.post("/v1/embeddings", dependencies=[Depends(_require_key)])
def embeddings(request: EmbeddingRequest) -> dict:
    if request.model != MODEL_ID:
        raise HTTPException(status_code=400, detail="Unsupported embedding model")
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts or len(texts) > 128 or any(not text or len(text) > 12000 for text in texts):
        raise HTTPException(status_code=422, detail="Expected 1-128 non-empty texts")
    if any(not text.startswith(("query: ", "passage: ")) for text in texts):
        raise HTTPException(status_code=422, detail="E5 inputs need a query: or passage: prefix")
    started = time.monotonic()
    # Tokenize validation as a batch instead of crossing Python/Rust per chunk.
    token_lengths = app.state.model.tokenizer(
        texts, add_special_tokens=True, return_length=True, truncation=False, padding=False,
    )["length"]
    if any(length > 512 for length in token_lengths):
        raise HTTPException(status_code=422, detail="E5 inputs must fit within 512 tokens")
    tokenized = time.monotonic()
    with _lock:
        inference_started = time.monotonic()
        vectors = app.state.model.encode(
            texts, batch_size=app.state.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        encoded = time.monotonic()
    return {
        "object": "list", "model": MODEL_ID,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector.tolist()}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": sum(token_lengths), "total_tokens": sum(token_lengths)},
        "elapsed_ms": round((encoded - started) * 1000, 2),
        "timing": {
            "tokenize_ms": round((tokenized - started) * 1000, 2),
            "queue_ms": round((inference_started - tokenized) * 1000, 2),
            "inference_ms": round((encoded - inference_started) * 1000, 2),
        },
    }
