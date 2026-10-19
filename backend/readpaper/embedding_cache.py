"""Short-lived embedding reuse across papers without storing source text."""

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .config import settings


def cache_key(text: str) -> str:
    namespace = [settings.embedding_model, settings.embedding_base_url.rstrip("/"), text]
    return hashlib.sha256(json.dumps(namespace, ensure_ascii=False).encode()).hexdigest()


@contextmanager
def _connect(data_dir: Path | None = None):
    directory = data_dir if data_dir is not None else settings.data_dir
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / "embedding-cache.sqlite3", timeout=30)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS embeddings "
                           "(key TEXT PRIMARY KEY, vector BLOB NOT NULL, last_used REAL NOT NULL)")
        connection.execute("CREATE INDEX IF NOT EXISTS embeddings_expiry ON embeddings(last_used)")
        with connection:
            yield connection
    finally:
        connection.close()


def read(keys: list[str]) -> dict[str, list[float]]:
    if not keys:
        return {}
    now = time.time()
    with _connect() as connection:
        values = {}
        for key in set(keys):
            row = connection.execute(
                "SELECT vector FROM embeddings WHERE key=? AND last_used>?",
                (key, now - settings.paper_ttl_seconds),
            ).fetchone()
            if row:
                vector = np.frombuffer(row[0], dtype="<f4")
                if len(vector) == 384 and np.isfinite(vector).all():
                    values[key] = vector.tolist()
                    connection.execute("UPDATE embeddings SET last_used=? WHERE key=?", (now, key))
    return values


def write(values: dict[str, list[float]]) -> None:
    if not values:
        return
    with _connect() as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO embeddings(key,vector,last_used) VALUES(?,?,?)",
            [(key, np.asarray(vector, dtype="<f4").tobytes(), time.time())
             for key, vector in values.items()],
        )


def cleanup(data_dir: Path | None = None, ttl_seconds: int | None = None) -> None:
    directory = data_dir if data_dir is not None else settings.data_dir
    if not (directory / "embedding-cache.sqlite3").exists():
        return
    with _connect(directory) as connection:
        connection.execute("DELETE FROM embeddings WHERE last_used<=?",
                           (time.time() - (ttl_seconds if ttl_seconds is not None else settings.paper_ttl_seconds),))
