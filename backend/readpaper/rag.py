"""Page-aware paper chunks and hybrid retrieval backed by Qdrant."""

import json
import math
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from huggingface_hub import hf_hub_download
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import Tokenizer

from .config import settings

INDEX_VERSION = 4
EMBEDDING_DIMENSIONS = 384
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 55
COLLECTION_PREFIX = "readpaper_"
_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:abstract|introduction|background|related work|"
    r"method(?:s|ology)?|approach|experiment(?:s|al setup)?|results?|"
    r"discussion|conclusion(?:s)?|references|bibliography|appendix(?: [A-Z])?)$",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    cache = settings.data_dir / "models" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=settings.embedding_model, filename="tokenizer.json",
                           cache_dir=str(cache))
    return Tokenizer.from_file(path)


def _token_count(text: str) -> int:
    return len(_tokenizer().encode(text).ids)


def _section_heading(text: str) -> str | None:
    candidate = text.strip().rstrip(":").strip()
    if len(candidate) > 80 or len(candidate.split()) > 8:
        return None
    if _HEADING.fullmatch(candidate):
        return candidate
    if re.fullmatch(r"\d+(?:\.\d+)*\s+[A-Z][\w\s-]{2,65}", candidate):
        return candidate
    return None


def build_chunks(sentences: list[dict[str, Any]], external_sources: list[dict[str, Any]],
                 paper_title: str) -> list[dict[str, Any]]:
    """Preserve page, section and sentence provenance while sizing by E5 tokens."""
    chunks: list[dict[str, Any]] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS, chunk_overlap=OVERLAP_TOKENS,
        length_function=_token_count, separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents = [{"id": "paper", "kind": "paper", "title": paper_title,
                  "url": None, "content_scope": "full_text", "sentences": sentences}] + external_sources
    for source in documents:
        rows = source.get("sentences") or [
            {"id": f"s{i}", "page": None, "text": text}
            for i, text in enumerate(re.split(r"(?<=[.!?])\s+", source.get("text", "")))
            if text.strip()
        ]
        section: str | None = None
        units: list[dict[str, Any]] = []
        for row in rows:
            heading = _section_heading(row["text"])
            if heading:
                if re.search(r"\b(?:references|bibliography)\b", heading, re.IGNORECASE):
                    break
                section = heading
                continue
            for part in splitter.split_text(row["text"].strip()):
                if part.strip():
                    units.append({"id": row["id"], "page": row.get("page"),
                                  "block": row.get("block"), "section": section,
                                  "text": part.strip()})

        group: list[dict[str, Any]] = []
        length = 0
        chunk_no = 0

        def save_group() -> None:
            nonlocal chunk_no
            if not group:
                return
            page = group[0]["page"]
            chunks.append({
                "id": f"{source['id']}-chunk-{chunk_no}", "document_id": source["id"],
                "kind": source["kind"], "title": source["title"], "url": source.get("url"),
                "content_scope": source.get("content_scope", "abstract"),
                "page": page + 1 if page is not None else None,
                "section": group[0]["section"],
                "text": " ".join(item["text"] for item in group),
                "sentences": [{"id": item["id"], "text": item["text"]} for item in group],
                "sentence_ids": list(dict.fromkeys(item["id"] for item in group)),
            })
            chunk_no += 1

        for unit in units:
            size = _token_count(unit["text"])
            boundary = bool(group and (
                unit["page"] != group[0]["page"] or
                unit["section"] != group[0]["section"]
            ))
            new_block = bool(group and unit["block"] != group[-1]["block"])
            if group and (boundary or length + size > CHUNK_TOKENS or
                          (new_block and length >= 320)):
                save_group()
                carry: list[dict[str, Any]] = []
                if not boundary and not new_block:
                    carried_tokens = 0
                    for previous in reversed(group):
                        token_count = _token_count(previous["text"])
                        if carried_tokens + token_count > OVERLAP_TOKENS:
                            break
                        carry.insert(0, previous)
                        carried_tokens += token_count
                if carry and carried_tokens + size > CHUNK_TOKENS:
                    carry = []
                group = carry
                length = sum(_token_count(item["text"]) for item in group)
            group.append(unit)
            length += size
        save_group()
    return chunks


class RemoteE5Embeddings(Embeddings):
    """LangChain embeddings adapter for the Mac mini's batched API."""

    def __init__(self) -> None:
        if not settings.embedding_api_key:
            raise ValueError("READPAPER_EMBEDDING_API_KEY가 필요합니다.")
        self.client = httpx.Client(
            base_url=settings.embedding_base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            timeout=httpx.Timeout(120, connect=8),
        )
        self.elapsed_ms = 0.0

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), 64):
            batch = texts[offset:offset + 64]
            started = time.monotonic()
            response = self.client.post("embeddings", json={
                "model": settings.embedding_model, "input": batch,
            })
            response.raise_for_status()
            self.elapsed_ms += (time.monotonic() - started) * 1000
            data = response.json().get("data", [])
            if len(data) != len(batch):
                raise ValueError("임베딩 서버의 결과 개수가 일치하지 않습니다.")
            for index, item in enumerate(data):
                vector = item.get("embedding")
                if (item.get("index") != index or not isinstance(vector, list) or
                        len(vector) != EMBEDDING_DIMENSIONS or
                        not all(isinstance(value, (int, float)) and math.isfinite(value)
                                for value in vector)):
                    raise ValueError("임베딩 서버가 유효하지 않은 벡터를 반환했습니다.")
                vectors.append(vector)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(["passage: " + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed(["query: " + text])[0]


@lru_cache(maxsize=1)
def _sparse_embeddings() -> FastEmbedSparse:
    cache = settings.data_dir / "models" / "fastembed"
    cache.mkdir(parents=True, exist_ok=True)
    return FastEmbedSparse(model_name="Qdrant/bm25", cache_dir=str(cache), threads=2)


def _qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=30)


def _collection_name(paper_id: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", paper_id):
        raise ValueError("유효하지 않은 논문 ID입니다.")
    return COLLECTION_PREFIX + paper_id


def _store(client: QdrantClient, paper_id: str, embeddings: Embeddings) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client, collection_name=_collection_name(paper_id),
        embedding=embeddings, sparse_embedding=_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID, vector_name="dense",
        sparse_vector_name="bm25", validate_collection_config=False,
    )


def _document(chunk: dict[str, Any]) -> Document:
    section = chunk.get("section")
    text = f"{section}\n{chunk['text']}" if section else chunk["text"]
    return Document(page_content=text, metadata=chunk)


def build_index(chunks: list[dict[str, Any]], directory: Path, paper_id: str,
                on_progress=None) -> dict[str, Any]:
    if not chunks:
        raise ValueError("검색할 본문이 없습니다")
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index.json").unlink(missing_ok=True)
    client = _qdrant_client()
    name = _collection_name(paper_id)
    try:
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(size=EMBEDDING_DIMENSIONS,
                                                           distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        with RemoteE5Embeddings() as embeddings:
            store = _store(client, paper_id, embeddings)
            for offset in range(0, len(chunks), 64):
                batch = chunks[offset:offset + 64]
                store.add_documents(
                    [_document(chunk) for chunk in batch], batch_size=64,
                    ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, f"{paper_id}/{chunk['id']}"))
                         for chunk in batch],
                )
                if on_progress:
                    on_progress(min(offset + len(batch), len(chunks)), len(chunks))
            embedding_ms = round(embeddings.elapsed_ms, 2)
        (directory / "chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        metadata = {
            "status": "ready", "model": settings.embedding_model,
            "chunks": len(chunks), "dimensions": EMBEDDING_DIMENSIONS,
            "version": INDEX_VERSION, "store": "qdrant", "collection": name,
            "embedding_ms": embedding_ms,
            "index_ms": round((time.monotonic() - started) * 1000, 2),
        }
        (directory / "index.json").write_text(json.dumps(metadata), encoding="utf-8")
        (directory / "vectors.npy").unlink(missing_ok=True)
        return metadata
    except Exception:
        try:
            if client.collection_exists(name):
                client.delete_collection(name)
        except Exception:
            pass
        raise
    finally:
        client.close()


def index_ready(directory: Path, paper_id: str, expected_chunks: int | None = None) -> bool:
    try:
        metadata = json.loads((directory / "index.json").read_text())
        if (metadata.get("version") != INDEX_VERSION or
                metadata.get("model") != settings.embedding_model or
                metadata.get("collection") != _collection_name(paper_id) or
                (expected_chunks is not None and metadata.get("chunks") != expected_chunks)):
            return False
        client = _qdrant_client()
        try:
            return client.collection_exists(metadata["collection"])
        finally:
            client.close()
    except (OSError, ValueError, KeyError, httpx.HTTPError, ApiException):
        return False


def prune_indexes(active_paper_ids: set[str]) -> None:
    """Remove Qdrant collections left by expired papers, including after an outage."""
    client = _qdrant_client()
    try:
        for item in client.get_collections().collections:
            if re.fullmatch(r"readpaper_[0-9a-f]{32}", item.name):
                if item.name[len(COLLECTION_PREFIX):] not in active_paper_ids:
                    client.delete_collection(item.name)
    finally:
        client.close()


@lru_cache(maxsize=8)
def _lexical(texts: tuple[str, ...]):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
    return vectorizer, vectorizer.fit_transform(texts)


def retrieve(query: str, chunks: list[dict[str, Any]], limit: int = 6,
             directory: Path | None = None, paper_id: str | None = None) -> list[dict[str, Any]]:
    if not chunks or not query.strip():
        return []
    if directory is None:
        vectorizer, matrix = _lexical(tuple(item["text"] for item in chunks))
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
        return [chunks[i] | {"score": round(float(scores[i]), 4)}
                for i in np.argsort(-scores)[:limit] if scores[i] > 0]
    if not paper_id or not index_ready(directory, paper_id, len(chunks)):
        raise ValueError("검색 색인이 변경되었거나 사용할 수 없습니다. 문헌을 다시 준비해 주세요.")
    client = _qdrant_client()
    try:
        with RemoteE5Embeddings() as embeddings:
            hits = _store(client, paper_id, embeddings).similarity_search_with_score(
                query, k=max(limit * 4, 20),
            )
    finally:
        client.close()
    by_id = {item["id"]: item for item in chunks}
    selected: list[dict[str, Any]] = []
    for doc, score in hits:
        candidate = by_id.get(doc.metadata.get("id"))
        if candidate is None:
            continue
        ids = set(candidate.get("sentence_ids", []))
        if any(candidate.get("document_id") == row.get("document_id") and ids
               and len(ids & set(row.get("sentence_ids", []))) / len(ids) > .6 for row in selected):
            continue
        selected.append(candidate | {"score": round(float(score), 4)})
        if len(selected) >= limit:
            break
    return selected
