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
from . import embedding_cache

INDEX_VERSION = 5
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
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


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
    # Splitters repeatedly count the same separators and sentence fragments.
    # Scope the cache to this build so paper text is released afterwards.
    token_count = lru_cache(maxsize=8192)(_token_count)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS, chunk_overlap=OVERLAP_TOKENS,
        length_function=token_count, separators=["\n\n", "\n", ". ", " ", ""],
    )

    def bounded_parts(text: str):
        if token_count(text) <= CHUNK_TOKENS:
            yield text
            return
        middle = text.rfind(" ", 0, len(text) // 2)
        if middle < 1:
            middle = len(text) // 2
        yield from bounded_parts(text[:middle].strip())
        yield from bounded_parts(text[middle:].strip())
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
                for bounded in bounded_parts(part.strip()):
                    if bounded:
                        units.append({"id": row["id"], "page": row.get("page"),
                                      "block": row.get("block"), "section": section,
                                      "text": bounded})

        group: list[dict[str, Any]] = []
        length = 0
        chunk_no = 0

        def save_group(items=None) -> None:
            nonlocal chunk_no
            current = group if items is None else items
            if not current:
                return
            text = " ".join(item["text"] for item in current)
            # Token counts can change at sentence joins. Check once per final
            # chunk, instead of re-tokenizing a growing paragraph every step.
            if len(current) > 1 and token_count(text) > CHUNK_TOKENS:
                save_group(current[:-1])
                save_group(current[-1:])
                return
            page = current[0]["page"]
            chunks.append({
                "id": f"{source['id']}-chunk-{chunk_no}", "document_id": source["id"],
                "kind": source["kind"], "title": source["title"], "url": source.get("url"),
                "content_scope": source.get("content_scope", "abstract"),
                "page": page + 1 if page is not None else None,
                "section": current[0]["section"],
                "text": text,
                "sentences": [{"id": item["id"], "text": item["text"]} for item in current],
                "sentence_ids": list(dict.fromkeys(item["id"] for item in current)),
            })
            chunk_no += 1

        for unit in units:
            size = token_count(unit["text"])
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
                        previous_size = token_count(previous["text"])
                        if carried_tokens + previous_size > OVERLAP_TOKENS:
                            break
                        carry.insert(0, previous)
                        carried_tokens += previous_size
                if carry and carried_tokens + size > CHUNK_TOKENS:
                    carry = []
                group = carry
                length = sum(token_count(item["text"]) for item in group)
            group.append(unit)
            length += size
        save_group()
    return chunks


class RemoteE5Embeddings(Embeddings):
    """LangChain embeddings adapter for the Mac mini's batched API."""

    def __init__(self, *, cache_documents: bool = True) -> None:
        if not settings.embedding_api_key:
            raise ValueError("READPAPER_EMBEDDING_API_KEY가 필요합니다.")
        self.client = httpx.Client(
            base_url=settings.embedding_base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            timeout=httpx.Timeout(120, connect=8),
        )
        self.elapsed_ms = 0.0
        self.cache_hits = 0
        self.generated_texts = 0
        self.cache_documents = cache_documents
        self.server_inference_ms = 0.0
        self.server_queue_ms = 0.0

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
            payload = response.json()
            timing = payload.get("timing") or {}
            for field, attribute in (("inference_ms", "server_inference_ms"), ("queue_ms", "server_queue_ms")):
                value = timing.get(field)
                if isinstance(value, (float, int)) and math.isfinite(value) and value >= 0:
                    setattr(self, attribute, getattr(self, attribute) + value)
            data = payload.get("data", [])
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
        prefixed = ["passage: " + text for text in texts]
        if not self.cache_documents:
            vectors = self._embed(prefixed)
            self.generated_texts += len(texts)
            return vectors
        keys = [embedding_cache.cache_key(text) for text in prefixed]
        cached = embedding_cache.read(keys)
        missing = {key: text for key, text in zip(keys, prefixed) if key not in cached}
        self.cache_hits += len(texts) - len(missing)
        if missing:
            vectors = self._embed(list(missing.values()))
            fresh = dict(zip(missing, vectors))
            embedding_cache.write(fresh)
            cached.update(fresh)
            self.generated_texts += len(fresh)
        return [cached[key] for key in keys]

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
                on_progress=None, *, extend: bool = False) -> dict[str, Any]:
    if not chunks:
        raise ValueError("검색할 본문이 없습니다")
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    if not extend:
        (directory / "index.json").unlink(missing_ok=True)
    client = _qdrant_client()
    name = _collection_name(paper_id)
    try:
        exists = client.collection_exists(name)
        if extend and not exists:
            raise ValueError("추가 색인의 원문 컬렉션이 없습니다.")
        if exists and not extend:
            client.delete_collection(name)
        if not extend:
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
            cache_hits = getattr(embeddings, "cache_hits", 0)
            generated_texts = getattr(embeddings, "generated_texts", len(chunks))
            server_inference_ms = getattr(embeddings, "server_inference_ms", 0.0)
            server_queue_ms = getattr(embeddings, "server_queue_ms", 0.0)
        (directory / "chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        metadata = {
            "status": "ready", "model": settings.embedding_model,
            "chunks": len(chunks), "dimensions": EMBEDDING_DIMENSIONS,
            "version": INDEX_VERSION, "store": "qdrant", "collection": name,
            "embedding_ms": embedding_ms,
            "cache_hits": cache_hits, "generated_texts": generated_texts,
            "server_inference_ms": round(server_inference_ms, 2),
            "server_queue_ms": round(server_queue_ms, 2),
            "index_ms": round((time.monotonic() - started) * 1000, 2),
        }
        metadata_path = directory / "index.tmp"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        metadata_path.replace(directory / "index.json")
        (directory / "vectors.npy").unlink(missing_ok=True)
        return metadata
    except Exception:
        try:
            if not extend and client.collection_exists(name):
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


def question_document(question: str) -> str | None:
    """Read the uploaded paper by default; widen explicit literature questions."""
    external = re.search(
        r"관련\s*(?:논문|연구)|다른\s*(?:논문|연구)|참고\s*문헌|피인용|인용한|인용된|비교|"
        r"\b(?:related|references?|citing|cited|compare|comparison|other papers)\b",
        question, re.IGNORECASE,
    )
    return None if external else "paper"


def retrieve(query: str, chunks: list[dict[str, Any]], limit: int = 6,
             directory: Path | None = None, paper_id: str | None = None,
             document_id: str | None = None) -> list[dict[str, Any]]:
    if not chunks or not query.strip():
        return []
    if directory is None:
        if document_id:
            chunks = [item for item in chunks if item.get("document_id") == document_id]
        if not chunks:
            return []
        vectorizer, matrix = _lexical(tuple(item["text"] for item in chunks))
        scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
        return [chunks[i] | {"score": round(float(scores[i]), 4)}
                for i in np.argsort(-scores)[:limit] if scores[i] > 0]
    if not paper_id or not index_ready(directory, paper_id):
        raise ValueError("검색 색인이 변경되었거나 사용할 수 없습니다. 문헌을 다시 준비해 주세요.")
    client = _qdrant_client()
    try:
        with RemoteE5Embeddings() as embeddings:
            scope = models.Filter(must=[models.FieldCondition(
                key="metadata.document_id", match=models.MatchValue(value=document_id),
            )]) if document_id else None
            hits = _store(client, paper_id, embeddings).similarity_search_with_score(
                query, k=max(limit * 4, 20), filter=scope,
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
