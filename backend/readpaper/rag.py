"""Page-aware evidence chunks and persistent multilingual semantic retrieval."""
import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import settings

_model_lock = threading.Lock()
INDEX_VERSION = 3


def build_chunks(sentences: list[dict[str, Any]], external_sources: list[dict[str, Any]],
                 paper_title: str) -> list[dict[str, Any]]:
    chunks = []
    documents = [{'id': 'paper', 'kind': 'paper', 'title': paper_title,
                  'url': None, 'content_scope': 'full_text', 'sentences': sentences}] + external_sources
    for source in documents:
        rows = source.get('sentences') or [
            {'id': f's{i}', 'page': None, 'text': text}
            for i, text in enumerate(re.split(r'(?<=[.!?])\s+', source.get('text', ''))) if text.strip()
        ]
        # Bound chunks even for PDFs with long unpunctuated blocks. Retain original sentence IDs.
        units = []
        for row in rows:
            words = row['text'].split()
            for offset in range(0, len(words), 70):
                units.append(dict(row, text=' '.join(words[offset:offset + 70])))
        offset = 0
        while offset < len(units):
            group, length = [], 0
            for row in units[offset:]:
                if group and (row.get('page') != group[0].get('page') or length + len(row['text'].split()) > 110):
                    break
                group.append(row)
                length += len(row['text'].split())
            page = group[0].get('page')
            chunks.append({
                'id': f"{source['id']}-chunk-{offset}", 'document_id': source['id'],
                'kind': source['kind'], 'title': source['title'], 'url': source.get('url'),
                'content_scope': source.get('content_scope', 'abstract'),
                'page': page + 1 if page is not None else None,
                'text': ' '.join(row['text'] for row in group),
                'sentences': [{'id': row['id'], 'text': row['text']} for row in group],
                'sentence_ids': list(dict.fromkeys(row['id'] for row in group)),
            })
            offset += max(1, len(group) - 1) if len(group) > 1 else 1
    return chunks


@lru_cache(maxsize=1)
def embedding_model():
    cache = settings.data_dir / 'models'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', str(cache / 'huggingface'))
    from fastembed import TextEmbedding
    return TextEmbedding(settings.embedding_model, cache_dir=str(cache), threads=2)


def embed(texts: list[str]) -> np.ndarray:
    with _model_lock:
        model = embedding_model()
        # MiniLM accepts 128 tokens. Encode every overflow window instead of
        # silently discarding the ends of scientific paragraphs or long queries.
        windows, owners = [], []
        for index, text in enumerate(texts):
            encoding = model.model.tokenizer.encode(text)
            for part in [encoding, *encoding.overflowing]:
                offsets = [(start, end) for start, end in part.offsets if end > start]
                windows.append(text[min(start for start, _ in offsets):max(end for _, end in offsets)] if offsets else text)
                owners.append(index)
        encoded = np.asarray(list(model.embed(windows, batch_size=32)), dtype=np.float32)
        values = np.stack([encoded[np.asarray(owners) == index].mean(axis=0) for index in range(len(texts))])
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def build_index(chunks: list[dict], directory: Path, on_progress=None) -> dict:
    if not chunks:
        raise ValueError('검색할 본문이 없습니다')
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    for offset in range(0, len(chunks), 64):
        batch = chunks[offset:offset + 64]
        parts.append(embed([item['text'] for item in batch]))
        if on_progress:
            on_progress(min(offset + 64, len(chunks)), len(chunks))
    vectors = np.concatenate(parts)
    with (directory / 'vectors.tmp').open('wb') as stream:
        np.save(stream, vectors, allow_pickle=False)
    (directory / 'chunks.json').write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding='utf-8')
    (directory / 'vectors.tmp').replace(directory / 'vectors.npy')
    metadata = {'status': 'ready', 'model': settings.embedding_model,
                'chunks': len(chunks), 'dimensions': vectors.shape[1], 'version': INDEX_VERSION}
    (directory / 'index.json').write_text(json.dumps(metadata), encoding='utf-8')
    _lexical.cache_clear()
    return metadata


@lru_cache(maxsize=8)
def _lexical(texts: tuple[str, ...]):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
    return vectorizer, vectorizer.fit_transform(texts)


def retrieve(query: str, chunks: list[dict[str, Any]], limit: int = 6,
             directory: Path | None = None) -> list[dict[str, Any]]:
    if not chunks or not query.strip():
        return []
    try:
        vectorizer, matrix = _lexical(tuple(item['text'] for item in chunks))
        lexical = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    except ValueError:
        lexical = np.zeros(len(chunks))
    if directory is not None:
        metadata = json.loads((directory / 'index.json').read_text())
        if (metadata.get('version') != INDEX_VERSION or metadata['model'] != settings.embedding_model
                or metadata['chunks'] != len(chunks)):
            raise ValueError('검색 색인이 변경되었습니다. 문헌을 다시 준비해 주세요.')
        vectors = np.load(directory / 'vectors.npy', allow_pickle=False)
        semantic = vectors @ embed([query])[0]
        # Semantic + lexical score fusion, then avoid near-identical overlapping chunks.
        scores = .8 * semantic + .2 * lexical
    else:
        scores = lexical
    selected = []
    for index in np.argsort(-scores):
        if scores[index] <= .025:
            continue
        candidate = chunks[index]
        ids = set(candidate.get('sentence_ids', []))
        if any(candidate.get('document_id') == row.get('document_id') and ids
               and len(ids & set(row.get('sentence_ids', []))) / len(ids) > .6 for row in selected):
            continue
        selected.append(candidate | {'score': round(float(scores[index]), 4)})
        if len(selected) >= limit:
            break
    return selected
