"""Deterministic retrieval substitutes that never download model weights."""

import re
import zlib
from collections import Counter

from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector


def fake_token_count(text: str) -> int:
    return len(text.split()) + 2


class FakeRemoteE5Embeddings(Embeddings):
    elapsed_ms = 1.0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    @staticmethod
    def _vector(text: str) -> list[float]:
        lower = text.lower()
        values = [0.0] * 384
        values[0] = 1 + lower.count("robot") + lower.count("control")
        values[1] = 1 + lower.count("attention") + lower.count("language")
        values[2] = 1 + lower.count("learning")
        return values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeSparseEmbeddings(SparseEmbeddings):
    @staticmethod
    def _vector(text: str) -> SparseVector:
        counts = Counter(re.findall(r"[a-z0-9]+", text.lower()))
        sparse = sorted((zlib.crc32(word.encode()), float(count))
                        for word, count in counts.items())
        return SparseVector(indices=[item[0] for item in sparse],
                            values=[item[1] for item in sparse])

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._vector(text)
