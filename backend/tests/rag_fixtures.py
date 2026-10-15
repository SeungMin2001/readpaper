"""Deterministic embeddings for API tests; live embeddings use verify_rag_pipeline.py."""
import numpy as np


def fake_embed(texts):
    values = np.array([[1.0 + text.lower().count('control'), 1.0 + text.lower().count('attention'),
                        1.0 + text.lower().count('learning')] for text in texts], dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)
