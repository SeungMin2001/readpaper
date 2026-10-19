"""Compare the previous CPU embedder with the Mac E5 API on identical chunks."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from readpaper.config import settings  # noqa: E402
from readpaper.rag import RemoteE5Embeddings  # noqa: E402


LEGACY_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64


def load_texts(path: Path, limit: int | None) -> list[str]:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json must contain a non-empty list")
    if limit is not None:
        chunks = chunks[:limit]
    texts = [chunk.get("text") for chunk in chunks]
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("every chunk needs non-empty text")
    return texts


def legacy_batch(model, texts: list[str]) -> None:
    """Reproduce the previous 128-token overflow and average path."""
    windows: list[str] = []
    owners: list[int] = []
    for index, value in enumerate(texts):
        encoding = model.model.tokenizer.encode(value)
        for part in [encoding, *encoding.overflowing]:
            offsets = [(start, end) for start, end in part.offsets if end > start]
            windows.append(value[min(start for start, _ in offsets):max(end for _, end in offsets)]
                           if offsets else value)
            owners.append(index)
    encoded = np.asarray(list(model.embed(windows, batch_size=32)), dtype=np.float32)
    vectors = np.stack([encoded[np.asarray(owners) == index].mean(axis=0)
                        for index in range(len(texts))])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-12)
    if vectors.shape != (len(texts), 384):
        raise ValueError("legacy embedder returned an unexpected shape")


def benchmark(mode: str, texts: list[str], repeats: int) -> dict:
    started = time.monotonic()
    if mode == "legacy-cpu":
        cache = settings.data_dir / "models"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
        from fastembed import TextEmbedding
        engine = TextEmbedding(LEGACY_MODEL, cache_dir=str(cache), threads=2)
        model = LEGACY_MODEL
    else:
        # Measure inference on every repeat, not the application's disk cache.
        engine = RemoteE5Embeddings(cache_documents=False)
        model = settings.embedding_model
    load_ms = round((time.monotonic() - started) * 1000, 2)
    runs = []
    try:
        for number in range(1, repeats + 1):
            if mode == "remote-e5":
                engine.elapsed_ms = 0.0
            started = time.monotonic()
            for offset in range(0, len(texts), BATCH_SIZE):
                batch = texts[offset:offset + BATCH_SIZE]
                if mode == "legacy-cpu":
                    legacy_batch(engine, batch)
                else:
                    vectors = engine.embed_documents(batch)
                    if len(vectors) != len(batch):
                        raise ValueError("remote embedder returned too few vectors")
                if (offset // BATCH_SIZE + 1) % 8 == 0:
                    print(f"run {number}/{repeats}: {min(offset + BATCH_SIZE, len(texts))}/{len(texts)} chunks",
                          file=sys.stderr, flush=True)
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            run = {"elapsed_ms": elapsed_ms,
                   "chunks_per_second": round(len(texts) * 1000 / max(elapsed_ms, 0.01), 2)}
            if mode == "remote-e5":
                run["api_ms"] = round(engine.elapsed_ms, 2)
            runs.append(run)
    finally:
        if mode == "remote-e5":
            engine.close()
    return {"mode": mode, "model": model, "chunks": len(texts),
            "characters": sum(map(len, texts)), "batch_size": BATCH_SIZE,
            "load_ms": load_ms, "runs": runs,
            "median_elapsed_ms": round(statistics.median(row["elapsed_ms"] for row in runs), 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True, help="Existing index/chunks.json")
    parser.add_argument("--mode", choices=("legacy-cpu", "remote-e5"), required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, help="First N chunks, for a quick connectivity check")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--repeats and --limit must be positive")
    texts = load_texts(args.chunks, args.limit)
    report = benchmark(args.mode, texts, args.repeats)
    report["source"] = str(args.chunks.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
