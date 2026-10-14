"""Compare warm Ollama glossary latency and inspect contextual term meanings."""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper import db  # noqa: E402
from readpaper.config import settings  # noqa: E402
from readpaper.llm import OllamaClient  # noqa: E402
from readpaper.main import _word_context  # noqa: E402


CASES = [
    {
        "label": "multiword attention",
        "title": "Neural Sequence Models",
        "sentence": "Multi-head attention allows the model to attend to different representation subspaces.",
        "focus": 1,
        "context": "Each attention head projects queries, keys, and values separately.",
    },
    {
        "label": "reinforcement learning policy",
        "title": "Learning to Act from Rewards",
        "sentence": "The policy is optimized to maximize expected reward.",
        "focus": 1,
        "context": "An agent samples actions from its policy and receives rewards.",
    },
    {
        "label": "attention query",
        "title": "Attention for Sequence Modeling",
        "sentence": "Each query is compared with all keys to compute attention weights.",
        "focus": 1,
        "context": "Attention uses query and key vectors to calculate compatibility scores.",
    },
    {
        "label": "positional embedding",
        "title": "Representing Token Positions",
        "sentence": "We use a learned positional embedding for each input token.",
        "focus": 5,
        "context": "Position information is added to input token embeddings.",
    },
    {
        "label": "residual connections",
        "title": "Training Deep Encoders",
        "sentence": "The encoder uses residual connections around each sub-layer.",
        "focus": 3,
        "context": "Residual connections help gradients flow through deep layers.",
    },
]


async def preload(model: str) -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": "30m"},
        )
        response.raise_for_status()


def latest_paper_cases(limit: int) -> list[dict]:
    with db.connect() as connection:
        row = connection.execute(
            "SELECT title, data_json FROM papers WHERE status = 'ready' "
            "ORDER BY last_accessed DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise SystemExit("No processed paper found. Upload a PDF first.")

    data = json.loads(row["data_json"])
    words = {word["id"]: word for page in data["pages"] for word in page["words"]}
    sentences = data["sentences"]
    by_id = {sentence["id"]: sentence for sentence in sentences}
    cases = []
    seen = set()
    for gloss in data.get("word_glosses", {}).values():
        ids = tuple(gloss["word_ids"])
        if not ids or ids in seen:
            continue
        seen.add(ids)
        clicked = words[ids[0]]
        sentence = by_id[clicked["sentence_id"]]
        cases.append({
            "label": f"{gloss['original']} (saved: {gloss['meaning']})",
            "title": row["title"],
            "sentence": sentence["text"],
            "tokens": [words[word_id]["text"] for word_id in sentence["word_ids"]],
            "focus": sentence["word_ids"].index(ids[0]),
            "passages": _word_context(
                sentences, clicked["text"], sentence["id"], data.get("chunks", []),
            ),
        })
        if len(cases) == limit:
            break
    if not cases:
        raise SystemExit("No saved word glosses found. Click a few terms in the paper first.")
    return cases


async def benchmark(models: list[str], cases: list[dict]) -> None:
    for model in models:
        print(f"\n{model} (warm)", flush=True)
        await preload(model)
        durations = []
        for case in cases:
            client = OllamaClient()
            sentence = case["sentence"]
            tokens = case.get("tokens") or sentence.split()
            passages = case["passages"] if "passages" in case else [sentence, case["context"]]
            started = time.monotonic()
            result = await client.translate_keyword(
                case["title"], sentence, tokens, case["focus"],
                passages, model=model,
            )
            wall_ms = round((time.monotonic() - started) * 1000)
            durations.append(wall_ms)
            metrics = client.last_metrics
            term = " ".join(tokens[result["start"]:result["end"]])
            prompt_ms = round(metrics.get("prompt_eval_duration", 0) / 1_000_000)
            generation_ms = round(metrics.get("eval_duration", 0) / 1_000_000)
            print(
                f"{case['label']}: {wall_ms}ms "
                f"(input {prompt_ms}ms, output {generation_ms}ms) "
                f"{term} → {result['meaning']}",
                flush=True,
            )
        print(f"median: {round(statistics.median(durations))}ms", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latest-paper", action="store_true",
        help="use up to five saved term locations from the most recently opened paper",
    )
    parser.add_argument("models", nargs="+", help="Ollama model names to compare")
    args = parser.parse_args()
    asyncio.run(benchmark(
        args.models, latest_paper_cases(5) if args.latest_paper else CASES,
    ))
