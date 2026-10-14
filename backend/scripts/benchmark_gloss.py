"""Measure vLLM glossary latency on sample or saved paper terms."""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper import db  # noqa: E402
from readpaper.config import settings  # noqa: E402
from readpaper.llm import VLLMClient, validate_keyword_gloss  # noqa: E402


CASES = [
    {
        "label": "multiword attention",
        "summary": "The Transformer replaces recurrence with multi-head attention to model relationships between tokens.",
        "sentence": "Multi-head attention allows the model to attend to different representation subspaces.",
        "clicked_token": "attention",
    },
    {
        "label": "reinforcement learning policy",
        "summary": "The paper trains a policy to maximize expected reward through reinforcement learning.",
        "sentence": "The policy is optimized to maximize expected reward.",
        "clicked_token": "policy",
    },
    {
        "label": "attention query",
        "summary": "The paper computes attention weights from query and key vectors in a sequence model.",
        "sentence": "Each query is compared with all keys to compute attention weights.",
        "clicked_token": "query",
    },
    {
        "label": "positional embedding",
        "summary": "The model adds learned position information to each input token representation.",
        "sentence": "We use a learned positional embedding for each input token.",
        "clicked_token": "embedding",
    },
    {
        "label": "residual connections",
        "summary": "The encoder uses skip connections around sub-layers to improve information flow.",
        "sentence": "The encoder uses residual connections around each sub-layer.",
        "clicked_token": "residual",
    },
]


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
    summary = data.get("paper_summary")
    if not summary:
        raise SystemExit("No saved paper summary found. Upload a PDF and wait for summary preparation.")
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
            "summary": summary,
            "sentence": sentence["text"],
            "tokens": [words[word_id]["text"] for word_id in sentence["word_ids"]],
            "focus": sentence["word_ids"].index(ids[0]),
            "clicked_token": clicked["text"],
        })
        if len(cases) == limit:
            break
    if not cases:
        raise SystemExit("No saved word glosses found. Click a few terms in the paper first.")
    return cases


async def benchmark(cases: list[dict]) -> None:
    print(f"\n{settings.llm_model} via {settings.llm_base_url}", flush=True)
    durations = []
    for case in cases:
        client = VLLMClient()
        sentence = case["sentence"]
        tokens = case.get("tokens") or sentence.split()
        focus = case.get("focus", next(
            (index for index, token in enumerate(tokens)
             if token.strip(".,;:!?()[]{}\\\"'").casefold()
             == case["clicked_token"].strip(".,;:!?()[]{}\\\"'").casefold()), 0,
        ))
        started = time.monotonic()
        result = await client.translate_keyword_from_summary(
            case["summary"], case["clicked_token"],
        )
        result = validate_keyword_gloss(result, tokens, focus)
        wall_ms = round((time.monotonic() - started) * 1000)
        durations.append(wall_ms)
        metrics = client.last_metrics
        term = " ".join(tokens[result["start"]:result["end"]])
        print(
            f"{case['label']}: {wall_ms}ms "
            f"(request {metrics.get('request_ms')}ms, "
            f"input {metrics.get('prompt_tokens')} tokens, "
            f"output {metrics.get('completion_tokens')} tokens) "
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
    args = parser.parse_args()
    asyncio.run(benchmark(latest_paper_cases(5) if args.latest_paper else CASES))
