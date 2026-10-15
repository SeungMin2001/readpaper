"""Measure completed translations, keeping inference and application cache hits separate."""

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper import db  # noqa: E402
from readpaper.config import settings  # noqa: E402
from readpaper.llm import (  # noqa: E402
    VLLMClient, close_model_transport, model_transport, start_model_transport, validate_keyword_gloss,
)
from readpaper.terms import resolve_term  # noqa: E402


TRANSFORMER_SUMMARY = (
    "The Transformer replaces recurrence with multi-head attention and positional encodings "
    "to improve parallel sequence transduction."
)
CASES = [
    {"summary": TRANSFORMER_SUMMARY, "sentence": sentence, "focus": focus}
    for sentence, focus in [
        ("Multi-head attention captures token relationships.", 1),
        ("Each query is compared with keys.", 1),
        ("The positional embedding represents input positions.", 2),
        ("We add residual connections around sublayers.", 2),
        ("The encoder processes the input sequence.", 1),
        ("The decoder predicts output tokens.", 1),
        ("Scaled dot-product attention combines weighted values.", 2),
        ("The attention weights select relevant information.", 2),
        ("We apply layer normalization to representations.", 3),
        ("The learning rate changes during training.", 2),
        ("We optimize cross entropy loss.", 3),
        ("The beam search generates candidate sequences.", 2),
        ("We use gradient clipping during training.", 3),
        ("The vocabulary defines the output symbols.", 1),
        ("The softmax produces normalized probabilities.", 1),
        ("Dropout prevents overfitting during training.", 2),
        ("Autoregressive decoding predicts tokens sequentially.", 0),
        ("The sequence length varies between examples.", 1),
    ]
] + [
    {"summary": "The paper learns a policy that maximizes reward in robotic control.",
     "sentence": "The policy is optimized to maximize reward.", "focus": 1},
    {"summary": "The paper learns a policy that maximizes reward in robotic control.",
     "sentence": "Reinforcement learning improves robotic control.", "focus": 0},
]


def latest_paper_cases(limit: int) -> list[dict]:
    with db.connect() as connection:
        row = connection.execute(
            "SELECT id, data_json FROM papers ORDER BY last_accessed DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise ValueError("No paper found. Upload a PDF first.")
    data = json.loads(row["data_json"])
    summary = data.get("paper_summary")
    if not summary:
        raise ValueError("The paper summary is not ready.")
    words = {word["id"]: word for page in data["pages"] for word in page["words"]}
    cases, seen = [], set()
    ignored = set("the a an and or of in on to for with by from we our is are was were it this that as at be".split())
    for sentence in data["sentences"]:
        tokens = [words[word_id]["text"] for word_id in sentence["word_ids"]]
        for focus, token in enumerate(tokens):
            if len(token) < 3 or token.casefold() in ignored or not any(c.isalpha() for c in token):
                continue
            selected = resolve_term(tokens, focus)
            if selected["term"] in seen:
                continue
            seen.add(selected["term"])
            cases.append({
                "paper_id": row["id"], "word_id": sentence["word_ids"][focus],
                "summary": summary, "tokens": tokens, "focus": focus,
            })
            if len(cases) == limit:
                return cases
    if not cases:
        raise ValueError("No terms found in the paper.")
    return cases


def summarize(rows: list[dict], target_ms: float) -> dict:
    durations = sorted(row["elapsed_ms"] for row in rows if row.get("ok"))
    return {
        "samples": len(durations), "errors": sum(not row.get("ok") for row in rows),
        "median_ms": round(statistics.median(durations), 2) if durations else None,
        "p95_ms": durations[math.ceil(len(durations) * .95) - 1] if durations else None,
        "max_ms": max(durations) if durations else None,
        "under_target": sum(value < target_ms for value in durations),
        "all_under_target": bool(durations) and len(durations) == len(rows) and max(durations) < target_ms,
    }


async def baseline_json(client: VLLMClient, summary: str, term: str) -> dict:
    """Previous production prompt, preserved only here for a real A/B measurement."""
    return await client.chat_json(
        "Translate clicked_token into a concise Korean gloss using paper_summary. "
        "Copy clicked_token as term unless the summary clearly identifies a multiword "
        "scientific term containing it. Return only JSON with term and meaning; no explanation.",
        json.dumps({"paper_summary": summary, "clicked_token": term}, ensure_ascii=False),
        schema={"type": "object", "properties": {"term": {"type": "string"}, "meaning": {"type": "string"}},
                "required": ["term", "meaning"], "additionalProperties": False},
        options={"temperature": 0, "num_predict": 32},
    )


async def benchmark(cases: list[dict], args) -> dict:
    report = {
        "model": settings.llm_model, "target_ms": args.target_ms, "rows": [],
        "scope": "web API round trip (excludes browser rendering)" if args.api_url else
                 "model request (no application result cache; excludes web API and browser)",
        "status": "running", "goal_met": False,
    }
    await start_model_transport()
    try:
        async with model_transport() as transport:
            if args.api_url:
                response = await transport.get(args.api_url + "/api/auth/me", timeout=5)
                if response.status_code == 401:
                    response = await transport.post(args.api_url + "/api/auth/login", json={
                        "username": settings.username, "password": settings.password,
                    }, timeout=5)
                response.raise_for_status()
            else:
                headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
                response = await transport.get(settings.llm_base_url + "/models", headers=headers, timeout=5)
                response.raise_for_status()
                if settings.llm_model not in [item["id"] for item in response.json()["data"]]:
                    raise ValueError("The configured model is not listed by the model server.")
            for repeat in range(args.repeats):
                for index, case in enumerate(cases):
                    tokens = case.get("tokens") or case["sentence"].split()
                    selected = resolve_term(tokens, case["focus"])
                    variants = ["api"] if args.api_url else (["json", "plain"] if args.compare else ["plain"])
                    # Alternate order to reduce systematic first-request/warmup bias.
                    if (index + repeat) % 2:
                        variants = list(reversed(variants))
                    for variant in variants:
                        row = {"variant": variant, "repeat": repeat + 1, "term": selected["term"], "ok": False}
                        client = VLLMClient()
                        started = time.monotonic()
                        try:
                            if variant == "api":
                                response = await transport.post(
                                    f"{args.api_url}/api/papers/{case['paper_id']}/translate-word",
                                    json={"word_id": case["word_id"]},
                                )
                                response.raise_for_status()
                                result = response.json()
                                row["cache"] = "hit" if result["cached"] else "miss"
                                row["timing"] = result.get("timing", {})
                            else:
                                result = await (baseline_json(client, case["summary"], selected["term"])
                                                if variant == "json" else client.translate_keyword_from_summary(
                                                    case["summary"], selected["term"]))
                                result = validate_keyword_gloss(result, tokens, case["focus"])
                                row["cache"] = "no_application_cache"
                                row["timing"] = client.last_metrics
                            row["meaning"] = result["meaning"]
                            row["ok"] = True
                        except Exception as exc:
                            row["error"] = type(exc).__name__
                        row["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
                        report["rows"].append(row)
                        print(json.dumps(row, ensure_ascii=False), flush=True)
        for variant in ("json", "plain", "api"):
            selected_rows = [r for r in report["rows"] if r["variant"] == variant]
            if selected_rows:
                report[variant] = summarize(selected_rows, args.target_ms)
        if args.api_url:
            fresh = [r for r in report["rows"] if r.get("cache") != "hit"]
            hits = [r for r in report["rows"] if r.get("cache") == "hit"]
            report["uncached_api"] = summarize(fresh, args.target_ms)
            report["cached_api"] = summarize(hits, args.target_ms)
            report["goal_met"] = report["uncached_api"]["all_under_target"]
        else:
            report["goal_met"] = report.get("plain", {}).get("all_under_target", False)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "blocked"
        report["error"] = type(exc).__name__
        print(f"Live benchmark blocked: {type(exc).__name__}. Check endpoint routing and authentication.", file=sys.stderr)
    finally:
        await close_model_transport()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-paper", action="store_true", help="select unique terms from the latest saved PDF")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--compare", action="store_true", help="compare the previous JSON prompt with meaning-only output")
    parser.add_argument("--target-ms", type=float, default=1000, help="fail if any completed uncached translation reaches this limit")
    parser.add_argument("--api-url", help="measure the web API using --latest-paper; result-cache hits are reported separately")
    parser.add_argument("--json-out", type=Path, help="save all measurements, including failure status")
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1 or args.target_ms <= 0:
        parser.error("samples, repeats and target-ms must be positive")
    if args.api_url and (not args.latest_paper or args.compare):
        parser.error("--api-url requires --latest-paper and cannot be combined with --compare")
    if args.api_url:
        args.api_url = args.api_url.rstrip("/")
    try:
        cases = latest_paper_cases(args.samples) if args.latest_paper else CASES[:args.samples]
        report = asyncio.run(benchmark(cases, args))
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))
    return 2 if report["status"] == "blocked" else (0 if report["goal_met"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
