"""Fetch and print OpenAlex/Wikipedia sources for a paper without running the LLM."""

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper.sources import collect_external_sources


def _report(sources: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    return {
        "counts": dict(Counter(source["kind"] for source in sources)),
        "sources": [
            {
                "kind": source["kind"],
                "title": source["title"],
                "year": source.get("year"),
                "publication_date": source.get("publication_date"),
                "content_scope": source.get("content_scope"),
                "openalex_id": source.get("openalex_id"),
                "doi": source.get("doi"),
                "url": source.get("url"),
                "text_chars": len(source.get("text", "")),
                "abstract_preview": source.get("text", "")[:240],
            }
            for source in sources
        ],
        "warnings": warnings,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    sources, warnings = await collect_external_sources(args.title, args.doi, args.keywords)
    return _report(sources, warnings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True, help="Title of the uploaded paper")
    parser.add_argument("--doi", help="DOI without the https://doi.org/ prefix")
    parser.add_argument("--keywords", nargs="+", required=True, help="English research concepts")
    parser.add_argument("--json-out", type=Path, help="Optional path to save the full diagnostic report")
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
