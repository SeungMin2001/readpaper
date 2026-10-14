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

REPO_ROOT = Path(__file__).resolve().parents[2]


def _report(sources: list[dict[str, Any]], warnings: list[str], include_text: bool) -> dict[str, Any]:
    report = {
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
                "is_oa": source.get("is_oa"),
                "oa_status": source.get("oa_status"),
                "open_access_locations": source.get("open_access_locations", []),
                "pdf_locations": source.get("pdf_locations", []),
                "text_chars": len(source.get("text", "")),
                "abstract_preview": source.get("text", "")[:240],
            }
            for source in sources
        ],
        "warnings": warnings,
    }
    if include_text:
        for item, source in zip(report["sources"], sources):
            item["text"] = source.get("text", "")
    return report


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    sources, warnings = await collect_external_sources(args.title, args.doi, args.keywords)
    return _report(sources, warnings, args.full_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True, help="Title of the uploaded paper")
    parser.add_argument("--doi", help="DOI without the https://doi.org/ prefix")
    parser.add_argument("--keywords", nargs="+", required=True, help="English research concepts")
    parser.add_argument(
        "--json-out", type=Path,
        help="Path inside the repository for the diagnostic report (default: data/inspection/sources-latest.json)",
    )
    parser.add_argument(
        "--full-text", "--include-source-text", dest="full_text", action="store_true",
        help="Include all text currently returned (abstract/background, not full papers)",
    )
    args = parser.parse_args()

    report = asyncio.run(_run(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    output_path = args.json_out or Path("data/inspection/sources-latest.json")
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path = output_path.resolve()
    if not output_path.is_relative_to(REPO_ROOT):
        parser.error("--json-out must point to a path inside the readpaper repository")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({
        "counts": report["counts"],
        "warnings": report["warnings"],
        "saved_to": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
