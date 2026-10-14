"""Download OpenAlex-listed open-access PDFs and save their extracted text for inspection."""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys
from typing import Any

import httpx
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper.sources import collect_external_sources
from readpaper.config import settings


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "data" / "papers" / "collected"
DEFAULT_REPORT = Path("data/inspection/fulltext-retrieval.json")
USER_AGENT = "Readpaper/0.1 (open-access paper text inspection)"
TITLE_STOPWORDS = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}


class DownloadFailure(Exception):
    pass


def _safe_openalex_id(source: dict[str, Any]) -> str:
    raw = str(source.get("openalex_id") or "")
    match = re.search(r"W\d+", raw)
    if match:
        return match.group(0)
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", source.get("title", "paper"))[:80] or "paper"


async def _download_pdf(client: httpx.AsyncClient, url: str, max_bytes: int) -> bytes:
    if not url.startswith("https://"):
        raise DownloadFailure("PDF URL is not HTTPS")
    chunks: list[bytes] = []
    total = 0
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            if response.url.scheme != "https":
                raise DownloadFailure("download redirected away from HTTPS")
            announced = response.headers.get("content-length")
            if announced and announced.isdigit() and int(announced) > max_bytes:
                raise DownloadFailure(f"PDF exceeds {max_bytes // (1024 * 1024)} MB limit")
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise DownloadFailure(f"PDF exceeds {max_bytes // (1024 * 1024)} MB limit")
                chunks.append(chunk)
    except httpx.HTTPError as exc:
        response = getattr(exc, "response", None)
        status = response.status_code if response is not None else "request error"
        raise DownloadFailure(f"HTTP {status}") from exc
    content = b"".join(chunks)
    if not content.startswith(b"%PDF-"):
        raise DownloadFailure("response did not contain a PDF file")
    return content


def _extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int, str, str]:
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            page_count = document.page_count
            if page_count == 0:
                raise DownloadFailure("PDF has no pages")
            if page_count > 500:
                raise DownloadFailure("PDF exceeds 500-page extraction limit")
            page_texts = [page.get_text("text").strip() for page in document]
            title = str((document.metadata or {}).get("title") or "")
    except DownloadFailure:
        raise
    except Exception as exc:
        raise DownloadFailure("PDF could not be parsed") from exc
    text = "\n\n".join(
        f"[Page {index + 1}]\n{page_text}" for index, page_text in enumerate(page_texts) if page_text
    )
    return text, page_count, title, page_texts[0] if page_texts else ""


def _title_match_score(expected_title: str, pdf_title: str, first_page_text: str) -> float:
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", value.lower()))
    expected = normalize(expected_title)
    content = normalize(f"{pdf_title} {first_page_text}")
    if expected and expected in content:
        return 1.0
    expected_terms = set(expected.split()) - TITLE_STOPWORDS
    content_terms = set(content.split())
    if not expected_terms:
        return 0.0
    return len(expected_terms & content_terms) / len(expected_terms)


def _source_record(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": source.get("kind"),
        "title": source.get("title"),
        "year": source.get("year"),
        "doi": source.get("doi"),
        "openalex_id": source.get("openalex_id"),
        "is_oa": source.get("is_oa", False),
        "oa_status": source.get("oa_status"),
        "open_access_locations": source.get("open_access_locations", []),
        "abstract_chars": len(source.get("text", "")),
        "pdf_locations": source.get("pdf_locations", []),
    }


async def _inspect_candidate(
    client: httpx.AsyncClient, source: dict[str, Any], max_bytes: int
) -> dict[str, Any]:
    record = _source_record(source)
    locations = source.get("pdf_locations") or []
    if not source.get("is_oa") or not locations:
        record["fulltext_status"] = "no_open_access_pdf_url_listed"
        return record

    file_stem = _safe_openalex_id(source)
    errors = []
    title_mismatches = []
    extraction_failures = []
    empty_text = False
    for location in locations:
        url = location.get("pdf_url")
        if not isinstance(url, str):
            continue
        try:
            pdf_bytes = await _download_pdf(client, url, max_bytes)
            try:
                text, page_count, pdf_title, first_page_text = _extract_pdf_text(pdf_bytes)
            except DownloadFailure as exc:
                extraction_failures.append({"url": url, "error": str(exc)})
                continue

            record["download_url"] = url
            record["download_bytes"] = len(pdf_bytes)
            record["page_count"] = page_count
            record["pdf_metadata_title"] = pdf_title
            record["extracted_chars"] = len(text)
            record["pdf_first_page_preview"] = " ".join(first_page_text.split())[:320]
            match_score = _title_match_score(source.get("title", ""), pdf_title, first_page_text)
            record["title_match_score"] = round(match_score, 3)
            if not text.strip():
                empty_text = True
                continue
            if match_score < 0.6:
                title_mismatches.append({
                    "url": url,
                    "title_match_score": round(match_score, 3),
                    "pdf_metadata_title": pdf_title,
                    "pdf_first_page_preview": record["pdf_first_page_preview"],
                })
                continue

            pdf_path = PAPER_DIR / f"{file_stem}.pdf"
            pdf_path.write_bytes(pdf_bytes)
            text_path = PAPER_DIR / f"{file_stem}.txt"
            text_path.write_text(text + "\n", encoding="utf-8")
            record["pdf_path"] = pdf_path.relative_to(REPO_ROOT).as_posix()
            record["text_path"] = text_path.relative_to(REPO_ROOT).as_posix()
            record["fulltext_status"] = "text_extracted"
            return record
        except DownloadFailure as exc:
            errors.append({"url": url, "error": str(exc)})

    record["attempt_errors"] = errors + extraction_failures
    if title_mismatches:
        record["fulltext_status"] = "downloaded_title_mismatch"
        record["title_mismatch_attempts"] = title_mismatches
    elif empty_text:
        record["fulltext_status"] = "pdf_downloaded_no_text_layer"
    elif extraction_failures:
        record["fulltext_status"] = "pdf_downloaded_text_extraction_failed"
    else:
        record["fulltext_status"] = "download_failed"
    return record


async def _root_relationships(title: str, doi: str | None) -> dict[str, list[str]]:
    params: dict[str, Any] = {"per-page": 1}
    if doi:
        params["filter"] = f"doi:https://doi.org/{doi}"
    else:
        params["search"] = title[:200]
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(15.0, connect=8.0),
            headers={"User-Agent": USER_AGENT}, follow_redirects=True,
        ) as client:
            response = await client.get(f"{settings.openalex_url}/works", params=params)
            response.raise_for_status()
            results = response.json().get("results", [])
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return {"referenced": [], "related": []}
    if not results:
        return {"referenced": [], "related": []}
    if not doi:
        normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        if SequenceMatcher(None, normalize(title), normalize(results[0].get("display_name", ""))).ratio() < 0.8:
            return {"referenced": [], "related": []}
    root = results[0]
    return {
        "referenced": [value for value in (root.get("referenced_works") or []) if isinstance(value, str)],
        "related": [value for value in (root.get("related_works") or []) if isinstance(value, str)],
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    sources, warnings = await collect_external_sources(args.title, args.doi, args.keywords)
    root_relations = await _root_relationships(args.title, args.doi)
    available_ids = {source.get("openalex_id") for source in sources if source.get("openalex_id")}
    relationship_coverage = {}
    for kind, work_ids in root_relations.items():
        found_ids = set(work_ids) & available_ids
        relationship_coverage[kind] = {
            "ids_reported_by_root_work": len(work_ids),
            "metadata_records_found": len(found_ids),
            "unresolved_openalex_ids": sorted(set(work_ids) - found_ids),
        }
    discovery_kinds = {"referenced", "related"} | ({"citing"} if args.include_citing else set())
    candidates = [source for source in sources if source.get("kind") in discovery_kinds]
    counts = Counter(source.get("kind", "unknown") for source in candidates)

    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for source in candidates:
        kind = source["kind"]
        if selected_counts[kind] >= args.per_kind_limit:
            continue
        selected.append(source)
        selected_counts[kind] += 1
    selected_ids = {source.get("openalex_id") for source in selected}

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    max_bytes = args.max_file_mb * 1024 * 1024
    timeout = httpx.Timeout(30.0, connect=10.0)
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        inspected = []
        for source in selected:
            inspected.append(await _inspect_candidate(client, source, max_bytes))

    all_candidates = [_source_record(source) for source in candidates]
    for record in all_candidates:
        record["selected_for_download"] = record.get("openalex_id") in selected_ids

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {"title": args.title, "doi": args.doi, "keywords": args.keywords},
        "relationship_coverage": relationship_coverage,
        "discovered_counts": dict(counts),
        "selected_counts": dict(selected_counts),
        "fulltext_status_counts": dict(Counter(row["fulltext_status"] for row in inspected)),
        "warnings": warnings,
        "candidates": all_candidates,
        "fulltext_attempts": inspected,
        "limitations": [
            "수집 대상은 OpenAlex가 공개 접근(open access) PDF로 표시한 URL뿐입니다.",
            "OpenAlex에 직접 PDF URL이 없으면 공개 저장소의 랜딩 페이지를 탐색하지 않습니다. 따라서 PDF URL 미등록은 전문 접근 불가를 뜻하지 않습니다.",
            "이 스크립트는 PDF 텍스트 레이어를 추출하며 스캔 페이지의 OCR은 하지 않습니다.",
            "저장 전에 PDF 첫 페이지 텍스트와 논문 제목을 대조합니다. 자동 대조가 완벽하지 않으므로 결과도 직접 확인해야 합니다.",
            "다운로드/추출 결과는 아직 청킹하거나 임베딩하지 않습니다.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True, help="Title of the uploaded paper")
    parser.add_argument("--doi", help="DOI without the https://doi.org/ prefix")
    parser.add_argument("--keywords", nargs="+", required=True, help="English research concepts")
    parser.add_argument(
        "--per-kind-limit", type=int, default=2,
        help="Maximum PDFs to test per relation type (default: 2)",
    )
    parser.add_argument(
        "--include-citing", action="store_true",
        help="Also try papers that cite the input paper (disabled by default)",
    )
    parser.add_argument("--max-file-mb", type=int, default=50, help="Maximum size of one PDF")
    parser.add_argument(
        "--json-out", type=Path, default=DEFAULT_REPORT,
        help="Manifest path inside the repository (default: data/inspection/fulltext-retrieval.json)",
    )
    args = parser.parse_args()
    if args.per_kind_limit < 1 or args.max_file_mb < 1:
        parser.error("--per-kind-limit and --max-file-mb must be at least 1")

    report = asyncio.run(_run(args))
    output_path = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
    output_path = output_path.resolve()
    if not output_path.is_relative_to(REPO_ROOT):
        parser.error("--json-out must point to a path inside the readpaper repository")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "discovered_counts": report["discovered_counts"],
        "selected_counts": report["selected_counts"],
        "fulltext_status_counts": report["fulltext_status_counts"],
        "saved_manifest": output_path.relative_to(REPO_ROOT).as_posix(),
        "papers_directory": PAPER_DIR.relative_to(REPO_ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
