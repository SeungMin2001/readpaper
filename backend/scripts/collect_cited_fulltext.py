"""Extract every bibliography item from a PDF and collect its available OA full text."""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET

import httpx
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper.citations import first_author_matches, parse_grobid_paper, title_similarity
from readpaper.config import settings


REPO_ROOT = Path(__file__).resolve().parents[2]
GROBID_DEFAULT = "http://127.0.0.1:8070"
USER_AGENT = "Readpaper/0.1 (open-access citation collection test)"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


class CollectionFailure(Exception):
    pass


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "paper"


def _load_source_overrides(path: Path | None) -> dict[int, list[dict[str, str]]]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionFailure(f"Could not read source overrides: {exc}") from exc
    if not isinstance(data, dict):
        raise CollectionFailure("Source overrides must be a JSON object keyed by bibliography number")
    overrides: dict[int, list[dict[str, str]]] = {}
    for key, value in data.items():
        try:
            number = int(key)
        except (TypeError, ValueError) as exc:
            raise CollectionFailure(f"Invalid bibliography number in source overrides: {key}") from exc
        candidates = value if isinstance(value, list) else [value]
        normalized = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise CollectionFailure(f"Source override {key} must contain an object or list of objects")
            url = candidate.get("pdf_url")
            if not isinstance(url, str) or not url.startswith("https://"):
                raise CollectionFailure(f"Source override {key} must use an HTTPS PDF URL")
            normalized.append({
                "pdf_url": url,
                "source": str(candidate.get("source") or "manual_source_override"),
                "version": str(candidate.get("version") or "unknown"),
            })
        overrides[number] = normalized
    return overrides


async def _extract_bibliography(client: httpx.AsyncClient, pdf_path: Path, grobid_url: str) -> str:
    endpoint = f"{grobid_url.rstrip('/')}/api/processFulltextDocument"
    try:
        response = await client.post(
            endpoint,
            data={"consolidateCitations": "0", "includeRawCitations": "1"},
            files={"input": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
            timeout=httpx.Timeout(180.0, connect=10.0),
        )
        response.raise_for_status()
    except (OSError, httpx.HTTPError) as exc:
        raise CollectionFailure(f"GROBID could not parse the root PDF: {exc}") from exc
    return response.text


async def _arxiv_metadata(client: httpx.AsyncClient, references: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ids = sorted({ref["arxiv_id"] for ref in references if ref.get("arxiv_id")})
    if not ids:
        return {}
    try:
        response = await client.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": ",".join(ids), "max_results": len(ids)},
            timeout=httpx.Timeout(45.0, connect=10.0),
        )
        response.raise_for_status()
        feed = ET.fromstring(response.text)
    except (httpx.HTTPError, ET.ParseError) as exc:
        print(f"arXiv metadata unavailable; will use bibliography titles and direct PDF URLs: {type(exc).__name__}", flush=True)
        return {}

    results: dict[str, dict[str, Any]] = {}
    for entry in feed.findall("atom:entry", ARXIV_NS):
        identifier = entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
        match = re.search(r"/(\d{4}\.\d{4,5})(?:v(\d+))?$", identifier)
        if not match:
            continue
        authors = [
            " ".join((author.findtext("atom:name", default="", namespaces=ARXIV_NS)).split())
            for author in entry.findall("atom:author", ARXIV_NS)
        ]
        results[match.group(1)] = {
            "title": " ".join(entry.findtext("atom:title", default="", namespaces=ARXIV_NS).split()),
            "authors": authors,
            "resolved_version": int(match.group(2)) if match.group(2) else None,
            "updated": entry.findtext("atom:updated", default=None, namespaces=ARXIV_NS),
            "landing_page_url": identifier,
        }
    return results


def _openalex_authors(work: dict[str, Any]) -> list[str]:
    names = []
    for authorship in work.get("authorships") or []:
        author = (authorship.get("author") or {}).get("display_name")
        if isinstance(author, str) and author:
            names.append(author)
    return names


def _openalex_pdf_locations(work: dict[str, Any]) -> list[dict[str, Any]]:
    locations = list(work.get("locations") or [])
    best = work.get("best_oa_location")
    if isinstance(best, dict) and best not in locations:
        locations.append(best)
    found = []
    seen = set()
    for location in locations:
        pdf_url = location.get("pdf_url")
        if not location.get("is_oa") or not isinstance(pdf_url, str) or not pdf_url.startswith("https://"):
            continue
        if pdf_url in seen:
            continue
        seen.add(pdf_url)
        found.append({
            "pdf_url": pdf_url,
            "landing_page_url": location.get("landing_page_url"),
            "version": location.get("version"),
            "license": location.get("license"),
        })
    return found


def _crossref_authors(work: dict[str, Any]) -> list[str]:
    return [
        " ".join(part for part in (author.get("given"), author.get("family")) if isinstance(part, str))
        for author in work.get("author") or []
        if isinstance(author, dict)
    ]


def _crossref_year(work: dict[str, Any]) -> int | None:
    for field in ("published-print", "published-online", "published", "issued"):
        date_parts = (work.get(field) or {}).get("date-parts")
        if date_parts and date_parts[0] and isinstance(date_parts[0][0], int):
            return date_parts[0][0]
    return None


def _crossref_pdf_locations(work: dict[str, Any]) -> list[dict[str, Any]]:
    found = []
    seen = set()
    for link in work.get("link") or []:
        if not isinstance(link, dict):
            continue
        url = link.get("URL")
        content_type = str(link.get("content-type") or "").lower()
        if not isinstance(url, str) or not url.startswith("https://") or "pdf" not in content_type:
            continue
        if url in seen:
            continue
        seen.add(url)
        found.append({
            "pdf_url": url,
            "landing_page_url": work.get("URL"),
            "version": link.get("intended-application") or "crossref_pdf_link",
            "license": None,
        })
    return found


def _select_crossref_match(reference: dict[str, Any], works: list[dict[str, Any]]) -> dict[str, Any] | None:
    expected_title = reference.get("title") or ""
    scored = []
    for work in works:
        titles = work.get("title") or []
        candidate_title = titles[0] if titles else ""
        score = title_similarity(expected_title, candidate_title)
        author_match = first_author_matches(reference.get("authors", []), _crossref_authors(work))
        year = reference.get("year")
        candidate_year = _crossref_year(work)
        year_match = year is None or candidate_year is None or abs(year - candidate_year) <= 2
        expected = " ".join(re.findall(r"[a-z0-9]+", expected_title.lower()))
        candidate = " ".join(re.findall(r"[a-z0-9]+", candidate_title.lower()))
        title_prefix = bool(expected and candidate.startswith(expected))
        if (score >= 0.82 and (author_match or year_match)) or (
            title_prefix and author_match and year_match
        ):
            scored.append((score, author_match, year_match, work))
    if not scored:
        return None
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return scored[0][3]


async def _crossref_search(
    client: httpx.AsyncClient, reference: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    query = reference.get("raw_reference") or reference.get("title") or ""
    if not query.strip():
        return None, "crossref_reference_text_missing"
    params: dict[str, Any] = {"query.bibliographic": query[:500], "rows": 10}
    if settings.crossref_email:
        params["mailto"] = settings.crossref_email
    try:
        response = await client.get("https://api.crossref.org/works", params=params)
        response.raise_for_status()
        works = response.json().get("message", {}).get("items", [])
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        return None, f"crossref_search_failed: {type(exc).__name__}"
    selected = _select_crossref_match(reference, works)
    return (selected, None) if selected else (None, "crossref_title_match_not_confident")


async def _unpaywall_pdf_location(
    client: httpx.AsyncClient, doi: str | None
) -> dict[str, Any] | None:
    if not doi or not settings.unpaywall_email:
        return None
    doi = doi.removeprefix("https://doi.org/").strip()
    try:
        response = await client.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": settings.unpaywall_email},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        work = response.json()
    except (httpx.HTTPError, ValueError, TypeError):
        return None
    locations = [work.get("best_oa_location"), *(work.get("oa_locations") or [])]
    for location in locations:
        if not isinstance(location, dict):
            continue
        url = location.get("url_for_pdf")
        if isinstance(url, str) and url.startswith("https://"):
            return {
                "pdf_url": url,
                "landing_page_url": location.get("landing_page_url"),
                "version": location.get("version") or "unpaywall_oa_location",
                "license": location.get("license"),
            }
    return None


async def _openalex_search(client: httpx.AsyncClient, reference: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    title = reference.get("title")
    if not title:
        return None, "reference_title_missing"
    query = " ".join(re.findall(r"[a-z0-9]+", title.lower()))
    if not query:
        return None, "reference_title_missing"

    params: dict[str, Any] = {"search": query, "per-page": 10}
    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key
    if reference.get("doi"):
        doi = reference["doi"].removeprefix("https://doi.org/").strip()
        doi_params = {"filter": f"doi:https://doi.org/{doi}", "per-page": 5}
        if settings.openalex_api_key:
            doi_params["api_key"] = settings.openalex_api_key
        try:
            doi_response = await client.get(
                f"{settings.openalex_url}/works",
                params=doi_params,
            )
            if doi_response.status_code < 400:
                doi_results = doi_response.json().get("results", [])
                selected = _select_openalex_match(reference, doi_results)
                if selected:
                    return selected, None
        except (httpx.HTTPError, ValueError):
            pass

    try:
        response = await client.get(f"{settings.openalex_url}/works", params=params)
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after", "1")
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 1.0
            # A long Retry-After usually means the shared unauthenticated API
            # budget is exhausted; waiting minutes or hours cannot help this run.
            if delay > 10:
                return None, "openalex_rate_limited"
            await asyncio.sleep(min(5.0, max(1.0, delay)))
            response = await client.get(f"{settings.openalex_url}/works", params=params)
        response.raise_for_status()
    except (httpx.HTTPError, ValueError) as exc:
        return None, f"openalex_search_failed: {type(exc).__name__}"
    selected = _select_openalex_match(reference, response.json().get("results", []))
    return (selected, None) if selected else (None, "openalex_title_match_not_confident")


def _select_openalex_match(reference: dict[str, Any], works: list[dict[str, Any]]) -> dict[str, Any] | None:
    expected_title = reference.get("title") or ""
    scored = []
    for work in works:
        candidate_title = work.get("display_name") or ""
        score = title_similarity(expected_title, candidate_title)
        authors = _openalex_authors(work)
        author_match = first_author_matches(reference.get("authors", []), authors)
        year = reference.get("year")
        candidate_year = work.get("publication_year")
        year_match = year is None or candidate_year is None or abs(year - candidate_year) <= 2
        normalized_expected = " ".join(re.findall(r"[a-z0-9]+", expected_title.lower()))
        normalized_candidate = " ".join(re.findall(r"[a-z0-9]+", candidate_title.lower()))
        title_is_prefix = bool(normalized_expected and normalized_candidate.startswith(normalized_expected))
        metadata_agrees = author_match or year_match
        accepted = (score >= 0.86 and metadata_agrees) or (title_is_prefix and author_match and year_match)
        if accepted:
            scored.append((score, author_match, year_match, work))
    if not scored:
        return None
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return scored[0][3]


async def _download_pdf(client: httpx.AsyncClient, url: str, max_bytes: int) -> bytes:
    if not url.startswith("https://"):
        raise CollectionFailure("PDF URL is not HTTPS")
    chunks = []
    total = 0
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            if response.url.scheme != "https":
                raise CollectionFailure("download redirected away from HTTPS")
            announced = response.headers.get("content-length")
            if announced and announced.isdigit() and int(announced) > max_bytes:
                raise CollectionFailure(f"PDF exceeds {max_bytes // (1024 * 1024)} MB limit")
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise CollectionFailure(f"PDF exceeds {max_bytes // (1024 * 1024)} MB limit")
                chunks.append(chunk)
    except httpx.HTTPError as exc:
        response = getattr(exc, "response", None)
        status = response.status_code if response is not None else "request error"
        raise CollectionFailure(f"HTTP {status}") from exc
    content = b"".join(chunks)
    if not content.startswith(b"%PDF-"):
        raise CollectionFailure("response did not contain a PDF file")
    return content


def _extract_pdf_text(pdf_bytes: bytes) -> tuple[str, int, str, str]:
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            page_count = document.page_count
            if page_count == 0 or page_count > 500:
                raise CollectionFailure("PDF page count is outside the 1–500 page limit")
            page_texts = [page.get_text("text").strip() for page in document]
            title = str((document.metadata or {}).get("title") or "")
    except CollectionFailure:
        raise
    except Exception as exc:
        raise CollectionFailure("PDF could not be parsed") from exc
    text = "\n\n".join(
        f"[Page {index + 1}]\n{page_text}" for index, page_text in enumerate(page_texts) if page_text
    )
    return text, page_count, title, page_texts[0] if page_texts else ""


def _title_match_score(expected: str, actual: str, first_page: str) -> float:
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", value.lower()))
    expected_normalized = normalize(expected)
    observed = normalize(f"{actual} {first_page}")
    if expected_normalized and expected_normalized in observed:
        return 1.0
    terms = set(expected_normalized.split())
    if not terms:
        return 0.0
    return len(terms & set(observed.split())) / len(terms)


async def _collect_one(
    client: httpx.AsyncClient,
    reference: dict[str, Any],
    arxiv_metadata: dict[str, dict[str, Any]],
    source_overrides: dict[int, list[dict[str, str]]],
    collection_dir: Path,
    max_bytes: int,
) -> dict[str, Any]:
    result = dict(reference)
    arxiv_id = reference.get("arxiv_id")
    openalex_work = reference.get("openalex_work")
    crossref_work = reference.get("crossref_work")
    arxiv = arxiv_metadata.get(arxiv_id) if arxiv_id else None
    crossref_titles = (crossref_work or {}).get("title") or []
    expected_title = (
        (arxiv or {}).get("title") or reference.get("title")
        or (crossref_titles[0] if crossref_titles else None)
    )
    if arxiv and expected_title:
        result["arxiv_title"] = arxiv.get("title")
        result["arxiv_landing_page_url"] = arxiv.get("landing_page_url")
        result["arxiv_updated"] = arxiv.get("updated")
        result["arxiv_latest_version"] = arxiv.get("resolved_version")

    urls: list[dict[str, Any]] = [
        {**candidate, "source": f"source_override:{candidate['source']}"}
        for candidate in source_overrides.get(reference["number"], [])
    ]
    if arxiv_id:
        version = reference.get("arxiv_version")
        version_suffix = f"v{version}" if version else ""
        arxiv_candidate = {
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}{version_suffix}",
            "version": f"v{version}" if version else "latest",
            "source": "arxiv_id_from_bibliography",
        }
        if all(arxiv_candidate["pdf_url"] != existing["pdf_url"] for existing in urls):
            urls.append(arxiv_candidate)
    unpaywall_location = await _unpaywall_pdf_location(
        client, reference.get("doi") or (crossref_work or {}).get("DOI")
    )
    if unpaywall_location and all(
        unpaywall_location["pdf_url"] != existing["pdf_url"] for existing in urls
    ):
        urls.append({**unpaywall_location, "source": "unpaywall_oa_location"})
    if openalex_work:
        for location in _openalex_pdf_locations(openalex_work):
            if all(location["pdf_url"] != existing["pdf_url"] for existing in urls):
                urls.append({**location, "source": "openalex_oa_location"})
    if crossref_work:
        for location in _crossref_pdf_locations(crossref_work):
            if all(location["pdf_url"] != existing["pdf_url"] for existing in urls):
                urls.append({**location, "source": "crossref_pdf_link"})
    result["resolved_title"] = expected_title
    if openalex_work:
        result["openalex"] = {
            "id": openalex_work.get("id"),
            "doi": openalex_work.get("doi"),
            "title": openalex_work.get("display_name"),
            "year": openalex_work.get("publication_year"),
            "is_oa": (openalex_work.get("open_access") or {}).get("is_oa"),
            "pdf_locations": _openalex_pdf_locations(openalex_work),
        }
    if crossref_work:
        result["crossref"] = {
            "doi": crossref_work.get("DOI"),
            "title": crossref_titles[0] if crossref_titles else None,
            "year": _crossref_year(crossref_work),
            "pdf_locations": _crossref_pdf_locations(crossref_work),
        }
    if not urls:
        warning = reference.get("resolution_warning") or ""
        if warning.startswith(("openalex_search_failed", "crossref_search_failed")) or "openalex_rate_limited" in warning:
            result["fulltext_status"] = "metadata_lookup_failed"
        elif warning:
            result["fulltext_status"] = "bibliographic_match_unresolved"
        else:
            result["fulltext_status"] = "no_open_access_pdf_url_listed"
        return result

    if not expected_title:
        result["fulltext_status"] = "reference_title_missing"
        return result
    errors = []
    mismatches = []
    for candidate in urls:
        try:
            pdf_bytes = await _download_pdf(client, candidate["pdf_url"], max_bytes)
            text, page_count, metadata_title, first_page = _extract_pdf_text(pdf_bytes)
            score = _title_match_score(expected_title, metadata_title, first_page)
            if score < 0.8:
                mismatches.append({
                    "url": candidate["pdf_url"],
                    "title_match_score": round(score, 3),
                    "pdf_metadata_title": metadata_title,
                    "pdf_first_page_preview": " ".join(first_page.split())[:320],
                })
                continue

            file_stem = f"ref-{reference['number']:02d}-{_safe_slug(expected_title)}"
            pdf_path = collection_dir / f"{file_stem}.pdf"
            text_path = collection_dir / f"{file_stem}.txt"
            pdf_path.write_bytes(pdf_bytes)
            text_path.write_text(text + "\n", encoding="utf-8")
            result.update({
                "fulltext_status": "text_extracted",
                "download_url": candidate["pdf_url"],
                "download_source": candidate.get("source"),
                "download_version": candidate.get("version"),
                "download_bytes": len(pdf_bytes),
                "page_count": page_count,
                "extracted_chars": len(text),
                "pdf_metadata_title": metadata_title,
                "title_match_score": round(score, 3),
                "pdf_path": pdf_path.relative_to(REPO_ROOT).as_posix(),
                "text_path": text_path.relative_to(REPO_ROOT).as_posix(),
            })
            return result
        except CollectionFailure as exc:
            errors.append({"url": candidate["pdf_url"], "error": str(exc)})

    result["attempt_errors"] = errors
    if mismatches:
        result["fulltext_status"] = "downloaded_title_mismatch"
        result["title_mismatch_attempts"] = mismatches
    else:
        result["fulltext_status"] = "download_failed"
    return result


async def _run(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    root_pdf = args.root_pdf.resolve()
    if not root_pdf.is_file():
        raise CollectionFailure(f"Root PDF not found: {root_pdf}")
    if args.source_overrides and not args.source_overrides.resolve().is_relative_to(REPO_ROOT):
        raise CollectionFailure("Source override files must be inside the readpaper repository")
    source_overrides = _load_source_overrides(args.source_overrides)
    timeout = httpx.Timeout(35.0, connect=10.0)
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        tei_xml = await _extract_bibliography(client, root_pdf, args.grobid_url)
        root_metadata = parse_grobid_paper(tei_xml)
        references = root_metadata["references"]
        root_title = args.title or root_metadata.get("title") or root_pdf.stem
        root_doi = args.root_doi or root_metadata.get("doi")
        root_arxiv_id = args.root_arxiv_id or root_metadata.get("arxiv_id")
        print(f"GROBID parsed {len(references)} bibliography entries.", flush=True)
        arxiv_metadata = await _arxiv_metadata(client, references)
        for reference in references:
            if reference.get("arxiv_id") and reference.get("arxiv_id") not in arxiv_metadata:
                reference["metadata_warning"] = "arxiv_id_not_resolved"
            if reference.get("arxiv_id") and not reference.get("title"):
                reference["title"] = (arxiv_metadata.get(reference["arxiv_id"]) or {}).get("title")

        for index, reference in enumerate(references, start=1):
            if reference.get("arxiv_id"):
                reference["resolution_method"] = "bibliography_arxiv_id"
                print(f"[{index}/{len(references)}] arXiv ID {reference['arxiv_id']}: {reference.get('title')}", flush=True)
                continue
            if source_overrides.get(reference["number"]):
                reference["resolution_method"] = "verified_pdf_source_override"
                print(f"[{index}/{len(references)}] verified PDF source: {reference.get('title')}", flush=True)
                continue
            work, warning = await _openalex_search(client, reference)
            if work:
                reference["openalex_work"] = work
                reference["resolution_method"] = "openalex_title_author_year_match"
                print(f"[{index}/{len(references)}] OpenAlex match: {reference.get('title')}", flush=True)
            else:
                crossref_work, crossref_warning = await _crossref_search(client, reference)
                if crossref_work:
                    reference["crossref_work"] = crossref_work
                    reference["resolution_method"] = "crossref_title_author_year_match"
                    print(f"[{index}/{len(references)}] Crossref match: {reference.get('title')}", flush=True)
                else:
                    reference["resolution_warning"] = f"{warning}; {crossref_warning}"
                    reference["resolution_method"] = "unresolved"
                    print(f"[{index}/{len(references)}] unresolved: {reference.get('title') or reference.get('raw_reference')}", flush=True)
            await asyncio.sleep(0.35)

        collection_dir = REPO_ROOT / "data" / "papers" / "collected" / f"{_safe_slug(root_title)}-citations"
        collection_dir.mkdir(parents=True, exist_ok=True)
        max_bytes = args.max_file_mb * 1024 * 1024
        results = []
        for index, reference in enumerate(references, start=1):
            result = await _collect_one(
                client, reference, arxiv_metadata, source_overrides, collection_dir, max_bytes
            )
            results.append(result)
            print(f"[{index}/{len(references)}] full text: {result['fulltext_status']} — {result.get('title') or result.get('resolved_title')}", flush=True)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root_paper": {
            "title": root_title,
            "doi": root_doi,
            "arxiv_id": root_arxiv_id,
            "authors": root_metadata.get("authors", []),
            "pdf_path": root_pdf.relative_to(REPO_ROOT).as_posix() if root_pdf.is_relative_to(REPO_ROOT) else str(root_pdf),
            "pdf_sha256": hashlib.sha256(root_pdf.read_bytes()).hexdigest(),
            "bibliography_source": "GROBID-parsed references from the supplied root PDF",
        },
        "reference_count": len(references),
        "source_overrides_file": (
            args.source_overrides.resolve().relative_to(REPO_ROOT).as_posix()
            if args.source_overrides else None
        ),
        "resolution_method_counts": dict(Counter(row.get("resolution_method", "unknown") for row in results)),
        "fulltext_status_counts": dict(Counter(row["fulltext_status"] for row in results)),
        "references": results,
        "limitations": [
            "참고문헌 항목은 업로드 PDF에 실제 인쇄된 목록을 GROBID로 구조화했습니다.",
            "인용된 자료 중 공개 PDF URL을 찾지 못했거나 다운로드가 실패한 자료는 전문이 저장되지 않습니다.",
            "수동으로 보완한 PDF 주소도 첫 페이지 제목 검사를 통과해야 저장합니다.",
            "PDF 첫 페이지 제목 유사도로 다른 논문 PDF를 걸러내며, DOI와 출판 판본까지 항상 증명하는 절차는 아닙니다.",
            "텍스트 추출은 PyMuPDF 텍스트 레이어 기준이며 OCR, 청킹, 임베딩은 수행하지 않습니다.",
            "참고문헌은 업로드 PDF에서 읽고, 메타데이터 후보는 OpenAlex와 Crossref에서 대조합니다.",
            "Unpaywall은 READPAPER_UNPAYWALL_EMAIL이 설정된 경우에만 조회합니다.",
            "메타데이터 조회에 실패하면 공개 PDF가 없다고 단정하지 않고 별도 실패 상태로 표시합니다.",
        ],
    }
    return report, tei_xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-pdf", type=Path, required=True, help="Root paper PDF to parse")
    parser.add_argument("--title", help="Root paper title (defaults to GROBID/PDF metadata)")
    parser.add_argument("--root-doi", help="Optional root-paper DOI (otherwise read from the PDF)")
    parser.add_argument("--root-arxiv-id", help="Optional root arXiv identifier for provenance")
    parser.add_argument("--grobid-url", default=GROBID_DEFAULT, help="GROBID base URL")
    parser.add_argument("--max-file-mb", type=int, default=50, help="Maximum size of one PDF")
    parser.add_argument(
        "--source-overrides", type=Path,
        help="Optional JSON of manually verified HTTPS PDF URLs keyed by bibliography number",
    )
    parser.add_argument(
        "--json-out", type=Path, default=Path("data/inspection/cited-fulltext.json"),
        help="Manifest path inside the repository",
    )
    parser.add_argument(
        "--xml-out", type=Path, default=Path("data/inspection/cited-references-grobid.xml"),
        help="Raw GROBID TEI path inside the repository",
    )
    args = parser.parse_args()
    if args.max_file_mb < 1:
        parser.error("--max-file-mb must be at least 1")
    try:
        report, tei_xml = asyncio.run(_run(args))
    except CollectionFailure as exc:
        parser.error(str(exc))
    json_path = (args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out).resolve()
    xml_path = (args.xml_out if args.xml_out.is_absolute() else REPO_ROOT / args.xml_out).resolve()
    if not json_path.is_relative_to(REPO_ROOT) or not xml_path.is_relative_to(REPO_ROOT):
        parser.error("output paths must point inside the readpaper repository")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    xml_path.write_text(tei_xml, encoding="utf-8")
    print(json.dumps({
        "reference_count": report["reference_count"],
        "resolution_method_counts": report["resolution_method_counts"],
        "fulltext_status_counts": report["fulltext_status_counts"],
        "manifest": json_path.relative_to(REPO_ROOT).as_posix(),
        "grobid_xml": xml_path.relative_to(REPO_ROOT).as_posix(),
        "papers_directory": (REPO_ROOT / "data" / "papers" / "collected" / f"{_safe_slug(report['root_paper']['title'])}-citations").relative_to(REPO_ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
