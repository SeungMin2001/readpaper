import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import date, timedelta
from difflib import SequenceMatcher
from typing import Any

import httpx

from .config import settings


ActivityReporter = Callable[[str, str], Awaitable[None]]
SOURCE_DISCOVERY_VERSION = 2


async def _report_activity(
    reporter: ActivityReporter | None, message: str, state: str = "info",
) -> None:
    if reporter:
        await reporter(message, state)


def reconstruct_abstract(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    positions: dict[int, str] = {}
    for word, offsets in index.items():
        if isinstance(offsets, list):
            for offset in offsets:
                if isinstance(offset, int) and offset >= 0:
                    positions[offset] = word
    return " ".join(positions[offset] for offset in sorted(positions))[:6000]


def _similar_title(left: str, right: str) -> bool:
    normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    first, second = normalize(left), normalize(right)
    return bool(first and second) and SequenceMatcher(None, first, second).ratio() >= 0.8


def _openalex_params(params: dict[str, Any]) -> dict[str, Any]:
    if settings.openalex_api_key:
        return params | {"api_key": settings.openalex_api_key}
    return params


def pdf_reference_dois(text: str) -> tuple[int, list[str]]:
    """Read numbered PDF bibliography entries without treating body citations as references."""
    heading = re.search(r"(?im)^[ \t]*(?:references|bibliography)[ \t]*$", text)
    if not heading:
        return 0, []
    section = text[heading.end():]
    markers = list(re.finditer(r"(?m)^\[(\d{1,4})\]\s*", section))
    if not markers or int(markers[0].group(1)) != 1:
        return 0, []
    identifiers: list[str] = []
    seen: set[str] = set()
    count = 0
    for index, marker in enumerate(markers):
        if int(marker.group(1)) != count + 1:
            break
        count += 1
        end = markers[index + 1].start() if index + 1 < len(markers) else len(section)
        entry = section[marker.end():end][:800]
        arxiv = re.search(r"(?i)(?:\barxiv\s*:\s*|arxiv\.org/abs/)(\d{4}\.\d{4,5})\b", entry)
        doi = re.search(r"(?i)\b10\.\d{4,9}/[^\s]+", entry)
        if arxiv:
            value = f"10.48550/arxiv.{arxiv.group(1)}"
        elif doi:
            value = doi.group(0).rstrip('.,;)]}')
        else:
            continue
        if value and value.lower() not in seen:
            seen.add(value.lower())
            identifiers.append(value)
    return count, identifiers


def _paper_source(work: dict[str, Any], kind: str, index: int) -> dict[str, Any] | None:
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    if not work.get("display_name"):
        return None
    url = work.get("doi") or (work.get("primary_location") or {}).get("landing_page_url") or work.get("id")
    if not isinstance(url, str) or not url.startswith("https://"):
        url = None
    open_access = work.get("open_access") or {}
    oa_locations = list(work.get("locations") or [])
    best_oa_location = work.get("best_oa_location")
    if isinstance(best_oa_location, dict):
        known_urls = {
            location.get("pdf_url") for location in oa_locations if isinstance(location, dict)
        }
        if best_oa_location.get("pdf_url") not in known_urls:
            oa_locations.append(best_oa_location)
    open_access_locations = []
    pdf_locations = []
    seen_pdf_urls: set[str] = set()
    seen_locations: set[tuple[str | None, str | None]] = set()
    for location in oa_locations:
        if not isinstance(location, dict) or not location.get("is_oa"):
            continue
        pdf_url = location.get("pdf_url")
        landing_page_url = location.get("landing_page_url")
        pdf_url = pdf_url if isinstance(pdf_url, str) and pdf_url.startswith("https://") else None
        landing_page_url = (
            landing_page_url if isinstance(landing_page_url, str)
            and landing_page_url.startswith("https://") else None
        )
        location_key = (pdf_url, landing_page_url)
        if location_key not in seen_locations:
            seen_locations.add(location_key)
            open_access_locations.append({
                "pdf_url": pdf_url,
                "landing_page_url": landing_page_url,
                "license": location.get("license"),
                "version": location.get("version"),
            })
        if not pdf_url or pdf_url in seen_pdf_urls:
            continue
        seen_pdf_urls.add(pdf_url)
        pdf_locations.append({
            "pdf_url": pdf_url,
            "landing_page_url": landing_page_url,
            "license": location.get("license"),
            "version": location.get("version"),
        })
    authors = [
        entry.get("author", {}).get("display_name", "")
        for entry in work.get("authorships", [])[:3] if isinstance(entry, dict)
    ]
    return {
        "id": f"external-{index}", "kind": kind,
        "openalex_id": work.get("id"), "doi": work.get("doi"),
        "referenced_works": work.get("referenced_works") or [],
        "title": work["display_name"],
        "text": abstract, "content_scope": "abstract" if abstract else "metadata", "url": url,
        "year": work.get("publication_year"), "publication_date": work.get("publication_date"),
        "is_oa": bool(
            open_access.get("is_oa")
            or (isinstance(best_oa_location, dict) and best_oa_location.get("is_oa"))
        ),
        "oa_status": open_access.get("oa_status"),
        "open_access_locations": open_access_locations,
        "pdf_locations": pdf_locations,
        "authors": [name for name in authors if name],
    }


async def collect_external_sources(
    title: str,
    doi: str | None,
    keywords: list[str],
    on_event: ActivityReporter | None = None,
    reference_text: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    sources: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_works: set[str] = set()

    def add_work(item: dict[str, Any], kind: str) -> None:
        key = item.get("id") or item.get("doi")
        if not isinstance(key, str) or key in seen_works:
            return
        source = _paper_source(item, kind, len(sources))
        if source:
            seen_works.add(key)
            sources.append(source)

    timeout = httpx.Timeout(8.0, connect=5.0)
    headers = {"User-Agent": "Readpaper/0.1 (research paper reading assistant)"}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        work: dict[str, Any] | None = None
        lookup_requests = []
        if doi:
            lookup_requests.append({"filter": f"doi:https://doi.org/{doi}", "per-page": 5})
        lookup_requests.append({"search": title[:200], "per-page": 5})
        lookup_errors = []
        for params in lookup_requests:
            lookup_kind = "DOI" if "filter" in params else "제목"
            await _report_activity(on_event, f"OpenAlex에서 {lookup_kind}으로 업로드 논문을 찾는 중")
            try:
                response = await client.get(
                    f"{settings.openalex_url}/works", params=_openalex_params(params)
                )
                response.raise_for_status()
                results = response.json().get("results", [])
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                lookup_errors.append(type(exc).__name__)
                continue
            work = next((
                item for item in results
                if _similar_title(title, item.get("display_name", ""))
            ), None)
            if work:
                break
        if work:
            await _report_activity(on_event, f"업로드 논문 확인: {work.get('display_name', title)}", "success")
        if not work and lookup_errors:
            warnings.append("OpenAlex에 연결할 수 없어 관련·인용 논문을 수집하지 못했습니다.")
            await _report_activity(on_event, warnings[-1], "warning")

        if work:
            if isinstance(work.get("id"), str):
                seen_works.add(work["id"])
            # OpenAlex provides the complete reference list and up to 20 related works.
            # Fetch the relation candidates in batches instead of silently truncating references.
            relation_kinds: dict[str, str] = {}
            relation_ids: list[str] = []
            for kind, work_ids in (
                ("referenced", work.get("referenced_works") or []),
                ("related", (work.get("related_works") or [])[:20]),
            ):
                for work_id in work_ids:
                    work_key = work_id.rsplit("/", 1)[-1] if isinstance(work_id, str) else ""
                    if re.fullmatch(r"W\d+", work_key) and work_key not in relation_kinds:
                        relation_kinds[work_key] = kind
                        relation_ids.append(work_key)

            reference_total = sum(kind == "referenced" for kind in relation_kinds.values())
            related_total = sum(kind == "related" for kind in relation_kinds.values())
            await _report_activity(
                on_event,
                f"OpenAlex 관계: 참고문헌 {reference_total}개 · 관련 연구 후보 {related_total}개",
                "success",
            )

            for offset in range(0, len(relation_ids), 50):
                batch = relation_ids[offset:offset + 50]
                batch_end = min(offset + len(batch), len(relation_ids))
                await _report_activity(
                    on_event,
                    f"참고·관련 논문 서지정보 조회 중 ({offset + 1}–{batch_end}/{len(relation_ids)})",
                )
                try:
                    response = await client.get(f"{settings.openalex_url}/works", params=_openalex_params({
                        "filter": f"openalex_id:{'|'.join(batch)}", "per-page": len(batch),
                    }))
                    response.raise_for_status()
                    batch_results = response.json().get("results", [])
                    by_id = {
                        item.get("id", "").rsplit("/", 1)[-1]: item
                        for item in batch_results if isinstance(item.get("id"), str)
                    }
                    for work_key in batch:
                        item = by_id.get(work_key)
                        if item:
                            add_work(item, relation_kinds[work_key])
                    await _report_activity(
                        on_event,
                        f"참고·관련 논문 서지정보 {batch_end}/{len(relation_ids)}개 처리 완료",
                        "success",
                    )
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    warnings.append("인용·관련 논문 일부의 서지 정보를 가져오지 못했습니다.")
                    await _report_activity(on_event, warnings[-1], "warning")
                    break

            if not reference_total and reference_text:
                bibliography_count, reference_dois = pdf_reference_dois(reference_text)
                if bibliography_count:
                    await _report_activity(
                        on_event,
                        f"PDF 참고문헌 {bibliography_count}개에서 DOI·arXiv 식별자 {len(reference_dois)}개 확인",
                        "success",
                    )
                for offset in range(0, len(reference_dois), 25):
                    batch = reference_dois[offset:offset + 25]
                    try:
                        response = await client.get(f"{settings.openalex_url}/works", params=_openalex_params({
                            "filter": "doi:" + "|".join(f"https://doi.org/{value}" for value in batch),
                            "per-page": len(batch),
                        }))
                        response.raise_for_status()
                        expected = {value.lower() for value in batch}
                        for item in response.json().get("results", []):
                            item_doi = str(item.get("doi") or "").removeprefix("https://doi.org/").lower()
                            if item_doi in expected:
                                add_work(item, "referenced")
                    except (httpx.HTTPError, ValueError, KeyError, TypeError):
                        warnings.append("PDF 참고문헌 일부의 서지 정보를 가져오지 못했습니다.")
                        await _report_activity(on_event, warnings[-1], "warning")
                if bibliography_count:
                    matched = sum(source["kind"] == "referenced" for source in sources)
                    await _report_activity(
                        on_event, f"PDF 참고문헌 중 {matched}/{bibliography_count}편 서지 확인", "success",
                    )

            if not related_total:
                terms = [item.get("display_name", "") for item in work.get("keywords") or []
                         if isinstance(item, dict) and item.get("display_name")]
                for offset in (0, 2):
                    query = " ".join(terms[offset:offset + 2]).strip()
                    if not query or sum(source["kind"] == "related" for source in sources) >= 8:
                        break
                    try:
                        response = await client.get(f"{settings.openalex_url}/works", params=_openalex_params({
                            "search": query, "per-page": 12,
                        }))
                        response.raise_for_status()
                        for item in response.json().get("results", []):
                            if sum(source["kind"] == "related" for source in sources) >= 8:
                                break
                            add_work(item, "related")
                    except (httpx.HTTPError, ValueError, KeyError, TypeError):
                        warnings.append("키워드 기반 관련 연구 검색에 실패했습니다.")
                        await _report_activity(on_event, warnings[-1], "warning")
                related = sum(source["kind"] == "related" for source in sources)
                if related:
                    await _report_activity(on_event, f"키워드 검색으로 관련 연구 {related}편 확인", "success")

            await _report_activity(on_event, "업로드 논문을 인용한 연구를 검색하는 중")
            try:
                response = await client.get(
                    f"{settings.openalex_url}/works",
                    params=_openalex_params({
                        "filter": f"cites:{work['id'].rsplit('/', 1)[-1]}", "per-page": 5,
                    }),
                )
                response.raise_for_status()
                citing_before = sum(source["kind"] == "citing" for source in sources)
                for item in response.json().get("results", []):
                    add_work(item, "citing")
                citing_count = sum(source["kind"] == "citing" for source in sources) - citing_before
                await _report_activity(
                    on_event, f"피인용 논문 {citing_count}편의 서지정보를 확인했습니다", "success",
                )
            except (httpx.HTTPError, ValueError, KeyError):
                warnings.append("이 논문을 인용한 연구 일부를 가져오지 못했습니다.")
                await _report_activity(on_event, warnings[-1], "warning")
        elif not warnings:
            warnings.append("OpenAlex에서 업로드 논문을 찾지 못해 참고·인용 관계를 확인하지 못했습니다.")
            await _report_activity(on_event, warnings[-1], "warning")

        if keywords:
            today = date.today().isoformat()
            cutoff = (date.today() - timedelta(days=3 * 365)).isoformat()
            query = " ".join(keywords[:2])[:160]
            query_terms = set(re.findall(r"[a-z0-9]+", query.lower())) - {"and", "of", "the", "for", "in"}
            try:
                response = await client.get(
                    f"{settings.openalex_url}/works", params=_openalex_params({
                    "search": query,
                    "filter": f"from_publication_date:{cutoff},to_publication_date:{today}",
                    "sort": "publication_date:desc", "per-page": 20,
                    }),
                )
                response.raise_for_status()
                for item in response.json().get("results", []):
                    if sum(source["kind"] == "recent" for source in sources) >= 5:
                        break
                    published = item.get("publication_date")
                    if not isinstance(published, str) or not cutoff <= published <= today:
                        continue
                    abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
                    if not abstract:
                        continue
                    candidate_terms = set(re.findall(
                        r"[a-z0-9]+", f"{item.get('display_name', '')} {abstract}".lower()
                    ))
                    if len(query_terms & candidate_terms) < min(2, len(query_terms)):
                        continue
                    add_work(item, "recent")
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                warnings.append("최신 관련 논문을 검색하지 못했습니다.")

        async def fetch_background(keyword: str) -> dict[str, Any] | None:
            try:
                response = await client.get(settings.wikipedia_url, params={
                    "action": "query", "format": "json", "generator": "search",
                    "gsrsearch": f"{keyword} artificial intelligence",
                    "gsrlimit": 1, "prop": "extracts|info", "inprop": "url",
                    "exintro": 1, "explaintext": 1,
                })
                response.raise_for_status()
                pages = response.json().get("query", {}).get("pages", {})
                if not pages:
                    return None
                page = next(iter(pages.values()))
                extract = page.get("extract", "")
                if not extract or not isinstance(extract, str):
                    return None
                url = page.get("fullurl")
                return {
                    "kind": "background", "title": page.get("title", keyword),
                    "text": extract[:4000], "content_scope": "background",
                    "url": url if isinstance(url, str) and url.startswith("https://") else None,
                    "year": None, "authors": [],
                }
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                return None

        backgrounds = await asyncio.gather(*(fetch_background(keyword) for keyword in keywords[:3]))
        for item in backgrounds:
            if item:
                item["id"] = f"external-{len(sources)}"
                sources.append(item)
        if keywords and not any(source["kind"] == "background" for source in sources):
            warnings.append("배경 지식 자료를 가져오지 못했습니다.")

    return sources, warnings
