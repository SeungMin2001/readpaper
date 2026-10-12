import asyncio
import re
from difflib import SequenceMatcher
from typing import Any

import httpx

from .config import settings


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


def _paper_source(work: dict[str, Any], kind: str, index: int) -> dict[str, Any] | None:
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    if not work.get("display_name"):
        return None
    url = work.get("doi") or (work.get("primary_location") or {}).get("landing_page_url") or work.get("id")
    if not isinstance(url, str) or not url.startswith("https://"):
        url = None
    authors = [
        entry.get("author", {}).get("display_name", "")
        for entry in work.get("authorships", [])[:3] if isinstance(entry, dict)
    ]
    return {
        "id": f"external-{index}", "kind": kind,
        "title": work["display_name"],
        "text": abstract, "url": url,
        "year": work.get("publication_year"),
        "authors": [name for name in authors if name],
    }


async def collect_external_sources(title: str, doi: str | None, keywords: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    sources: list[dict[str, Any]] = []
    warnings: list[str] = []
    timeout = httpx.Timeout(8.0, connect=5.0)
    headers = {"User-Agent": "Readpaper/0.1 (research paper reading assistant)"}
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        work: dict[str, Any] | None = None
        try:
            params = {"per-page": 1}
            if doi:
                params["filter"] = f"doi:https://doi.org/{doi}"
            else:
                params["search"] = title[:200]
            response = await client.get(f"{settings.openalex_url}/works", params=params)
            response.raise_for_status()
            results = response.json().get("results", [])
            if results and (doi or _similar_title(title, results[0].get("display_name", ""))):
                work = results[0]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            warnings.append("OpenAlex에 연결할 수 없어 관련·인용 논문을 수집하지 못했습니다.")

        if work:
            references = [("referenced", work_id) for work_id in work.get("referenced_works", [])[:4]]
            related = [("related", work_id) for work_id in work.get("related_works", [])[:4]]

            async def fetch_work(kind: str, work_id: str) -> tuple[str, dict[str, Any] | None]:
                work_key = work_id.rsplit("/", 1)[-1]
                if not re.fullmatch(r"W\d+", work_key):
                    return kind, None
                try:
                    response = await client.get(f"{settings.openalex_url}/works/{work_key}")
                    response.raise_for_status()
                    return kind, response.json()
                except (httpx.HTTPError, ValueError):
                    return kind, None

            results = await asyncio.gather(*(fetch_work(kind, work_id) for kind, work_id in references + related))
            for kind, item in results:
                if item:
                    source = _paper_source(item, kind, len(sources))
                    if source:
                        sources.append(source)

            try:
                response = await client.get(
                    f"{settings.openalex_url}/works",
                    params={"filter": f"cites:{work['id'].rsplit('/', 1)[-1]}", "per-page": 4},
                )
                response.raise_for_status()
                for item in response.json().get("results", []):
                    source = _paper_source(item, "citing", len(sources))
                    if source:
                        sources.append(source)
            except (httpx.HTTPError, ValueError, KeyError):
                warnings.append("이 논문을 인용한 연구 일부를 가져오지 못했습니다.")
        elif not warnings:
            warnings.append("OpenAlex에서 이 논문을 찾지 못해 외부 논문을 수집하지 못했습니다.")

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
                    "text": extract[:4000], "url": url if isinstance(url, str) and url.startswith("https://") else None,
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
