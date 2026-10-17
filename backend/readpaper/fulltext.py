"""Collect publicly accessible PDFs, retaining the original file and page provenance."""
import asyncio
import ipaddress
import json
import re
import socket
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
import pymupdf

from .pdf import extract_pdf


async def public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.port not in (None, 443):
        raise ValueError('공개 HTTPS PDF 주소가 아닙니다')
    try:
        literal_address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal_address = None
    if literal_address is not None and not literal_address.is_global:
        raise ValueError('사설 네트워크 주소는 수집할 수 없습니다')
    addresses = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError('사설 네트워크 주소는 수집할 수 없습니다')


async def download_pdf(client: httpx.AsyncClient, url: str) -> bytes:
    for _ in range(6):
        await public_url(url)
        async with client.stream('GET', url) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers['location'])
                continue
            response.raise_for_status()
            content = bytearray()
            async for part in response.aiter_bytes():
                content.extend(part)
                if len(content) > 25 * 1024 * 1024:
                    raise ValueError('PDF가 25MB를 초과합니다')
            if not content.startswith(b'%PDF-'):
                raise ValueError('PDF 파일이 아닌 응답입니다')
            return bytes(content)
    raise ValueError('PDF 주소의 이동 횟수를 초과했습니다')


def parse_fulltext(content: bytes, title: str) -> dict:
    # Verify identity before introducing downloaded content into the evidence store.
    with pymupdf.open(stream=content, filetype='pdf') as document:
        if not 0 < len(document) <= 100:
            raise ValueError('본문 추출은 100페이지 이하만 지원합니다')
        head = str(document.metadata.get('title') or '') + ' ' + document[0].get_text()
    normalize = lambda value: set(re.findall(r'[a-z0-9]+', value.lower())) - {'the', 'of', 'and', 'a', 'in', 'for', 'to', 'on', 'with'}
    expected = normalize(title)
    if not expected or len(expected & normalize(head)) / len(expected) < .7:
        raise ValueError('다운로드한 PDF의 제목이 일치하지 않습니다')
    return extract_pdf(content)


async def collect_fulltext(sources: list[dict], directory: Path, on_event=None) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(3)
    async with httpx.AsyncClient(timeout=httpx.Timeout(25, connect=8), follow_redirects=False,
                                 headers={'User-Agent': 'Readpaper/0.2 (open-access research reader)'}) as client:
        async def collect(source):
            result = dict(source)
            urls = list(dict.fromkeys(location.get('pdf_url') for location in source.get('pdf_locations', []) if location.get('pdf_url')))
            result['fulltext_status'] = 'unavailable'
            result['fulltext_error'] = '공개 PDF 주소가 없습니다'
            async with semaphore:
                for url in urls[:4]:
                    try:
                        async with asyncio.timeout(90):
                            content = await download_pdf(client, url)
                            parsed = await asyncio.to_thread(parse_fulltext, content, source['title'])
                        stem = re.sub(r'[^a-zA-Z0-9_-]', '_', source['id'])
                        (directory / f'{stem}.pdf').write_bytes(content)
                        (directory / f'{stem}.txt').write_text(parsed['full_text'], encoding='utf-8')
                        result.update(text=parsed['full_text'], sentences=parsed['sentences'],
                                      content_scope='full_text', fulltext_status='ready',
                                      fulltext_error=None, page_count=len(parsed['pages']),
                                      download_url=url, pdf_file=f'{stem}.pdf')
                        break
                    except (httpx.HTTPError, ValueError, OSError, TimeoutError, KeyError) as exc:
                        result['fulltext_status'] = 'failed'
                        result['fulltext_error'] = str(exc)[:180]
            if on_event:
                ok = result['fulltext_status'] == 'ready'
                await on_event(f"{'전문 수집 완료' if ok else '전문 미확보'}: {source['title']}" + ('' if ok else f" · {result['fulltext_error']}"), 'success' if ok else 'warning')
            return result
        results = await asyncio.gather(*(collect(source) for source in sources))
    (directory / 'manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    return results
