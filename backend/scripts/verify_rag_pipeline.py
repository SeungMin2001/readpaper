"""Inspect live discovery, OA collection and semantic indexing without an answer LLM."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from readpaper.fulltext import collect_fulltext
from readpaper.pdf import extract_pdf
from readpaper.rag import build_chunks, build_index, retrieve
from readpaper.sources import collect_external_sources


async def run(args):
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    root = extract_pdf(args.pdf.read_bytes()) if args.pdf else None
    title = root['title'] if root else args.title
    if not title:
        raise ValueError('--title 또는 --pdf가 필요합니다.')
    warnings = []
    if args.reuse:
        results = json.loads((folder / 'sources' / 'manifest.json').read_text())
    else:
        sources, warnings = await collect_external_sources(
            title, root.get('doi') if root else args.doi, [],
            reference_text=root['full_text'] if root else '',
        )
        selected, related = [], 0
        for source in sources:
            if source['kind'] == 'related':
                related += 1
                if related > 8:
                    continue
            selected.append(source)
        async def event(message, state):
            print(f'{state}: {message}', flush=True)
        results = await collect_fulltext(selected, folder / 'sources', event)
    chunks = build_chunks(root['sentences'] if root else [], results, title)
    metadata = await asyncio.to_thread(build_index, chunks, folder / 'index',
                                      lambda done, total: print(f'embedding: {done}/{total}', flush=True))
    hits = await asyncio.to_thread(retrieve, args.query, chunks, 6, folder / 'index')
    report = {'title': title, 'source_count': len(results),
              'kinds': dict(Counter(source['kind'] for source in results)),
              'fulltext': dict(Counter(source['fulltext_status'] for source in results)),
              'index': metadata, 'warnings': warnings, 'query': args.query, 'hits': hits}
    (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'hits'}, ensure_ascii=False, indent=2))
    print(f"Report: {folder / 'report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--title')
    parser.add_argument('--doi')
    parser.add_argument('--pdf', type=Path)
    parser.add_argument('--out', type=Path, default=Path('data/inspection/rag-check'))
    parser.add_argument('--reuse', action='store_true', help='Reuse the saved collection manifest; rebuild the index')
    parser.add_argument('--query', default='어텐션과 순환 신경망의 차이는 무엇인가요?')
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
