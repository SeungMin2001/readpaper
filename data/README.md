# Local research data

Keep collection, extraction, and retrieval artifacts inside this directory. Generated data is ignored by Git so paper PDFs and extracted text stay local.

- `papers/input/`: root PDFs used for ingestion tests.
- `papers/collected/`: retrieved open-access PDFs and extracted text.
- `chunks/`: chunk inspection exports.
- `inspection/`: GROBID XML, collection manifests, and retrieval diagnostics.
- `runtime/`: local SQLite database, analyzed papers, extracted text, chunk metadata, and model caches. Qdrant keeps the actual vectors in its separate Docker volume.

## Run citation collection for any PDF

`backend/scripts/collect_cited_fulltext.py` uses GROBID to read the supplied paper's title and printed bibliography. It resolves references with bibliography arXiv IDs first, then OpenAlex, followed by Crossref when OpenAlex cannot find a confident match. If `READPAPER_UNPAYWALL_EMAIL` is configured, it also asks Unpaywall for an open-access PDF location. The collector downloads candidates over HTTPS, checks that the first page matches the cited title, then saves both the PDF and page-labeled text. Unavailable, paywalled, mismatched, and metadata-lookup failures are kept as separate statuses.

Start GROBID once:

```sh
docker run -d --name readpaper-grobid -p 127.0.0.1:8070:8070 \
  -e JAVA_TOOL_OPTIONS=-XX:-UseContainerSupport lfoppiano/grobid:0.8.2
```

If the named container already exists but is stopped, use `docker start readpaper-grobid`. Put any research PDF under `data/papers/input/` and run:

```sh
./.venv/bin/python backend/scripts/collect_cited_fulltext.py \
  --root-pdf data/papers/input/your-paper.pdf \
  --json-out data/inspection/your-paper-citations.json \
  --xml-out data/inspection/your-paper-grobid.xml
```

GROBID supplies the title and identifiers by default. Pass `--title`, `--root-doi`, or `--root-arxiv-id` when its extracted metadata needs correction. The optional `--source-overrides` is a manual exception list keyed to one PDF's bibliography numbering; the Attention Is All You Need example file is specific to that PDF and must not be reused for other papers.

Metadata lookup can be configured in the local `.env` file. `READPAPER_OPENALEX_API_KEY` improves OpenAlex lookup reliability, `READPAPER_UNPAYWALL_EMAIL` enables Unpaywall PDF lookup, and `READPAPER_CROSSREF_EMAIL` supplies Crossref's polite-pool contact. These are optional; no credentials are committed.

The manifest includes each reference, its resolver, selected download URL, title-match score, page count, extracted character count, and failure reason. To print statuses:

```sh
./.venv/bin/python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("data/inspection/your-paper-citations.json").read_text())
for ref in report["references"]:
    print(f'{ref["number"]:02}  {ref["fulltext_status"]:30}  {ref["title"]}')
PY
```

## Related-paper discovery

The web analysis flow checks the uploaded paper's title before accepting an OpenAlex DOI match, then gathers referenced and related candidates. It selects up to eight related candidates in OpenAlex order, downloads verified public PDFs when available, and indexes their full text alongside the uploaded paper. Records without a public PDF are marked as abstract-only evidence. The GROBID script above reads printed citations independently.

As a generic collection check, GROBID processed the unrelated arXiv paper “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding” and extracted its title, arXiv ID, and 55 printed references without using the Attention-specific override list. The generic OpenAlex/Crossref/arXiv path resolved 30 references and saved 9 full texts in this run; the rest were marked as metadata failures, no listed public PDF, or download failure. This environment had exhausted its shared OpenAlex quota and had no Unpaywall email configured, so this run is a parser/fallback check rather than a completeness benchmark. Its manifest and XML are in `inspection/bert-pretraining-2018-citations*`.

The upload flow saves the original PDF temporarily under `runtime/papers/<paper_id>/` and stores translations, abstracts, and chunk metadata in SQLite. The actual dense and BM25 vectors are stored in the Qdrant Docker volume. Paper files and collections expire after 24 hours of inactivity; vocabulary remains. Docker sets `READPAPER_DATA_DIR=/data`.
