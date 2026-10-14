# Local research data

Keep files used for collection and chunking inside this directory. Data artifacts are ignored by Git; this README is tracked so the layout is documented.

- `papers/input/`: PDFs supplied for local ingestion experiments.
- `papers/collected/`: title-checked open-access PDFs and their extracted text.
- `chunks/`: extracted text and chunk inspection exports.
- `inspection/`: collection and retrieval diagnostic reports, such as JSON source lists.
- `runtime/`: local app SQLite database, including analyzed paper data and chunks.

The source inspector saves its report under `inspection/`. `backend/scripts/fetch_open_access_fulltext.py` tests open-access PDF retrieval and writes a manifest under `inspection/`, with downloaded PDFs and extracted `.txt` files under `papers/collected/`. It checks the PDF's first-page text against the expected title before saving. This diagnostic step does not chunk or embed the text yet.

The web upload flow currently processes the uploaded PDF in memory and does not preserve that original PDF. It stores extracted paper data, translations, external abstracts, and chunks in the local SQLite database under `runtime/`. Docker can override the runtime location with `READPAPER_DATA_DIR=/data`.
