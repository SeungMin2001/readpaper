import re
from collections import OrderedDict
from typing import Any

import pymupdf as fitz


END_SENTENCE = re.compile(r"[.!?][\"'’”)]*$")
ABBREVIATIONS = {"e.g.", "i.e.", "et", "al.", "fig.", "eq.", "dr.", "vs.", "cf."}
DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def _ends_sentence(token: str) -> bool:
    lower = token.lower()
    return bool(END_SENTENCE.search(token)) and lower not in ABBREVIATIONS and not re.match(r"^\d+\.\d+$", token)


def _reading_order(blocks: OrderedDict[int, list[dict[str, Any]]], page_width: float) -> list[list[dict[str, Any]]]:
    """Order simple one- and two-column pages without changing word coordinates."""
    entries = []
    midpoint = page_width / 2
    for block_words in blocks.values():
        x0 = min(word["x0"] for word in block_words)
        x1 = max(word["x1"] for word in block_words)
        y0 = min(word["y0"] for word in block_words)
        y1 = max(word["y1"] for word in block_words)
        side = "left" if x1 < midpoint + 5 else "right" if x0 > midpoint - 5 else "wide"
        entries.append((side, y0, y1, x0, block_words))
    left = [entry for entry in entries if entry[0] == "left"]
    right = [entry for entry in entries if entry[0] == "right"]
    if (
        len(left) >= 2 and len(right) >= 2
        and min(max(entry[2] for entry in left), max(entry[2] for entry in right))
        > max(min(entry[1] for entry in left), min(entry[1] for entry in right))
    ):
        column_top = min(entry[1] for entry in left + right)
        top = [entry for entry in entries if entry[0] == "wide" and entry[2] <= column_top + 5]
        rest = [entry for entry in entries if entry[0] == "wide" and entry not in top]
        ordered = (
            sorted(top, key=lambda entry: (entry[1], entry[3]))
            + sorted(left, key=lambda entry: (entry[1], entry[3]))
            + sorted(right, key=lambda entry: (entry[1], entry[3]))
            + sorted(rest, key=lambda entry: (entry[1], entry[3]))
        )
    else:
        ordered = sorted(entries, key=lambda entry: (entry[1], entry[3]))
    return [entry[4] for entry in ordered]


def extract_pdf(data: bytes) -> dict[str, Any]:
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("유효한 PDF 파일을 읽을 수 없습니다.") from exc
    if document.page_count == 0:
        raise ValueError("PDF에 페이지가 없습니다.")
    if document.page_count > 100:
        raise ValueError("첫 버전에서는 100페이지 이하 PDF를 지원합니다.")

    pages: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []
    all_text: list[str] = []
    word_counter = 0

    for page_index, page in enumerate(document):
        page_words: list[dict[str, Any]] = []
        blocks: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        for raw in page.get_text("words", sort=True):
            x0, y0, x1, y1, token, block_no, _line_no, _word_no = raw[:8]
            if not token.strip() or not any(char.isalnum() for char in token):
                continue
            word = {
                "id": f"w{word_counter}", "text": token, "page": page_index,
                "x0": round(x0, 2), "y0": round(y0, 2),
                "x1": round(x1, 2), "y1": round(y1, 2), "sentence_id": "",
            }
            word_counter += 1
            page_words.append(word)
            blocks.setdefault(block_no, []).append(word)

        for block_words in _reading_order(blocks, page.rect.width):
            pending: list[dict[str, Any]] = []

            def flush() -> None:
                if not pending:
                    return
                sentence_id = f"s{len(sentences)}"
                text = " ".join(word["text"] for word in pending)
                for word in pending:
                    word["sentence_id"] = sentence_id
                sentences.append({
                    "id": sentence_id, "page": page_index,
                    "text": text, "word_ids": [word["id"] for word in pending],
                })
                all_text.append(text)
                pending.clear()

            for word in block_words:
                pending.append(word)
                if _ends_sentence(word["text"]):
                    flush()
            flush()

        pages.append({
            "index": page_index,
            "width": round(page.rect.width, 2),
            "height": round(page.rect.height, 2),
            "words": page_words,
        })

    text = "\n".join(all_text)
    if len(re.findall(r"[A-Za-z]", text)) < 100:
        raise ValueError("선택 가능한 영문 텍스트가 부족합니다. 스캔 PDF는 아직 지원하지 않습니다.")
    first_lines = [s["text"] for s in sentences[:3]]
    metadata_title = (document.metadata or {}).get("title") or ""
    title = metadata_title.strip() if len(metadata_title.strip()) > 8 else (first_lines[0] if first_lines else "Untitled paper")
    doi_match = DOI.search(text[:15000])
    return {
        "title": title[:300], "doi": doi_match.group(0).rstrip(".,;)") if doi_match else None,
        "pages": pages, "sentences": sentences, "full_text": text,
    }
