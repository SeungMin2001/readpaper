"""Choose a small set of likely useful terms before opening a paper."""

import re
from collections import Counter
from typing import Any


MAX_PREPARED_TERMS = 24
MAX_TERMS_PER_PAGE = 6

_STOP_WORDS = set("""
about above after again against all also among and another are because been before
being below between both but can could did does doing during each either else even
every figure figures first for from further had has have having here how however
into its itself just least less like made make many may might more most must neither
other our ours over paper papers page pages part per perhaps present previous results
same several should since some such table tables than that their them there these
they this those through thus toward under until upon using used uses very were what
when where which while with within would your
conference proceedings journal volume references reference available published
copyright university author authors section appendix introduction conclusion
""".split())


def _term_key(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def select_paper_candidates(data: dict[str, Any], limit: int = MAX_PREPARED_TERMS) -> list[str]:
    """Rank technical-looking source words across the paper, with page coverage.

    This is deliberately local and fast; the LLM chooses each phrase boundary and
    Korean meaning using document context during the subsequent translation step.
    """
    pages = data.get("pages", [])
    if not pages or limit <= 0:
        return []
    counts = Counter(
        key for page in pages for word in page.get("words", [])
        if (key := _term_key(word["text"]))
    )
    title_keys = {_term_key(word) for word in data.get("title", "").split()}
    existing = data.get("word_glosses", {})
    candidates: list[tuple[float, int, int, str, str, str]] = []

    for page_index, page in enumerate(pages):
        seen_on_page: set[str] = set()
        for position, word in enumerate(page.get("words", [])):
            key = _term_key(word["text"])
            letters = re.sub(r"[^A-Za-z]", "", word["text"])
            acronym = ((2 <= len(letters) <= 6 and letters.isupper())
                       or (3 <= len(letters) <= 8 and letters[0].isupper()
                           and any(character.isupper() for character in letters[1:])))
            if ((len(key) < 5 and not acronym) or key in _STOP_WORDS or key in seen_on_page
                    or word["id"] in existing or not word.get("sentence_id")):
                continue
            seen_on_page.add(key)
            score = (
                min(counts[key], 8) * 2
                + min(len(key), 13) * 0.7
                + (12 if key in title_keys else 0)
                + (4 if "-" in word["text"] else 0)
                + (6 if acronym else 0)
                + (3 if key.endswith(("tion", "ment", "ing", "ence", "ity")) else 0)
            )
            candidates.append((score, page_index, position, word["id"], key,
                               word["sentence_id"]))

    ranked = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))
    selected: list[tuple[float, int, int, str, str, str]] = []
    page_counts: Counter[int] = Counter()
    term_counts: Counter[str] = Counter()
    selected_sentences: set[str] = set()

    # Cover the whole text before filling remaining slots with the best terms.
    page_order = sorted({item[1] for item in ranked})
    if limit == 1:
        page_order = page_order[:1]
    elif len(page_order) > limit:
        page_order = [page_order[round(index * (len(page_order) - 1) / (limit - 1))]
                      for index in range(limit)]
    for page_index in page_order:
        choice = next((item for item in ranked
                       if item[1] == page_index and term_counts[item[4]] == 0), None)
        if choice:
            selected.append(choice)
            page_counts[page_index] += 1
            term_counts[choice[4]] += 1
            selected_sentences.add(choice[5])

    for item in ranked:
        if len(selected) >= limit:
            break
        if (item in selected or item[5] in selected_sentences
                or page_counts[item[1]] >= MAX_TERMS_PER_PAGE
                or term_counts[item[4]] >= 2):
            continue
        selected.append(item)
        page_counts[item[1]] += 1
        term_counts[item[4]] += 1
        selected_sentences.add(item[5])
    return [item[3] for item in selected]
