from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


def build_chunks(sentences: list[dict[str, Any]], external_sources: list[dict[str, Any]],
                 paper_title: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for offset in range(0, len(sentences), 3):
        group = sentences[offset:offset + 3]
        chunks.append({
            "id": f"paper-{offset // 3}", "kind": "paper", "title": paper_title,
            "text": " ".join(item["text"] for item in group),
            "content_scope": "full_text",
            "url": None, "page": group[0]["page"] + 1,
        })
    chunks.extend(source for source in external_sources if source.get("text"))
    return chunks


def retrieve(query: str, chunks: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    if not chunks or not query.strip():
        return []
    documents = [item["text"] for item in chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=30000)
        matrix = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])
    except ValueError:
        return []
    similarities = (matrix @ query_vector.T).toarray().ravel()
    ranked = sorted(range(len(chunks)), key=lambda index: similarities[index], reverse=True)
    return [chunks[index] for index in ranked[:limit] if similarities[index] > 0.025]
