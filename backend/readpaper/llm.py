import json
import re
from typing import Any

import httpx

from .config import settings


class ModelUnavailable(RuntimeError):
    pass


class InvalidModelOutput(RuntimeError):
    pass


def _parse_json(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InvalidModelOutput("모델 응답이 JSON 형식이 아닙니다.") from exc
    if not isinstance(value, dict):
        raise InvalidModelOutput("모델 응답은 JSON 객체여야 합니다.")
    return value


class OllamaClient:
    async def chat_json(self, system: str, user: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(240.0, connect=10.0)) as client:
                response = await client.post(
                    f"{settings.ollama_url}/api/chat",
                    json={
                        "model": settings.ollama_model,
                        "stream": False,
                        "format": "json",
                        "think": False,
                        "options": {"temperature": 0, "num_ctx": 8192},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    },
                )
                response.raise_for_status()
                return _parse_json(response.json()["message"]["content"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ModelUnavailable(
                "Mac mini의 Ollama 서버에 연결하거나 응답을 읽을 수 없습니다. "
                "READPAPER_OLLAMA_URL과 모델 이름을 확인하세요."
            ) from exc

    async def profile(self, title: str, full_text: str) -> dict[str, Any]:
        sample = full_text[:12000]
        if len(full_text) > 16000:
            sample += "\n[후반부 발췌]\n" + full_text[-3000:]
        result = await self.chat_json(
            "You analyze English AI research papers for accurate Korean reading support. "
            "Return only JSON with keys summary (Korean, max 800 characters) and "
            "keywords (array of 2 to 5 English technical concepts useful for background research). "
            "Do not invent facts absent from the paper.",
            f"Title: {title}\nPaper excerpt:\n{sample}",
        )
        summary = result.get("summary")
        keywords = result.get("keywords")
        if not isinstance(summary, str) or not isinstance(keywords, list):
            raise InvalidModelOutput("논문 맥락 분석 결과가 불완전합니다.")
        return {
            "summary": summary[:1600],
            "keywords": [str(item)[:80] for item in keywords[:5] if isinstance(item, str)],
        }

    async def translate_sentence(
        self, sentence: dict[str, Any], tokens: list[str], profile: dict[str, Any],
        previous: str, following: str,
    ) -> dict[str, Any]:
        request = {
            "paper_summary": profile["summary"],
            "previous_sentence": previous,
            "sentence_id": sentence["id"],
            "sentence": sentence["text"],
            "following_sentence": following,
            "tokens": [{"index": i, "text": token} for i, token in enumerate(tokens)],
        }
        system = (
            "You translate English AI research papers into Korean. Return ONLY JSON: "
            '{"translation":"natural Korean sentence","units":[{"start":0,"end":1,"meaning":"short Korean gloss"}]}. '
            "Token indices are zero-based; end is exclusive. Units must form an exact, ordered, "
            "non-overlapping partition of every input token. Use one token per unit except multiword "
            "technical terms (e.g. reinforcement learning) which must be one unit. Provide the single "
            "best contextual Korean gloss, not definitions or alternatives. Keep names, numbers and "
            "equation tokens unchanged if translation is inappropriate. Use the paper summary and "
            "neighboring sentences to resolve domain-specific meanings. Never omit tokens."
        )
        last_error: Exception | None = None
        for attempt in range(2):
            user = json.dumps(request, ensure_ascii=False)
            if attempt:
                user += "\nPrevious response failed validation. Return a complete exact partition."
            try:
                result = await self.chat_json(system, user)
                return validate_translation(result, len(tokens))
            except InvalidModelOutput as exc:
                last_error = exc
        raise InvalidModelOutput(
            f"문장 {sentence['id']}의 단어 번역을 검증할 수 없습니다: {last_error}"
        )

    async def search_query(self, korean_question: str) -> str:
        result = await self.chat_json(
            "Turn the user's Korean or English research-paper question into one concise English "
            'search query for retrieving relevant scientific paragraphs. Return JSON {"query":"..."}.',
            korean_question,
        )
        query = result.get("query")
        return query[:300] if isinstance(query, str) and query.strip() else korean_question

    async def select_related_papers(
        self, profile: dict[str, Any], candidates: list[dict[str, Any]], limit: int = 8,
    ) -> list[dict[str, Any]]:
        if not candidates or limit < 1:
            return []
        candidates = candidates[:25]
        candidate_payload = [{
            "id": item["id"],
            "title": item["title"],
            "year": item.get("year"),
            "abstract": item.get("text", "")[:500],
            "candidate_type": item.get("kind"),
        } for item in candidates]
        expected_count = min(limit, len(candidate_payload))
        system = (
            "You rank scientific papers for relevance to an uploaded research paper. "
            f"Select exactly the {expected_count} most relevant items from the supplied candidates. "
            "Use the paper's research question, methods, and findings, not keyword overlap alone. "
            "Never invent or alter candidate IDs. Return only JSON with key selections, an array "
            'of objects shaped as {"id":"candidate-id","reason":"short reason"}, ordered most relevant first.'
        )
        request = {
            "uploaded_paper": profile.get("summary", ""),
            "keywords": profile.get("keywords", []),
            "candidates": candidate_payload,
        }
        result = await self.chat_json(system, json.dumps(request, ensure_ascii=False))
        return validate_related_selection(result, candidates, limit)

    async def answer(self, question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        evidence = [
            {"id": item["id"], "title": item["title"], "text": item["text"][:1600]}
            for item in contexts
        ]
        result = await self.chat_json(
            "Answer the user's question in Korean using ONLY the supplied evidence. "
            "Clearly say when the evidence is insufficient. Every substantive answer must cite "
            'source IDs. Return JSON {"answer":"...","citations":["source-id"]}. '
            "Do not invent citations or quote unavailable full text.",
            json.dumps({"question": question, "evidence": evidence}, ensure_ascii=False),
        )
        valid_ids = {item["id"] for item in contexts}
        answer = result.get("answer")
        citations = result.get("citations")
        if not isinstance(answer, str) or not isinstance(citations, list):
            raise InvalidModelOutput("질문 답변 형식이 올바르지 않습니다.")
        clean_citations = [value for value in citations if isinstance(value, str) and value in valid_ids]
        if not clean_citations:
            return {"answer": "수집한 자료에서 이 질문의 근거를 확인하지 못했습니다.", "citations": []}
        return {"answer": answer, "citations": list(dict.fromkeys(clean_citations))}


def validate_translation(result: dict[str, Any], token_count: int) -> dict[str, Any]:
    translation = result.get("translation")
    units = result.get("units")
    if not isinstance(translation, str) or not translation.strip() or not isinstance(units, list):
        raise InvalidModelOutput("문장 번역 또는 단어 번역이 없습니다.")
    cursor = 0
    clean_units: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, dict):
            raise InvalidModelOutput("단어 번역 항목이 객체가 아닙니다.")
        start, end, meaning = unit.get("start"), unit.get("end"), unit.get("meaning")
        if type(start) is not int or type(end) is not int or start != cursor or end <= start or end > token_count:
            raise InvalidModelOutput("단어 번역 범위가 누락되거나 겹칩니다.")
        if not isinstance(meaning, str) or not meaning.strip() or len(meaning) > 100:
            raise InvalidModelOutput("단어 뜻이 없거나 너무 깁니다.")
        clean_units.append({"start": start, "end": end, "meaning": meaning.strip()})
        cursor = end
    if cursor != token_count:
        raise InvalidModelOutput("마지막 단어까지 번역되지 않았습니다.")
    return {"translation": translation.strip(), "units": clean_units}


def validate_related_selection(
    result: dict[str, Any], candidates: list[dict[str, Any]], limit: int = 8,
) -> list[dict[str, Any]]:
    selections = result.get("selections")
    if not isinstance(selections, list):
        raise InvalidModelOutput("관련 논문 선택 결과에 selections 배열이 없습니다.")
    by_id = {item.get("id"): item for item in candidates if isinstance(item.get("id"), str)}
    expected_count = min(limit, len(by_id))
    if len(selections) != expected_count:
        raise InvalidModelOutput("관련 논문 선택 개수가 후보 수 또는 요청한 개수와 다릅니다.")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, selection in enumerate(selections, start=1):
        if not isinstance(selection, dict):
            raise InvalidModelOutput("관련 논문 선택 항목이 객체가 아닙니다.")
        paper_id = selection.get("id")
        reason = selection.get("reason")
        if not isinstance(paper_id, str) or paper_id not in by_id or paper_id in seen:
            raise InvalidModelOutput("관련 논문 선택에 알 수 없거나 중복된 후보 ID가 있습니다.")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
            raise InvalidModelOutput("관련 논문 선택 사유가 없거나 너무 깁니다.")
        seen.add(paper_id)
        selected.append(by_id[paper_id] | {
            "llm_relevance_rank": rank,
            "llm_relevance_reason": reason.strip(),
        })
    return selected
