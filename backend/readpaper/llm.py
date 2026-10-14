import json
import re
import time
import unicodedata
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


class VLLMClient:
    def __init__(self) -> None:
        self.last_metrics: dict[str, int] = {}

    async def chat_json(
        self, system: str, user: str, *, schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.last_metrics = {}
        request_started = time.monotonic()
        options = options or {}
        payload: dict[str, Any] = {
            "model": settings.llm_model,
            "stream": False,
            "temperature": options.get("temperature", 0),
            "max_tokens": options.get("num_predict", 512),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "readpaper_response",
                        "strict": True,
                        "schema": schema,
                    },
                }
                if schema else {"type": "json_object"}
            ),
        }
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(240.0, connect=10.0)) as client:
                response = await client.post(
                    f"{settings.llm_base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()
                usage = result.get("usage") or {}
                self.last_metrics = {
                    "request_ms": round((time.monotonic() - request_started) * 1000),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
                content = result["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                if not isinstance(content, str):
                    raise TypeError("vLLM returned a non-text response")
                return _parse_json(content)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelUnavailable(
                "vLLM 서버에 연결하거나 응답을 읽을 수 없습니다. "
                "READPAPER_LLM_BASE_URL과 READPAPER_LLM_MODEL 설정을 확인하세요."
            ) from exc

    async def translate_keyword(
        self, title: str, sentence: str, tokens: list[str], focus_index: int,
        paper_passages: list[str],
    ) -> dict[str, Any]:
        if not tokens or not 0 <= focus_index < len(tokens):
            raise InvalidModelOutput("선택한 단어의 문장 위치가 올바르지 않습니다.")
        schema = {
            "type": "object",
            "properties": {
                "term": {"type": "string"},
                "meaning": {"type": "string"},
            },
            "required": ["term", "meaning"],
            "additionalProperties": False,
        }
        request = {
            "paper_title": title,
            "sentence": sentence,
            "clicked_token": tokens[focus_index],
            "before_clicked_token": tokens[max(0, focus_index - 3):focus_index],
            "after_clicked_token": tokens[focus_index + 1:focus_index + 4],
            "other_passages_from_this_paper": [
                passage for passage in paper_passages if passage != sentence
            ],
        }
        result = await self.chat_json(
            "Return the exact English scientific term containing clicked_token and its short "
            "Korean meaning as JSON with keys term and meaning. Include adjacent modifiers "
            "when they form a compound term. Copy term from the sentence, and use the paper "
            "context to choose its meaning. No explanation.",
            json.dumps(request, ensure_ascii=False),
            schema=schema,
            options={"temperature": 0, "num_predict": 96},
        )
        return validate_keyword_gloss(result, tokens, focus_index)

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


def _word_parts(value: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).casefold())


def validate_keyword_gloss(
    result: dict[str, Any], tokens: list[str], focus_index: int,
) -> dict[str, Any]:
    meaning = result.get("meaning")
    if not isinstance(meaning, str) or not meaning.strip() or len(meaning) > 100:
        raise InvalidModelOutput("용어 뜻이 없거나 너무 깁니다.")
    if not tokens or not 0 <= focus_index < len(tokens):
        raise InvalidModelOutput("선택한 단어의 문장 위치가 올바르지 않습니다.")

    # Match the model's copied phrase to PDF tokens. A phrase must include the
    # clicked token, even when the same words occur elsewhere in the sentence.
    term = result.get("term")
    term_parts = _word_parts(term) if isinstance(term, str) else []
    sentence_parts: list[str] = []
    source_indices: list[int] = []
    for index, token in enumerate(tokens):
        for part in _word_parts(token):
            sentence_parts.append(part)
            source_indices.append(index)

    if term_parts:
        for offset in range(len(sentence_parts) - len(term_parts) + 1):
            after = offset + len(term_parts)
            if sentence_parts[offset:after] != term_parts:
                continue
            start, end = source_indices[offset], source_indices[after - 1] + 1
            if (start <= focus_index < end and end - start <= 6
                    and (offset == 0 or source_indices[offset - 1] != start)
                    and (after == len(source_indices) or source_indices[after] != end - 1)):
                return {"start": start, "end": end, "meaning": meaning.strip()}

    # The model may paraphrase a phrase instead of copying it. The gloss is
    # still for the clicked word, so preserve that translation without
    # highlighting unrelated words or failing the request.
    return {"start": focus_index, "end": focus_index + 1, "meaning": meaning.strip()}


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
