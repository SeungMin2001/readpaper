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
    def __init__(self) -> None:
        self.last_metrics: dict[str, int] = {}

    async def chat_json(
        self, system: str, user: str, *, schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None, keep_alive: str = "30m",
    ) -> dict[str, Any]:
        self.last_metrics = {}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(240.0, connect=10.0)) as client:
                response = await client.post(
                    f"{settings.ollama_url}/api/chat",
                    json={
                        "model": settings.ollama_model,
                        "stream": False,
                        "format": schema or "json",
                        "think": False,
                        "keep_alive": keep_alive,
                        "options": options or {"temperature": 0, "num_ctx": 8192},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    },
                )
                response.raise_for_status()
                payload = response.json()
                self.last_metrics = {
                    key: payload[key]
                    for key in (
                        "total_duration", "load_duration", "prompt_eval_duration",
                        "eval_duration", "prompt_eval_count", "eval_count",
                    )
                    if isinstance(payload.get(key), int)
                }
                return _parse_json(payload["message"]["content"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ModelUnavailable(
                "Mac mini의 Ollama 서버에 연결하거나 응답을 읽을 수 없습니다. "
                "READPAPER_OLLAMA_URL과 모델 이름을 확인하세요."
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
                "start": {"type": "integer", "minimum": 0},
                "end": {"type": "integer", "minimum": 1},
                "meaning": {"type": "string"},
            },
            "required": ["start", "end", "meaning"],
            "additionalProperties": False,
        }
        request = {
            "paper_title": title,
            "clicked_token": {"index": focus_index, "text": tokens[focus_index]},
            "sentence_tokens": [
                {"index": index, "text": token} for index, token in enumerate(tokens)
            ],
            "sentence": sentence,
            "other_passages_from_this_paper": paper_passages,
        }
        result = await self.chat_json(
            "You are a concise dictionary for reading scientific papers. Translate ONLY the clicked "
            "English keyword or technical term into a short Korean gloss; do not translate the sentence. "
            "Use the paper title, the sentence, and retrieved passages from anywhere in the same paper "
            "to resolve the author's intended sense. If the clicked token belongs to a standard "
            "multiword technical term, return the smallest complete term span containing it. Otherwise "
            "return only the clicked token. Do not include surrounding grammar words. Return token start "
            "(inclusive), end (exclusive), and a natural short Korean meaning. Never explain your reasoning.",
            json.dumps(request, ensure_ascii=False),
            schema=schema,
            options={"temperature": 0, "num_ctx": 4096, "num_predict": 96},
        )
        return validate_keyword_gloss(result, len(tokens), focus_index)

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


def validate_keyword_gloss(
    result: dict[str, Any], token_count: int, focus_index: int,
) -> dict[str, Any]:
    start, end, meaning = result.get("start"), result.get("end"), result.get("meaning")
    if (type(start) is not int or type(end) is not int or start < 0 or end <= start
            or end > token_count or not start <= focus_index < end):
        raise InvalidModelOutput("용어 범위가 선택한 단어를 포함하지 않거나 문장 범위를 벗어났습니다.")
    if not isinstance(meaning, str) or not meaning.strip() or len(meaning) > 100:
        raise InvalidModelOutput("용어 뜻이 없거나 너무 깁니다.")
    return {"start": start, "end": end, "meaning": meaning.strip()}


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
