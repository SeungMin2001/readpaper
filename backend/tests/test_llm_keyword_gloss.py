import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper.llm import InvalidModelOutput, ModelUnavailable, VLLMClient, validate_keyword_gloss  # noqa: E402
from readpaper.config import settings  # noqa: E402


class KeywordGlossTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_openai_compatible_vllm_chat_endpoint(self):
        response_payload = {
            "choices": [{"message": {"content": '{"term":"attention","meaning":"주의"}'}}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 20},
        }
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=response_payload)

        transport = httpx.MockTransport(respond)
        real_async_client = httpx.AsyncClient
        client = VLLMClient()

        with patch(
            "readpaper.llm.httpx.AsyncClient",
            side_effect=lambda **kwargs: real_async_client(transport=transport, **kwargs),
        ):
            result = await client.chat_json(
                "system", "user",
                schema={"type": "object", "properties": {"meaning": {"type": "string"}}},
            )

        self.assertEqual(result["meaning"], "주의")
        self.assertEqual(requests[0]["model"], settings.llm_model)
        self.assertEqual(requests[0]["response_format"]["type"], "json_schema")
        self.assertEqual(client.last_metrics["prompt_tokens"], 400)
        self.assertEqual(client.last_metrics["completion_tokens"], 20)

    async def stream_gloss(self, parts, *, finish="stop", status=200, requests=None):
        events = [{"choices": [{"delta": {"content": part}}]} for part in parts]
        if finish is not None:
            events.append({"choices": [{"delta": {}, "finish_reason": finish}]})
        events.append({"choices": [], "usage": {
            "prompt_tokens": 75, "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 64},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }})
        wire = "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in events)
        wire += "data: [DONE]\n\n"

        def respond(request):
            if requests is not None:
                requests.append(json.loads(request.content))
            return httpx.Response(status, text=wire, headers={"Content-Type": "text/event-stream"})

        real_client = httpx.AsyncClient
        client = VLLMClient()
        with patch("readpaper.llm.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(
            transport=httpx.MockTransport(respond), **kwargs,
        )):
            result = await client.translate_keyword_from_summary(
                "The paper studies reinforcement learning for robotic control.", "reinforcement learning",
            )
        return result, client.last_metrics

    async def test_translation_generates_only_meaning_with_reusable_prefix(self):
        requests = []
        result, metrics = await self.stream_gloss(["강화", "학습"], requests=requests)
        self.assertEqual(result, {"term": "reinforcement learning", "meaning": "강화학습"})
        request = requests[0]
        self.assertNotIn("response_format", request)
        self.assertTrue(request["stream"])
        self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(request["messages"][1]["content"], "reinforcement learning")
        self.assertIn("robotic control", request["messages"][0]["content"])
        self.assertNotIn("start", request["messages"][0]["content"])
        self.assertEqual(metrics["completion_tokens"], 3)
        self.assertEqual(metrics["cached_prompt_tokens"], 64)
        self.assertEqual(metrics["reasoning_tokens"], 0)
        self.assertGreaterEqual(metrics["request_ms"], metrics["first_token_ms"])

    async def test_rejects_partial_and_invalid_glosses(self):
        for parts, finish in [(["강화"], "length"), (["강화학습"], None),
                              (["English only"], "stop"), ([""], "stop"),
                              (['{"meaning":"강화학습"}'], "stop"),
                              (["강화학습\n이 용어는"], "stop")]:
            with self.subTest(parts=parts, finish=finish):
                with self.assertRaises(InvalidModelOutput):
                    await self.stream_gloss(parts, finish=finish)

    async def test_reports_model_http_error(self):
        with self.assertRaises(ModelUnavailable):
            await self.stream_gloss([], status=503)

    def test_matches_repeated_term_at_clicked_occurrence(self):
        result = validate_keyword_gloss(
            {"term": "attention", "meaning": "주의"},
            ["Attention", "improves", "attention", "models."], 2,
        )
        self.assertEqual(result, {"start": 2, "end": 3, "meaning": "주의"})

    def test_matches_hyphenated_pdf_token(self):
        result = validate_keyword_gloss(
            {"term": "self attention", "meaning": "자기 주의"},
            ["self-attention", "layers."], 0,
        )
        self.assertEqual(result, {"start": 0, "end": 1, "meaning": "자기 주의"})

    def test_unmatched_term_falls_back_to_clicked_word(self):
        result = validate_keyword_gloss(
            {"term": "unrelated term", "meaning": "문맥 임베딩"},
            ["contextual", "embedding", "model"], 1,
        )
        self.assertEqual(result, {"start": 1, "end": 2, "meaning": "문맥 임베딩"})

    def test_rejects_empty_meaning(self):
        with self.assertRaises(InvalidModelOutput):
            validate_keyword_gloss(
                {"term": "attention", "meaning": " "}, ["attention"], 0,
            )


if __name__ == "__main__":
    unittest.main()
