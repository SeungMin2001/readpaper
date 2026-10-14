import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper.llm import InvalidModelOutput, VLLMClient, validate_keyword_gloss  # noqa: E402
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

    async def test_requests_only_short_contextual_keyword_gloss(self):
        client = VLLMClient()
        client.chat_json = AsyncMock(return_value={
            "term": "Reinforcement learning", "meaning": "강화학습",
        })

        result = await client.translate_keyword(
            "A Paper About Control",
            "Reinforcement learning improves robotic control.",
            ["Reinforcement", "learning", "improves", "robotic", "control."],
            1,
            [
                "Reinforcement learning improves robotic control.",
                "The agent learns a policy from rewards.",
                "We evaluate learned policies.",
            ],
        )

        self.assertEqual(result, {"start": 0, "end": 2, "meaning": "강화학습"})
        args, kwargs = client.chat_json.await_args
        request = json.loads(args[1])
        self.assertEqual(request["clicked_token"], "learning")
        self.assertEqual(request["before_clicked_token"], ["Reinforcement"])
        self.assertEqual(request["after_clicked_token"], ["improves", "robotic", "control."])
        self.assertNotIn("tokens", request)
        self.assertEqual(len(request["other_passages_from_this_paper"]), 2)
        self.assertEqual(kwargs["options"]["num_predict"], 96)
        self.assertEqual(set(kwargs["schema"]["properties"]), {"term", "meaning"})

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
