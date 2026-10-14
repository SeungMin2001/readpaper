import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper.llm import InvalidModelOutput, OllamaClient, validate_keyword_gloss  # noqa: E402
from readpaper.config import settings  # noqa: E402


class KeywordGlossTests(unittest.IsolatedAsyncioTestCase):
    async def test_captures_ollama_timing_metadata(self):
        response_payload = {
            "message": {"content": '{"start":0,"end":1,"meaning":"주의"}'},
            "total_duration": 900_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_duration": 300_000_000,
            "eval_duration": 450_000_000,
            "prompt_eval_count": 400,
            "eval_count": 20,
        }
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=response_payload)

        transport = httpx.MockTransport(respond)
        real_async_client = httpx.AsyncClient
        client = OllamaClient()

        with patch(
            "readpaper.llm.httpx.AsyncClient",
            side_effect=lambda **kwargs: real_async_client(transport=transport, **kwargs),
        ):
            result = await client.chat_json("system", "user", model="gemma4:e2b")

        self.assertEqual(result["meaning"], "주의")
        self.assertEqual(requests[0]["model"], "gemma4:e2b")
        self.assertEqual(client.last_metrics["total_duration"], 900_000_000)
        self.assertEqual(client.last_metrics["prompt_eval_count"], 400)
        self.assertEqual(client.last_metrics["eval_count"], 20)

    async def test_requests_only_short_contextual_keyword_gloss(self):
        client = OllamaClient()
        client.chat_json = AsyncMock(return_value={
            "start": 0, "end": 2, "meaning": "강화학습",
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
        self.assertEqual(request["focus_index"], 1)
        self.assertEqual(request["tokens"][1], "learning")
        self.assertEqual(len(request["other_passages_from_this_paper"]), 2)
        self.assertEqual(kwargs["options"]["num_predict"], 96)
        self.assertEqual(kwargs["options"]["num_ctx"], 4096)
        self.assertEqual(kwargs["model"], settings.ollama_gloss_model)

    async def test_gloss_model_can_be_overridden_for_comparison(self):
        client = OllamaClient()
        client.chat_json = AsyncMock(return_value={
            "start": 0, "end": 1, "meaning": "정책",
        })
        await client.translate_keyword(
            "Learning to Act", "Policy improves reward.",
            ["Policy", "improves", "reward."], 0, [], model="gemma4:e2b",
        )
        self.assertEqual(client.chat_json.await_args.kwargs["model"], "gemma4:e2b")

    def test_gloss_span_must_contain_clicked_token(self):
        with self.assertRaises(InvalidModelOutput):
            validate_keyword_gloss({"start": 0, "end": 1, "meaning": "강화"}, 3, 2)


if __name__ == "__main__":
    unittest.main()
