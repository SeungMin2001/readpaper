import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper.llm import InvalidModelOutput, OllamaClient, validate_keyword_gloss  # noqa: E402


class KeywordGlossTests(unittest.IsolatedAsyncioTestCase):
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
            ["The agent learns a policy from rewards.", "We evaluate learned policies."],
        )

        self.assertEqual(result, {"start": 0, "end": 2, "meaning": "강화학습"})
        args, kwargs = client.chat_json.await_args
        request = json.loads(args[1])
        self.assertEqual(request["clicked_token"], {"index": 1, "text": "learning"})
        self.assertEqual(len(request["other_passages_from_this_paper"]), 2)
        self.assertEqual(kwargs["options"]["num_predict"], 96)
        self.assertEqual(kwargs["options"]["num_ctx"], 4096)

    def test_gloss_span_must_contain_clicked_token(self):
        with self.assertRaises(InvalidModelOutput):
            validate_keyword_gloss({"start": 0, "end": 1, "meaning": "강화"}, 3, 2)


if __name__ == "__main__":
    unittest.main()
