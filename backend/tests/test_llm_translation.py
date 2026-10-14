import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readpaper.llm import InvalidModelOutput, OllamaClient  # noqa: E402


class TranslationOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_token_output_is_retried_and_compounds_are_grouped(self):
        client = OllamaClient()
        client.chat_json = AsyncMock(side_effect=[
            {"s2": {"translation": "번역", "token_glosses": [
                    {"join_previous": False, "meaning": "강화학습"},
                    {"join_previous": True, "meaning": "강화학습"},
                ]}},
            {"s2": {"translation": "번역", "token_glosses": [
                    {"join_previous": False, "meaning": "강화학습"},
                    {"join_previous": True, "meaning": "강화학습"},
                    {"join_previous": False, "meaning": "개선"},
                ]}},
        ])

        result = await client.translate_sentence(
            {"id": "s2", "text": "Learning improves models."},
            ["Learning", "improves", "models."],
            {"summary": "기계학습 논문"}, "", "",
        )

        self.assertEqual(result["units"], [
            {"start": 0, "end": 2, "meaning": "강화학습"},
            {"start": 2, "end": 3, "meaning": "개선"},
        ])
        self.assertEqual(client.chat_json.await_count, 2)
        first_call = client.chat_json.await_args_list[0]
        token_schema = first_call.kwargs["schema"]["properties"]["s2"]["properties"]["token_glosses"]
        self.assertEqual(token_schema["minItems"], 3)
        self.assertEqual(token_schema["maxItems"], 3)
        retry_prompt = client.chat_json.await_args_list[1].args[1]
        self.assertIn("기대=3, 받은=2", retry_prompt)
        self.assertIn("exactly one token_glosses entry per input token", retry_prompt)

    async def test_translation_still_fails_if_retry_omits_token_glosses(self):
        invalid = {"s2": {"translation": "번역", "token_glosses": [
            {"join_previous": False, "meaning": "뜻"},
        ]}}
        client = OllamaClient()
        client.chat_json = AsyncMock(side_effect=[invalid, invalid])

        with self.assertRaisesRegex(InvalidModelOutput, "문장 s2.*기대=3, 받은=1"):
            await client.translate_sentence(
                {"id": "s2", "text": "Learning improves models."},
                ["Learning", "improves", "models."],
                {"summary": "기계학습 논문"}, "", "",
            )

    async def test_join_is_used_only_when_phrase_glosses_match(self):
        client = OllamaClient()
        client.chat_json = AsyncMock(return_value={
            "s0": {
                "translation": "번역",
                "token_glosses": [
                    {"join_previous": False, "meaning": "강화"},
                    {"join_previous": True, "meaning": "학습"},
                ],
            },
        })

        result = await client.translate_sentence(
            {"id": "s0", "text": "Reinforcement learning."},
            ["Reinforcement", "learning."],
            {"summary": "기계학습 논문"}, "", "",
        )

        self.assertEqual(result["units"], [
            {"start": 0, "end": 1, "meaning": "강화"},
            {"start": 1, "end": 2, "meaning": "학습"},
        ])

    async def test_multiple_sentences_share_one_model_call(self):
        client = OllamaClient()
        client.chat_json = AsyncMock(return_value={
            "s0": {"translation": "번역 0", "token_glosses": [
                {"join_previous": False, "meaning": "주의"},
                {"join_previous": False, "meaning": "기제"},
            ]},
            "s1": {"translation": "번역 1", "token_glosses": [
                {"join_previous": False, "meaning": "기계"},
            ]},
        })

        results = await client.translate_sentences([
            {"id": "s0", "sentence": "Attention mechanisms.", "tokens": ["Attention", "mechanisms."],
             "previous_sentence": "", "following_sentence": "They work."},
            {"id": "s1", "sentence": "They work.", "tokens": ["They"],
             "previous_sentence": "Attention mechanisms.", "following_sentence": ""},
        ], {"summary": "기계학습 논문"})

        self.assertEqual(client.chat_json.await_count, 1)
        self.assertEqual(set(results), {"s0", "s1"})
        schema = client.chat_json.await_args.kwargs["schema"]
        self.assertEqual(schema["required"], ["s0", "s1"])


if __name__ == "__main__":
    unittest.main()
