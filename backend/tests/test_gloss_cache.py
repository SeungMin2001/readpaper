import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from readpaper import db, main
from readpaper.config import Settings
from readpaper.llm import ModelUnavailable
from readpaper.terms import gloss_cache_key, resolve_term


class TermResolutionTests(unittest.TestCase):
    def test_longest_compound_contains_clicked_word(self):
        tokens = "We use scaled dot-product attention and attention weights.".split()
        for index in (2, 3, 4):
            self.assertEqual(resolve_term(tokens, index), {
                "start": 2, "end": 5, "term": "scaled dot-product attention",
            })
        self.assertEqual(resolve_term(tokens, 6)["term"], "attention weights")

    def test_unknown_terms_and_phrase_boundaries(self):
        self.assertEqual(resolve_term(["novel", "widget"], 1)["term"], "widget")
        self.assertEqual(resolve_term(["residual,", "connections"], 0)["term"], "residual")
        self.assertEqual(resolve_term(["C++", "and", "C#"], 0)["term"], "C++")
        self.assertEqual(resolve_term(["C++", "and", "C#"], 2)["term"], "C#")

    def test_cache_key_does_not_conflate_contexts_models_or_case(self):
        def key(summary="paper", term="US", model="gemma", endpoint="local"):
            return gloss_cache_key(summary, term, model=model, endpoint=endpoint)
        values = [key(), key(summary="other paper"), key(term="us"),
                  key(model="other"), key(endpoint="other")]
        self.assertEqual(len(set(values)), len(values))


class CountingModel:
    def __init__(self):
        self.calls = []
        self.gate = None
        self.started = asyncio.Event()
        self.two_started = asyncio.Event()
        self.fail = False

    async def translate_keyword_from_summary(self, summary, term):
        self.calls.append((summary, term))
        self.started.set()
        if len(self.calls) >= 2:
            self.two_started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise ModelUnavailable("test outage")
        return {"term": term, "meaning": "문맥상 뜻"}


class GlossCacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.config = Settings(data_dir=Path(self.directory.name), dev_mode=True)
        self.model = CountingModel()
        for target, value in (("readpaper.db.settings", self.config),
                              ("readpaper.main.settings", self.config)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("readpaper.main.VLLMClient", return_value=self.model)
        patcher.start()
        self.addCleanup(patcher.stop)
        db.init_db()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")
        self.seed("paper")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.directory.cleanup()
        self.assertEqual(main._gloss_locks, {})
        self.assertEqual(main._gloss_lock_users, {})

    def seed(self, paper_id):
        sentences, pages = [], []
        for index, text in enumerate(("We use multi-head attention.", "The multi-head attention works.",
                                      "The policy improves.", "A policy gradient improves.")):
            words = [{"id": f"s{index}w{n}", "text": token, "sentence_id": f"s{index}"}
                     for n, token in enumerate(text.split())]
            pages.append({"words": words})
            sentences.append({"id": f"s{index}", "word_ids": [w["id"] for w in words], "text": text})
        db.create_paper(paper_id, "A paper", {
            "pages": pages, "sentences": sentences,
            "paper_summary": "The paper studies multi-head attention and reinforcement learning.",
        })

    async def click(self, word, paper="paper"):
        return await self.client.post(f"/api/papers/{paper}/translate-word", json={"word_id": word})

    async def test_cross_page_cache_rebinds_ids_and_persists(self):
        first = await self.click("s0w3")
        self.assertEqual(first.status_code, 200, first.text)
        self.assertFalse(first.json()["cached"])
        self.assertEqual(first.json()["word_ids"], ["s0w2", "s0w3"])
        db.init_db()  # An existing cache also survives schema initialization on restart.
        second = await self.click("s1w2")
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["word_ids"], ["s1w1", "s1w2"])
        self.assertEqual(second.json()["sentence_id"], "s1")
        self.assertEqual(second.json()["timing"]["cache"], "term")
        self.assertEqual(second.json()["timing"]["model_request_ms"], 0)
        third = await self.click("s1w1")
        self.assertEqual(third.json()["timing"]["cache"], "position")
        self.assertGreater(third.json()["elapsed_ms"], 0)
        self.assertEqual(len(self.model.calls), 1)

    async def test_same_term_concurrent_locations_generate_once(self):
        self.model.gate = asyncio.Event()
        first = asyncio.create_task(self.click("s0w3"))
        await asyncio.wait_for(self.model.started.wait(), 1)
        second = asyncio.create_task(self.click("s1w2"))
        await asyncio.sleep(0)
        self.model.gate.set()
        results = await asyncio.gather(first, second)
        self.assertTrue(all(r.status_code == 200 for r in results))
        self.assertEqual(len(self.model.calls), 1)
        self.assertNotEqual(results[0].json()["word_ids"], results[1].json()["word_ids"])

    async def test_different_terms_do_not_serialize(self):
        self.model.gate = asyncio.Event()
        requests = [asyncio.create_task(self.click(word)) for word in ("s0w3", "s2w1")]
        try:
            await asyncio.wait_for(self.model.two_started.wait(), 1)
        finally:
            self.model.gate.set()
        results = await asyncio.gather(*requests)
        self.assertTrue(all(r.status_code == 200 for r in results))
        self.assertEqual(len(self.model.calls), 2)

    async def test_context_model_and_paper_changes_do_not_reuse_cache(self):
        await self.click("s0w3")
        db.merge_paper_data("paper", {"paper_summary": "A different scientific context."})
        self.assertFalse((await self.click("s0w3")).json()["cached"])
        with patch("readpaper.main.settings", replace(self.config, llm_model="different-model")):
            self.assertFalse((await self.click("s0w3")).json()["cached"])
        self.seed("other")
        self.assertFalse((await self.click("s0w3", paper="other")).json()["cached"])
        self.assertEqual(len(self.model.calls), 4)

    async def test_single_word_and_compound_do_not_share_meaning(self):
        await self.click("s2w1")
        await self.click("s3w1")
        self.assertEqual([term for _, term in self.model.calls], ["policy", "policy gradient"])

    async def test_failed_request_can_retry_and_does_not_poison_cache(self):
        self.model.fail = True
        self.assertEqual((await self.click("s0w3")).status_code, 503)
        self.assertFalse(db.get_paper("paper")["data"].get("word_glosses"))
        self.model.fail = False
        result = await self.click("s0w3")
        self.assertEqual(result.status_code, 200, result.text)
        self.assertFalse(result.json()["cached"])
        self.assertEqual(len(self.model.calls), 2)

    async def test_expired_paper_deletes_shared_glosses(self):
        await self.click("s0w3")
        with db.connect() as connection:
            connection.execute("UPDATE papers SET last_accessed = 0")
        db.cleanup_expired()
        with db.connect() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM term_glosses").fetchone()[0], 0)
        self.assertEqual((await self.click("s0w3")).status_code, 404)


if __name__ == "__main__":
    unittest.main()
