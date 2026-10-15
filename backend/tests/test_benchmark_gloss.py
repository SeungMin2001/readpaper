import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_gloss import benchmark, summarize  # noqa: E402


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def test_one_slow_result_cannot_hide_behind_a_fast_p95(self):
        rows = [{"ok": True, "elapsed_ms": 200}] * 19 + [{"ok": True, "elapsed_ms": 1200}]
        result = summarize(rows, 1000)
        self.assertEqual(result["p95_ms"], 200)
        self.assertEqual(result["max_ms"], 1200)
        self.assertFalse(result["all_under_target"])
        self.assertFalse(summarize([], 1000)["all_under_target"])
        self.assertFalse(summarize([{"ok": False, "elapsed_ms": 1}], 1000)["all_under_target"])

    async def test_cached_api_results_do_not_prove_new_word_latency(self):
        def respond(request):
            if request.url.path.endswith("/api/auth/me"):
                return httpx.Response(200, json={"username": "test"})
            return httpx.Response(200, json={"meaning": "어텐션", "cached": True})

        real_client = httpx.AsyncClient
        with patch("readpaper.llm.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(
            transport=httpx.MockTransport(respond), **kwargs,
        )):
            report = await benchmark([{
                "tokens": ["attention"], "focus": 0, "summary": "Paper", "paper_id": "p", "word_id": "w",
            }], SimpleNamespace(api_url="http://test", target_ms=1000, repeats=1, compare=False))
        self.assertFalse(report["goal_met"])
        self.assertEqual(report["uncached_api"]["samples"], 0)
        self.assertEqual(report["cached_api"]["samples"], 1)

    async def test_unreachable_model_is_blocked_not_a_passing_empty_run(self):
        def respond(request):
            raise httpx.ConnectError("unreachable", request=request)

        real_client = httpx.AsyncClient
        with patch("readpaper.llm.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(
            transport=httpx.MockTransport(respond), **kwargs,
        )):
            report = await benchmark([], SimpleNamespace(api_url=None, target_ms=1000, repeats=1, compare=False))
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["goal_met"])


if __name__ == "__main__":
    unittest.main()
