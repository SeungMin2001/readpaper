import os
import asyncio
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

os.environ["READPAPER_DEV_MODE"] = "1"

import pymupdf as fitz  # noqa: E402
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from readpaper import db  # noqa: E402
from readpaper.config import Settings  # noqa: E402
from readpaper.llm import InvalidModelOutput, validate_translation  # noqa: E402
from readpaper.main import app  # noqa: E402
from readpaper.pdf import extract_pdf  # noqa: E402
from readpaper.rag import build_chunks, retrieve  # noqa: E402
from readpaper.sources import collect_external_sources, reconstruct_abstract  # noqa: E402


def example_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    left = [
        "Reinforcement learning improves robotic control.",
        "Our method uses policy optimization for decisions.",
        "Experiments demonstrate useful gains for agents.",
    ]
    right = [
        "Transformers encode contextual language features.",
        "Attention mechanisms support scientific analysis.",
        "These results motivate further research.",
    ]
    for index, line in enumerate(left):
        page.insert_text((40, 80 + 20 * index), line)
    for index, line in enumerate(right):
        page.insert_text((315, 80 + 20 * index), line)
    return document.tobytes()


class FakeOllama:
    async def profile(self, title, full_text):
        return {"summary": "로봇 제어와 언어 분석 연구", "keywords": ["reinforcement learning"]}

    async def translate_sentence(self, sentence, tokens, profile, previous, following):
        units = []
        index = 0
        while index < len(tokens):
            end = 2 if index == 0 and tokens[:2] == ["Reinforcement", "learning"] else index + 1
            units.append({"start": index, "end": end, "meaning": "강화학습" if end == 2 and index == 0 else "뜻"})
            index = end
        return validate_translation({"translation": "문맥에 맞는 문장 번역", "units": units}, len(tokens))

    async def search_query(self, question):
        return "robotic control"

    async def answer(self, question, contexts):
        return {"answer": "근거에 따르면 로봇 제어를 개선합니다.", "citations": [contexts[0]["id"]]}


async def fake_sources(title, doi, keywords):
    return [{
        "id": "external-0", "kind": "related", "title": "A related study",
        "text": "Robotic control with reinforcement learning.",
        "url": "https://example.org/paper", "year": 2024, "authors": ["A. Researcher"],
    }], []


class ReadpaperFlowTests(unittest.TestCase):
    def test_single_user_login_required_outside_development(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Settings(
                data_dir=Path(directory), dev_mode=False,
                username="reader", password="correct-password",
                session_secret="a-test-secret-long-enough-for-signing",
            )
            with patch("readpaper.db.settings", config), patch("readpaper.main.settings", config):
                with TestClient(app) as client:
                    self.assertEqual(client.get("/api/auth/me").status_code, 401)
                    self.assertEqual(client.post("/api/auth/login", json={
                        "username": "reader", "password": "wrong",
                    }).status_code, 401)
                    self.assertEqual(client.post("/api/auth/login", json={
                        "username": "reader", "password": "correct-password",
                    }).status_code, 200)
                    self.assertEqual(client.get("/api/auth/me").json()["username"], "reader")
                    client.cookies.set("readpaper_session", "tampered")
                    self.assertEqual(client.get("/api/auth/me").status_code, 401)

    def test_two_column_pdf_order_and_word_mapping(self):
        parsed = extract_pdf(example_pdf())
        texts = [sentence["text"] for sentence in parsed["sentences"]]
        self.assertEqual(texts[0], "Reinforcement learning improves robotic control.")
        self.assertEqual(texts[2], "Experiments demonstrate useful gains for agents.")
        self.assertEqual(texts[3], "Transformers encode contextual language features.")
        self.assertTrue(all(word["sentence_id"] for word in parsed["pages"][0]["words"]))

    def test_upload_translation_rag_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Settings(data_dir=Path(directory), dev_mode=True)
            with patch("readpaper.db.settings", config), patch("readpaper.main.settings", config), \
                    patch("readpaper.main.OllamaClient", FakeOllama), \
                    patch("readpaper.main.collect_external_sources", fake_sources):
                with TestClient(app) as client:
                    pdf_bytes = example_pdf()
                    upload = client.post(
                        "/api/papers/upload",
                        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
                    )
                    self.assertEqual(upload.status_code, 200, upload.text)
                    paper_id = upload.json()["paper_id"]
                    paper = client.get(f"/api/papers/{paper_id}").json()
                    self.assertEqual(paper["status"], "ready")
                    self.assertEqual(len(paper["translations"]), 6)
                    first_unit = paper["translations"]["s0"]["units"][0]
                    self.assertEqual(first_unit["original"], "Reinforcement learning")
                    self.assertEqual(first_unit["meaning"], "강화학습")
                    self.assertEqual(len(first_unit["word_ids"]), 2)

                    answer = client.post(
                        f"/api/papers/{paper_id}/ask", json={"question": "로봇 제어는 어떻게 개선되나요?"}
                    )
                    self.assertEqual(answer.status_code, 200, answer.text)
                    self.assertTrue(answer.json()["citations"])
                    self.assertTrue(answer.json()["sources"])

                    item = {
                        "paper_title": paper["title"], "term": "Reinforcement learning",
                        "meaning": "강화학습", "source_sentence": paper["sentences"][0]["text"],
                    }
                    self.assertEqual(client.post("/api/vocabulary", json=item).status_code, 201)
                    self.assertEqual(len(client.get("/api/vocabulary").json()), 1)

                    cached = client.post(
                        "/api/papers/upload",
                        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
                    )
                    self.assertTrue(cached.json()["cached"])
                    self.assertEqual(cached.json()["paper_id"], paper_id)

                    with db.connect() as conn:
                        conn.execute("UPDATE papers SET last_accessed = 0")
                    db.cleanup_expired()
                    self.assertEqual(client.get(f"/api/papers/{paper_id}").status_code, 404)
                    self.assertEqual(len(client.get("/api/vocabulary").json()), 1)

    def test_invalid_gloss_alignment_rejected(self):
        with self.assertRaises(InvalidModelOutput):
            validate_translation({
                "translation": "번역", "units": [
                    {"start": 0, "end": 1, "meaning": "가"},
                    {"start": 2, "end": 3, "meaning": "다"},
                ],
            }, 3)

    def test_retrieval_and_abstract_reconstruction(self):
        self.assertEqual(
            reconstruct_abstract({"learning": [1], "Machine": [0], "works.": [2]}),
            "Machine learning works.",
        )
        chunks = [
            {"id": "a", "text": "reinforcement learning controls robotic arms"},
            {"id": "b", "text": "transformer language modeling uses attention"},
        ]
        self.assertEqual(retrieve("robotic control", chunks)[0]["id"], "a")

    def test_external_sources_are_labeled_and_title_matched(self):
        def work(work_id, title):
            return {
                "id": f"https://openalex.org/{work_id}", "display_name": title,
                "abstract_inverted_index": {"Useful": [0], "evidence.": [1]},
                "authorships": [], "doi": None,
            }

        def respond(request):
            path = request.url.path
            if path.endswith("/works") and "from_publication_date" in request.url.params.get("filter", ""):
                return httpx.Response(200, json={"results": [
                    {**work("W1", "Contextual Language Research"), "publication_date": date.today().isoformat()},
                    {**work("W5", "Modern language model methods"), "publication_date": date.today().isoformat()},
                    {**work("W6", "Old language model methods"), "publication_date": "2020-01-01"},
                    {**work("W7", "Unrelated vision study"), "publication_date": date.today().isoformat()},
                    {**work("W8", "Future transformer attention method"), "publication_date": "2099-01-01"},
                ]})
            if path.endswith("/works") and "search" in request.url.params:
                return httpx.Response(200, json={"results": [{
                    **work("W1", "Contextual Language Research"),
                    "referenced_works": ["https://openalex.org/W2"],
                    "related_works": ["https://openalex.org/W3"],
                }]})
            if path.endswith("/works"):
                return httpx.Response(200, json={"results": [work("W4", "A citing paper")]})
            if path.endswith("/works/W2"):
                return httpx.Response(200, json=work("W2", "A referenced paper"))
            if path.endswith("/works/W3"):
                return httpx.Response(200, json=work("W3", "A related paper"))
            return httpx.Response(200, json={"query": {"pages": {"1": {
                "title": "Language model", "extract": "Background knowledge.",
                "fullurl": "https://en.wikipedia.org/wiki/Language_model",
            }}}})

        client_class = httpx.AsyncClient
        transport = httpx.MockTransport(respond)
        with patch("readpaper.sources.httpx.AsyncClient", side_effect=lambda **kwargs: client_class(
            transport=transport, **kwargs,
        )):
            sources, warnings = asyncio.run(collect_external_sources(
                "Contextual Language Research", None, ["language model"],
            ))
            unrelated, unrelated_warnings = asyncio.run(collect_external_sources(
                "A completely different subject", None, [],
            ))
        self.assertEqual([item["kind"] for item in sources], [
            "referenced", "related", "citing", "recent", "background",
        ])
        self.assertEqual(len({item["id"] for item in sources}), 5)
        self.assertEqual(sources[3]["title"], "Modern language model methods")
        self.assertEqual(sources[3]["content_scope"], "abstract")
        self.assertEqual(sources[3]["openalex_id"], "https://openalex.org/W5")
        self.assertIn(sources[3], build_chunks([], sources, "Uploaded paper"))
        self.assertFalse(warnings)
        self.assertFalse(any(item["kind"] != "background" for item in unrelated))
        self.assertTrue(unrelated_warnings)


if __name__ == "__main__":
    unittest.main()
