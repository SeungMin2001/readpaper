import os
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from embedding_service.server import MODEL_ID, app


class EmbeddingServiceTests(unittest.TestCase):
    def test_openai_embedding_contract_and_input_validation(self):
        seen = []

        class Model:
            tokenizer = staticmethod(lambda texts, **_kwargs: {"length": [len(text.split()) for text in texts]})

            def encode(self, texts, **_kwargs):
                seen.extend(texts)
                return np.array([[1.0] + [0.0] * 383 for _ in texts])

        app.state.model = Model()
        app.state.batch_size = 32
        with patch.dict(os.environ, {"READPAPER_EMBEDDING_API_KEY": "test-key"}):
            client = TestClient(app)
            body = {"model": MODEL_ID, "input": ["passage: paper text", "query: question"]}
            self.assertEqual(client.post("/v1/embeddings", json=body).status_code, 401)
            headers = {"Authorization": "Bearer test-key"}
            result = client.post("/v1/embeddings", json=body, headers=headers)
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual([item["index"] for item in result.json()["data"]], [0, 1])
            self.assertEqual(len(result.json()["data"][0]["embedding"]), 384)
            self.assertEqual(seen, body["input"])
            self.assertEqual(client.post("/v1/embeddings", json={
                "model": MODEL_ID, "input": ["missing prefix"],
            }, headers=headers).status_code, 422)
