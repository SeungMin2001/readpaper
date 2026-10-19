import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from readpaper.config import Settings
from readpaper.rag import RemoteE5Embeddings
from readpaper import embedding_cache


class EmbeddingCacheTests(unittest.TestCase):
    def test_reuse_order_duplicates_expiry_and_model_namespace(self):
        requests = []
        def respond(request):
            texts = json.loads(request.content)['input']
            requests.append(texts)
            return httpx.Response(200, json={'data': [
                {'index': i, 'embedding': [float(len(text))] + [0.0] * 383}
                for i, text in enumerate(texts)
            ]})
        real_client = httpx.Client
        with tempfile.TemporaryDirectory() as temporary:
            config = Settings(data_dir=Path(temporary), embedding_api_key='test',
                              embedding_base_url='http://mini/v1', paper_ttl_seconds=10)
            with patch('readpaper.rag.settings', config), patch('readpaper.embedding_cache.settings', config), \
                 patch('readpaper.rag.httpx.Client', side_effect=lambda **kwargs:
                       real_client(transport=httpx.MockTransport(respond), **kwargs)), \
                 patch('readpaper.embedding_cache.time.time', return_value=100):
                with RemoteE5Embeddings() as adapter:
                    first = adapter.embed_documents(['alpha', 'long beta', 'alpha'])
                    self.assertEqual(adapter.cache_hits, 1)
                    self.assertEqual(adapter.generated_texts, 2)
                with RemoteE5Embeddings() as adapter:
                    second = adapter.embed_documents(['long beta', 'alpha'])
                    self.assertEqual(second, [first[1], first[0]])
                    self.assertEqual(adapter.cache_hits, 2)
                    self.assertEqual(adapter.elapsed_ms, 0)
                    adapter.embed_query('alpha')
                self.assertEqual(requests, [['passage: alpha', 'passage: long beta'], ['query: alpha']])
                original_key = embedding_cache.cache_key('passage: alpha')
                with patch('readpaper.embedding_cache.settings', Settings(embedding_model='other')):
                    self.assertNotEqual(embedding_cache.cache_key('passage: alpha'), original_key)
                with patch('readpaper.embedding_cache.time.time', return_value=111):
                    embedding_cache.cleanup()
                    with RemoteE5Embeddings() as adapter:
                        adapter.embed_documents(['alpha'])
                        self.assertEqual(adapter.generated_texts, 1)
                self.assertEqual(len(requests), 3)
                with RemoteE5Embeddings(cache_documents=False) as adapter:
                    adapter.embed_documents(['alpha'])
                    adapter.embed_documents(['alpha'])
                    self.assertEqual(adapter.generated_texts, 2)
                    self.assertEqual(adapter.cache_hits, 0)
                self.assertEqual(len(requests), 5)
