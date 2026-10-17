import asyncio
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pymupdf
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from readpaper import db
from readpaper.config import Settings
from readpaper.fulltext import collect_fulltext, download_pdf, parse_fulltext, public_url
from readpaper.main import app
from readpaper.llm import VLLMClient, InvalidModelOutput
from readpaper.rag import RemoteE5Embeddings, build_chunks, build_index, retrieve
from test_flow import example_pdf, FakeVLLM, fake_sources


from rag_fixtures import FakeRemoteE5Embeddings, FakeSparseEmbeddings, fake_token_count


class RagWorkspaceTests(unittest.TestCase):
    def setUp(self):
        qdrant_directory = tempfile.TemporaryDirectory()
        self.addCleanup(qdrant_directory.cleanup)
        for target, replacement in (
            ('readpaper.rag._token_count', fake_token_count),
            ('readpaper.rag.RemoteE5Embeddings', FakeRemoteE5Embeddings),
            ('readpaper.rag._sparse_embeddings', lambda: FakeSparseEmbeddings()),
            ('readpaper.rag._qdrant_client',
             lambda: QdrantClient(path=str(Path(qdrant_directory.name) / 'qdrant'))),
        ):
            substitute = patch(target, replacement)
            substitute.start()
            self.addCleanup(substitute.stop)

    def test_remote_embedding_adapter_adds_e5_prefixes_and_batches(self):
        requests = []
        def respond(request):
            body = json.loads(request.content)
            requests.append(body['input'])
            return httpx.Response(200, json={'data': [
                {'index': index, 'embedding': [1.0] + [0.0] * 383}
                for index in range(len(body['input']))
            ]})
        fake_client = httpx.Client(transport=httpx.MockTransport(respond),
                                   base_url='http://mini:8002/v1/')
        config = Settings(embedding_api_key='test-key',
                          embedding_base_url='http://mini:8002/v1')
        with patch('readpaper.rag.settings', config), \
             patch('readpaper.rag.httpx.Client', return_value=fake_client):
            with RemoteE5Embeddings() as adapter:
                self.assertEqual(len(adapter.embed_documents(['alpha', 'beta'])), 2)
                self.assertEqual(len(adapter.embed_query('gamma')), 384)
        self.assertEqual(requests, [['passage: alpha', 'passage: beta'],
                                    ['query: gamma']])

    def test_answer_sends_real_image_input_and_rejects_unknown_citations(self):
        requests = []
        citation = ['selection-0']
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps({
                'answer': '그림 설명 [selection-0]', 'citations': citation,
            })}}]})
        contexts = [{'id': 'selection-0', 'title': 'Paper', 'text': 'Selected figure',
                     'image_data': 'data:image/png;base64,aGVsbG8='}]
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                with patch('readpaper.llm._transport', client):
                    result = await VLLMClient().answer('그림을 설명해줘', contexts)
                    self.assertEqual(result['citations'], ['selection-0'])
                    citation[:] = ['invented-source']
                    with self.assertRaises(InvalidModelOutput):
                        await VLLMClient().answer('질문', contexts)
        asyncio.run(run())
        self.assertEqual(requests[0]['messages'][1]['content'][1]['type'], 'image_url')
        self.assertEqual(requests[0]['messages'][1]['content'][1]['image_url']['url'], contexts[0]['image_data'])

    def test_chunks_keep_every_sentence_and_page_with_bounded_sizes(self):
        sentences = [{'id': f's{i}', 'page': i//4, 'text': f'Sentence {i}. ' + 'attention ' * 50} for i in range(12)]
        sources = [{'id':'external-1','title':'External','kind':'referenced','content_scope':'full_text','sentences':sentences}]
        chunks = build_chunks(sentences, sources, 'Root')
        for document in ('paper', 'external-1'):
            own = [chunk for chunk in chunks if chunk['document_id']==document]
            self.assertEqual({sid for c in own for sid in c['sentence_ids']}, {s['id'] for s in sentences})
            self.assertTrue(all(fake_token_count(c['text']) <= 400 for c in own))
            self.assertTrue(all(all(int(sid[1:])//4+1 == c['page'] for sid in c['sentence_ids']) for c in own))

    def test_qdrant_hybrid_index_and_model_mismatch(self):
        chunks = build_chunks([{'id':'s0','page':0,'text':'Robotic control learning.'},
                               {'id':'s1','page':1,'text':'Attention in language.'}], [], 'Root')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            paper_id=uuid.uuid4().hex
            result=build_index(chunks,path,paper_id)
            self.assertEqual(result['dimensions'],384)
            self.assertEqual(result['store'],'qdrant')
            self.assertFalse((path/'vectors.npy').exists())
            stored=json.loads((path/'chunks.json').read_text())
            self.assertEqual(stored,chunks)
            found=retrieve('robotic control', stored, directory=path, paper_id=paper_id)
            self.assertEqual(found[0]['page'],1)
            result['model']='wrong-model'
            (path/'index.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError): retrieve('control',stored,directory=path,paper_id=paper_id)
            with patch('readpaper.rag.RemoteE5Embeddings', side_effect=ValueError('embedding unavailable')):
                with self.assertRaisesRegex(ValueError, 'embedding unavailable'):
                    build_index(chunks,path,paper_id)
            self.assertFalse((path/'index.json').exists())

    def test_download_does_not_accept_private_hosts_or_redirects(self):
        with self.assertRaises(ValueError): asyncio.run(public_url('http://example.com/file.pdf'))
        with self.assertRaises(ValueError): asyncio.run(public_url('https://127.0.0.1/file.pdf'))
        checked=[]
        async def check(url):
            checked.append(url)
            if '127.0.0.1' in url: raise ValueError('private')
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(302,headers={'location':'https://127.0.0.1/secret'}))) as client:
                with patch('readpaper.fulltext.public_url',check):
                    with self.assertRaises(ValueError): await download_pdf(client,'https://example.com/paper.pdf')
        asyncio.run(run())
        self.assertEqual(len(checked),2)

    def test_fulltext_preserves_pdf_text_and_failure_scope(self):
        title='Reinforcement learning improves robotic control'
        source={'id':'external-0','kind':'referenced','title':title,'text':'Abstract only',
                'content_scope':'abstract','pdf_locations':[{'pdf_url':'https://example.com/paper.pdf'}]}
        content = example_pdf()
        with tempfile.TemporaryDirectory() as directory, patch('readpaper.fulltext.download_pdf', AsyncMock(return_value=content)):
            records=asyncio.run(collect_fulltext([source],Path(directory)))
            self.assertEqual(records[0]['fulltext_status'],'ready')
            self.assertEqual(records[0]['content_scope'],'full_text')
            self.assertEqual(len(records[0]['sentences']),6)
            self.assertEqual((Path(directory)/'external-0.pdf').read_bytes(), content)
            self.assertIn('Transformers', (Path(directory)/'external-0.txt').read_text())
            self.assertTrue((Path(directory)/'manifest.json').is_file())
        with self.assertRaises(ValueError): parse_fulltext(example_pdf(),'Unrelated medieval history')
        with tempfile.TemporaryDirectory() as directory, patch('readpaper.fulltext.download_pdf', AsyncMock(side_effect=ValueError('blocked'))):
            result=asyncio.run(collect_fulltext([source],Path(directory)))[0]
            self.assertEqual(result['content_scope'],'abstract')
            self.assertEqual(result['text'],'Abstract only')
            self.assertEqual(result['fulltext_status'],'failed')

    def test_upload_selection_image_history_citations_and_expiry(self):
        class RecordingModel(FakeVLLM):
            async def answer(self,question,contexts):
                self.question=question; self.contexts=contexts
                return {'answer':f"설명 [{contexts[0]['id']}]",'citations':[contexts[0]['id']]}
        with tempfile.TemporaryDirectory() as directory:
            config=Settings(data_dir=Path(directory),dev_mode=True)
            model=RecordingModel()
            with patch('readpaper.db.settings',config), patch('readpaper.main.settings',config), \
                 patch('readpaper.main.VLLMClient',return_value=model), patch('readpaper.main.collect_external_sources',fake_sources):
                with TestClient(app) as client:
                    uploaded=client.post('/api/papers/upload',files={'file':('paper.pdf',example_pdf(),'application/pdf')})
                    pid=uploaded.json()['paper_id']
                    paper=client.get(f'/api/papers/{pid}').json()
                    self.assertEqual(paper['rag_index']['status'],'ready')
                    root=Path(directory)/'papers'/pid
                    self.assertEqual(paper['rag_index']['store'],'qdrant')
                    attachment={'kind':'text','page':1,'word_ids':['w0','w1']}
                    result=client.post(f'/api/papers/{pid}/ask',json={'question':'이 뜻은?', 'attachments':[attachment],
                        'history':[{'role':'user','content':'이전 질문'}]})
                    self.assertEqual(result.status_code,200,result.text)
                    evidence=result.json()['sources'][0]
                    self.assertEqual(evidence['excerpt'],'Reinforcement learning')
                    self.assertEqual(evidence['page'],1)
                    self.assertEqual(evidence['sentence_ids'],['s0'])
                    self.assertIn('#page=1',evidence['url'])
                    self.assertIn('이전 질문', model.question)
                    self.assertEqual(client.get(evidence['url'].split('#')[0]).headers['content-type'],'application/pdf')
                    image=client.post(f'/api/papers/{pid}/ask',json={'question':'그림 설명', 'attachments':[{'kind':'figure','page':1,'rect':[35,65,300,145]}]})
                    self.assertEqual(image.status_code,200,image.text)
                    self.assertTrue(model.contexts[0]['image_data'].startswith('data:image/png;base64,'))
                    self.assertNotIn('image_data',image.json()['sources'][0])
                    for bad in ({'kind':'text','page':2,'word_ids':['w0']}, {'kind':'text','page':1,'word_ids':['forged']}, {'kind':'figure','page':1,'rect':[-1,0,200,200]}):
                        response=client.post(f'/api/papers/{pid}/ask',json={'question':'설명','attachments':[bad]})
                        self.assertEqual(response.status_code,422)
                    with db.connect() as connection: connection.execute('UPDATE papers SET last_accessed=0')
                    db.cleanup_expired()
                    self.assertFalse(root.exists())

    def test_embedding_failure_is_not_marked_ready_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Settings(data_dir=Path(directory),dev_mode=True)
            with patch('readpaper.db.settings',config), patch('readpaper.main.settings',config), \
                 patch('readpaper.main.VLLMClient',return_value=FakeVLLM()), patch('readpaper.main.collect_external_sources',fake_sources):
                with TestClient(app) as client:
                    with patch('readpaper.rag.RemoteE5Embeddings',side_effect=ValueError('embedding unavailable')):
                        uploaded=client.post('/api/papers/upload',files={'file':('paper.pdf',example_pdf(),'application/pdf')})
                    pid=uploaded.json()['paper_id']
                    self.assertEqual(client.get(f'/api/papers/{pid}').json()['status'],'failed')
                    self.assertEqual(client.post(f'/api/papers/{pid}/ask',json={'question':'질문'}).status_code,409)
                    self.assertEqual(client.post(f'/api/papers/{pid}/retry').status_code,200)
                    self.assertEqual(client.get(f'/api/papers/{pid}').json()['status'],'ready')


if __name__=='__main__': unittest.main()
