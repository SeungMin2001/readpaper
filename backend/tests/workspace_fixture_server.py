"""Isolated UI test fixture: fake LLM/metadata, actual application routes and PDF parsing."""
import os,sys
os.environ['READPAPER_DEV_MODE']='1'
os.environ['READPAPER_DATA_DIR']=os.environ['READPAPER_TEST_DATA_DIR']
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import readpaper.main as main
from test_flow import FakeVLLM
from test_rag_workspace import fake_embed
import readpaper.rag as rag
rag.embed=fake_embed
class DemoModel(FakeVLLM):
    async def translate_keyword_from_summary(self,summary,term):
        return {'term':term,'meaning':'다중 헤드 어텐션 메커니즘'}
    async def answer(self,question,contexts):
        return {'answer':f"이 논문은 **어텐션 메커니즘**으로 입력 간 관계를 학습합니다. [{contexts[0]['id']}]\n\n선택한 내용은 다음과 같이 이해할 수 있습니다.\n\n- 쿼리와 키로 관련성을 계산합니다.\n- 가중합으로 정보를 통합합니다.", 'citations':[contexts[0]['id']]}
async def sources(*args,**kwargs):
    return [dict(id=f'external-{i}',kind='referenced' if i<5 else 'related',title=title,text='Attention computes weighted representations for sequence learning.',content_scope='abstract',url='https://arxiv.org/abs/1706.03762',year=2017,openalex_id=f'https://openalex.org/W{i}',referenced_works=['https://openalex.org/W0'] if i==6 else []) for i,title in enumerate(['Neural Machine Translation by Jointly Learning to Align and Translate','Sequence to Sequence Learning with Neural Networks','Learning Phrase Representations using RNN Encoder–Decoder','Deep Residual Learning for Image Recognition','Layer Normalization','BERT: Pre-training of Deep Bidirectional Transformers','Language Models are Few-Shot Learners'])],[]
main.VLLMClient=DemoModel
main.collect_external_sources=sources
import uvicorn
uvicorn.run(main.app,host='127.0.0.1',port=int(os.environ['READPAPER_TEST_PORT']))
