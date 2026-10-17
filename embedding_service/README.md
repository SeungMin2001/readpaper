# Mac mini 임베딩 서버

Apple Silicon Mac의 PyTorch MPS GPU에서 `intfloat/multilingual-e5-small`을 실행합니다. 웹 백엔드는 논문 청크를 64개씩 `/v1/embeddings`에 보내고, 서버는 기본 32개씩 모델에 넣습니다. 응답은 정규화된 384차원 벡터입니다.

Mac mini의 저장소 루트에서 Python 3.12 가상 환경을 준비합니다.

```bash
python3.12 -m venv .venv-embedding
source .venv-embedding/bin/activate
pip install -r embedding_service/requirements.txt
export READPAPER_EMBEDDING_API_KEY='별도로-생성한-긴-무작위-키'
python -c 'import torch; print("MPS available:", torch.backends.mps.is_available())'
uvicorn embedding_service.server:app --host 127.0.0.1 --port 8002
```

MPS가 `True`여야 서버가 시작됩니다. 첫 실행에는 Hugging Face에서 모델 가중치를 받습니다. 로컬 확인 후 클라우드와 연결할 때는 `--host 127.0.0.1`을 Mac mini의 사설 VPN IP로 바꿉니다. API 키를 설정하고 8002 포트는 공개 인터넷에 열지 않습니다. vLLM은 별도 터미널의 8001 포트에서 실행합니다.

다른 터미널에서 인증과 실제 임베딩을 확인합니다.

```bash
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/v1/embeddings \
  -H "Authorization: Bearer $READPAPER_EMBEDDING_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"intfloat/multilingual-e5-small","input":["query: attention mechanism"]}'
```

클라우드의 `.env`에는 `READPAPER_EMBEDDING_BASE_URL=http://<Mac의 사설 VPN IP>:8002/v1`과 동일한 `READPAPER_EMBEDDING_API_KEY`를 설정합니다. 로컬 Docker Desktop에서 웹 앱을 실행한다면 기본 주소 `http://host.docker.internal:8002/v1`을 사용합니다. 서버는 E5의 `query: `·`passage: ` 접두부와 512토큰 한도를 검사합니다.

색인 완료 시 웹 앱 진행 기록에 임베딩 API 호출 시간과 색인 전체 시간이 남고, 각 논문의 `index/index.json`에도 `embedding_ms`와 `index_ms`가 기록됩니다. 실제 성능 비교에는 같은 PDF·동일한 수집 결과로 반복 실행하고 Mac mini의 vLLM 동시 실행 여부를 기록해야 합니다.
