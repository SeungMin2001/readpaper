# Readpaper

영문 AI 논문 PDF를 원본 형태로 읽으며 단어·복합 전문 용어의 문맥별 한국어 뜻을 확인하는 개인용 웹 앱입니다. 논문 전체를 미리 번역하지 않고, 사용자가 용어를 누를 때 논문 문맥을 바탕으로 짧은 뜻을 생성해 저장합니다. 현재 LLM 연결은 vLLM의 OpenAI 호환 API를 사용합니다.

## 현재 구현

- 브라우저에서 PDF.js로 원본 PDF를 렌더링합니다. 서버는 PyMuPDF로 단어 위치와 문장을 추출하고, 선택 가능한 텍스트가 없는 PDF는 거부합니다.
- 용어를 누르면 제목, 문장, 논문 안의 반복 용례와 검색된 문맥을 vLLM에 보내 짧은 한국어 뜻을 생성합니다. 선택된 용어 범위를 검증하고 저장하므로 다시 누르면 모델을 호출하지 않습니다.
- 서로 다른 단어 요청은 백엔드에서 직렬화하지 않아 vLLM이 동시 요청을 묶어 처리할 수 있습니다. 같은 단어의 중복 요청만 합칩니다.
- OpenAlex에서 참고·관련·후속 인용 논문과 최근 3년의 키워드 관련 논문을 찾고, 공개 초록과 Wikipedia 배경 설명을 수집합니다. 외부 논문의 전문은 아직 수집하지 않습니다.
- 추가 질문에는 논문 문단과 수집한 초록·배경 설명에서 TF-IDF로 근거를 찾습니다. 답변에는 검색된 출처만 표시합니다.
- 단어장은 SQLite에 저장합니다. PDF 원본은 업로드 처리 후 보관하지 않으며 추출 텍스트와 외부 자료는 마지막 사용 후 24시간이 지나면 삭제합니다.

## Mac mini에서 vLLM-Metal 실행

Mac mini M4에서는 CUDA용 vLLM이 아니라 Apple Silicon용 vLLM-Metal을 사용합니다. 아래 예시는 Gemma 4 12B 4-bit 체크포인트를 Hugging Face에서 내려받아 서버를 시작합니다. Ollama의 `gemma4:12b`와 같은 모델 계열·크기이지만 정확한 원본 체크포인트와 양자화가 다를 수 있으므로, 기존 모델을 지우기 전에 뜻의 품질을 비교하세요.

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
export VLLM_API_KEY='replace-with-a-long-random-secret'
vllm serve mlx-community/gemma-4-12B-it-4bit \
  --host 0.0.0.0 --port 8001 \
  --served-model-name gemma4-12b \
  --max-model-len 8192 \
  --api-key "$VLLM_API_KEY"
```

첫 실행은 모델 가중치를 내려받아 Mac 저장 공간을 사용합니다. 서버가 준비되면 다른 터미널에서 API를 확인합니다.

```bash
curl http://localhost:8001/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

기존 Ollama 체크포인트 정보를 확인하려면 `ollama show --modelfile gemma4:12b`를 실행합니다. vLLM 연결과 뜻 품질을 확인한 뒤 기존 Gemma 파일을 지우려면 `ollama rm gemma4:12b`를 실행합니다. `ollama list`에 있는 다른 모델도 지우려면 각 모델 이름을 확인해 별도로 `ollama rm <모델명>`을 실행합니다.

## 웹 앱 실행

Python 3.12와 Node.js 24가 필요합니다. vLLM-Metal을 위 명령으로 실행한 상태에서:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
READPAPER_DEV_MODE=1 \
READPAPER_LLM_BASE_URL=http://127.0.0.1:8001/v1 \
READPAPER_LLM_MODEL=gemma4-12b \
READPAPER_LLM_API_KEY="$VLLM_API_KEY" \
uvicorn readpaper.main:app --reload
```

다른 터미널에서 프론트엔드를 실행합니다.

```bash
cd frontend
npm ci
npm run dev
```

## Docker와 클라우드 배포

Docker로 웹 앱을 실행할 때 `.env.example`을 `.env`로 복사합니다. `READPAPER_LLM_BASE_URL`은 Mac에서 실행 중인 vLLM 서버 주소로 둡니다. 기본값 `http://host.docker.internal:8001/v1`은 Docker Desktop에서 Mac 호스트로 연결할 때 사용합니다. `READPAPER_LLM_API_KEY`에는 `vllm serve --api-key`에 전달한 동일한 키를 넣습니다.

```bash
docker compose up -d --build
docker compose exec web python -c 'import os,urllib.request; req=urllib.request.Request("http://host.docker.internal:8001/v1/models",headers={"Authorization":"Bearer "+os.getenv("READPAPER_LLM_API_KEY","")}); print(urllib.request.urlopen(req).read().decode())'
```

클라우드 앱 서버에서 Mac mini vLLM에 연결할 때는 두 기기를 Tailscale 같은 사설망에 연결하고, `.env`의 주소를 `http://<Mac의 사설 IP>:8001/v1`로 설정합니다. vLLM API 키를 설정하고 포트 8001을 공개 인터넷에 열지 마세요. Mac이 꺼져 있으면 새 뜻 생성과 질문 답변이 멈춥니다. 상시 서비스가 필요하면 AWS GPU 인스턴스에 CUDA용 vLLM을 배포할 수 있지만, GPU를 켜 둔 시간만큼 비용이 발생합니다.

HTTPS 공개 배포 절차는 [배포 안내](docs/deployment.md)를 참고하세요. 데이터 수집 흐름은 [RAG 수집 점검 기록](docs/rag-source-check.md), 제품 범위는 [제품 구상](docs/product-brief.md), 진행 순서는 [로드맵](docs/roadmap.md), 검색 설계는 [RAG 설계](docs/rag-design.md)에 기록합니다.

## 성능 확인

웹 컨테이너가 실행된 뒤 샘플 용어로 vLLM 응답과 중앙값을 확인할 수 있습니다.

```bash
docker compose exec web python /app/backend/scripts/benchmark_gloss.py
docker compose exec web python /app/backend/scripts/benchmark_gloss.py --latest-paper
```

이 측정은 현재 vLLM 서버의 응답 시간을 기록합니다. vLLM을 쓴다고 첫 단어 클릭이 자동으로 1초 안에 끝나는 것은 아니므로, Mac mini에서 실제 용어로 지연 시간과 번역 품질을 함께 평가해야 합니다.
