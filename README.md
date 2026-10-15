# Readpaper

영문 AI 논문 PDF를 원본 형태로 읽으며 단어·복합 전문 용어의 문맥별 한국어 뜻을 확인하는 개인용 웹 앱입니다. 업로드 후 논문 핵심 요약과 인용·관련 문헌 검색을 준비하고, 용어는 누를 때 번역합니다. 현재 LLM 연결은 vLLM의 OpenAI 호환 API를 사용합니다.

## 현재 구현

- 브라우저에서 PDF.js로 원본 PDF를 렌더링합니다. 서버는 PyMuPDF로 단어 위치와 문장을 추출하고, 선택 가능한 텍스트가 없는 PDF는 거부합니다.
- 업로드 시 핵심 요약과 문헌 수집·색인을 준비한 뒤 원문을 엽니다. 준비 중 원문을 먼저 열 수도 있습니다. 용어 번역은 미리 하지 않습니다.
- 용어를 누르면 서버가 원문에서 단어·일반적인 AI 복합 용어의 범위를 정하고, 저장한 논문 요약과 그 용어만 모델에 보냅니다. 모델은 JSON이나 원문을 복사하지 않고 짧은 한국어 뜻만 생성합니다. 등록되지 않은 복합 용어는 클릭한 단어만 번역합니다.
- 같은 논문의 동일한 요약·모델·용어는 다른 페이지에서도 저장된 뜻을 재사용합니다. 동일 입력의 동시 요청은 추론을 한 번만 실행하고 각 위치에 결과를 연결합니다. 서로 다른 용어 요청은 병렬로 보냅니다.
- 번역 요청은 Thinking을 명시적으로 끄고 HTTP 연결을 재사용합니다. 고정 지시문과 논문 요약을 공통 접두부로 배치해 서버가 지원하는 prefix caching을 활용할 수 있습니다. 실제 캐시 적중 여부는 모델 서버 측정값으로 확인해야 합니다.
- OpenAlex에서 참고문헌과 관련 최대 8편·피인용 최대 5편을 찾고, 공개 PDF를 다운로드해 제목을 대조한 뒤 본문을 수집합니다. 전문 미확보 자료는 초록/서지 정보로 구분합니다.
- 본문을 페이지·문장별로 청킹하고 CPU 다국어 MiniLM 임베딩(384차원)을 저장합니다. 의미 검색과 TF-IDF를 결합해 채팅 근거를 찾고 문서·페이지·문장 ID·원문을 출처로 표시합니다.
- 문장과 그림 영역을 선택해 오른쪽 채팅에 첨부할 수 있습니다. 그림은 원본 PDF에서 잘라 모델의 이미지 입력으로 보냅니다. 채팅은 최근 대화와 Markdown 답변을 지원합니다.
- 단어장은 SQLite에 저장합니다. PDF 원본·수집 PDF·추출 텍스트·벡터는 마지막 사용 후 24시간이 지나면 함께 삭제합니다.

화면 사용법, 데이터 저장 위치, 실제 수집 결과는 [채팅·RAG 사용 및 검증](docs/workspace-rag.md)을 참고하세요. 최초 문헌 준비 시 웹 서버에서 임베딩 모델을 다운로드합니다. 그림 질문에는 vLLM의 이미지 입력 지원이 필요합니다.

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

웹 컨테이너가 실행된 뒤 20개 용어로 이전 JSON 요청과 새 뜻 전용 요청을 비교합니다. 이 검사는 모델을 실제로 호출하며 앱의 번역 결과 캐시를 사용하지 않습니다.

```bash
docker compose exec web python /app/backend/scripts/benchmark_gloss.py --compare
docker compose exec web python /app/backend/scripts/benchmark_gloss.py --latest-paper --api-url http://127.0.0.1:8000 --repeats 2
```

두 번째 명령은 업로드된 논문의 용어를 웹 API에 요청합니다. 처음 생성한 뜻과 저장된 뜻의 시간을 별도로 집계합니다. 중앙값·p95·최대값·오류를 출력하고, 새 뜻 중 하나라도 1초 이상이면 종료 코드 1, 서버에 연결할 수 없으면 2를 반환합니다. 모두 캐시 적중인 경우에는 새 뜻의 1초 목표를 달성한 것으로 판정하지 않습니다. `--json-out /data/gloss-benchmark.json`으로 전체 결과를 저장할 수 있습니다.

API의 `timing.first_token_ms`는 모델 요청부터 첫 텍스트 조각까지, `after_first_token_ms`는 그 뒤 응답 완료까지입니다. GPU 순수 연산 시간과는 다르며 네트워크·대기 시간을 포함합니다. 사용자 화면에는 브라우저에서 잰 클릭 후 응답 시간이 표시됩니다. 스트림은 서버에서 끝까지 받은 뒤 검증하므로 첫 토큰만 나온 시점을 번역 완료로 세지 않습니다. 실측 범위와 제한은 [성능 검증 기록](docs/gloss-performance.md)에 정리합니다.

업로드 준비 시간은 PDF 길이와 Mac mini의 동시 처리 성능에 따라 달라집니다. 초기 로딩은 한 문장 요약과 문헌 전문 수집·임베딩을 준비합니다. 용어는 클릭할 때 번역하므로 첫 응답은 모델 추론 시간을 기다리고, 요약만 제공하므로 문장 단위 문맥이 필요한 다의어는 정확도가 낮아질 수 있습니다.
