# Readpaper

영문 AI 논문 PDF를 원본 형태로 읽으며 단어·복합 전문 용어의 문맥별 한국어 뜻을 확인하는 개인용 웹 앱입니다. 논문은 번역하지 않고, 사용자가 용어를 누른 순간 논문 전체에서 관련 문맥을 검색해 짧은 뜻을 생성하고 저장합니다. 현재 논문과 수집 자료를 근거로 추가 질문도 할 수 있습니다. 제품 범위는 [제품 구상](docs/product-brief.md), 진행 순서는 [로드맵](docs/roadmap.md), RAG 확장 계획은 [RAG 설계](docs/rag-design.md)에 기록합니다.

## 현재 구현

- 브라우저에서 PDF.js로 원본 PDF를 렌더링합니다. 서버는 PyMuPDF로 단어 위치와 문장을 추출하고, 선택 가능한 텍스트가 없는 PDF는 거부합니다. 문헌 수집 중에도 원문을 읽을 수 있습니다.
- 단어를 누르면 Mac mini의 Ollama에 그 단어·복합 용어의 짧은 뜻만 요청합니다. 제목, 클릭한 문장, 논문 안에서 해당 단어가 다시 나오는 곳과 검색된 본문 문맥을 함께 보내 분야별 의미를 구분합니다. 선택된 용어 범위를 검사하고 뜻을 저장하므로 다시 누르면 모델을 호출하지 않습니다.
- OpenAlex에서 참고·관련·후속 인용 논문과 최근 3년의 키워드 관련 논문을 찾고 서지 정보·공개 초록을 수집합니다. Wikipedia에서 배경 설명도 가져옵니다. 접근할 수 없으면 현재 논문만으로 읽기와 질문 기능이 동작하고 경고를 표시합니다. 외부 논문의 전문은 아직 수집하지 않으며 화면에서 초록 근거임을 구분합니다.
- 추가 질문에는 한국어 질문을 영어 검색어로 바꾼 뒤, 현재 논문 문단과 수집한 초록·배경 설명에서 TF-IDF로 근거를 찾습니다. Ollama 답변에는 실제 검색된 출처만 표시합니다.
- 누른 단어는 단어장에 저장하고 직접 단어를 추가하거나 삭제할 수 있습니다. 단어장은 SQLite에 유지합니다. PDF 원본은 업로드 처리 후 보관하지 않습니다. 추출된 텍스트·용어 뜻·외부 자료는 마지막 사용 후 24시간이 지나면 주기적으로 삭제합니다.

외부 자료 수집을 따로 점검하려면 [RAG 수집 점검 기록](docs/rag-source-check.md)을 참고하세요. DOI와 키워드를 넣어 인용·관련 후보를 확인하고, 공개 PDF의 전문을 실제로 가져와 텍스트를 추출하는 테스트 명령도 있습니다. 이 진단 수집은 아직 웹 앱의 RAG 흐름에 연결되지는 않았습니다.

수집·청킹 실험에 쓰는 로컬 파일은 저장소 안의 [`data/`](data/README.md)에서 관리합니다. 이 폴더의 데이터 파일은 Git에 포함되지 않습니다. 현재 수집 점검 JSON과 로컬 앱 데이터베이스는 각각 `data/inspection/`, `data/runtime/`에 저장됩니다.

## 개발 환경에서 실행

Python 3.12와 Node.js 24가 필요합니다. Mac mini에서 Ollama를 실행하고 `gemma4:12b` 모델을 내려받습니다. Ollama와 모델은 Mac에서 로컬로 실행되며, 모델 파일은 Mac의 저장 공간을 사용합니다. 전체 모델을 바꾸려면 `READPAPER_OLLAMA_MODEL`, 클릭한 용어 번역에만 다른 모델을 쓰려면 `READPAPER_GLOSS_MODEL` 환경 변수를 설정합니다.

```bash
ollama pull gemma4:12b
ollama run gemma4:12b
```

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
READPAPER_DEV_MODE=1 READPAPER_OLLAMA_URL=http://127.0.0.1:11434 uvicorn readpaper.main:app --reload
```

다른 터미널에서:

```bash
cd frontend
npm ci
npm run dev
```

Vite 개발 서버의 `/api` 요청은 `localhost:8000`으로 전달됩니다. `READPAPER_DEV_MODE=1`은 로컬 개발 전용이며 로그인을 건너뜁니다. 운영 환경에서는 사용하지 않습니다. 프론트엔드 빌드는 `cd frontend && npm run build`, 백엔드 검증은 `cd backend && python -m unittest discover -s tests -v`로 실행합니다.

## Docker로 실행

`.env.example`을 `.env`로 복사해 사용자 이름, 긴 비밀번호, 무작위 세션 서명 키와 Ollama 주소를 채운 뒤 `docker compose up --build`를 실행합니다. `.env`는 Git에 포함되지 않습니다. 기본 Compose 설정은 웹 앱을 **서버 자신의 `127.0.0.1:8000`**에만 연결합니다. 인터넷에 공개할 때는 HTTPS 역방향 프록시를 앞에 둬야 하며 `READPAPER_SECURE_COOKIES=1`로 설정합니다.

Mac mini와 클라우드 서버는 Tailscale 같은 인증된 전용 네트워크로 연결합니다. Mac의 Ollama 포트 `11434`를 공개 인터넷에 열지 마세요. 클라우드 컨테이너에서 Mac의 전용 네트워크 주소로 `/api/tags` 요청이 성공하는지 확인한 뒤 `READPAPER_OLLAMA_URL`을 그 주소로 설정합니다. 새로운 용어의 첫 번역이나 새 질문에 답할 때 Mac mini가 켜져 있어야 합니다. 이미 만든 용어 뜻은 저장본에서 조회하므로 Ollama를 다시 호출하지 않습니다. Ollama에는 모델을 30분 동안 메모리에 유지하도록 요청해 연속 클릭 때 재로딩을 줄입니다.

용어 번역 속도와 뜻의 정확도를 같은 입력으로 비교하려면 Mac에서 `ollama pull gemma4:e2b`를 실행한 뒤 다음 명령을 사용합니다. 각 모델을 먼저 메모리에 올리고 실제 업로드 논문에서 이미 눌러 본 용어 최대 다섯 개의 번역 결과와 응답 시간 중앙값을 출력합니다. 저장된 뜻도 함께 출력되며 서비스의 모델 설정은 바뀌지 않습니다.

```bash
docker compose up -d --build
docker compose exec web python /app/backend/scripts/benchmark_gloss.py --latest-paper gemma4:12b gemma4:e2b
```

뜻이 적절하고 응답 시간이 목표에 도달하면 `.env`에 `READPAPER_GLOSS_MODEL=gemma4:e2b`를 추가하고 `docker compose up -d --force-recreate web`을 실행합니다. 채팅과 관련 논문 판단은 계속 `READPAPER_OLLAMA_MODEL`을 사용합니다. 모델을 바꾼 직후 첫 요청에는 모델 로딩 시간이 추가될 수 있으므로 다음 요청부터 측정합니다.

관련 자료 수집에는 `api.openalex.org`와 `en.wikipedia.org`로 나가는 HTTPS 연결이 필요합니다. 사설망 연결과 외부 자료 접근이 불가능할 때에도 오류가 설명되도록 구성했습니다. AWS EC2와 HTTPS 공개 배포 절차는 [배포 안내](docs/deployment.md)에 정리했습니다.

## 아직 검증하거나 보완할 부분

현재 테스트는 생성한 2단 PDF와 **모의 Ollama 응답**으로 업로드, 용어 클릭·캐시, 근거 표시, 단어장, 로그인 흐름을 확인합니다. 실제 Mac mini 모델의 뜻 정확도와 클릭 응답 시간, 실제 학술 논문에서의 좌표·읽기 순서, OpenAlex·Wikipedia 응답, 클라우드의 HTTPS·전용 연결은 아직 검증하지 않았습니다. 실측 응답 시간은 첫 모델 로딩 여부, Ollama 상태, 문맥 길이에 따라 달라집니다.

첫 버전의 분석 작업은 웹 프로세스의 백그라운드 작업이며 SQLite를 사용합니다. 서버 재시작 중인 작업의 재개, 다중 서버 배포, 임베딩 기반 검색은 이후 작업 대기열과 저장소를 도입할 때 보완해야 합니다. 실제 논문으로 성능을 재고 필요하면 Docling/GROBID와 임베딩 검색을 비교합니다.
