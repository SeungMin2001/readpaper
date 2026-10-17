# RAG 설계와 구현 범위

현재 웹 앱은 **전문 수집 → 페이지·문장 청킹 → 다국어 임베딩 저장 → 의미/키워드 검색 → 출처를 포함한 답변**으로 연결돼 있다. 상세한 데이터 흐름과 실제 검증 결과는 [채팅·RAG 사용 및 검증](workspace-rag.md)을 참고한다.

## 구성과 역할

| 단계 | 현재 구현 | 실행 위치 |
|---|---|---|
| 원문 표시·선택 | PDF.js, 단어/문장 선택, 그림 영역 크롭 미리보기 | 브라우저 |
| PDF 본문·좌표 추출 | PyMuPDF, 원문 페이지·문장 ID 유지 | 웹 백엔드 |
| 문헌 식별·관계 수집 | OpenAlex DOI/제목 조회, 빈 관계 목록은 PDF DOI·arXiv ID와 키워드 검색으로 보완, 관련 최대 8편·피인용 최대 5편 | 웹 백엔드 |
| 전문 확보 | 공개 PDF 다운로드, 크기·페이지·제목 검사, 실패 시 초록/서지 구분 | 웹 백엔드 |
| 청킹 | LangChain 텍스트 분할기, 페이지·절 경계, E5 기준 400토큰·55토큰 겹침 | 웹 백엔드 |
| 의미 벡터 | multilingual-e5-small, 정규화된 384차원, API 배치 처리 | Mac mini MPS GPU |
| 저장 | PDF/TXT/JSON과 상태·로그는 SQLite, dense·BM25 벡터는 Qdrant | Docker 영구 볼륨 |
| 검색 | LangChain QdrantVectorStore의 RRF 하이브리드 검색, 겹치는 문장 제외 | Qdrant + 웹 백엔드 |
| 질문 정규화·답변 | Gemma 4 12B, OpenAI 호환 vLLM API | Mac mini |
| 선택한 그림 | 서버가 원본 PDF를 크롭해 image_url로 전송 | 웹 백엔드 → Mac mini |
| 출처 | 실제 제공된 청크 ID 검증, 문서·페이지·문장·원문과 PDF 링크 | 백엔드 + 브라우저 |

논문 요약과 문헌 수집은 병렬로 시작한다. 초기 화면에서 진행을 보고 완료 후 읽기를 시작하며, 원문 먼저 열기도 제공한다. 번역은 기존의 한 문장 요약+용어 요청을 유지한다. 논문별 PDF·텍스트·색인은 마지막 사용 후 24시간에 삭제하며 단어장과 임베딩 모델 캐시는 유지한다.

## 코드를 읽는 순서

1. `backend/readpaper/main.py`: 업로드, 진행 기록, 준비 작업, 질문 API의 연결.
2. `backend/readpaper/sources.py`: 업로드 논문 식별, OpenAlex 관계 후보 조회.
3. `backend/readpaper/fulltext.py`: 공개 PDF 다운로드·제목 검사·본문과 파일 저장.
4. `backend/readpaper/rag.py`: 청킹, 원격 E5 임베딩, Qdrant 색인·검색과 중복 제거.
5. `embedding_service/server.py`: Mac mini MPS 임베딩 API.
6. `backend/readpaper/llm.py`: 질문 검색어, 선택한 이미지, 근거 기반 답변과 인용 ID 검사.
7. `frontend/src/PdfReader.tsx`: 원문 위 좌표, 번역 영역, 문장/그림 선택.
8. `frontend/src/App.tsx`: 준비 화면, 채팅 첨부·대화 이력·출처, 문헌 목록.

## 아직 적용하지 않은 확장

- **Cross-encoder 리랭커:** 현재의 RRF 결합·중복 제거는 별도의 학습된 리랭커가 아니다. 정답 근거 목록을 만들어 Recall@5/10과 지연을 먼저 측정하고 리랭커의 추가 이득을 평가한다.
- **LLM 관련 논문 선정:** 현재는 OpenAlex 관련 후보 순서를 사용한다. 기존 `select_related_papers`를 채택하려면 주제 관련성·최신성 정답 사례로 비교한다.
- **구조 파서 확대:** PyMuPDF 기준선은 모든 PDF의 다단 편집·표·수식·각주를 정확히 이해하지 못한다. GROBID/Docling, 절·표·캡션 분리를 별도 평가한다. 현재 PDF 참고문헌 보완은 번호 목록의 DOI·arXiv ID만 확인하므로 식별자가 없는 서지를 검증해 연결하는 작업도 후속 과제다.
- **작업 큐:** 현재 FastAPI 백그라운드 작업이며 중단 시 실패와 재시도를 제공한다. 다중 사용자/다중 워커 운영에서는 Redis+작업자 큐, 동시 수집 제한, 원자적 작업 선점을 추가한다.
- **답변 품질 평가:** 출처 ID가 유효하더라도 답변이 그 근거를 올바르게 해석했는지는 별도 검증한다. 문헌 검색·본문 파싱·근거 적합성·이미지 이해를 따로 평가한다.
