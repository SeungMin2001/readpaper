# RAG 수집 기준선 점검

## 실행 방법

백엔드 폴더에서 수집기만 실행한다. 이 명령은 Ollama를 부르지 않으므로 DOI와 검색 키워드를 명시해 OpenAlex/Wikipedia 수집을 분리해서 확인할 수 있다.

```bash
cd backend
../.venv/bin/python scripts/inspect_sources.py \
  --title "Attention Is All You Need" \
  --doi 10.48550/arXiv.1706.03762 \
  --keywords transformer "attention mechanism"
```

JSON에는 수집 경로, 연도·날짜, DOI/OpenAlex ID, 초록/메타데이터 구분, 초록 미리보기, 경고가 나온다. `--json-out /tmp/sources.json`을 추가하면 같은 보고서를 파일로 저장한다.

## 실제 API 점검 기록

- 입력: *Attention Is All You Need*, DOI `10.48550/arXiv.1706.03762`; 키워드 `transformer`, `attention mechanism`.
- OpenAlex와 Wikipedia가 실제 응답을 반환했고 수집기 경고는 없었다.
- 참고 논문 8편, OpenAlex 관련 논문 3편, 이 논문을 인용한 논문 4편, 최근 관련 논문 1편, Wikipedia 배경 자료 2건이 나왔다.
- 가장 최근 후보는 2026년 *ViXNet: Vision Transformer with Xception Network for Facial Expression Recognition*이었다. 최근 Transformer 응용 연구로는 관련이 있지만, 원 논문인 자연어 처리와 분야가 다르다.
- OpenAlex의 related 결과에는 BERT와 함께 논문 주제에서 멀어 보이는 언어학·실험 자동화 논문도 포함됐다. 인용 관계는 실제 연결을 뜻하지만 모든 인용 논문이 질문에 유용한 것은 아니다.
- 외부 논문은 초록만 사용했다. 인용 논문 한 편은 초록이 없어 서지 정보만 보였고 RAG 문맥으로는 들어가지 않았다.
- 질문 `multi-head self attention architecture`를 현재 TF-IDF 검색기에 넣었을 때 `Self-training PCFG grammars...`가 반환됐다. `self`라는 단어가 겹친 결과로, 관련도 오탐이다. TF-IDF 점수와 검색 로그를 아직 UI/API에서 확인할 수 없다.
- 첫 수집 실행에서 OpenAlex의 미래 날짜 논문(2027)이 최근 목록에 들어오는 것을 발견했다. 최근 검색에 오늘 날짜 상한을 추가했고, 수집 테스트에서 미래 날짜 항목을 제외하는 것을 확인했다.

## 지금 알 수 있는 점과 다음 검증

현재 수집기는 DOI 기반 원 논문 조회와 참고·관련·인용 관계 확장을 수행한다. 관계별 논문 발견은 동작하지만, OpenAlex의 관련도 순서를 그대로 사용하므로 주제에 먼 결과도 섞일 수 있다. 최근 연구는 날짜 제한과 키워드 일치로 후보를 거르지만 의미상 관련도를 완전히 판단하지 못한다. 이후 임베딩과 리랭킹으로 개선하되, 먼저 실제 사용자 PDF 한 편으로 제목·DOI 식별과 예상 참고문헌을 대조한다.

다음 실험에서는 각 질문에 정답 근거가 되는 논문/문단을 미리 표시한다. 같은 질문을 TF-IDF, 임베딩, 하이브리드 검색, 리랭킹으로 실행하고, 상위 5개 안에 정답 근거가 있는지와 순위가 어떻게 바뀌는지 비교한다. 전문 수집은 공개 접근성과 이용 조건을 확인한 뒤 별도 단계로 추가한다.
