# RAG 수집 기준선 점검

## 실행 방법

저장소 루트에서 실행한다. 첫 명령은 논문이 인용한 연구와 OpenAlex 관련 연구 후보를 확인한다. 참고문헌은 OpenAlex가 제공한 목록 전체, 관련 후보는 최대 20편까지 요청한다. `--full-text`는 아래 단계에서 받은 초록·배경 설명 텍스트 전체를 JSON에 넣는 옵션이며 외부 논문 PDF 전문을 뜻하지 않는다.

```bash
./.venv/bin/python backend/scripts/inspect_sources.py \
  --title "Attention Is All You Need" \
  --doi 10.48550/arXiv.1706.03762 \
  --keywords transformer "attention mechanism" \
  --full-text \
  --json-out data/inspection/attention-is-all-you-need.json
```

JSON에는 수집 경로, 연도·날짜, DOI/OpenAlex ID, OA 상태와 공개 PDF 위치, 초록/메타데이터 구분, 초록 미리보기, 경고가 나온다. `--full-text`는 현재 수집기가 가져오는 초록·배경 설명 전체를 넣는다. 결과 파일과 청킹 실험 데이터는 저장소의 `data/` 아래에서 관리하며 Git에는 올리지 않는다.

## 공개 전문을 실제로 가져오는 테스트

두 번째 명령은 후보 목록에서 참고·관련 유형별 최대 2편씩 골라 OpenAlex가 공개 접근 PDF로 표시한 URL만 시도한다. PDF 텍스트를 추출하고 첫 페이지 제목이 예상 논문 제목과 맞는지 확인한 뒤 통과한 PDF와 `.txt`만 `data/papers/collected/`에 저장한다. 결과 보고서는 전문을 얻지 못한 사유까지 포함해 `data/inspection/`에 기록한다. 현재는 OpenAlex가 직접 PDF URL을 제공하지 않는 후보의 랜딩 페이지를 탐색하지 않으므로, 그런 후보가 곧 전문 접근 불가라는 뜻은 아니다.

```bash
./.venv/bin/python backend/scripts/fetch_open_access_fulltext.py \
  --title "Attention Is All You Need" \
  --doi "10.48550/arXiv.1706.03762" \
  --keywords transformer "attention mechanism" \
  --per-kind-limit 2 \
  --json-out data/inspection/attention-is-all-you-need-fulltext.json
```

전문 추출 결과와 수집 성공·실패를 확인한다.

```bash
jq '{relationship_coverage, discovered_counts, fulltext_status_counts, attempts: [.fulltext_attempts[] | {kind, title, fulltext_status, pdf_path, text_path, extracted_chars, pdf_first_page_preview}]}' \
  data/inspection/attention-is-all-you-need-fulltext.json
```

`text_path`에 나온 `.txt` 파일을 VS Code에서 열어 실제 본문이 맞는지 확인한다. 직접 PDF URL이 없거나 다운로드가 실패하면 그 상태를 보고서에 남긴다. PDF에 텍스트 레이어가 없는 경우 OCR은 하지 않는다. 이 단계는 다운로드와 텍스트 추출을 검증하는 용도이며 아직 청킹·임베딩·RAG 인덱싱은 하지 않는다.

## 실제 API 점검 기록

- 입력: *Attention Is All You Need*, DOI `10.48550/arXiv.1706.03762`; 키워드 `transformer`, `attention mechanism`.
- OpenAlex와 Wikipedia가 실제 응답을 반환했고 수집기 경고는 없었다.
- 초기 기준선 수집기는 참고문헌을 8편, OpenAlex 관련 후보를 4편으로 제한했다. 이 제한을 제거하기 전 실행에서는 참고 8편, 관련 3편, 이 논문을 인용한 논문 4편, 최근 관련 1편, Wikipedia 배경 자료 2건이 나왔다.
- 가장 최근 후보는 2026년 *ViXNet: Vision Transformer with Xception Network for Facial Expression Recognition*이었다. 최근 Transformer 응용 연구로는 관련이 있지만, 원 논문인 자연어 처리와 분야가 다르다.
- OpenAlex의 related 결과에는 BERT와 함께 논문 주제에서 멀어 보이는 언어학·실험 자동화 논문도 포함됐다. 인용 관계는 실제 연결을 뜻하지만 모든 인용 논문이 질문에 유용한 것은 아니다.
- *Attention Is All You Need*의 OpenAlex 응답은 참고문헌 ID 28개와 관련 후보 20개를 제공했다. 배치 서지 조회에서 참고문헌 21개, 관련 후보 중 중복을 제외한 11개가 반환됐고, 관계 중복을 포함해 관련 ID 15개가 메타데이터에서 확인됐다.
- 참고문헌 2편과 관련 후보 2편의 전문을 시험했다. OpenAlex에 직접 PDF URL이 등록되지 않은 후보 2편, 텍스트를 추출한 후보 1편, 다른 논문의 PDF가 연결돼 제목 검증에서 거른 후보 1편으로 나왔다. 성공한 텍스트는 `data/papers/collected/`에 PDF와 TXT로 저장했다. 직접 PDF URL이 없다는 결과는 공개 랜딩 페이지에도 전문이 없다는 뜻은 아니다.
- 질문 `multi-head self attention architecture`를 현재 TF-IDF 검색기에 넣었을 때 `Self-training PCFG grammars...`가 반환됐다. `self`라는 단어가 겹친 결과로, 관련도 오탐이다. TF-IDF 점수와 검색 로그를 아직 UI/API에서 확인할 수 없다.
- 첫 수집 실행에서 OpenAlex의 미래 날짜 논문(2027)이 최근 목록에 들어오는 것을 발견했다. 최근 검색에 오늘 날짜 상한을 추가했고, 수집 테스트에서 미래 날짜 항목을 제외하는 것을 확인했다.

## 지금 알 수 있는 점과 다음 검증

현재 수집기는 DOI 기반 원 논문 조회, OpenAlex에 등록된 참고문헌 목록, 관련 후보를 가져온다. 관련 후보는 OpenAlex가 주는 최대 20편으로 한정된다. 이 사례에서는 OpenAlex가 알려준 참고문헌 28개 중 21개만 메타데이터를 찾았다. 보고서의 `unresolved_openalex_ids`에서 못 찾은 ID를 확인할 수 있다. 관련 순서만으로 학술 주제가 맞다고 판단할 수 없고, OpenAlex의 PDF URL과 논문 정보가 서로 맞지 않을 수도 있어 제목 검증을 거친다. 공개 전문 접근 성공 여부와 추출 텍스트를 먼저 확인한 뒤 청킹을 진행한다.

다음에는 `unresolved_openalex_ids`의 원인을 확인하고, 직접 PDF 주소가 없는 후보의 공개 랜딩 페이지에서 PDF 링크를 찾을 수 있는지 검사한다. 제목 검증을 통과한 TXT를 직접 살펴본 다음, 페이지·절 출처를 붙여 청킹하고 청크별로 검색 품질을 확인한다. 임베딩과 리랭킹은 이 기준선 결과가 준비된 뒤 비교한다.
