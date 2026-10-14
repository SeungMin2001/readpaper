# 첫 공개 배포: AWS 서버 + Mac mini Ollama

이 안내는 웹/API/SQLite를 AWS EC2 한 대에 두고, Mac mini에서만 Ollama를 실행하는 개인용 첫 버전용이다. 트래픽이나 동시 분석이 늘면 작업 대기열과 관리형 저장소로 분리한다. 실제 배포에는 AWS 계정, 본인 도메인, Mac mini와 서버의 비공개 네트워크 연결이 필요하다.

1. AWS 계정에서 MFA와 예산 알림을 설정한다. EC2 인스턴스 한 대(예: Ubuntu LTS, `t3.medium` 이상, 20GB 이상 EBS)를 만들고 도메인의 A 레코드를 그 인스턴스의 고정 공인 IP에 연결한다. 인스턴스, EBS, 고정 IP, 네트워크 전송, 도메인에 비용이 발생할 수 있으므로 AWS 요금 계산기에서 리전별 가격을 먼저 확인한다.
2. 보안 그룹의 인바운드는 웹용 TCP 80/443만 전체에 열고, SSH를 쓴다면 자신의 IP에서만 허용한다. Ollama의 11434 포트는 공개하지 않는다. 서버에 Docker Engine과 Compose 플러그인, Tailscale을 설치한다. Mac mini에도 Tailscale을 설치해 같은 tailnet에 로그인한다. tailnet 정책에서 서버가 Mac의 Ollama 포트에만 접근하도록 제한한다.
3. Mac mini에서 모델을 준비한다. 현재 설정 예시는 `ollama pull gemma4:12b`이다. Ollama가 tailnet에서만 접근 가능한 주소에 바인딩되도록 설정하고, 서버에서 `curl http://<Mac의 tailnet IP>:11434/api/tags`로 연결을 확인한다. 서버 컨테이너에서도 같은 요청을 해 본다. 연결 실패 시 Tailscale ACL, Mac 방화벽, Ollama 리스닝 주소, 서버의 Docker→tailnet 라우팅을 차례로 확인한다. Tailscale 계정 키나 Ollama 포트를 공개 저장소에 넣지 않는다.
4. 서버에 이 저장소를 체크아웃하고 `.env.example`을 `.env`로 복사한다. `READPAPER_USERNAME`, 긴 `READPAPER_PASSWORD`, 무작위 `READPAPER_SESSION_SECRET`, `READPAPER_OLLAMA_URL=http://<Mac의 tailnet IP>:11434`, `READPAPER_DOMAIN=<본인 도메인>`, `READPAPER_SECURE_COOKIES=1`, `READPAPER_DEV_MODE=0`을 설정한다. `.env`는 Git에 올리지 않는다.
5. 도메인 DNS가 준비된 뒤 `docker compose -f compose.prod.yaml config`로 설정을 확인하고 `docker compose -f compose.prod.yaml up -d --build`로 실행한다. Caddy가 80/443에서 HTTPS 인증서를 발급하고 웹 컨테이너로 요청을 전달한다. `https://<도메인>/api/health`, 로그인, 샘플 PDF 업로드, 단어 클릭, 질문·출처 표시를 실제 Mac 모델로 확인한다. `docker compose -f compose.prod.yaml logs --tail=100 web caddy`로 오류를 본다.

`/data` Docker 볼륨에는 단어장과 24시간 분석 캐시가 있다. 운영 중 서버를 교체하기 전에는 볼륨을 백업한다. PDF 원본은 보관하지 않는다. OpenAlex/Wikipedia에는 PDF 파일을 보내지 않고 논문 제목·DOI·키워드만 조회한다. Mac mini가 꺼져도 이미 생성된 용어 뜻은 캐시 만료 전까지 볼 수 있지만, 처음 보는 용어의 뜻과 새 질문을 생성하려면 연결이 필요하다.

배포가 끝난 뒤에는 예산 알림, HTTPS, 인증, 캐시 만료, Mac 연결 실패 안내와 재시도를 실제 서비스에서 점검한다. 이 저장소의 로컬 테스트와 Docker 검증만으로 AWS 배포가 완료되었다고 판단하지 않는다.
