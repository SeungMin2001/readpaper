# 첫 공개 배포: AWS 웹 앱 + vLLM

현재 구성은 웹/API/SQLite를 AWS EC2에 두고, LLM은 사설망을 통해 별도 vLLM 서버에 요청한다. 처음에는 Mac mini의 vLLM-Metal을 연결할 수 있고, 상시 서비스가 필요하면 AWS GPU 인스턴스의 vLLM으로 옮긴다. GPU 서버는 실행 시간에 따라 비용이 발생한다.

1. AWS에서 MFA와 예산 알림을 설정한다. Ubuntu EC2 한 대, 도메인, 고정 공인 IP를 준비한다. 리전별 인스턴스·디스크·네트워크 요금을 확인한다.
2. EC2 보안 그룹에는 웹용 TCP 80/443만 공개하고 SSH는 자신의 IP에서만 허용한다. 웹/API는 Docker Compose와 Caddy로 운영한다.
3. Mac mini에서 vLLM-Metal을 실행한다. EC2와 Mac mini를 Tailscale 같은 사설망에 연결하고, TCP 8001 접근은 앱 서버로 제한한다. vLLM에 API 키를 설정하고 모델 서버 포트를 공개 인터넷에 열지 않는다. 서버에서 `curl -H 'Authorization: Bearer <키>' http://<Mac의 사설 IP>:8001/v1/models`로 연결을 확인한다.
4. 저장소의 `.env.example`을 `.env`로 복사한다. `READPAPER_USERNAME`, 긴 `READPAPER_PASSWORD`, 무작위 `READPAPER_SESSION_SECRET`, `READPAPER_LLM_BASE_URL=http://<모델 서버의 사설 IP>:8001/v1`, `READPAPER_LLM_MODEL=gemma4-12b`, `READPAPER_LLM_API_KEY=<vLLM 키>`, `READPAPER_DOMAIN=<도메인>`, `READPAPER_SECURE_COOKIES=1`, `READPAPER_DEV_MODE=0`을 설정한다. `.env`는 Git에 올리지 않는다.
5. EC2에서 `docker compose -f compose.prod.yaml config`로 설정을 확인하고 `docker compose -f compose.prod.yaml up -d --build`로 실행한다. HTTPS, 로그인, PDF 업로드, 단어 뜻 생성과 질문·출처 표시를 확인한다. 오류는 `docker compose -f compose.prod.yaml logs --tail=100 web caddy`로 확인한다.

`/data` Docker 볼륨에는 단어장과 24시간 분석 캐시가 있다. PDF 원본은 보관하지 않는다. Mac mini가 꺼져 있으면 처음 보는 단어 뜻과 새 질문을 생성할 수 없다. 24시간 운영이 필요하면 AWS GPU 서버에서 vLLM을 실행하고, 웹 컨테이너는 같은 VPC의 내부 주소로 연결한다. GPU를 계속 켜 두면 해당 시간만큼 비용이 발생한다.

배포 전후에 AWS 예산 알림, HTTPS, 인증, 캐시 만료, 모델 서버 연결 실패와 복구를 확인한다. 로컬 설정만으로 AWS 배포가 완료된 것으로 판단하지 않는다.
