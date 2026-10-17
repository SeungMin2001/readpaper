# 첫 공개 배포: AWS 웹 앱·Qdrant + Mac mini 모델 서버

현재 구성은 웹/API/SQLite/Qdrant를 AWS EC2에 두고, LLM과 E5 임베딩은 사설망을 통해 Mac mini에 요청한다. 상시 서비스가 필요하면 AWS GPU 인스턴스의 모델 서버로 옮길 수 있다. GPU 서버는 실행 시간에 따라 비용이 발생한다.

1. AWS에서 MFA와 예산 알림을 설정한다. Ubuntu EC2 한 대, 도메인, 고정 공인 IP를 준비한다. 리전별 인스턴스·디스크·네트워크 요금을 확인한다.
2. EC2 보안 그룹에는 웹용 TCP 80/443만 공개하고 SSH는 자신의 IP에서만 허용한다. 웹/API는 Docker Compose와 Caddy로 운영한다.
3. Mac mini에서 vLLM-Metal과 [E5 임베딩 서버](../embedding_service/README.md)를 실행한다. EC2와 Mac mini를 Tailscale 같은 사설망에 연결하고, TCP 8001·8002 접근을 앱 서버로 제한한다. 두 서버에 API 키를 설정하고 포트를 공개 인터넷에 열지 않는다. EC2에서 인증된 vLLM `/v1/models`와 임베딩 `/v1/embeddings` 호출로 연결을 확인한다.
4. 저장소의 `.env.example`을 `.env`로 복사한다. `READPAPER_USERNAME`, 긴 `READPAPER_PASSWORD`, 무작위 `READPAPER_SESSION_SECRET`, `READPAPER_LLM_BASE_URL=http://<Mac의 사설 IP>:8001/v1`, `READPAPER_LLM_MODEL=gemma4-12b`, `READPAPER_LLM_API_KEY=<vLLM 키>`, `READPAPER_EMBEDDING_BASE_URL=http://<Mac의 사설 IP>:8002/v1`, `READPAPER_EMBEDDING_API_KEY=<임베딩 키>`, `READPAPER_QDRANT_URL=http://qdrant:6333`, `READPAPER_DOMAIN=<도메인>`, `READPAPER_SECURE_COOKIES=1`, `READPAPER_DEV_MODE=0`을 설정한다. `.env`는 Git에 올리지 않는다.
5. EC2에서 `docker compose -f compose.prod.yaml config`로 설정을 확인하고 `docker compose -f compose.prod.yaml up -d --build`로 실행한다. Qdrant는 Compose의 내부 네트워크에 두고 `qdrant-data` 볼륨에 색인을 보관한다. HTTPS, 로그인, PDF 업로드, 색인 완료, 단어 뜻 생성과 질문·출처 표시를 확인한다. 오류는 `docker compose -f compose.prod.yaml logs --tail=100 web qdrant caddy`로 확인한다.

`/data` Docker 볼륨에는 단어장과 24시간 분석 캐시가 있고, Qdrant 볼륨에는 논문별 벡터 컬렉션이 있다. PDF 원본은 만료 시 삭제한다. Mac mini가 꺼져 있으면 처음 보는 단어 뜻, 새 질문, 색인 준비를 처리할 수 없다. 24시간 운영이 필요하면 AWS GPU 서버에서 모델을 실행하고, 웹 컨테이너는 같은 VPC의 내부 주소로 연결한다. GPU를 계속 켜 두면 해당 시간만큼 비용이 발생한다.

배포 전후에 AWS 예산 알림, HTTPS, 인증, 캐시 만료, 모델 서버 연결 실패와 복구를 확인한다. 로컬 설정만으로 AWS 배포가 완료된 것으로 판단하지 않는다.
