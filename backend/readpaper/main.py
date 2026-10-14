import asyncio
import base64
import hashlib
import hmac
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import db
from .config import settings
from .llm import InvalidModelOutput, ModelUnavailable, OllamaClient
from .pdf import extract_pdf
from .rag import build_chunks, retrieve
from .sources import collect_external_sources


app = FastAPI(title="Readpaper API", version="0.1.0")


class LoginRequest(BaseModel):
    username: str
    password: str


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class VocabularyRequest(BaseModel):
    paper_title: str = Field(min_length=1, max_length=300)
    term: str = Field(min_length=1, max_length=150)
    meaning: str = Field(min_length=1, max_length=150)
    source_sentence: str = Field(default="", max_length=1000)


def _sign_session(username: str) -> str:
    expires = int(time.time()) + 7 * 24 * 60 * 60
    payload = f"{username}:{expires}".encode()
    signature = hmac.new(settings.session_secret.encode(), payload, hashlib.sha256).digest()
    return (
        base64.urlsafe_b64encode(payload).decode()
        + "."
        + base64.urlsafe_b64encode(signature).decode()
    )


def _session_valid(token: str | None) -> bool:
    if not token or not settings.session_secret:
        return False
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        payload = base64.urlsafe_b64decode(encoded_payload.encode())
        signature = base64.urlsafe_b64decode(encoded_signature.encode())
        expected = hmac.new(settings.session_secret.encode(), payload, hashlib.sha256).digest()
        username, expires = payload.decode().rsplit(":", 1)
        return (
            hmac.compare_digest(signature, expected)
            and hmac.compare_digest(username, settings.username)
            and int(expires) > time.time()
        )
    except (ValueError, UnicodeDecodeError):
        return False


def require_user(request: Request) -> str:
    if settings.dev_mode:
        return "developer"
    if not _session_valid(request.cookies.get("readpaper_session")):
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return settings.username


_cleanup_task: asyncio.Task[None] | None = None


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(60 * 60)
        await asyncio.to_thread(db.cleanup_expired)


@app.on_event("startup")
async def startup() -> None:
    global _cleanup_task
    if not settings.dev_mode and not all((settings.username, settings.password, settings.session_secret)):
        raise RuntimeError("READPAPER_USERNAME, READPAPER_PASSWORD, READPAPER_SESSION_SECRET이 필요합니다.")
    db.init_db()
    db.cleanup_expired()
    db.fail_interrupted_jobs()
    _cleanup_task = asyncio.create_task(_cleanup_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/login")
def login(credentials: LoginRequest, response: Response) -> dict[str, str]:
    if settings.dev_mode:
        return {"username": "developer"}
    if not (
        hmac.compare_digest(credentials.username, settings.username)
        and hmac.compare_digest(credentials.password, settings.password)
    ):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
    response.set_cookie(
        "readpaper_session", _sign_session(settings.username),
        httponly=True, secure=settings.secure_cookies,
        samesite="strict", max_age=7 * 24 * 60 * 60,
    )
    return {"username": settings.username}


@app.post("/api/auth/logout")
def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie("readpaper_session")
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: str = Depends(require_user)) -> dict[str, str]:
    return {"username": user}


async def process_paper(paper_id: str) -> None:
    paper = db.get_paper(paper_id, touch=False)
    if paper is None:
        return
    payload = paper["data"]
    try:
        model = OllamaClient()
        db.set_paper(paper_id, status="processing", progress=3, error="")
        profile = await model.profile(payload["title"], payload["full_text"])
        payload["profile"] = profile
        db.set_paper(paper_id, progress=8)
        external, warnings = await collect_external_sources(
            payload["title"], payload.get("doi"), profile["keywords"]
        )
        related_candidates = [
            source for source in external if source.get("kind") in {"related", "recent"}
        ]
        selected_related = await model.select_related_papers(profile, related_candidates, limit=8)
        external = [
            source for source in external if source.get("kind") not in {"related", "recent"}
        ] + selected_related
        payload["external_sources"] = external
        payload["source_warnings"] = warnings
        db.set_paper(paper_id, progress=12)

        words = {word["id"]: word for page in payload["pages"] for word in page["words"]}
        translations: dict[str, Any] = {}
        sentences = payload["sentences"]
        for index, sentence in enumerate(sentences):
            tokens = [words[word_id]["text"] for word_id in sentence["word_ids"]]
            result = await model.translate_sentence(
                sentence, tokens, profile,
                sentences[index - 1]["text"] if index else "",
                sentences[index + 1]["text"] if index + 1 < len(sentences) else "",
            )
            units = []
            for unit in result["units"]:
                unit_word_ids = sentence["word_ids"][unit["start"]:unit["end"]]
                units.append({
                    "word_ids": unit_word_ids,
                    "original": " ".join(words[word_id]["text"] for word_id in unit_word_ids),
                    "meaning": unit["meaning"],
                })
            translations[sentence["id"]] = {
                "translation": result["translation"], "units": units,
            }
            if index % 3 == 0 or index + 1 == len(sentences):
                db.set_paper(paper_id, progress=12 + int(86 * (index + 1) / len(sentences)))

        payload["translations"] = translations
        payload["chunks"] = build_chunks(sentences, external, payload["title"])
        db.set_paper(paper_id, payload=payload, status="ready", progress=100, error="")
    except (ModelUnavailable, InvalidModelOutput) as exc:
        db.set_paper(paper_id, status="failed", error=str(exc))
    except Exception:
        db.set_paper(paper_id, status="failed", error="논문 분석 중 오류가 발생했습니다. 서버 로그를 확인하세요.")
        raise


@app.post("/api/papers/upload")
async def upload_paper(background: BackgroundTasks, file: UploadFile = File(...),
                       _user: str = Depends(require_user)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일을 선택하세요.")
    data = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="PDF는 25MB 이하로 업로드하세요.")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="유효한 PDF 파일이 아닙니다.")
    db.cleanup_expired()
    content_hash = hashlib.sha256(data).hexdigest()
    cached = db.find_cached_paper(content_hash)
    if cached:
        return {"paper_id": cached["id"], "status": "ready", "cached": True}
    try:
        parsed = extract_pdf(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    parsed["content_hash"] = content_hash
    paper_id = uuid.uuid4().hex
    db.create_paper(paper_id, parsed["title"], parsed)
    background.add_task(process_paper, paper_id)
    return {"paper_id": paper_id, "status": "processing", "cached": False}


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: str, _user: str = Depends(require_user)) -> dict[str, Any]:
    db.cleanup_expired()
    paper = db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="논문 분석 결과가 만료되었거나 없습니다.")
    response = {
        "id": paper_id, "title": paper["title"], "status": paper["status"],
        "progress": paper["progress"], "error": paper["error"],
    }
    if paper["status"] == "ready":
        data = paper["data"]
        response.update({
            "pages": data["pages"], "sentences": data["sentences"],
            "translations": data["translations"], "profile": data["profile"],
            "sources": [
                {key: value for key, value in source.items() if key != "text"}
                for source in data["external_sources"]
            ],
            "source_warnings": data["source_warnings"],
        })
    return response


@app.post("/api/papers/{paper_id}/retry")
def retry_paper(paper_id: str, background: BackgroundTasks,
                _user: str = Depends(require_user)) -> dict[str, str]:
    paper = db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="논문 분석 결과가 없습니다.")
    if paper["status"] != "failed":
        raise HTTPException(status_code=409, detail="실패한 분석만 다시 시도할 수 있습니다.")
    db.set_paper(paper_id, status="processing", progress=0, error="")
    background.add_task(process_paper, paper_id)
    return {"status": "processing"}


@app.post("/api/papers/{paper_id}/ask")
async def ask_paper(paper_id: str, body: QuestionRequest,
                    _user: str = Depends(require_user)) -> dict[str, Any]:
    paper = db.get_paper(paper_id)
    if paper is None or paper["status"] != "ready":
        raise HTTPException(status_code=404, detail="읽을 준비가 된 논문이 없습니다.")
    model = OllamaClient()
    try:
        english_query = await model.search_query(body.question)
        contexts = retrieve(english_query, paper["data"]["chunks"])
        if not contexts:
            return {"answer": "수집한 자료에서 이 질문의 근거를 찾지 못했습니다.", "citations": [], "sources": []}
        result = await model.answer(body.question, contexts)
        selected = [
            {key: value for key, value in source.items() if key != "text"}
            | {"excerpt": source["text"][:350]}
            for source in contexts if source["id"] in result["citations"]
        ]
        return result | {"sources": selected}
    except (ModelUnavailable, InvalidModelOutput) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/vocabulary")
def vocabulary(_user: str = Depends(require_user)) -> list[dict[str, Any]]:
    return db.list_vocabulary()


@app.post("/api/vocabulary", status_code=201)
def add_vocabulary(body: VocabularyRequest,
                   _user: str = Depends(require_user)) -> dict[str, bool]:
    db.save_vocabulary(body.paper_title, body.term, body.meaning, body.source_sentence)
    return {"ok": True}


@app.delete("/api/vocabulary/{item_id}")
def remove_vocabulary(item_id: int, _user: str = Depends(require_user)) -> dict[str, bool]:
    if not db.delete_vocabulary(item_id):
        raise HTTPException(status_code=404, detail="단어를 찾을 수 없습니다.")
    return {"ok": True}


_frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str) -> FileResponse:
    candidate = (_frontend / path).resolve()
    if candidate.is_file() and candidate.is_relative_to(_frontend.resolve()):
        return FileResponse(candidate)
    index = _frontend / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="프론트엔드를 먼저 빌드하세요.")
