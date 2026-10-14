import asyncio
import base64
import hashlib
import hmac
import re
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

_ollama_lock = asyncio.Lock()


class LoginRequest(BaseModel):
    username: str
    password: str


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class WordTranslationRequest(BaseModel):
    word_id: str = Field(min_length=1, max_length=40)


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


def _word_context(sentences: list[dict[str, Any]], word: str, sentence_id: str,
                  chunks: list[dict[str, Any]]) -> list[str]:
    """Gather compact evidence from the clicked sentence and the full paper."""
    selected = next((index for index, item in enumerate(sentences) if item["id"] == sentence_id), None)
    if selected is None:
        return []

    contexts = [sentences[selected]["text"]]
    normalized = re.sub(r"^\W+|\W+$", "", word).casefold()
    matches = []
    for index, sentence in enumerate(sentences):
        if index == selected:
            continue
        if any(re.sub(r"^\W+|\W+$", "", token).casefold() == normalized
               for token in sentence["text"].split()):
            matches.append(index)

    # Spread repeated occurrences across the paper to preserve document-level sense.
    if len(matches) > 3:
        matches = [matches[round(offset * (len(matches) - 1) / 2)] for offset in range(3)]
    contexts.extend(sentences[index]["text"] for index in matches)

    # Add a few lexically related passages when the term appears only once.
    if len(contexts) < 4:
        paper_chunks = [chunk for chunk in chunks if chunk.get("kind") == "paper"]
        related = retrieve(f"{word} {sentences[selected]['text']}", paper_chunks, limit=5)
        for item in related:
            excerpt = item.get("text", "")
            if excerpt and excerpt not in contexts:
                contexts.append(excerpt)
            if len(contexts) == 4:
                break
    return [context[:900] for context in contexts[:4]]


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
        db.clear_paper_activity(paper_id)
        db.set_paper(paper_id, status="processing", progress=3, error="")
        db.add_paper_activity(
            paper_id,
            f"PDF 읽기 완료: {len(payload['pages'])}페이지, {len(payload['sentences'])}문장",
            "success",
        )

        async def report_collection_event(message: str, state: str = "info") -> None:
            db.add_paper_activity(paper_id, message, state)

        # Paper reading and local RAG preparation must not depend on an LLM call.
        external, warnings = await collect_external_sources(
            payload["title"], payload.get("doi"), [], on_event=report_collection_event,
        )
        # OpenAlex returns related_works ordered by its own relevance signal.
        # Use the first eight as an explicit, non-LLM baseline for this stage.
        related_count = 0
        discovered = []
        for source in external:
            if source.get("kind") == "related":
                if related_count >= 8:
                    continue
                related_count += 1
            discovered.append(source)
        external = discovered
        db.merge_paper_data(paper_id, {
            "external_sources": external,
            "source_warnings": warnings,
        })
        source_counts = {
            kind: sum(source.get("kind") == kind for source in external)
            for kind in ("referenced", "related", "citing")
        }
        db.add_paper_activity(
            paper_id,
            "수집 완료: 참고문헌 {referenced}편 · 관련 연구 {related}편 · 피인용 {citing}편".format(**source_counts),
            "success",
        )
        for warning in warnings:
            db.add_paper_activity(paper_id, warning, "warning")

        db.set_paper(paper_id, progress=75)
        chunks = build_chunks(payload["sentences"], external, payload["title"])
        db.merge_paper_data(paper_id, {"chunks": chunks})
        db.set_paper(paper_id, status="ready", progress=100, error="")
        db.add_paper_activity(paper_id, "원문·문헌·검색 문맥 준비 완료. 용어 뜻은 클릭할 때 생성합니다.", "success")
    except Exception:
        db.set_paper(paper_id, status="failed", error="논문 분석 중 오류가 발생했습니다. 서버 로그를 확인하세요.")
        db.add_paper_activity(paper_id, "분석 중 예기치 못한 오류가 발생했습니다", "error")
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
        "activity": db.list_paper_activity(paper_id),
    }
    data = paper["data"]
    sources_ready = "external_sources" in data
    response["sources_ready"] = sources_ready
    if sources_ready:
        response.update({
            "sources": [
                {key: value for key, value in source.items() if key != "text"}
                for source in data["external_sources"]
            ],
            "source_warnings": data.get("source_warnings", []),
        })
    response.update({
        "pages": data["pages"], "sentences": data["sentences"],
        "word_glosses": data.get("word_glosses", {}),
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


@app.post("/api/papers/{paper_id}/translate-word")
async def translate_word(paper_id: str, body: WordTranslationRequest,
                         _user: str = Depends(require_user)) -> dict[str, Any]:
    paper = db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")
    data = paper["data"]
    known_words = {
        word["id"]: word for page in data.get("pages", []) for word in page.get("words", [])
    }
    word = known_words.get(body.word_id)
    if word is None:
        raise HTTPException(status_code=404, detail="선택한 단어를 논문에서 찾을 수 없습니다.")

    cached = data.get("word_glosses", {}).get(body.word_id)
    if cached:
        return cached | {
            "cached": True, "elapsed_ms": 0,
            "activity": db.list_paper_activity(paper_id),
        }

    sentences = data.get("sentences", [])
    sentence_index = next(
        (index for index, item in enumerate(sentences) if item["id"] == word["sentence_id"]), None,
    )
    if sentence_index is None:
        raise HTTPException(status_code=422, detail="단어가 속한 문장을 찾을 수 없습니다.")
    sentence = sentences[sentence_index]
    tokens = [known_words[word_id]["text"] for word_id in sentence["word_ids"]]
    focus_index = sentence["word_ids"].index(body.word_id)
    context_started = time.monotonic()
    passages = _word_context(sentences, word["text"], sentence["id"], data.get("chunks", []))
    context_ms = round((time.monotonic() - context_started) * 1000)

    started = time.monotonic()
    async with _ollama_lock:
        lock_wait_ms = round((time.monotonic() - started) * 1000)
        # Another click may have completed while this request waited for the local model.
        latest = db.get_paper(paper_id, touch=False)
        cached = (latest or {}).get("data", {}).get("word_glosses", {}).get(body.word_id)
        if cached:
            return cached | {
                "cached": True, "elapsed_ms": 0,
                "activity": db.list_paper_activity(paper_id),
            }
        model = OllamaClient()
        try:
            result = await model.translate_keyword(
                data["title"], sentence["text"], tokens, focus_index, passages,
            )
        except (ModelUnavailable, InvalidModelOutput) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        model_request_ms = round((time.monotonic() - started) * 1000) - lock_wait_ms
        model_metrics = getattr(model, "last_metrics", {})

        start, end = result.get("start"), result.get("end")
        meaning = result.get("meaning")
        if (type(start) is not int or type(end) is not int or start < 0 or end <= start
                or end > len(tokens) or not start <= focus_index < end
                or not isinstance(meaning, str) or not meaning.strip() or len(meaning) > 100):
            raise HTTPException(status_code=502, detail="용어 번역 결과를 검증할 수 없습니다.")

        word_ids = sentence["word_ids"][start:end]
        clean_term = [re.sub(r"^[\W_]+|[\W_]+$", "", token) for token in tokens[start:end]]
        unit = {
            "word_ids": word_ids,
            "original": " ".join(token for token in clean_term if token),
            "meaning": meaning.strip(),
            "sentence_id": sentence["id"],
        }
        db.save_word_gloss(paper_id, word_ids, unit)

    elapsed_ms = round((time.monotonic() - started) * 1000)
    db.add_paper_activity(
        paper_id, f"용어 뜻 생성: {unit['original']} · {elapsed_ms / 1000:.1f}초", "success",
    )
    return unit | {
        "cached": False, "elapsed_ms": elapsed_ms, "context_count": len(passages),
        "timing": {
            "model": settings.ollama_gloss_model,
            "context_ms": context_ms,
            "lock_wait_ms": lock_wait_ms,
            "model_request_ms": model_request_ms,
            "ollama_total_ms": round(model_metrics["total_duration"] / 1_000_000)
            if "total_duration" in model_metrics else None,
            "ollama_load_ms": round(model_metrics["load_duration"] / 1_000_000)
            if "load_duration" in model_metrics else None,
            "prompt_eval_ms": round(model_metrics["prompt_eval_duration"] / 1_000_000)
            if "prompt_eval_duration" in model_metrics else None,
            "generation_ms": round(model_metrics["eval_duration"] / 1_000_000)
            if "eval_duration" in model_metrics else None,
            "prompt_tokens": model_metrics.get("prompt_eval_count"),
            "generated_tokens": model_metrics.get("eval_count"),
        },
        "activity": db.list_paper_activity(paper_id),
    }


@app.post("/api/papers/{paper_id}/ask")
async def ask_paper(paper_id: str, body: QuestionRequest,
                    _user: str = Depends(require_user)) -> dict[str, Any]:
    paper = db.get_paper(paper_id)
    if paper is None or paper["status"] != "ready":
        raise HTTPException(status_code=404, detail="읽을 준비가 된 논문이 없습니다.")
    model = OllamaClient()
    try:
        async with _ollama_lock:
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
