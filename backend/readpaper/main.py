import asyncio
import base64
import hashlib
import hmac
import logging
import math
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import pymupdf

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import db
from .config import settings
from .llm import InvalidModelOutput, ModelUnavailable, VLLMClient, close_model_transport, start_model_transport
from .pdf import extract_pdf
from .rag import INDEX_VERSION, build_chunks, build_index, retrieve
from .fulltext import collect_fulltext
from .sources import SOURCE_DISCOVERY_VERSION, collect_external_sources
from .terms import gloss_cache_key, resolve_term


app = FastAPI(title="Readpaper API", version="0.1.0")
logger = logging.getLogger(__name__)

_gloss_locks: dict[tuple[str, str], asyncio.Lock] = {}
_gloss_lock_users: dict[tuple[str, str], int] = {}
_gloss_locks_guard = asyncio.Lock()


@asynccontextmanager
async def _gloss_request_lock(key: tuple[str, str]):
    """Deduplicate identical model inputs, including terms at different locations."""
    async with _gloss_locks_guard:
        lock = _gloss_locks.setdefault(key, asyncio.Lock())
        _gloss_lock_users[key] = _gloss_lock_users.get(key, 0) + 1
    try:
        async with lock:
            yield
    finally:
        async with _gloss_locks_guard:
            users = _gloss_lock_users[key] - 1
            if users:
                _gloss_lock_users[key] = users
            else:
                _gloss_lock_users.pop(key, None)
                _gloss_locks.pop(key, None)


class LoginRequest(BaseModel):
    username: str
    password: str


class Attachment(BaseModel):
    kind: Literal['text', 'figure']
    page: int = Field(ge=1, le=100)
    word_ids: list[str] = Field(default_factory=list, max_length=300)
    rect: list[float] = Field(default_factory=list, max_length=4)


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(max_length=2000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    attachments: list[Attachment] = Field(default_factory=list, max_length=4)
    history: list[ChatMessage] = Field(default_factory=list, max_length=6)


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
    await start_model_transport()
    _cleanup_task = asyncio.create_task(_cleanup_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    await close_model_transport()
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
            reference_text=payload.get("full_text", ""),
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
            "서지 조회 완료: 참고문헌 {referenced}편 · 관련 연구 {related}편 · 피인용 {citing}편".format(**source_counts),
            "success",
        )
        for warning in warnings:
            db.add_paper_activity(paper_id, warning, "warning")

        db.set_paper(paper_id, progress=50)
        directory = settings.data_dir / 'papers' / paper_id
        external = await collect_fulltext(external, directory / 'sources', report_collection_event)
        db.merge_paper_data(paper_id, {'external_sources': external})
        db.set_paper(paper_id, progress=75)
        db.add_paper_activity(paper_id, '본문 청킹 및 다국어 임베딩 생성 중')
        chunks = build_chunks(payload["sentences"], external, payload["title"])
        db.merge_paper_data(paper_id, {"chunks": chunks})
        def report_index_progress(done: int, total: int) -> None:
            db.add_paper_activity(paper_id, f'임베딩 생성 {done}/{total}개')
            db.set_paper(paper_id, progress=75 + int(24 * done / total))
        index = await asyncio.to_thread(build_index, chunks, directory / 'index', report_index_progress)
        index['fulltext_documents'] = sum(source.get('fulltext_status') == 'ready' for source in external)
        db.merge_paper_data(paper_id, {
            'rag_index': index, 'source_discovery_version': SOURCE_DISCOVERY_VERSION,
        })
        db.add_paper_activity(paper_id, f"검색 색인 저장: {len(chunks)}개 청크 · {index['dimensions']}차원", 'success')
        db.set_paper(paper_id, status="ready", progress=100, error="")
        db.add_paper_activity(paper_id, "원문·문헌·검색 문맥 준비 완료.", "success")
    except Exception:
        db.set_paper(paper_id, status="failed", error="논문 분석 중 오류가 발생했습니다. 서버 로그를 확인하세요.")
        db.add_paper_activity(paper_id, "분석 중 예기치 못한 오류가 발생했습니다", "error")
        raise


async def prepare_paper_summary(paper_id: str) -> None:
    """Create and persist one compact paper summary before opening the reader."""
    db.update_gloss_preparation(paper_id, status="running")
    paper = db.get_paper(paper_id, touch=False)
    if paper is None:
        db.update_gloss_preparation(paper_id, status="failed", error="논문을 찾을 수 없습니다.")
        return
    data = paper["data"]
    if not data.get("paper_summary"):
        try:
            summary = await VLLMClient().summarize_paper(
                paper["title"], _paper_summary_source(data),
            )
            data["paper_summary"] = summary
        except (ModelUnavailable, InvalidModelOutput) as exc:
            db.update_gloss_preparation(paper_id, status="failed", error=str(exc))
            return
        except Exception:
            logger.exception("Paper summary failed for paper %s", paper_id)
            db.update_gloss_preparation(
                paper_id, status="failed", error="논문 핵심 요약 중 오류가 발생했습니다.",
            )
            return
    if data.get("gloss_context_version") != "paper_summary_v1":
        db.merge_paper_data(paper_id, {
            "paper_summary": data["paper_summary"],
            "gloss_context_version": "paper_summary_v1",
        })
    db.update_gloss_preparation(paper_id, status="completed")


async def prepare_new_paper(paper_id: str) -> None:
    # Source discovery and the one-time summary can run independently.
    results = await asyncio.gather(
        process_paper(paper_id), prepare_paper_summary(paper_id),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            logger.error("Paper background job failed for %s: %s", paper_id, result)


def _paper_summary_source(data: dict[str, Any]) -> str:
    sentences = data.get("sentences", [])
    body = []
    for sentence in sentences:
        text = sentence.get("text", "").strip()
        if re.fullmatch(r"(?:references|bibliography|works cited)", text, re.IGNORECASE):
            break
        if text:
            body.append(text)
    if len(body) > 24:
        selected = sorted({round(index * (len(body) - 1) / 23) for index in range(24)})
        body = [body[index] for index in selected]
    return "\n".join(body)[:9000]


def _start_gloss_preparation(paper_id: str, data: dict[str, Any],
                             background: BackgroundTasks) -> None:
    status = db.get_gloss_preparation(paper_id)
    legacy_context = data.get("gloss_context_version") != "paper_summary_v1"
    legacy_glosses = legacy_context and bool(data.get("word_glosses"))
    if legacy_glosses:
        # Older cached translations were generated from sentence passages, so
        # clear them before preparing the same terms with the compact summary.
        db.merge_paper_data(paper_id, {"word_glosses": {}})
    force = legacy_context and bool(
        status and status["status"] not in {"queued", "running"}
    )
    if db.claim_gloss_preparation(paper_id, 0, force=force):
        background.add_task(prepare_paper_summary, paper_id)


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
        directory = settings.data_dir / 'papers' / cached['id']
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'original.pdf').write_bytes(data)
        old_empty_sources = (not cached['data'].get('external_sources') and
                             cached['data'].get('source_discovery_version') != SOURCE_DISCOVERY_VERSION)
        needs_index = (cached['data'].get('rag_index', {}).get('version') != INDEX_VERSION or
                       cached['data'].get('rag_index', {}).get('model') != settings.embedding_model or
                       not (directory / 'index' / 'vectors.npy').is_file())
        if old_empty_sources or needs_index:
            db.set_paper(cached['id'], status='processing', progress=0)
            background.add_task(process_paper, cached['id'])
        _start_gloss_preparation(cached["id"], cached["data"], background)
        return {"paper_id": cached["id"], "status": "processing" if old_empty_sources or needs_index else "ready", "cached": True}
    try:
        parsed = extract_pdf(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    parsed["content_hash"] = content_hash
    parsed["chunks"] = build_chunks(parsed["sentences"], [], parsed["title"])
    paper_id = uuid.uuid4().hex
    directory = settings.data_dir / 'papers' / paper_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'original.pdf').write_bytes(data)
    db.create_paper(paper_id, parsed["title"], parsed)
    db.claim_gloss_preparation(paper_id, 0)
    background.add_task(prepare_new_paper, paper_id)
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
    response["rag_index"] = data.get("rag_index")
    if sources_ready:
        response.update({
            "sources": [
                {key: value for key, value in source.items() if key not in {"text", "sentences"}}
                for source in data["external_sources"]
            ],
            "source_warnings": data.get("source_warnings", []),
        })
    response.update({
        "pages": data["pages"], "sentences": data["sentences"],
        "word_glosses": data.get("word_glosses", {}),
    })
    return response


def _gloss_preparation_response(paper_id: str) -> dict[str, Any]:
    paper = db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")
    status = db.get_gloss_preparation(paper_id)
    preparation = status or {"status": "pending", "total": 0, "ready": 0,
                             "failed": 0, "error": None}
    current = paper["data"].get("paper_summary")
    if preparation["status"] == "running":
        stage = "summarizing"
    else:
        stage = preparation["status"]
    return preparation | {
        "stage": stage,
        "summary": current,
        "word_glosses": paper["data"].get("word_glosses", {}),
    }


@app.get("/api/papers/{paper_id}/gloss-preparation")
def get_gloss_preparation(paper_id: str,
                          _user: str = Depends(require_user)) -> dict[str, Any]:
    return _gloss_preparation_response(paper_id)


@app.post("/api/papers/{paper_id}/gloss-preparation")
def retry_gloss_preparation(paper_id: str, background: BackgroundTasks,
                            _user: str = Depends(require_user)) -> dict[str, Any]:
    paper = db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")
    _start_gloss_preparation(paper_id, paper["data"], background)
    return _gloss_preparation_response(paper_id)


@app.post("/api/papers/{paper_id}/retry")
def retry_paper(paper_id: str, background: BackgroundTasks,
                _user: str = Depends(require_user)) -> dict[str, str]:
    paper = db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="논문 분석 결과가 없습니다.")
    if paper["status"] == "processing":
        raise HTTPException(status_code=409, detail="문헌 수집이 이미 진행 중입니다.")
    db.set_paper(paper_id, status="processing", progress=0, error="")
    background.add_task(process_paper, paper_id)
    return {"status": "processing"}


@app.post("/api/papers/{paper_id}/translate-word")
async def translate_word(paper_id: str, body: WordTranslationRequest,
                         _user: str = Depends(require_user)) -> dict[str, Any]:
    return await _translate_word(paper_id, body.word_id)


async def _translate_word(paper_id: str, word_id: str) -> dict[str, Any]:
    started = time.monotonic()
    paper = db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="논문을 찾을 수 없습니다.")
    data = paper["data"]
    known_words = {
        word["id"]: word for page in data.get("pages", []) for word in page.get("words", [])
    }
    word = known_words.get(word_id)
    if word is None:
        raise HTTPException(status_code=404, detail="선택한 단어를 논문에서 찾을 수 없습니다.")
    sentence = next(
        (item for item in data.get("sentences", []) if item["id"] == word["sentence_id"]), None,
    )
    if sentence is None:
        raise HTTPException(status_code=422, detail="단어가 속한 문장을 찾을 수 없습니다.")
    tokens = [known_words[item]["text"] for item in sentence["word_ids"]]
    selected = resolve_term(tokens, sentence["word_ids"].index(word_id))
    paper_summary = data.get("paper_summary")
    if not paper_summary:
        raise HTTPException(status_code=503, detail="논문 핵심 요약을 준비하고 있습니다. 잠시 후 다시 시도해 주세요.")
    key = gloss_cache_key(
        paper_summary, selected["term"], model=settings.llm_model, endpoint=settings.llm_base_url,
    )
    word_ids = sentence["word_ids"][selected["start"]:selected["end"]]
    context_ms = (time.monotonic() - started) * 1000
    cached = data.get("word_glosses", {}).get(word_id)
    meaning = cached["meaning"] if cached and cached.get("translation_key") == key else None
    cache_source = "position" if meaning else "term"
    if meaning is None:
        meaning = db.get_term_gloss(paper_id, key)
    model_metrics: dict[str, Any] = {}
    lock_wait_ms = model_request_ms = 0.0
    if meaning is None:
        lock_started = time.monotonic()
        async with _gloss_request_lock((paper_id, key)):
            lock_wait_ms = (time.monotonic() - lock_started) * 1000
            meaning = db.get_term_gloss(paper_id, key)
            cache_source = "shared_request" if meaning is not None else "miss"
            if meaning is None:
                model = VLLMClient()
                model_started = time.monotonic()
                try:
                    result = await model.translate_keyword_from_summary(paper_summary, selected["term"])
                    meaning = result.get("meaning")
                    if not isinstance(meaning, str) or not meaning.strip() or len(meaning) > 100:
                        raise InvalidModelOutput("용어 뜻이 없거나 너무 깁니다.")
                    meaning = meaning.strip()
                except (ModelUnavailable, InvalidModelOutput) as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                model_request_ms = (time.monotonic() - model_started) * 1000
                model_metrics = getattr(model, "last_metrics", {})
            unit = {
                "word_ids": word_ids, "original": selected["term"], "meaning": meaning,
                "sentence_id": sentence["id"], "context_source": "paper_summary",
                "translation_key": key,
            }
            if not db.save_word_gloss(paper_id, word_ids, unit, cache_key=key):
                raise HTTPException(status_code=404, detail="논문이 만료되었습니다.")
    else:
        # Bind a shared meaning to this occurrence, never to an earlier page's word IDs.
        unit = {
            "word_ids": word_ids, "original": selected["term"], "meaning": meaning,
            "sentence_id": sentence["id"], "context_source": "paper_summary",
            "translation_key": key,
        }
        if cache_source != "position" and not db.save_word_gloss(paper_id, word_ids, unit):
            raise HTTPException(status_code=404, detail="논문이 만료되었습니다.")
    if cache_source == "miss":
        db.add_paper_activity(
            paper_id, f"용어 뜻 생성: {unit['original']} · {model_request_ms / 1000:.1f}초", "success",
        )
    activity = db.list_paper_activity(paper_id)
    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    return unit | {
        "cached": cache_source != "miss", "elapsed_ms": elapsed_ms,
        "timing": {
            "provider": "vllm", "model": settings.llm_model, "cache": cache_source,
            "server_ms": elapsed_ms, "context_ms": round(context_ms, 2),
            "lock_wait_ms": round(lock_wait_ms, 2), "model_request_ms": round(model_request_ms, 2),
            "provider_request_ms": model_metrics.get("request_ms"),
            "first_token_ms": model_metrics.get("first_token_ms"),
            "after_first_token_ms": model_metrics.get("after_first_token_ms"),
            "prompt_tokens": model_metrics.get("prompt_tokens"),
            "generated_tokens": model_metrics.get("completion_tokens"),
            "cached_prompt_tokens": model_metrics.get("cached_prompt_tokens"),
            "reasoning_tokens": model_metrics.get("reasoning_tokens"),
        },
        "activity": activity,
    }


def selection_contexts(paper: dict, attachments: list[Attachment]) -> list[dict]:
    contexts = []
    if sum(item.kind == 'figure' for item in attachments) > 2:
        raise HTTPException(422, '그림은 한 번에 2개까지 첨부할 수 있습니다.')
    pages = paper['data']['pages']
    for index, attachment in enumerate(attachments):
        if attachment.page > len(pages):
            raise HTTPException(422, '선택한 페이지가 없습니다.')
        page = pages[attachment.page - 1]
        by_id = {word['id']: word for word in page['words']}
        if any(word_id not in by_id for word_id in attachment.word_ids):
            raise HTTPException(422, '선택한 단어가 이 페이지에 없습니다.')
        words = [word for word in page['words'] if word['id'] in attachment.word_ids]
        context = {'id': f'selection-{index}', 'document_id': 'paper', 'kind': 'paper',
                   'title': paper['title'], 'page': attachment.page, 'url': None,
                   'content_scope': 'full_text'}
        if attachment.kind == 'figure':
            rect = attachment.rect
            if (len(rect) != 4 or not all(math.isfinite(v) for v in rect)
                    or not 0 <= rect[0] < rect[2] <= page['width']
                    or not 0 <= rect[1] < rect[3] <= page['height']
                    or rect[2] - rect[0] < 5 or rect[3] - rect[1] < 5):
                raise HTTPException(422, '올바른 그림 영역을 선택해 주세요.')
            path = settings.data_dir / 'papers' / paper['id'] / 'original.pdf'
            if not path.exists():
                raise HTTPException(409, '그림 첨부를 위해 PDF를 다시 업로드해 주세요.')
            with pymupdf.open(path) as document:
                scale = min(1.5, 1024 / max(rect[2]-rect[0], rect[3]-rect[1]))
                pix = document[attachment.page - 1].get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=pymupdf.Rect(rect))
                context['image_data'] = 'data:image/png;base64,' + base64.b64encode(pix.tobytes('png')).decode()
            words = [word for word in page['words'] if rect[0] <= (word['x0']+word['x1'])/2 <= rect[2]
                     and rect[1] <= (word['y0']+word['y1'])/2 <= rect[3]]
            context['rect'] = rect
        if not words and attachment.kind == 'text':
            raise HTTPException(422, '첨부할 단어나 문장을 선택해 주세요.')
        words.sort(key=lambda word: (int(word['sentence_id'][1:]), int(word['id'][1:])))
        context['text'] = ' '.join(word['text'] for word in words) or '사용자가 선택한 그림 영역 (이미지 참조)'
        context['sentence_ids'] = list(dict.fromkeys(word['sentence_id'] for word in words))
        contexts.append(context)
    return contexts


@app.get('/api/papers/{paper_id}/sources/{source_id}/pdf')
def source_pdf(paper_id: str, source_id: str, _user: str = Depends(require_user)):
    paper = db.get_paper(paper_id)
    if not paper:
        raise HTTPException(404, '논문이 없습니다.')
    directory = settings.data_dir / 'papers' / paper_id
    if source_id == 'paper':
        path = directory / 'original.pdf'
    else:
        source = next((item for item in paper['data'].get('external_sources', []) if item['id'] == source_id), None)
        if not source or not source.get('pdf_file'):
            raise HTTPException(404, '수집된 PDF가 없습니다.')
        path = directory / 'sources' / Path(source['pdf_file']).name
    if not path.is_file():
        raise HTTPException(404, 'PDF가 없습니다.')
    return FileResponse(path, media_type='application/pdf')


@app.post("/api/papers/{paper_id}/ask")
async def ask_paper(paper_id: str, body: QuestionRequest,
                    _user: str = Depends(require_user)) -> dict[str, Any]:
    paper = db.get_paper(paper_id)
    if paper is None:
        raise HTTPException(404, '논문이 없습니다.')
    if paper['status'] != 'ready' or not paper['data'].get('rag_index'):
        raise HTTPException(409, '문헌 검색 준비를 완료한 뒤 질문해 주세요.')
    model = VLLMClient()
    selected_context = selection_contexts(paper, body.attachments)
    history = [{'role': message.role, 'content': message.content[:800]} for message in body.history[-4:]]
    # History is conversational context, never indexed evidence.
    import json
    question = json.dumps({'question': body.question, 'history': history,
                           'selected_text': [item['text'][:300] for item in selected_context]}, ensure_ascii=False)
    try:
        english_query = await model.search_query(question)
        contexts = await asyncio.to_thread(retrieve, english_query + ' ' + body.question,
                                           paper['data']['chunks'], 6,
                                           settings.data_dir / 'papers' / paper_id / 'index')
        contexts = selected_context + contexts[:6 - len(selected_context)]
        if not contexts:
            return {'answer': '수집한 자료에서 이 질문의 근거를 찾지 못했습니다.', 'citations': [], 'sources': []}
        result = await model.answer(question, contexts)
        selected = []
        for source in contexts:
            if source['id'] not in result['citations']:
                continue
            item = {key: value for key, value in source.items() if key not in {'text', 'image_data'}}
            item['excerpt'] = source['text']
            document_id = source.get('document_id', '')
            external = next((s for s in paper['data'].get('external_sources', []) if s['id'] == document_id), {})
            if document_id == 'paper' or external.get('pdf_file'):
                item['url'] = f'/api/papers/{paper_id}/sources/{document_id}/pdf' + (f"#page={source['page']}" if source.get('page') else '')
            selected.append(item)
        return result | {'sources': selected}
    except (ModelUnavailable, InvalidModelOutput, ValueError, OSError) as exc:
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
