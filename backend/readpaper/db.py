import json
import sqlite3
import shutil
import time
from pathlib import Path
from typing import Any

from .config import settings
from . import embedding_cache


def connect() -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(settings.data_dir / "readpaper.sqlite3", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db() -> None:
    with connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                data_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS vocabulary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_title TEXT NOT NULL,
                term TEXT NOT NULL,
                meaning TEXT NOT NULL,
                source_sentence TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(paper_title, term, source_sentence)
            );
            CREATE TABLE IF NOT EXISTS paper_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                created_at REAL NOT NULL,
                state TEXT NOT NULL,
                message TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS gloss_preparation (
                paper_id TEXT PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
                status TEXT NOT NULL,
                total INTEGER NOT NULL,
                ready INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS term_glosses (
                paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                cache_key TEXT NOT NULL,
                meaning TEXT NOT NULL,
                PRIMARY KEY (paper_id, cache_key)
            );
        """)


def cleanup_expired() -> None:
    embedding_cache.cleanup(settings.data_dir, settings.paper_ttl_seconds)
    with connect() as db:
        cutoff = time.time() - settings.paper_ttl_seconds
        expired = db.execute("SELECT id FROM papers WHERE last_accessed < ?", (cutoff,)).fetchall()
        db.execute("DELETE FROM papers WHERE last_accessed < ?", (cutoff,))
    root = (settings.data_dir / 'papers').resolve()
    for row in expired:
        directory = (root / row['id']).resolve()
        if directory.parent == root:
            shutil.rmtree(directory, ignore_errors=True)


def list_paper_ids() -> set[str]:
    with connect() as db:
        return {row["id"] for row in db.execute("SELECT id FROM papers")}


def fail_interrupted_jobs() -> None:
    with connect() as db:
        db.execute(
            "UPDATE papers SET status = 'failed', "
            "error = '서버 재시작으로 분석이 중단되었습니다. 다시 시도해 주세요.' "
            "WHERE status = 'processing'"
        )
        db.execute(
            "UPDATE gloss_preparation SET status = 'failed', "
            "error = '서버 재시작으로 논문 요약 준비가 중단되었습니다.', updated_at = ? "
            "WHERE status IN ('queued', 'running')",
            (time.time(),),
        )


def get_gloss_preparation(paper_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute(
            "SELECT status, total, ready, failed, error FROM gloss_preparation "
            "WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
    return dict(row) if row else None


def claim_gloss_preparation(paper_id: str, total: int, *, force: bool = False) -> bool:
    """Only one request may launch or retry a preparation job."""
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT status FROM gloss_preparation WHERE paper_id = ?", (paper_id,),
        ).fetchone()
        if row and row["status"] != "failed" and not force:
            return False
        db.execute(
            "INSERT INTO gloss_preparation "
            "(paper_id, status, total, ready, failed, error, updated_at) "
            "VALUES (?, 'queued', ?, 0, 0, NULL, ?) "
            "ON CONFLICT(paper_id) DO UPDATE SET "
            "status = 'queued', total = excluded.total, ready = 0, failed = 0, "
            "error = NULL, updated_at = excluded.updated_at",
            (paper_id, total, time.time()),
        )
    return True


def update_gloss_preparation(paper_id: str, *, status: str | None = None,
                             ready_delta: int = 0, failed_delta: int = 0,
                             error: str | None = None) -> None:
    fields = ["updated_at = ?", "ready = ready + ?", "failed = failed + ?"]
    values: list[Any] = [time.time(), ready_delta, failed_delta]
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if error is not None:
        fields.append("error = ?")
        values.append(error[:300])
    values.append(paper_id)
    with connect() as db:
        db.execute(
            f"UPDATE gloss_preparation SET {', '.join(fields)} WHERE paper_id = ?",
            values,
        )


def create_paper(paper_id: str, title: str, payload: dict[str, Any]) -> None:
    now = time.time()
    with connect() as db:
        db.execute(
            "INSERT INTO papers (id, title, status, created_at, last_accessed, data_json) "
            "VALUES (?, ?, 'processing', ?, ?, ?)",
            (paper_id, title, now, now, json.dumps(payload, ensure_ascii=False)),
        )


def get_paper(paper_id: str, touch: bool = True) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if row is None:
            return None
        if touch:
            db.execute("UPDATE papers SET last_accessed = ? WHERE id = ?", (time.time(), paper_id))
        item = dict(row)
        item["data"] = json.loads(item.pop("data_json"))
        return item


def set_paper(paper_id: str, *, status: str | None = None, progress: int | None = None,
              error: str | None = None, payload: dict[str, Any] | None = None) -> None:
    fields: list[str] = []
    values: list[Any] = []
    for name, value in (("status", status), ("progress", progress), ("error", error)):
        if value is not None:
            fields.append(f"{name} = ?")
            values.append(value)
    if payload is not None:
        fields.append("data_json = ?")
        values.append(json.dumps(payload, ensure_ascii=False))
    if not fields:
        return
    values.append(paper_id)
    with connect() as db:
        db.execute(f"UPDATE papers SET {', '.join(fields)} WHERE id = ?", values)


def merge_paper_data(paper_id: str, updates: dict[str, Any]) -> bool:
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT data_json FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if row is None:
            return False
        data = json.loads(row["data_json"])
        data.update(updates)
        db.execute(
            "UPDATE papers SET data_json = ? WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), paper_id),
        )
    return True


def get_term_gloss(paper_id: str, cache_key: str) -> str | None:
    with connect() as db:
        row = db.execute(
            "SELECT meaning FROM term_glosses WHERE paper_id = ? AND cache_key = ?",
            (paper_id, cache_key),
        ).fetchone()
    return row["meaning"] if row else None


def save_word_gloss(paper_id: str, word_ids: list[str], gloss: dict[str, Any],
                    *, cache_key: str | None = None) -> bool:
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT data_json FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if row is None:
            return False
        data = json.loads(row["data_json"])
        word_glosses = data.setdefault("word_glosses", {})
        for word_id in word_ids:
            word_glosses[word_id] = gloss
        db.execute(
            "UPDATE papers SET data_json = ? WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), paper_id),
        )
        if cache_key is not None:
            db.execute(
                "INSERT INTO term_glosses (paper_id, cache_key, meaning) VALUES (?, ?, ?) "
                "ON CONFLICT(paper_id, cache_key) DO UPDATE SET meaning = excluded.meaning",
                (paper_id, cache_key, gloss["meaning"]),
            )
    return True


def clear_paper_activity(paper_id: str) -> None:
    with connect() as db:
        db.execute("DELETE FROM paper_activity WHERE paper_id = ?", (paper_id,))


def add_paper_activity(paper_id: str, message: str, state: str = "info") -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO paper_activity (paper_id, created_at, state, message) VALUES (?, ?, ?, ?)",
            (paper_id, time.time(), state, message[:500]),
        )
        db.execute(
            "DELETE FROM paper_activity WHERE paper_id = ? AND id NOT IN "
            "(SELECT id FROM paper_activity WHERE paper_id = ? ORDER BY id DESC LIMIT 80)",
            (paper_id, paper_id),
        )


def list_paper_activity(paper_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT id, created_at, state, message FROM ("
            "SELECT id, created_at, state, message FROM paper_activity "
            "WHERE paper_id = ? ORDER BY id DESC LIMIT 80) ORDER BY id ASC",
            (paper_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def find_cached_paper(content_hash: str) -> dict[str, Any] | None:
    with connect() as db:
        rows = db.execute("SELECT id, data_json FROM papers WHERE status = 'ready'").fetchall()
    for row in rows:
        payload = json.loads(row["data_json"])
        if payload.get("content_hash") == content_hash:
            return get_paper(row["id"])
    return None


def list_vocabulary() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM vocabulary ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def save_vocabulary(paper_title: str, term: str, meaning: str, source_sentence: str) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO vocabulary (paper_title, term, meaning, source_sentence, created_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(paper_title, term, source_sentence) "
            "DO UPDATE SET meaning = excluded.meaning",
            (paper_title[:300], term[:150], meaning[:150], source_sentence[:1000], time.time()),
        )


def delete_vocabulary(item_id: int) -> bool:
    with connect() as db:
        cursor = db.execute("DELETE FROM vocabulary WHERE id = ?", (item_id,))
        return cursor.rowcount > 0
