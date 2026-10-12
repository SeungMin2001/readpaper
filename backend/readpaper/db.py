import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .config import settings


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
        """)


def cleanup_expired() -> None:
    with connect() as db:
        db.execute(
            "DELETE FROM papers WHERE last_accessed < ?",
            (time.time() - settings.paper_ttl_seconds,),
        )


def fail_interrupted_jobs() -> None:
    with connect() as db:
        db.execute(
            "UPDATE papers SET status = 'failed', "
            "error = '서버 재시작으로 분석이 중단되었습니다. 다시 시도해 주세요.' "
            "WHERE status = 'processing'"
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
