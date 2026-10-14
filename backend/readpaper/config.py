import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv(
        "READPAPER_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data" / "runtime")
    ))
    ollama_url: str = os.getenv("READPAPER_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model: str = os.getenv("READPAPER_OLLAMA_MODEL", "qwen3:8b")
    openalex_url: str = os.getenv("READPAPER_OPENALEX_URL", "https://api.openalex.org").rstrip("/")
    wikipedia_url: str = os.getenv("READPAPER_WIKIPEDIA_URL", "https://en.wikipedia.org/w/api.php")
    max_upload_bytes: int = 25 * 1024 * 1024
    paper_ttl_seconds: int = 24 * 60 * 60
    dev_mode: bool = os.getenv("READPAPER_DEV_MODE") == "1"
    username: str = os.getenv("READPAPER_USERNAME", "")
    password: str = os.getenv("READPAPER_PASSWORD", "")
    session_secret: str = os.getenv("READPAPER_SESSION_SECRET", "")
    secure_cookies: bool = os.getenv("READPAPER_SECURE_COOKIES") == "1"


settings = Settings()
