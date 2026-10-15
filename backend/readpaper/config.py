import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv(
        "READPAPER_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data" / "runtime")
    ))
    llm_base_url: str = os.getenv(
        "READPAPER_LLM_BASE_URL", "http://127.0.0.1:8001/v1"
    ).rstrip("/")
    llm_model: str = os.getenv("READPAPER_LLM_MODEL", "gemma4-12b")
    llm_api_key: str = os.getenv("READPAPER_LLM_API_KEY", "")
    openalex_url: str = os.getenv("READPAPER_OPENALEX_URL", "https://api.openalex.org").rstrip("/")
    openalex_api_key: str = os.getenv("READPAPER_OPENALEX_API_KEY", "")
    unpaywall_email: str = os.getenv("READPAPER_UNPAYWALL_EMAIL", "")
    crossref_email: str = os.getenv("READPAPER_CROSSREF_EMAIL", "")
    wikipedia_url: str = os.getenv("READPAPER_WIKIPEDIA_URL", "https://en.wikipedia.org/w/api.php")
    embedding_model: str = os.getenv("READPAPER_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    max_upload_bytes: int = 25 * 1024 * 1024
    paper_ttl_seconds: int = 24 * 60 * 60
    dev_mode: bool = os.getenv("READPAPER_DEV_MODE") == "1"
    username: str = os.getenv("READPAPER_USERNAME", "")
    password: str = os.getenv("READPAPER_PASSWORD", "")
    session_secret: str = os.getenv("READPAPER_SESSION_SECRET", "")
    secure_cookies: bool = os.getenv("READPAPER_SECURE_COOKIES") == "1"


settings = Settings()
