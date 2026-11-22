from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OMNIRAG_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data/local")
    auth_enabled: bool = False
    serve_frontend: bool = False
    frontend_dist: Path = Path("frontend/dist")
    ollama_base_url: HttpUrl = HttpUrl("http://127.0.0.1:11434")
    ollama_model: str = "qwen3-vl:2b-instruct"
    # Coding assistant / general chat: plain-text generation with one of the two
    # existing provider integrations. "ollama" is the local gateway above;
    # "openrouter" is the OpenRouter client ported from the RAG project.
    code_provider: Literal["ollama", "openrouter"] = "ollama"
    ollama_code_model: str = ""
    openrouter_base_url: HttpUrl = HttpUrl("https://openrouter.ai/api/v1")
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-5.2"
    openrouter_http_referer: str = ""
    openrouter_title: str = ""
    code_max_output_tokens: int = Field(default=2048, ge=256, le=8192)
    url_fetch_timeout_seconds: int = Field(default=20, ge=5, le=60)
    url_fetch_max_mb: int = Field(default=5, ge=1, le=25)
    # "grounded" writes answers with validated citations; "extractive" replies with
    # verbatim selected excerpts only (the Phase 5-11 behavior, kept selectable).
    answer_mode: Literal["grounded", "extractive"] = "grounded"
    model_timeout_seconds: int = Field(default=120, ge=5, le=600)
    context_tokens: int = Field(default=4096, ge=2048, le=8192)
    max_output_tokens: int = Field(default=768, ge=128, le=2048)
    top_k: int = Field(default=4, ge=1, le=5)
    upload_max_mb: int = Field(default=10, ge=1, le=50)
    max_pages: int = Field(default=100, ge=1, le=300)
    ingest_timeout_seconds: int = Field(default=90, ge=5, le=300)
    ocr_enabled: bool = True
    ocr_max_pages: int = Field(default=20, ge=1, le=100)
    ocr_page_timeout_seconds: int = Field(default=20, ge=1, le=60)
    retrieval_enabled: bool = False
    default_retrieval_mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "bm25"
    retrieval_models_dir: Path = Path("data/models")
    retrieval_manifest: Path = Path("models/retrieval.lock.json")
    retrieval_candidates: int = Field(default=12, ge=5, le=30)
    retrieval_threads: int = Field(default=2, ge=1, le=4)

    @property
    def vectors(self) -> Path:
        return self.data_dir.resolve() / "qdrant"

    @property
    def database(self) -> Path:
        return self.data_dir.resolve() / "omnirag.sqlite3"

    @property
    def uploads(self) -> Path:
        return self.data_dir.resolve() / "uploads"

    def prepare(self) -> None:
        self.uploads.mkdir(parents=True, exist_ok=True)
