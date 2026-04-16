"""Centralized application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for bootstrap environments
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    medical_terms_path: Path
    artifacts_dir: Path
    corpus_path: Path
    retrieval_corpus_path: Path
    evaluation_path: Path
    rag_demos_path: Path
    embeddings_path: Path
    embeddings_metadata_path: Path
    llm_provider: str
    llm_api_key: str
    llm_model: str
    llm_timeout_seconds: int
    llm_retry_attempts: int
    llm_retry_backoff_seconds: float
    llm_min_interval_seconds: float
    rag_abstract_char_limit: int
    pubmed_email: str
    pubmed_tool_name: str
    pubmed_rate_limit_per_sec: int
    pubmed_timeout_seconds: int
    pubmed_max_results: int
    pubmed_retry_attempts: int
    pubmed_retry_backoff_seconds: float
    embedding_model: str
    bm25_k1: float
    bm25_b: float
    rrf_k: int
    top_k: int


def get_settings() -> Settings:
    artifacts_dir = PROJECT_ROOT / _get_env_str("ARTIFACTS_DIR", "artifacts")

    return Settings(
        project_root=PROJECT_ROOT,
        medical_terms_path=PROJECT_ROOT / _get_env_str("MEDICAL_TERMS_PATH", "medical_terms.csv"),
        artifacts_dir=artifacts_dir,
        corpus_path=PROJECT_ROOT / _get_env_str("CORPUS_PATH", "artifacts/corpus.json"),
        retrieval_corpus_path=PROJECT_ROOT / _get_env_str(
            "RETRIEVAL_CORPUS_PATH", "artifacts/retrieval_corpus.json"
        ),
        evaluation_path=PROJECT_ROOT / _get_env_str("EVALUATION_PATH", "artifacts/evaluation.json"),
        rag_demos_path=PROJECT_ROOT / _get_env_str("RAG_DEMOS_PATH", "artifacts/rag_demos.json"),
        embeddings_path=PROJECT_ROOT / _get_env_str("EMBEDDINGS_PATH", "artifacts/document_embeddings.npy"),
        embeddings_metadata_path=PROJECT_ROOT
        / _get_env_str("EMBEDDINGS_METADATA_PATH", "artifacts/document_embeddings_metadata.json"),
        llm_provider=_get_env_str("LLM_PROVIDER", "gemini"),
        llm_api_key=_get_env_str("LLM_API_KEY", ""),
        llm_model=_get_env_str("LLM_MODEL", "gemini-2.0-flash"),
        llm_timeout_seconds=_get_env_int("LLM_TIMEOUT_SECONDS", 90),
        llm_retry_attempts=_get_env_int("LLM_RETRY_ATTEMPTS", 4),
        llm_retry_backoff_seconds=_get_env_float("LLM_RETRY_BACKOFF_SECONDS", 2.0),
        llm_min_interval_seconds=_get_env_float("LLM_MIN_INTERVAL_SECONDS", 6.0),
        rag_abstract_char_limit=_get_env_int("RAG_ABSTRACT_CHAR_LIMIT", 1200),
        pubmed_email=_get_env_str("PUBMED_EMAIL", ""),
        pubmed_tool_name=_get_env_str("PUBMED_TOOL_NAME", "doctor-follow-assessment"),
        pubmed_rate_limit_per_sec=_get_env_int("PUBMED_RATE_LIMIT_PER_SEC", 3),
        pubmed_timeout_seconds=_get_env_int("PUBMED_TIMEOUT_SECONDS", 30),
        pubmed_max_results=_get_env_int("PUBMED_MAX_RESULTS", 5),
        pubmed_retry_attempts=_get_env_int("PUBMED_RETRY_ATTEMPTS", 3),
        pubmed_retry_backoff_seconds=_get_env_float("PUBMED_RETRY_BACKOFF_SECONDS", 1.5),
        embedding_model=_get_env_str("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        bm25_k1=_get_env_float("BM25_K1", 1.5),
        bm25_b=_get_env_float("BM25_B", 0.75),
        rrf_k=_get_env_int("RRF_K", 60),
        top_k=_get_env_int("TOP_K", 5),
    )


settings = get_settings()
