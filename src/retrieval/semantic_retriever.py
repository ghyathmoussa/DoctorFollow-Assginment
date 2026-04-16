"""Semantic retrieval using sentence-transformers embeddings."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.retrieval.corpus import RetrievalDocument, load_retrieval_documents, normalize_text
from src.utils.io import ensure_parent_dir


@dataclass(frozen=True)
class SemanticSearchResult:
    rank: int
    score: float
    pmid: str
    title: str
    year: int | None
    journal: str | None
    matched_terms: list[str]
    has_abstract: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "pmid": self.pmid,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "matched_terms": self.matched_terms,
            "has_abstract": self.has_abstract,
        }


def _is_e5_model(model_name: str) -> bool:
    return "e5" in model_name.lower()


def _format_document_text(document: RetrievalDocument, model_name: str) -> str:
    text = normalize_text(document.retrieval_text)
    if _is_e5_model(model_name):
        return f"passage: {text}"
    return text


def _format_query_text(query: str, model_name: str) -> str:
    text = normalize_text(query)
    if _is_e5_model(model_name):
        return f"query: {text}"
    return text


class SemanticRetriever:
    """Semantic retriever with cached document embeddings."""

    def __init__(
        self,
        documents: list[RetrievalDocument],
        *,
        model_name: str | None = None,
        embeddings_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        self.documents = documents
        self.model_name = model_name or settings.embedding_model
        self.embeddings_path = embeddings_path or settings.embeddings_path
        self.metadata_path = metadata_path or settings.embeddings_metadata_path
        self.model = self._load_model()
        self.document_embeddings = self._load_or_create_document_embeddings()

    def search(self, query: str, top_k: int | None = None) -> list[SemanticSearchResult]:
        query_text = _format_query_text(query, self.model_name)
        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        scores = self.document_embeddings @ query_embedding
        limit = settings.top_k if top_k is None else top_k
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: (float(scores[idx]), self.documents[idx].year or 0),
            reverse=True,
        )

        results: list[SemanticSearchResult] = []
        for rank, idx in enumerate(ranked_indices[:limit], start=1):
            document = self.documents[idx]
            results.append(
                SemanticSearchResult(
                    rank=rank,
                    score=float(scores[idx]),
                    pmid=document.pmid,
                    title=document.title,
                    year=document.year,
                    journal=document.journal,
                    matched_terms=document.matched_terms,
                    has_abstract=document.has_abstract,
                )
            )
        return results

    def _load_or_create_document_embeddings(self) -> np.ndarray:
        if self._cache_is_usable():
            return np.load(self.embeddings_path)

        texts = [_format_document_text(document, self.model_name) for document in self.documents]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self._write_cache(embeddings)
        return embeddings

    def _load_model(self) -> SentenceTransformer:
        local_snapshot = self._resolve_local_model_snapshot()
        if local_snapshot is not None:
            return SentenceTransformer(str(local_snapshot), local_files_only=True)

        try:
            return SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            return SentenceTransformer(self.model_name)

    def _resolve_local_model_snapshot(self) -> Path | None:
        if os.path.isabs(self.model_name) and Path(self.model_name).exists():
            return Path(self.model_name)

        if "/" not in self.model_name:
            return None

        namespace, repo = self.model_name.split("/", maxsplit=1)
        hub_root = Path.home() / ".cache" / "huggingface" / "hub"
        model_root = hub_root / f"models--{namespace}--{repo}"
        ref_file = model_root / "refs" / "main"
        if not ref_file.exists():
            return None

        snapshot_id = ref_file.read_text(encoding="utf-8").strip()
        snapshot_path = model_root / "snapshots" / snapshot_id
        if snapshot_path.exists():
            return snapshot_path
        return None

    def _cache_is_usable(self) -> bool:
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            return False

        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        expected_pmids = [document.pmid for document in self.documents]
        return (
            metadata.get("model_name") == self.model_name
            and metadata.get("document_count") == len(self.documents)
            and metadata.get("pmids") == expected_pmids
        )

    def _write_cache(self, embeddings: np.ndarray) -> None:
        ensure_parent_dir(self.embeddings_path)
        np.save(self.embeddings_path, embeddings)
        ensure_parent_dir(self.metadata_path)
        metadata = {
            "model_name": self.model_name,
            "document_count": len(self.documents),
            "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "pmids": [document.pmid for document in self.documents],
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic search over the retrieval corpus.")
    parser.add_argument("query", type=str, help="Search query.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=settings.retrieval_corpus_path,
        help="Retrieval corpus JSON path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.embedding_model,
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.top_k,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=settings.embeddings_path,
        help="Path to the cached document embeddings .npy file.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=settings.embeddings_metadata_path,
        help="Path to the embeddings metadata JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    documents = load_retrieval_documents(args.corpus)
    retriever = SemanticRetriever(
        documents,
        model_name=args.model,
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
    )
    results = retriever.search(args.query, top_k=args.top_k)
    for result in results:
        print(result.to_dict())


if __name__ == "__main__":
    main()
