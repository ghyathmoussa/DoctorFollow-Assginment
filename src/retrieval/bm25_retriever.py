"""BM25 retrieval over the prepared retrieval corpus."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:  # pragma: no cover - dependency availability varies by environment
    BM25Okapi = None  # type: ignore[assignment]

from src.config import settings
from src.retrieval.corpus import RetrievalDocument, load_retrieval_documents, normalize_text


TOKEN_PATTERN = re.compile(r"\w+(?:[-']\w+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class BM25SearchResult:
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


def tokenize_text(text: str) -> list[str]:
    lowered = normalize_text(text).lower()
    return TOKEN_PATTERN.findall(lowered)


class _FallbackBM25Okapi:
    """Small BM25 implementation used only when rank_bm25 is unavailable locally."""

    def __init__(self, corpus: list[list[str]], k1: float, b: float) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self._initialize()

    def _initialize(self) -> None:
        term_document_counts: dict[str, int] = {}

        for document in self.corpus:
            freqs: dict[str, int] = {}
            for token in document:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            for token in freqs:
                term_document_counts[token] = term_document_counts.get(token, 0) + 1

        corpus_size = len(self.corpus)
        for token, freq in term_document_counts.items():
            # Standard BM25 idf with +1 inside the log for numerical stability.
            self.idf[token] = math.log(1.0 + (corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.corpus)
        if not query_tokens:
            return scores

        for index, freqs in enumerate(self.doc_freqs):
            doc_len = self.doc_lengths[index]
            norm = 1.0 - self.b + self.b * doc_len / self.avgdl if self.avgdl else 1.0
            score = 0.0
            for token in query_tokens:
                term_freq = freqs.get(token, 0)
                if term_freq == 0:
                    continue
                numerator = term_freq * (self.k1 + 1.0)
                denominator = term_freq + self.k1 * norm
                score += self.idf.get(token, 0.0) * (numerator / denominator)
            scores[index] = score
        return scores


class BM25Retriever:
    """Simple BM25 retriever over retrieval-ready documents."""

    def __init__(
        self,
        documents: list[RetrievalDocument],
        *,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:
        self.documents = documents
        self.k1 = settings.bm25_k1 if k1 is None else k1
        self.b = settings.bm25_b if b is None else b
        self.tokenized_corpus = [tokenize_text(doc.retrieval_text) for doc in documents]
        bm25_cls = BM25Okapi or _FallbackBM25Okapi
        self.backend_name = "rank_bm25" if BM25Okapi is not None else "fallback"
        self.index = bm25_cls(self.tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int | None = None) -> list[BM25SearchResult]:
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        limit = settings.top_k if top_k is None else top_k
        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: (scores[idx], self.documents[idx].year or 0),
            reverse=True,
        )

        results: list[BM25SearchResult] = []
        for rank, idx in enumerate(ranked_indices[:limit], start=1):
            document = self.documents[idx]
            results.append(
                BM25SearchResult(
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 search over the retrieval corpus.")
    parser.add_argument("query", type=str, help="Search query.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=settings.retrieval_corpus_path,
        help="Retrieval corpus JSON path.",
    )
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="Number of results to return.")
    parser.add_argument("--k1", type=float, default=settings.bm25_k1, help="BM25 term frequency scaling.")
    parser.add_argument("--b", type=float, default=settings.bm25_b, help="BM25 document length normalization.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    documents = load_retrieval_documents(args.corpus)
    retriever = BM25Retriever(documents, k1=args.k1, b=args.b)
    results = retriever.search(args.query, top_k=args.top_k)
    for result in results:
        print(result.to_dict())


if __name__ == "__main__":
    main()
