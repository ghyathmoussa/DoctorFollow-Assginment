"""Hybrid retrieval using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import settings
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.corpus import RetrievalDocument, load_retrieval_documents
from src.retrieval.semantic_retriever import SemanticRetriever


@dataclass(frozen=True)
class HybridRRFResult:
    rank: int
    rrf_score: float
    pmid: str
    title: str
    year: int | None
    journal: str | None
    matched_terms: list[str]
    has_abstract: bool
    bm25_rank: int | None
    semantic_rank: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "rrf_score": self.rrf_score,
            "pmid": self.pmid,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "matched_terms": self.matched_terms,
            "has_abstract": self.has_abstract,
            "bm25_rank": self.bm25_rank,
            "semantic_rank": self.semantic_rank,
        }


def reciprocal_rank_fusion(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


class HybridRRFRetriever:
    """Hybrid retriever that fuses BM25 and semantic rankings with RRF."""

    def __init__(
        self,
        documents: list[RetrievalDocument],
        rrf_k: int | None = None,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        semantic_model_name: str | None = None,
        embeddings_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        self.documents = documents
        self.rrf_k = settings.rrf_k if rrf_k is None else rrf_k
        self.document_by_pmid = {document.pmid: document for document in documents}
        self.bm25 = BM25Retriever(documents, k1=bm25_k1, b=bm25_b)
        self.semantic = SemanticRetriever(
            documents,
            model_name=semantic_model_name,
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
        )

    def search(self, query: str, top_k: int | None = None, candidate_pool: int | None = None) -> list[HybridRRFResult]:
        limit = settings.top_k if top_k is None else top_k
        pool = max(limit, candidate_pool or max(limit * 3, 10))

        bm25_results = self.bm25.search(query, top_k=pool)
        semantic_results = self.semantic.search(query, top_k=pool)

        fused_scores: dict[str, float] = defaultdict(float)
        bm25_ranks: dict[str, int] = {}
        semantic_ranks: dict[str, int] = {}

        for result in bm25_results:
            fused_scores[result.pmid] += reciprocal_rank_fusion(result.rank, self.rrf_k)
            bm25_ranks[result.pmid] = result.rank

        for result in semantic_results:
            fused_scores[result.pmid] += reciprocal_rank_fusion(result.rank, self.rrf_k)
            semantic_ranks[result.pmid] = result.rank

        ranked_pmids = sorted(
            fused_scores,
            key=lambda pmid: (
                fused_scores[pmid],
                -(self.document_by_pmid[pmid].year or 0),
                pmid,
            ),
            reverse=True,
        )

        results: list[HybridRRFResult] = []
        for rank, pmid in enumerate(ranked_pmids[:limit], start=1):
            document = self.document_by_pmid[pmid]
            results.append(
                HybridRRFResult(
                    rank=rank,
                    rrf_score=float(fused_scores[pmid]),
                    pmid=pmid,
                    title=document.title,
                    year=document.year,
                    journal=document.journal,
                    matched_terms=document.matched_terms,
                    has_abstract=document.has_abstract,
                    bm25_rank=bm25_ranks.get(pmid),
                    semantic_rank=semantic_ranks.get(pmid),
                )
            )
        return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid RRF retrieval over the corpus.")
    parser.add_argument("query", type=str, help="Search query.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=settings.retrieval_corpus_path,
        help="Retrieval corpus JSON path.",
    )
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="Number of fused results to return.")
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=max(settings.top_k * 3, 10),
        help="How many results to take from each component retriever before fusing.",
    )
    parser.add_argument("--rrf-k", type=int, default=settings.rrf_k, help="RRF constant k.")
    parser.add_argument("--bm25-k1", type=float, default=settings.bm25_k1, help="BM25 term frequency scaling.")
    parser.add_argument("--bm25-b", type=float, default=settings.bm25_b, help="BM25 document length normalization.")
    parser.add_argument(
        "--model",
        type=str,
        default=settings.embedding_model,
        help="Sentence-transformers model name for semantic retrieval.",
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
    retriever = HybridRRFRetriever(
        documents,
        rrf_k=args.rrf_k,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        semantic_model_name=args.model,
        embeddings_path=args.embeddings_path,
        metadata_path=args.metadata_path,
    )
    results = retriever.search(args.query, top_k=args.top_k, candidate_pool=args.candidate_pool)
    for result in results:
        print(result.to_dict())


if __name__ == "__main__":
    main()
