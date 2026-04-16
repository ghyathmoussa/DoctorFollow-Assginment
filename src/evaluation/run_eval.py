"""Run retrieval evaluation across BM25, semantic, and hybrid methods."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import settings
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.corpus import RetrievalDocument, load_retrieval_documents, normalize_text
from src.retrieval.hybrid_rrf import HybridRRFRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.utils.io import write_json


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    text: str
    expected_terms: list[str]
    high_relevance_keywords: list[str]


EVALUATION_QUERIES: list[EvaluationQuery] = [
    EvaluationQuery(
        query_id="q1",
        text="What are the latest guidelines for managing type 2 diabetes?",
        expected_terms=["type 2 diabetes mellitus"],
        high_relevance_keywords=["guideline", "guidelines", "management", "treatment", "therapy", "consensus"],
    ),
    EvaluationQuery(
        query_id="q2",
        text="Çocuklarda akut otitis media tedavisi nasıl yapılır?",
        expected_terms=["acute otitis media"],
        high_relevance_keywords=["pediatric", "children", "treatment", "therapy", "management", "otitis"],
    ),
    EvaluationQuery(
        query_id="q3",
        text="Iron supplementation dosing for anemia during pregnancy",
        expected_terms=["iron deficiency anemia"],
        high_relevance_keywords=["iron", "anemia", "pregnancy", "dosing", "supplementation", "ferrous"],
    ),
    EvaluationQuery(
        query_id="q4",
        text="Çölyak hastalığı tanı kriterleri nelerdir?",
        expected_terms=["celiac disease diagnosis"],
        high_relevance_keywords=["celiac", "coeliac", "diagnosis", "diagnostic", "criteria", "screening"],
    ),
    EvaluationQuery(
        query_id="q5",
        text="Antibiotic resistance patterns in community acquired pneumonia",
        expected_terms=["community acquired pneumonia"],
        high_relevance_keywords=["antibiotic", "resistance", "community-acquired pneumonia", "pneumonia", "pathogen"],
    ),
]


def precision_at_k(relevances: list[int], k: int = 5) -> float:
    window = relevances[:k]
    if not window:
        return 0.0
    return sum(1 for rel in window if rel > 0) / len(window)


def ndcg_at_k(relevances: list[int], ideal_relevances: list[int], k: int = 5) -> float:
    window = relevances[:k]
    if not window:
        return 0.0

    dcg = 0.0
    for index, rel in enumerate(window, start=1):
        gain = (2**rel - 1) / math.log2(index + 1)
        dcg += gain

    ideal = sorted(ideal_relevances, reverse=True)[:k]
    idcg = 0.0
    for index, rel in enumerate(ideal, start=1):
        gain = (2**rel - 1) / math.log2(index + 1)
        idcg += gain

    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_relevance(query: EvaluationQuery, document: RetrievalDocument) -> int:
    normalized_terms = {normalize_text(term).lower() for term in document.matched_terms}
    expected_terms = {normalize_text(term).lower() for term in query.expected_terms}
    if not normalized_terms.intersection(expected_terms):
        return 0

    text = normalize_text(f"{document.title} {document.abstract}").lower()
    keyword_hits = sum(1 for keyword in query.high_relevance_keywords if keyword.lower() in text)
    if keyword_hits >= 2:
        return 2
    return 1


def _result_to_payload(result: Any, relevance: int) -> dict[str, Any]:
    payload = result.to_dict()
    payload["relevance"] = relevance
    return payload


def run_evaluation(
    documents: list[RetrievalDocument],
    *,
    top_k: int = 5,
    candidate_pool: int = 10,
) -> dict[str, Any]:
    bm25 = BM25Retriever(documents)
    semantic = SemanticRetriever(documents)
    hybrid = HybridRRFRetriever(documents)
    document_by_pmid = {document.pmid: document for document in documents}

    methods = {
        "bm25": lambda text: bm25.search(text, top_k=top_k),
        "semantic": lambda text: semantic.search(text, top_k=top_k),
        "hybrid_rrf": lambda text: hybrid.search(text, top_k=top_k, candidate_pool=candidate_pool),
    }

    evaluation_results: dict[str, Any] = {
        "queries": [],
        "aggregate_metrics": {},
        "metric_notes": {
            "judgment_method": "bootstrap_heuristic",
            "relevance_scale": "0=not relevant, 1=topically relevant, 2=highly relevant",
            "precision_at_5": "Counts results with relevance > 0 in the top 5.",
            "ndcg_at_5": "Uses graded relevance to reward highly relevant results appearing earlier.",
        },
    }

    per_method_precision: dict[str, list[float]] = {name: [] for name in methods}
    per_method_ndcg: dict[str, list[float]] = {name: [] for name in methods}

    for query in EVALUATION_QUERIES:
        corpus_relevances = [bootstrap_relevance(query, document) for document in documents]
        query_payload: dict[str, Any] = {
            "query_id": query.query_id,
            "text": query.text,
            "expected_terms": query.expected_terms,
            "methods": {},
        }

        for method_name, runner in methods.items():
            results = runner(query.text)
            relevances: list[int] = []
            result_payloads: list[dict[str, Any]] = []

            for result in results:
                document = document_by_pmid[result.pmid]
                relevance = bootstrap_relevance(query, document)
                relevances.append(relevance)
                result_payloads.append(_result_to_payload(result, relevance))

            method_metrics = {
                "precision_at_5": precision_at_k(relevances, k=top_k),
                "ndcg_at_5": ndcg_at_k(relevances, corpus_relevances, k=top_k),
            }
            per_method_precision[method_name].append(method_metrics["precision_at_5"])
            per_method_ndcg[method_name].append(method_metrics["ndcg_at_5"])
            query_payload["methods"][method_name] = {
                "metrics": method_metrics,
                "results": result_payloads,
            }

        evaluation_results["queries"].append(query_payload)

    for method_name in methods:
        precision_scores = per_method_precision[method_name]
        ndcg_scores = per_method_ndcg[method_name]
        evaluation_results["aggregate_metrics"][method_name] = {
            "mean_precision_at_5": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
            "mean_ndcg_at_5": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        }

    return evaluation_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods on the assignment queries.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=settings.retrieval_corpus_path,
        help="Retrieval corpus JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.evaluation_path,
        help="Where to write the evaluation JSON artifact.",
    )
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="Number of results to evaluate per query.")
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=max(settings.top_k * 3, 10),
        help="Candidate pool used by hybrid RRF before fusion.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    documents = load_retrieval_documents(args.corpus)
    payload = run_evaluation(documents, top_k=args.top_k, candidate_pool=args.candidate_pool)
    write_json(args.output, payload)
    print(f"Saved evaluation to {args.output}")
    print(payload["aggregate_metrics"])


if __name__ == "__main__":
    main()
