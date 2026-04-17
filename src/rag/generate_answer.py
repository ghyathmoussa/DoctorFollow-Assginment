"""RAG answer generation using the best evaluated retrieval method."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any

import requests

from src.config import settings
from src.evaluation.run_eval import EVALUATION_QUERIES, run_evaluation
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.corpus import RetrievalDocument, load_retrieval_documents, normalize_text
from src.retrieval.hybrid_rrf import HybridRRFRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.utils.io import read_json, write_json


SYSTEM_PROMPT = """You are a medical literature QA assistant for Turkish-speaking doctors.
Answer only from the provided context.
Do not use outside knowledge.
If the context is insufficient, say so clearly.
Cite every substantive claim using either PMID or article title from the provided sources.
Keep the answer concise, clinically oriented, and faithful to the retrieved evidence."""


class LLMClientError(Exception):
    """Raised when the configured LLM provider fails after retries."""


@dataclass(frozen=True)
class RetrievedContext:
    rank: int
    pmid: str
    title: str
    year: int | None
    journal: str | None
    matched_terms: list[str]
    abstract: str
    has_abstract: bool
    score: float
    score_label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "pmid": self.pmid,
            "title": self.title,
            "year": self.year,
            "journal": self.journal,
            "matched_terms": self.matched_terms,
            "abstract": self.abstract,
            "has_abstract": self.has_abstract,
            "score": self.score,
            "score_label": self.score_label,
        }


def _truncate_text(text: str, limit: int) -> str:
    normalized = text.strip()
    if limit <= 0 or len(normalized) <= limit:
        return normalized

    cutoff = max(limit - 3, 1)
    return normalized[:cutoff].rstrip() + "..."


def choose_best_method(evaluation_path: Path | None = None) -> str:
    path = evaluation_path or settings.evaluation_path
    if not path.exists():
        return "semantic"

    payload = read_json(path)
    metrics = payload.get("aggregate_metrics", {})
    if not metrics:
        return "semantic"

    ranked = sorted(
        metrics.items(),
        key=lambda item: (
            item[1].get("mean_ndcg_at_5", 0.0),
            item[1].get("mean_precision_at_5", 0.0),
        ),
        reverse=True,
    )
    return ranked[0][0]


def choose_best_method_from_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "semantic"

    ranked = sorted(
        metrics.items(),
        key=lambda item: (
            item[1].get("mean_ndcg_at_5", 0.0),
            item[1].get("mean_precision_at_5", 0.0),
        ),
        reverse=True,
    )
    return ranked[0][0]


def choose_best_method_for_setting(
    documents: list[RetrievalDocument],
    *,
    top_k: int,
    enable_query_translation: bool,
) -> str:
    evaluation_payload = run_evaluation(
        documents,
        top_k=top_k,
        candidate_pool=max(top_k * 3, 10),
        enable_query_translation=enable_query_translation,
    )
    return choose_best_method_from_metrics(evaluation_payload.get("aggregate_metrics", {}))


def retrieve_context(
    query: str,
    documents: list[RetrievalDocument],
    method_name: str,
    top_k: int,
    *,
    enable_query_translation: bool = True,
) -> list[RetrievedContext]:
    document_by_pmid = {document.pmid: document for document in documents}

    if method_name == "bm25":
        retriever = BM25Retriever(documents, enable_query_translation=enable_query_translation)
        results = retriever.search(query, top_k=top_k)
        return [
            RetrievedContext(
                rank=result.rank,
                pmid=result.pmid,
                title=result.title,
                year=result.year,
                journal=result.journal,
                matched_terms=result.matched_terms,
                abstract=document_by_pmid[result.pmid].abstract,
                has_abstract=result.has_abstract,
                score=result.score,
                score_label="bm25_score",
            )
            for result in results
        ]

    if method_name == "hybrid_rrf":
        retriever = HybridRRFRetriever(documents, enable_query_translation=enable_query_translation)
        results = retriever.search(query, top_k=top_k)
        return [
            RetrievedContext(
                rank=result.rank,
                pmid=result.pmid,
                title=result.title,
                year=result.year,
                journal=result.journal,
                matched_terms=result.matched_terms,
                abstract=document_by_pmid[result.pmid].abstract,
                has_abstract=result.has_abstract,
                score=result.rrf_score,
                score_label="rrf_score",
            )
            for result in results
        ]

    retriever = SemanticRetriever(documents, enable_query_translation=enable_query_translation)
    results = retriever.search(query, top_k=top_k)
    return [
        RetrievedContext(
            rank=result.rank,
            pmid=result.pmid,
            title=result.title,
            year=result.year,
            journal=result.journal,
            matched_terms=result.matched_terms,
            abstract=document_by_pmid[result.pmid].abstract,
            has_abstract=result.has_abstract,
            score=result.score,
            score_label="cosine_similarity",
        )
        for result in results
    ]


def build_context_block(contexts: list[RetrievedContext]) -> str:
    blocks: list[str] = []
    abstract_limit = settings.rag_abstract_char_limit
    for item in contexts:
        abstract = item.abstract if item.abstract else "No abstract available."
        abstract = _truncate_text(abstract, abstract_limit)
        blocks.append(
            "\n".join(
                [
                    f"Source {item.rank}",
                    f"PMID: {item.pmid}",
                    f"Title: {item.title}",
                    f"Year: {item.year if item.year is not None else 'Unknown'}",
                    f"Journal: {item.journal or 'Unknown'}",
                    f"Matched Terms: {', '.join(item.matched_terms) if item.matched_terms else 'Unknown'}",
                    f"Abstract: {abstract}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_user_prompt(query: str, contexts: list[RetrievedContext]) -> str:
    context_block = build_context_block(contexts)
    return (
        f"User query: {normalize_text(query)}\n\n"
        "Use the following retrieved medical literature context and answer in the same language as the user query when reasonable.\n\n"
        f"{context_block}\n\n"
        "Answer with short clinically useful bullets or a compact paragraph. Cite each key point with PMID or title."
    )


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str) -> None:
        self.provider = provider.strip().lower()
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_seconds = settings.llm_timeout_seconds
        self.retry_attempts = settings.llm_retry_attempts
        self.retry_backoff_seconds = settings.llm_retry_backoff_seconds
        self.min_interval_seconds = settings.llm_min_interval_seconds
        self._last_request_ts = 0.0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError("LLM_API_KEY is not configured.")

        if self.provider == "gemini":
            return self._generate_gemini(system_prompt, user_prompt)
        if self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _post_with_retries(
        self,
        url: str,
        *,
        json_payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        provider_name: str,
    ) -> requests.Response:
        last_error: Exception | None = None

        for attempt in range(1, self.retry_attempts + 1):
            response: requests.Response | None = None
            try:
                self._respect_rate_limit()
                response = requests.post(url, json=json_payload, headers=headers, timeout=self.timeout_seconds)
                self._last_request_ts = time.monotonic()
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    detail = self._extract_error_message(response)
                    raise LLMClientError(
                        f"{provider_name} returned retryable status {response.status_code}: {detail}"
                    )
                response.raise_for_status()
                return response
            except (requests.RequestException, LLMClientError) as exc:
                last_error = exc
                if attempt == self.retry_attempts:
                    raise LLMClientError(
                        f"{provider_name} request failed after {self.retry_attempts} attempts: {exc}"
                    ) from exc

                sleep_seconds = self._retry_delay_seconds(attempt, response=response, exc=exc)
                time.sleep(sleep_seconds)

        raise LLMClientError(f"{provider_name} request failed unexpectedly: {last_error}")

    def _respect_rate_limit(self) -> None:
        if self.min_interval_seconds <= 0:
            return

        now = time.monotonic()
        elapsed = now - self._last_request_ts
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _retry_delay_seconds(
        self,
        attempt: int,
        *,
        response: requests.Response | None = None,
        exc: Exception | None = None,
    ) -> float:
        retry_after = None
        if response is not None:
            retry_after = response.headers.get("Retry-After")
        elif isinstance(exc, requests.HTTPError) and exc.response is not None:
            retry_after = exc.response.headers.get("Retry-After")

        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
        return self.retry_backoff_seconds * attempt

    def _extract_error_message(self, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip() or "No error details returned."

        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if message:
                    return str(message)
        return response.text.strip() or "No error details returned."

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        response = self._post_with_retries(url, json_payload=payload, provider_name="Gemini")
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise ValueError("Gemini returned an empty response.")
        return text

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        response = self._post_with_retries(
            url,
            json_payload=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            provider_name="OpenAI",
        )
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("OpenAI returned no choices.")
        text = choices[0].get("message", {}).get("content", "").strip()
        if not text:
            raise ValueError("OpenAI returned an empty response.")
        return text


def run_rag_query(
    query: str,
    documents: list[RetrievalDocument],
    *,
    method_name: str,
    top_k: int,
    llm_client: LLMClient,
    enable_query_translation: bool = True,
) -> dict[str, Any]:
    contexts = retrieve_context(
        query,
        documents,
        method_name=method_name,
        top_k=top_k,
        enable_query_translation=enable_query_translation,
    )
    user_prompt = build_user_prompt(query, contexts)
    answer = llm_client.generate(SYSTEM_PROMPT, user_prompt)
    return {
        "query": query,
        "retrieval_method": method_name,
        "query_translation_enabled": enable_query_translation,
        "retrieved_documents": [context.to_dict() for context in contexts],
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "answer": answer,
        "status": "completed",
    }


def _default_demo_queries() -> list[str]:
    return [EVALUATION_QUERIES[1].text, EVALUATION_QUERIES[4].text]


def _all_evaluation_queries() -> list[str]:
    return [query.text for query in EVALUATION_QUERIES]


def _run_demo_batch(
    queries: list[str],
    documents: list[RetrievalDocument],
    *,
    method_name: str,
    top_k: int,
    llm_client: LLMClient,
    enable_query_translation: bool,
) -> list[dict[str, Any]]:
    demos: list[dict[str, Any]] = []
    for query in queries:
        contexts = retrieve_context(
            query,
            documents,
            method_name=method_name,
            top_k=top_k,
            enable_query_translation=enable_query_translation,
        )
        user_prompt = build_user_prompt(query, contexts)
        try:
            answer = llm_client.generate(SYSTEM_PROMPT, user_prompt)
            demos.append(
                {
                    "query": query,
                    "retrieval_method": method_name,
                    "query_translation_enabled": enable_query_translation,
                    "retrieved_documents": [context.to_dict() for context in contexts],
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                    "answer": answer,
                    "status": "completed",
                }
            )
        except Exception as exc:
            demos.append(
                {
                    "query": query,
                    "retrieval_method": method_name,
                    "query_translation_enabled": enable_query_translation,
                    "retrieved_documents": [context.to_dict() for context in contexts],
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                    "answer": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return demos


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cited RAG answers using the best retrieval method.")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to answer. Can be provided multiple times.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="best",
        choices=["best", "bm25", "semantic", "hybrid_rrf"],
        help="Retrieval method to use.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=settings.retrieval_corpus_path,
        help="Retrieval corpus JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.rag_demos_path,
        help="Where to write the RAG demo JSON artifact.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=settings.rag_comparison_path,
        help="Where to write the before/after bonus RAG comparison artifact.",
    )
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="How many documents to retrieve.")
    parser.add_argument(
        "--disable-query-translation",
        action="store_true",
        help="Use raw queries without Turkish-to-English expansion.",
    )
    parser.add_argument(
        "--compare-query-translation",
        action="store_true",
        help="Generate side-by-side before/after bonus RAG outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    documents = load_retrieval_documents(args.corpus)
    llm_client = LLMClient(settings.llm_provider, settings.llm_api_key, settings.llm_model)
    query_translation_enabled = not args.disable_query_translation

    if args.compare_query_translation:
        queries = args.queries or _all_evaluation_queries()
        before_method = (
            choose_best_method_for_setting(documents, top_k=args.top_k, enable_query_translation=False)
            if args.method == "best"
            else args.method
        )
        after_method = (
            choose_best_method_for_setting(documents, top_k=args.top_k, enable_query_translation=True)
            if args.method == "best"
            else args.method
        )
        comparison_payload = {
            "comparison_type": "query_translation_before_after",
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "llm_min_interval_seconds": settings.llm_min_interval_seconds,
            "rag_abstract_char_limit": settings.rag_abstract_char_limit,
            "queries": queries,
            "before_bonus": {
                "label": "Before Bonus",
                "retrieval_method": before_method,
                "query_translation_enabled": False,
                "demos": _run_demo_batch(
                    queries,
                    documents,
                    method_name=before_method,
                    top_k=args.top_k,
                    llm_client=llm_client,
                    enable_query_translation=False,
                ),
            },
            "after_bonus": {
                "label": "After Bonus",
                "retrieval_method": after_method,
                "query_translation_enabled": True,
                "demos": _run_demo_batch(
                    queries,
                    documents,
                    method_name=after_method,
                    top_k=args.top_k,
                    llm_client=llm_client,
                    enable_query_translation=True,
                ),
            },
        }
        write_json(args.comparison_output, comparison_payload)
        print(f"Saved RAG comparison to {args.comparison_output}")
        print(f"Queries: {len(queries)}")
        print(f"Before bonus method: {before_method}")
        print(f"After bonus method: {after_method}")
        return

    queries = args.queries or _default_demo_queries()
    method_name = (
        choose_best_method_for_setting(
            documents,
            top_k=args.top_k,
            enable_query_translation=query_translation_enabled,
        )
        if args.method == "best"
        else args.method
    )

    demos = _run_demo_batch(
        queries,
        documents,
        method_name=method_name,
        top_k=args.top_k,
        llm_client=llm_client,
        enable_query_translation=query_translation_enabled,
    )
    payload = {
        "retrieval_method": method_name,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_min_interval_seconds": settings.llm_min_interval_seconds,
        "rag_abstract_char_limit": settings.rag_abstract_char_limit,
        "query_translation_enabled": query_translation_enabled,
        "demos": demos,
    }
    write_json(args.output, payload)
    print(f"Saved RAG demos to {args.output}")
    print(f"Method: {method_name}")
    print(f"Queries: {len(demos)}")


if __name__ == "__main__":
    main()
