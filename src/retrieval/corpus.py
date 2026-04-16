"""Shared retrieval-ready corpus schema and loaders."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings
from src.utils.io import read_json, write_json
from src.utils.logging import configure_logging


@dataclass(frozen=True)
class RetrievalDocument:
    pmid: str
    title: str
    abstract: str
    retrieval_text: str
    normalized_text: str
    matched_terms: list[str]
    authors: list[str]
    first_author: str | None
    journal: str | None
    year: int | None
    doi: str | None
    has_abstract: bool
    abstract_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def build_retrieval_text(title: str, abstract: str) -> str:
    title_text = normalize_text(title)
    abstract_text = normalize_text(abstract)
    if abstract_text:
        return f"{title_text} {abstract_text}".strip()
    return title_text


def abstract_status(abstract: str) -> str:
    return "present" if normalize_text(abstract) else "missing"


def build_retrieval_document(article: dict[str, Any]) -> RetrievalDocument:
    title = normalize_text(str(article.get("title", "")))
    abstract = normalize_text(str(article.get("abstract", "")))
    retrieval_text = build_retrieval_text(title, abstract)
    normalized = normalize_text(retrieval_text.lower())
    terms = sorted({normalize_text(str(term)) for term in article.get("matched_terms", []) if str(term).strip()})
    authors = [normalize_text(str(author)) for author in article.get("authors", []) if str(author).strip()]

    return RetrievalDocument(
        pmid=normalize_text(str(article.get("pmid", ""))),
        title=title,
        abstract=abstract,
        retrieval_text=retrieval_text,
        normalized_text=normalized,
        matched_terms=terms,
        authors=authors,
        first_author=_clean_optional_text(article.get("first_author")),
        journal=_clean_optional_text(article.get("journal")),
        year=_coerce_year(article.get("year")),
        doi=_clean_optional_text(article.get("doi")),
        has_abstract=bool(abstract),
        abstract_status=abstract_status(abstract),
    )


def validate_retrieval_document(document: RetrievalDocument) -> list[str]:
    issues: list[str] = []
    if not document.pmid:
        issues.append("Missing PMID.")
    if not document.title:
        issues.append("Missing title.")
    if not document.retrieval_text:
        issues.append("Missing retrieval text.")
    if document.has_abstract and document.abstract_status != "present":
        issues.append("Abstract status is inconsistent with abstract content.")
    if not document.has_abstract and document.abstract_status != "missing":
        issues.append("Missing abstract was not labeled correctly.")
    return issues


def load_retrieval_documents(path: Path | None = None) -> list[RetrievalDocument]:
    corpus_path = path or settings.retrieval_corpus_path
    payload = read_json(corpus_path)
    documents = payload.get("documents")
    if documents is None:
        documents = payload.get("articles", [])
    return [build_retrieval_document(article) for article in documents]


def build_retrieval_corpus_payload(corpus_path: Path | None = None) -> dict[str, Any]:
    source_path = corpus_path or settings.corpus_path
    source_payload = read_json(source_path)

    documents: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for article in source_payload.get("articles", []):
        document = build_retrieval_document(article)
        document_issues = validate_retrieval_document(document)
        if document_issues:
            issues.append({"pmid": document.pmid, "issues": document_issues})
        documents.append(document.to_dict())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_corpus_path": str(source_path),
        "summary": {
            "documents": len(documents),
            "documents_with_abstract": sum(1 for doc in documents if doc["has_abstract"]),
            "documents_without_abstract": sum(1 for doc in documents if not doc["has_abstract"]),
            "validation_issues": len(issues),
        },
        "documents": documents,
        "issues": issues,
    }


def _coerce_year(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = normalize_text(str(value))
    if not cleaned or cleaned.lower() == "none":
        return None
    return cleaned


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a retrieval-ready corpus artifact.")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.corpus_path,
        help="Source corpus JSON from the PubMed pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.retrieval_corpus_path,
        help="Where to write the retrieval-ready corpus JSON artifact.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()
    payload = build_retrieval_corpus_payload(args.input)
    write_json(args.output, payload)
    print(f"Prepared retrieval corpus: {args.output}")
    print(payload["summary"])


if __name__ == "__main__":
    main()
