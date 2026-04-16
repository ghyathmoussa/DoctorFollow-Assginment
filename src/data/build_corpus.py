"""Build a deduplicated PubMed corpus from medical terms."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import Settings, settings
from src.data.pubmed_client import ArticleRecord, PubMedClient, PubMedClientError
from src.utils.io import write_json
from src.utils.logging import configure_logging


LOGGER = logging.getLogger(__name__)


@dataclass
class BuildStats:
    terms_processed: int = 0
    unique_articles: int = 0
    duplicates_removed: int = 0
    error_count: int = 0
    matched_pmids: int = 0


@dataclass
class CorpusBuilderResult:
    payload: dict[str, Any]
    stats: BuildStats


def read_terms(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["term"].strip() for row in reader if row.get("term", "").strip()]


def build_corpus(app_settings: Settings) -> CorpusBuilderResult:
    client = PubMedClient(app_settings)
    terms = read_terms(app_settings.medical_terms_path)
    articles_by_pmid: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    stats = BuildStats()

    for term in terms:
        LOGGER.info("Fetching PubMed articles for term: %s", term)
        try:
            pmids = client.search_recent_pmids(term, max_results=app_settings.pubmed_max_results)
            stats.matched_pmids += len(pmids)
            records = client.fetch_articles(pmids)
            _merge_records(term, records, articles_by_pmid, stats)
        except PubMedClientError as exc:
            stats.error_count += 1
            errors.append({"term": term, "error": str(exc)})
            LOGGER.exception("Failed to fetch PubMed data for term: %s", term)
        finally:
            stats.terms_processed += 1

    deduplicated_articles = sorted(
        articles_by_pmid.values(),
        key=lambda item: (item.get("year") is None, -(item.get("year") or 0), item["pmid"]),
    )
    stats.unique_articles = len(deduplicated_articles)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "PubMed E-utilities",
            "medical_terms_path": str(app_settings.medical_terms_path),
            "max_results_per_term": app_settings.pubmed_max_results,
        },
        "summary": {
            "terms_processed": stats.terms_processed,
            "unique_articles": stats.unique_articles,
            "duplicates_removed": stats.duplicates_removed,
            "matched_pmids": stats.matched_pmids,
            "errors": stats.error_count,
        },
        "terms": terms,
        "articles": deduplicated_articles,
        "errors": errors,
    }
    return CorpusBuilderResult(payload=payload, stats=stats)


def _merge_records(
    term: str,
    records: list[ArticleRecord],
    articles_by_pmid: dict[str, dict[str, Any]],
    stats: BuildStats,
) -> None:
    for record in records:
        existing = articles_by_pmid.get(record.pmid)
        if existing is None:
            articles_by_pmid[record.pmid] = _record_to_document(record, term)
            continue

        stats.duplicates_removed += 1
        matched_terms = set(existing["matched_terms"])
        matched_terms.add(term)
        existing["matched_terms"] = sorted(matched_terms)


def _record_to_document(record: ArticleRecord, term: str) -> dict[str, Any]:
    retrieval_text = " ".join(part for part in [record.title.strip(), record.abstract.strip()] if part).strip()
    return {
        "pmid": record.pmid,
        "title": record.title,
        "abstract": record.abstract,
        "authors": record.authors,
        "first_author": record.first_author,
        "journal": record.journal,
        "year": record.year,
        "doi": record.doi,
        "matched_terms": [term],
        "retrieval_text": retrieval_text,
        "has_abstract": bool(record.abstract.strip()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the PubMed article corpus.")
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.corpus_path,
        help="Where to write the corpus JSON artifact.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = _parse_args()
    result = build_corpus(settings)
    write_json(args.output, result.payload)

    summary = result.payload["summary"]
    LOGGER.info("Terms processed: %s", summary["terms_processed"])
    LOGGER.info("Unique articles: %s", summary["unique_articles"])
    LOGGER.info("Duplicates removed: %s", summary["duplicates_removed"])
    LOGGER.info("Errors: %s", summary["errors"])
    LOGGER.info("Corpus written to %s", args.output)


if __name__ == "__main__":
    main()
