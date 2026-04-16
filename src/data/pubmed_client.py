"""PubMed E-utilities client and XML parsing helpers."""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import requests

from src.config import Settings


LOGGER = logging.getLogger(__name__)


class PubMedClientError(Exception):
    """Raised when PubMed operations fail after retries."""


@dataclass
class ArticleRecord:
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    first_author: str | None
    journal: str | None
    year: int | None
    doi: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PubMedClient:
    """Minimal client for PubMed esearch and efetch endpoints."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self._last_request_ts = 0.0
        self._min_interval = 1.0 / max(1, self.settings.pubmed_rate_limit_per_sec)

    def search_recent_pmids(self, term: str, max_results: int | None = None) -> list[str]:
        response = self._request_json(
            self.ESEARCH_URL,
            params={
                "db": "pubmed",
                "term": term,
                "sort": "pub date",
                "retmax": max_results or self.settings.pubmed_max_results,
                "retmode": "json",
                **self._common_params(),
            },
        )
        return response.get("esearchresult", {}).get("idlist", [])

    def fetch_articles(self, pmids: list[str]) -> list[ArticleRecord]:
        if not pmids:
            return []

        root = self._request_xml(
            self.EFETCH_URL,
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                **self._common_params(),
            },
        )
        records: list[ArticleRecord] = []
        for node in root.findall(".//PubmedArticle"):
            try:
                records.append(self._parse_pubmed_article(node))
            except PubMedClientError:
                LOGGER.warning("Skipping malformed PubMed article record.", exc_info=True)
        return records

    def _common_params(self) -> dict[str, str]:
        params = {"tool": self.settings.pubmed_tool_name}
        if self.settings.pubmed_email:
            params["email"] = self.settings.pubmed_email
        return params

    def _request_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self._request(url, params=params)
        try:
            return response.json()
        except ValueError as exc:
            raise PubMedClientError("PubMed returned invalid JSON") from exc

    def _request_xml(self, url: str, params: dict[str, Any]) -> ET.Element:
        response = self._request(url, params=params)
        try:
            return ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise PubMedClientError("PubMed returned invalid XML") from exc

    def _request(self, url: str, params: dict[str, Any]) -> requests.Response:
        attempts = self.settings.pubmed_retry_attempts
        backoff = self.settings.pubmed_retry_backoff_seconds

        for attempt in range(1, attempts + 1):
            self._respect_rate_limit()
            try:
                response = self.session.get(url, params=params, timeout=self.settings.pubmed_timeout_seconds)
                self._last_request_ts = time.monotonic()
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    raise PubMedClientError(f"PubMed returned retryable status {response.status_code}")
                response.raise_for_status()
                return response
            except (requests.RequestException, PubMedClientError) as exc:
                if attempt == attempts:
                    raise PubMedClientError(
                        f"PubMed request failed after {attempts} attempts: {exc}"
                    ) from exc
                sleep_seconds = backoff * attempt
                LOGGER.warning(
                    "PubMed request failed on attempt %s/%s. Retrying in %.1f seconds.",
                    attempt,
                    attempts,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        raise PubMedClientError("PubMed request failed unexpectedly")

    def _respect_rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _parse_pubmed_article(self, article_node: ET.Element) -> ArticleRecord:
        medline = article_node.find("MedlineCitation")
        article = medline.find("Article") if medline is not None else None
        if medline is None or article is None:
            raise PubMedClientError("Missing MedlineCitation or Article block in PubMed XML")

        pmid = self._extract_text(medline.find("PMID"))
        title = self._extract_text(article.find("ArticleTitle"))
        abstract = self._extract_abstract(article.find("Abstract"))
        authors = self._extract_authors(article.find("AuthorList"))

        return ArticleRecord(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            first_author=authors[0] if authors else None,
            journal=self._extract_text(article.find("Journal/Title")) or None,
            year=self._extract_year(article, article_node),
            doi=self._extract_doi(article, article_node),
        )

    def _extract_abstract(self, abstract_node: ET.Element | None) -> str:
        if abstract_node is None:
            return ""

        parts: list[str] = []
        for item in abstract_node.findall("AbstractText"):
            label = item.attrib.get("Label")
            text = self._extract_text(item)
            if not text:
                continue
            parts.append(f"{label}: {text}" if label else text)
        return "\n".join(parts).strip()

    def _extract_authors(self, author_list_node: ET.Element | None) -> list[str]:
        if author_list_node is None:
            return []

        authors: list[str] = []
        for author_node in author_list_node.findall("Author"):
            collective_name = self._extract_text(author_node.find("CollectiveName"))
            if collective_name:
                authors.append(collective_name)
                continue

            last_name = self._extract_text(author_node.find("LastName"))
            fore_name = self._extract_text(author_node.find("ForeName"))
            initials = self._extract_text(author_node.find("Initials"))
            full_name = " ".join(part for part in [fore_name, last_name] if part).strip()
            if full_name:
                authors.append(full_name)
            elif last_name:
                authors.append(last_name)
            elif initials:
                authors.append(initials)
        return authors

    def _extract_year(self, article: ET.Element, article_node: ET.Element) -> int | None:
        candidate_years: list[int] = []
        year_paths = (
            "Journal/JournalIssue/PubDate/Year",
            "ArticleDate/Year",
        )

        for path in year_paths:
            value = self._extract_text(article.find(path))
            year = self._coerce_year(value)
            if year is not None:
                candidate_years.append(year)

        for pub_date in article_node.findall(".//PubMedPubDate"):
            year = self._coerce_year(self._extract_text(pub_date.find("Year")))
            if year is not None:
                candidate_years.append(year)

        medline_date = self._extract_text(article.find("Journal/JournalIssue/PubDate/MedlineDate"))
        if medline_date:
            for token in medline_date.split():
                year = self._coerce_year("".join(ch for ch in token if ch.isdigit()))
                if year is not None:
                    candidate_years.append(year)

        if not candidate_years:
            return None
        return min(candidate_years)

    def _extract_doi(self, article: ET.Element, article_node: ET.Element) -> str | None:
        for doi_node in article.findall("ELocationID"):
            if doi_node.attrib.get("EIdType") == "doi":
                value = self._extract_text(doi_node)
                if value:
                    return value

        for article_id in article_node.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                value = self._extract_text(article_id)
                if value:
                    return value
        return None

    @staticmethod
    def _extract_text(node: ET.Element | None) -> str:
        if node is None:
            return ""
        return "".join(node.itertext()).strip()

    @staticmethod
    def _coerce_year(value: str) -> int | None:
        if not value or not value.isdigit():
            return None

        year = int(value)
        current_year = datetime.now().year
        if 1800 <= year <= current_year + 1:
            return year
        return None
