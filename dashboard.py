"""Basic Streamlit dashboard for browsing project artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.config import settings
from src.utils.io import read_json


st.set_page_config(
    page_title="Medical RAG Dashboard",
    layout="wide",
)


def load_artifact(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"Missing artifact: `{path.name}`"

    try:
        payload = read_json(path)
    except Exception as exc:  # pragma: no cover - UI fallback
        return None, f"Could not read `{path.name}`: {exc}"

    if not isinstance(payload, dict):
        return None, f"`{path.name}` does not contain a JSON object."
    return payload, None


def format_metric(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def choose_best_method(evaluation_payload: dict[str, Any] | None) -> str:
    if not evaluation_payload:
        return "semantic"

    aggregate_metrics = evaluation_payload.get("aggregate_metrics", {})
    if not aggregate_metrics:
        return "semantic"

    winner = max(
        aggregate_metrics.items(),
        key=lambda item: (
            item[1].get("mean_ndcg_at_5", 0.0),
            item[1].get("mean_precision_at_5", 0.0),
        ),
    )
    return winner[0]


def build_aggregate_rows(evaluation_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate_metrics = evaluation_payload.get("aggregate_metrics", {})
    for method_name, metrics in aggregate_metrics.items():
        rows.append(
            {
                "Method": method_name,
                "Mean Precision": metrics.get("mean_precision_at_5", 0.0),
                "Mean nDCG": metrics.get("mean_ndcg_at_5", 0.0),
            }
        )
    return sorted(rows, key=lambda row: row["Mean nDCG"], reverse=True)


def build_query_options(evaluation_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return evaluation_payload.get("queries", [])


def render_overview(
    corpus_payload: dict[str, Any] | None,
    evaluation_payload: dict[str, Any] | None,
    rag_payload: dict[str, Any] | None,
) -> None:
    corpus_summary = (corpus_payload or {}).get("summary", {})
    demos = (rag_payload or {}).get("demos", [])
    completed_demos = sum(1 for demo in demos if demo.get("status") == "completed")
    failed_demos = sum(1 for demo in demos if demo.get("status") == "failed")
    best_method = choose_best_method(evaluation_payload)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Articles", corpus_summary.get("unique_articles", 0))
    col2.metric("Terms Processed", corpus_summary.get("terms_processed", 0))
    col3.metric("Best Retrieval", best_method)
    col4.metric("Completed Demos", completed_demos)

    if failed_demos:
        st.warning(f"{failed_demos} RAG demo runs failed. The dashboard still shows their retrieved context and errors.")

    st.subheader("Project Snapshot")
    snapshot_left, snapshot_right = st.columns([1, 1])

    with snapshot_left:
        st.markdown(
            "\n".join(
                [
                    f"- LLM Provider: `{(rag_payload or {}).get('llm_provider', settings.llm_provider)}`",
                    f"- LLM Model: `{(rag_payload or {}).get('llm_model', settings.llm_model)}`",
                    f"- Dashboard Data Source: `{settings.artifacts_dir}`",
                    f"- Default Top-K: `{settings.top_k}`",
                ]
            )
        )

    with snapshot_right:
        if evaluation_payload and evaluation_payload.get("aggregate_metrics"):
            st.dataframe(build_aggregate_rows(evaluation_payload), use_container_width=True, hide_index=True)
        else:
            st.info("Run `python -m src.evaluation.run_eval` to populate evaluation metrics.")


def render_corpus(corpus_payload: dict[str, Any] | None) -> None:
    st.subheader("Corpus")
    if not corpus_payload:
        st.info("Run `python -m src.data.build_corpus` to generate corpus artifacts.")
        return

    terms = corpus_payload.get("terms", [])
    articles = corpus_payload.get("articles", [])
    summary = corpus_payload.get("summary", {})

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    stats_col1.metric("Unique Articles", summary.get("unique_articles", 0))
    stats_col2.metric("Duplicates Removed", summary.get("duplicates_removed", 0))
    stats_col3.metric("Errors", summary.get("errors", 0))

    st.markdown("**Tracked Medical Terms**")
    if terms:
        st.caption(", ".join(terms))
    else:
        st.caption("No terms found in artifact.")

    preview_rows = [
        {
            "PMID": article.get("pmid"),
            "Year": article.get("year"),
            "Title": article.get("title"),
            "Journal": article.get("journal"),
            "Matched Terms": ", ".join(article.get("matched_terms", [])),
            "Has Abstract": article.get("has_abstract"),
        }
        for article in articles[:15]
    ]
    st.markdown("**Article Preview**")
    st.dataframe(preview_rows, use_container_width=True, hide_index=True)


def render_evaluation(evaluation_payload: dict[str, Any] | None) -> None:
    st.subheader("Evaluation")
    if not evaluation_payload:
        st.info("Run `python -m src.evaluation.run_eval` to generate evaluation results.")
        return

    aggregate_rows = build_aggregate_rows(evaluation_payload)
    st.markdown("**Aggregate Metrics**")
    st.dataframe(aggregate_rows, use_container_width=True, hide_index=True)

    queries = build_query_options(evaluation_payload)
    if not queries:
        st.info("No query-level evaluation results found.")
        return

    query_labels = {
        f"{query.get('query_id', 'query')} - {query.get('text', 'Untitled query')}": query for query in queries
    }
    selected_label = st.selectbox("Choose an evaluation query", list(query_labels.keys()))
    selected_query = query_labels[selected_label]

    st.caption("Expected terms: " + ", ".join(selected_query.get("expected_terms", [])))

    methods = selected_query.get("methods", {})
    for method_name, method_payload in methods.items():
        metrics = method_payload.get("metrics", {})
        st.markdown(f"**{method_name}**")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Precision", format_metric(metrics.get("precision_at_5")))
        metric_col2.metric("nDCG", format_metric(metrics.get("ndcg_at_5")))

        result_rows = [
            {
                "Rank": result.get("rank"),
                "PMID": result.get("pmid"),
                "Title": result.get("title"),
                "Year": result.get("year"),
                "Matched Terms": ", ".join(result.get("matched_terms", [])),
                "Relevance": result.get("relevance"),
            }
            for result in method_payload.get("results", [])
        ]
        st.dataframe(result_rows, use_container_width=True, hide_index=True)


def render_rag_demos(rag_payload: dict[str, Any] | None) -> None:
    st.subheader("RAG Demos")
    if not rag_payload:
        st.info("Run `python -m src.rag.generate_answer` to generate demo outputs.")
        return

    demos = rag_payload.get("demos", [])
    if not demos:
        st.info("No RAG demos were found.")
        return

    labels = {demo.get("query", f"Demo {index + 1}"): demo for index, demo in enumerate(demos)}
    selected_label = st.selectbox("Choose a RAG demo", list(labels.keys()))
    selected_demo = labels[selected_label]

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Status", selected_demo.get("status", "unknown"))
    info_col2.metric("Method", selected_demo.get("retrieval_method", rag_payload.get("retrieval_method", "-")))
    info_col3.metric("Retrieved Docs", len(selected_demo.get("retrieved_documents", [])))

    if selected_demo.get("status") == "completed":
        st.markdown("**Generated Answer**")
        st.write(selected_demo.get("answer") or "No answer returned.")
    else:
        st.error(selected_demo.get("error", "The demo failed without an error message."))

    st.markdown("**Retrieved Context**")
    for document in selected_demo.get("retrieved_documents", []):
        title = f"#{document.get('rank', '?')} PMID {document.get('pmid', '-')}: {document.get('title', 'Untitled')}"
        with st.expander(title):
            st.markdown(
                "\n".join(
                    [
                        f"- Year: `{document.get('year', '-')}`",
                        f"- Journal: `{document.get('journal', '-')}`",
                        f"- Matched Terms: `{', '.join(document.get('matched_terms', [])) or '-'}`",
                        f"- Score: `{format_metric(document.get('score'))}` ({document.get('score_label', 'score')})",
                    ]
                )
            )
            st.write(document.get("abstract") or "No abstract available.")


def render_sidebar(errors: list[str]) -> None:
    st.sidebar.title("Artifacts")
    st.sidebar.caption("This dashboard is read-only and uses the generated JSON files in `artifacts/`.")
    st.sidebar.code(f"ARTIFACTS_DIR={settings.artifacts_dir}")

    if errors:
        st.sidebar.warning("\n".join(errors))
    else:
        st.sidebar.success("All expected dashboard artifacts were loaded.")


def main() -> None:
    corpus_payload, corpus_error = load_artifact(settings.corpus_path)
    evaluation_payload, evaluation_error = load_artifact(settings.evaluation_path)
    rag_payload, rag_error = load_artifact(settings.rag_demos_path)

    errors = [error for error in [corpus_error, evaluation_error, rag_error] if error]

    st.title("Medical RAG Dashboard")
    st.caption("Basic project dashboard for corpus health, retrieval evaluation, and RAG demo inspection.")

    render_sidebar(errors)
    render_overview(corpus_payload, evaluation_payload, rag_payload)

    overview_tab, corpus_tab, evaluation_tab, rag_tab = st.tabs(
        ["Overview", "Corpus", "Evaluation", "RAG Demos"]
    )

    with overview_tab:
        st.info("Use the tabs to inspect the corpus, compare retrieval methods, and review generated demo outputs.")

    with corpus_tab:
        render_corpus(corpus_payload)

    with evaluation_tab:
        render_evaluation(evaluation_payload)

    with rag_tab:
        render_rag_demos(rag_payload)


if __name__ == "__main__":
    main()
