SHELL := /bin/bash

VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
INSTALL_STAMP := $(VENV)/.deps-installed

.DEFAULT_GOAL := help

.PHONY: help setup install check-env build-corpus prepare-corpus evaluate rag pipeline show-results dashboard run

help:
	@printf "Available targets:\n"
	@printf "  make setup           Create the virtual environment\n"
	@printf "  make install         Install Python dependencies\n"
	@printf "  make build-corpus    Fetch PubMed articles and save artifacts/corpus.json\n"
	@printf "  make prepare-corpus  Build artifacts/retrieval_corpus.json\n"
	@printf "  make evaluate        Run retrieval evaluation\n"
	@printf "  make rag             Generate RAG demo outputs\n"
	@printf "  make pipeline        Run the full data, evaluation, and RAG pipeline\n"
	@printf "  make show-results    Print a readable summary from generated artifacts\n"
	@printf "  make dashboard       Launch the Streamlit dashboard\n"
	@printf "  make run             Run the full pipeline, show results, then start the dashboard\n"

setup: $(VENV_PYTHON)

install: $(INSTALL_STAMP)

$(VENV_PYTHON):
	python3 -m venv $(VENV)

$(INSTALL_STAMP): requirements.txt | $(VENV_PYTHON)
	$(VENV_PIP) install -r requirements.txt
	touch $(INSTALL_STAMP)

check-env:
	@if [ -f .env ]; then \
		echo "Using configuration from .env"; \
	else \
		echo "No .env file found. Falling back to exported environment variables and defaults."; \
	fi

build-corpus: check-env install
	$(VENV_PYTHON) -m src.data.build_corpus

prepare-corpus: check-env install
	$(VENV_PYTHON) -m src.retrieval.corpus

evaluate: check-env install
	$(VENV_PYTHON) -m src.evaluation.run_eval

rag: check-env install
	$(VENV_PYTHON) -m src.rag.generate_answer

pipeline: build-corpus prepare-corpus evaluate rag

show-results: install
	$(VENV_PYTHON) -m src.utils.show_results

dashboard: install
	$(STREAMLIT) run dashboard.py

run: pipeline show-results dashboard
