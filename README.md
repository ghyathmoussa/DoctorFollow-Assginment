# DoctorFollow AI Technical Assessment

## Overview
This project builds a small medical RAG pipeline for Turkish-speaking doctors querying English medical literature.

It includes:
- a PubMed ingestion pipeline using E-utilities
- a deduplicated JSON corpus
- three retrieval methods: BM25, semantic search, and hybrid RRF
- an evaluation pipeline over the 5 required queries
- a grounded RAG generation pipeline with provider-configurable LLM access

## Repo Structure
- `src/data/pubmed_client.py`: PubMed API client, retry logic, XML parsing
- `src/data/build_corpus.py`: reads `medical_terms.csv`, fetches articles, deduplicates, writes `artifacts/corpus.json`
- `src/retrieval/corpus.py`: retrieval-ready document schema and normalization
- `src/retrieval/bm25_retriever.py`: BM25 retrieval
- `src/retrieval/semantic_retriever.py`: sentence-transformers embeddings + cosine similarity + cache
- `src/retrieval/hybrid_rrf.py`: reciprocal rank fusion
- `src/evaluation/run_eval.py`: evaluation runner for the 5 assignment queries
- `src/rag/generate_answer.py`: retrieval + prompt building + LLM client + demo artifact generation
- `artifacts/`: generated corpus, retrieval corpus, embeddings, evaluation, and RAG demo outputs

## Setup
### 1. Configure environment variables
You can either:
- copy `.env.example` to `.env`
- or pass the variables from your shell / Docker `--env-file`

To use a local `.env` file:

```bash
cp .env.example .env
```

Important variables:
- `LLM_PROVIDER`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_TIMEOUT_SECONDS`
- `LLM_RETRY_ATTEMPTS`
- `LLM_RETRY_BACKOFF_SECONDS`
- `LLM_MIN_INTERVAL_SECONDS`
- `RAG_ABSTRACT_CHAR_LIMIT`
- `EMBEDDING_MODEL`
- `BM25_K1`
- `BM25_B`
- `RRF_K`
- `TOP_K`

### 2. Local Python setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run
### Recommended local run
This is the easiest local command:

```bash
make run
```

It will:
- create `.venv` if needed
- install dependencies
- build the PubMed corpus
- prepare the retrieval corpus
- run evaluation
- generate RAG demos
- print a readable summary in the terminal
- start the Streamlit dashboard

Open the dashboard at `http://localhost:8501`.

### Useful local commands
If you want more control:

```bash
make pipeline
make show-results
make dashboard
make help
```

### Docker run
Build the image:

```bash
docker build -t medical-rag .
```

Run the full pipeline and dashboard with your `.env` file:

```bash
docker run --rm -it \
  --env-file .env \
  -p 8501:8501 \
  medical-rag
```

Then open `http://localhost:8501`.

The container defaults to `make run`, so it will:
- run the full pipeline
- print the results summary
- start the dashboard on port `8501`

If you only want the dashboard for already-generated artifacts:

```bash
docker run --rm -it \
  --env-file .env \
  -p 8501:8501 \
  medical-rag \
  make dashboard
```

### Run step by step manually
If you prefer running each stage yourself:

#### Build the PubMed corpus
```bash
python -m src.data.build_corpus
```

#### Prepare the retrieval corpus
```bash
python -m src.retrieval.corpus
```

#### BM25 retrieval
```bash
python -m src.retrieval.bm25_retriever "type 2 diabetes management guidelines" --top-k 5
```

#### Semantic retrieval
```bash
python -m src.retrieval.semantic_retriever "Çölyak hastalığı tanı kriterleri nelerdir?" --top-k 5
```

#### Hybrid RRF retrieval
```bash
python -m src.retrieval.hybrid_rrf "community acquired pneumonia antibiotic resistance" --top-k 5 --rrf-k 60
```

#### Run evaluation
```bash
python -m src.evaluation.run_eval
```

#### Run RAG demos
```bash
python -m src.rag.generate_answer
```

#### Run the dashboard
```bash
streamlit run dashboard.py
```

The dashboard is intentionally basic and reads from the generated files in `artifacts/`:
- corpus summary and article preview from `artifacts/corpus.json`
- retrieval metrics from `artifacts/evaluation.json`
- RAG demo results from `artifacts/rag_demos.json`

## Part 1: Data Pipeline
For each term in `medical_terms.csv`, the pipeline:
- calls PubMed `esearch` to get the 5 most recent PMIDs
- calls PubMed `efetch` to retrieve article metadata and abstracts
- normalizes:
  - PMID
  - title
  - abstract
  - authors
  - first author
  - journal
  - year
  - DOI
- deduplicates by PMID
- tracks `matched_terms`
- writes a retrieval-ready text field: `title + abstract`

### Pipeline Output
Current corpus summary from `artifacts/corpus.json`:
- `terms_processed`: 10
- `unique_articles`: 50
- `duplicates_removed`: 0
- `matched_pmids`: 50
- `errors`: 0

## Approach
I split the system into stages so each part can be rerun independently:
- fetch once, save corpus
- normalize once, save retrieval corpus
- build retrieval methods over the same document schema
- evaluate retrieval before choosing a RAG backend

This makes the system easier to debug and avoids repeated PubMed calls.

### Model Choice
I selected `intfloat/multilingual-e5-small` for semantic retrieval.

Why:
- the assignment includes Turkish queries over English literature
- E5 is strong for multilingual retrieval
- it is much smaller and more practical than `bge-m3`

## BM25 Analysis
### Tokenization
BM25 uses a regex-based multilingual word tokenizer:

```python
\w+(?:[-']\w+)*
```

Why:
- simple, deterministic lexical baseline
- supports Turkish characters through Unicode tokenization
- preserves useful medical forms like hyphenated terms

### What `k1` controls
`k1` controls term frequency scaling.

Interpretation:
- lower `k1`: repeating the same term many times helps less
- higher `k1`: repeated term matches keep contributing more strongly

### What `b` controls
`b` controls document length normalization by adjusting the influence of document length relative to the average document length (avgdlavgdl).

Interpretation:
- `b = 0`: little or no penalty for longer documents
- higher `b`: stronger normalization, so long abstracts get discounted more

## RRF Analysis
I implemented Reciprocal Rank Fusion as:

```text
RRF(doc) = sum(1 / (k + rank_i(doc)))
```

where `rank_i(doc)` is the document’s rank in each retrieval list.

### What does `k` do
`k` smooths how much advantage top ranks get.

Interpretation:
- smaller `k`: top positions dominate more strongly
- larger `k`: differences between rank positions flatten out

### What happens at `k=0`
With `k=0`, the score becomes:

```text
1/rank
```

This makes rank 1 extremely valuable and creates much sharper score differences between rank positions.

### What happens at `k=1000`
With `k=1000`, all contributions become tiny and much closer together:
- rank 1 and rank 5 are no longer very different
- fusion becomes less sensitive to top-rank disagreements

### Why rank positions instead of raw scores
BM25 and cosine similarity are not on a comparable numeric scale:
- BM25 is unbounded and depends on term statistics
- cosine similarity is bounded and behaves differently

Using raw scores would make one retrieval system dominate due to scale, not relevance. Using rank positions avoids that calibration problem and makes the fusion more stable.

## Evaluation
### Metrics
I used:
- `Precision`
- `nDCG`

Why:
- `Precision` measures how useful the shown results are
- `nDCG` rewards putting the most relevant items earlier

### Judgment Method
The current benchmark uses a transparent bootstrap heuristic, not a manually labeled gold set.

Relevance scale:
- `0`: not relevant
- `1`: topically relevant
- `2`: highly relevant

This is saved in `artifacts/evaluation.json` under `metric_notes`.

### Aggregate Results
From `artifacts/evaluation.json`:

| Method | Mean Precision | Mean nDCG |
|---|---:|---:|
| BM25 | 0.56 | 0.5785 |
| Semantic | 0.76 | 0.7738 |
| Hybrid RRF | 0.68 | 0.7153 |

### Best Method
The current best method is `semantic`.

Why:
- best aggregate `Precision`
- best aggregate `nDCG`
- strongest behavior on cross-lingual and broader conceptual queries

### Qualitative Pattern
- BM25 did very well on strongly lexical queries such as acute otitis media and iron supplementation during pregnancy.
- Semantic retrieval performed much better on Turkish queries like celiac diagnosis and on broader conceptual pneumonia queries.
- Hybrid improved over BM25 overall, but in the current setup it did not beat pure semantic retrieval.

## RAG Generation
The RAG pipeline currently:
- chooses the best retrieval method from the evaluation artifact
- retrieves top context documents
- builds a strict grounded prompt
- calls the configured LLM provider
- writes results to `artifacts/rag_demos.json`

As a result, `artifacts/rag_demos.json` currently contains:
- retrieved documents
- prompts
- failed generation status
- explicit error messages

Once a real API key is configured, rerunning:

```bash
python -m src.rag.generate_answer
```

should produce the final cited answers for the required 2 queries.

## Hardest Problem
The hardest part was making semantic retrieval reliable in an offline / partially restricted environment.

Two issues had to be solved:
- first: run model download required network access and storage.
- later: runs still attempted Hugging Face resolution even after caching

I solved it by:
- caching document embeddings locally
- storing embedding metadata to validate cache reuse
- resolving the local Hugging Face snapshot path directly after first download

That made semantic retrieval repeatable and much faster after the first successful model fetch.

## Scenario Question
Your team needs to benchmark a 70B open-source LLM for medical QA. Your usual GPU provider doesn't have L40S available today. Your manager is busy all day. Results needed by end of week. What do you do?

I would move immediately on parallel tracks and optimize for getting benchmarkable results, not waiting for a perfect GPU match.

Concrete plan:
1. Check alternative inference platforms the same day:
   - Together AI
   - Fireworks AI
   - Groq if the target model is supported
   - Modal or RunPod for on-demand GPUs
   - Lambda Labs, Vast.ai, or Paperspace for temporary GPU availability
   - Hugging Face Inference Endpoints if the target model is supported
2. If the exact 70B model is unavailable, pick the closest viable alternative and document the trade-off:
   - same family, different quantization
   - same family on A100/H100 instead of L40S
   - smaller model for pipeline shakeout, then 70B for final benchmark
3. Separate benchmarking goals:
   - quality benchmark
   - latency benchmark
   - cost benchmark
   This matters because a non-L40S platform may still be acceptable for quality evaluation even if latency is not directly comparable.
4. Freeze the benchmark protocol before running:
   - dataset
   - prompts
   - decoding parameters
   - metrics
   - timeout / retry rules
5. Start a small pilot run immediately on the best available platform to de-risk formatting, prompt issues, and logging.
6. Send a concise update to the manager and team with:
   - unavailable original provider
   - selected fallback platform
   - expected trade-offs
   - benchmark ETA
   - any comparability limitations

Trade-off summary:
- L40S availability matters for apples-to-apples latency and cost comparisons.
- It matters much less if the urgent goal is model quality benchmarking by end of week.
- I would prioritize getting a documented, reproducible benchmark done now, while clearly labeling any hardware mismatch in the final report.

## AI Usage
AI tools were used during development for implementation assistance and iteration. The code, structure, and final documentation were reviewed and adjusted to fit the assignment requirements and the actual project outputs.

## Submission Notes
- Python only
- no API keys hardcoded in source files
- environment-driven configuration for credentials and tunable parameters
- artifacts included for corpus build, evaluation, and RAG demo wiring

## Remaining Gaps
- Phase 8 still needs a real `LLM_API_KEY` to generate the final 2 cited answers
- Phase 9 bonus work has not been implemented yet
