# Technical Report — Production RAG System Roadmap

**Project:** Paddington RAG (Project 1)
**Author:** _(you)_
**Date:** 2026-07-15
**Purpose:** Gap analysis and implementation plan to turn the current RAG MVP into a
production-grade, portfolio-ready system suitable for a Senior AI Engineer application.

---

## 1. Executive Summary

The repository already contains a **working, well-structured RAG happy path**: text is
chunked, embedded with OpenAI `text-embedding-3-small`, stored in Postgres/pgvector, and
retrieved by cosine similarity to ground an LLM answer with source attribution and
per-request cost tracking. The engineering fundamentals are strong — clean
service/repository layering, async throughout, LiteLLM abstraction with retries and
fallbacks, structured logging.

However, the RAG feature is currently a **sub-component of a larger web-automation agent
("Paddington")**, and it stops at the MVP boundary. To meet the stated goal of a
*production RAG system* — and to be a convincing Senior-level portfolio piece — the
following are missing entirely: a **vector index** (today every query is a sequential
scan), **document loaders** (only raw text via JSON is accepted), **reranking**,
**hybrid/keyword retrieval**, an **evaluation pipeline**, **containerized deployment +
CI/CD**, and **documentation/diagrams**.

This report maps each checklist requirement to its current state and proposes a
prioritized, file-level implementation plan.

---

## 2. Current-State Inventory

| Component | File | Status |
|---|---|---|
| Chunking | `services/document_service.py` (`RecursiveCharacterTextSplitter`, 1000/200) | ✅ Works, single strategy |
| Embedding | `llm/embedding_service.py` (LiteLLM, `text-embedding-3-small`, 1536-d, batched) | ✅ Solid |
| Vector store | `models/document.py` (`Vector(1536)`), pgvector | ⚠️ No index |
| Retrieval | `repositories/document_repository.py::search_similar_chunks` (cosine top-k) | ⚠️ Vector-only |
| Generation | `services/document_service.py::query` (grounded prompt, sources, cost) | ✅ Solid |
| Agent tools | `agent/tools.py` (`search_documents`, `list_documents`, `get_document_content`) | ✅ Works |
| API | `routes/documents.py` (upload / query / list, JWT-auth) | ⚠️ Partial |
| Ingestion input | `schemas/document.py` (raw `content` string only) | ❌ No file upload |
| Reranking | — | ❌ Missing |
| Evaluation | — | ❌ Missing |
| Deployment | `docker-compose.yml` (Postgres only) | ❌ No app image |
| CI/CD | `.github/` | ❌ Empty |
| Docs | `README.md` (2 lines) | ❌ Missing |

---

## 3. Gap Analysis Against the Goal

Goal checklist: *"document ingestion, chunking strategy, embedding, vector store,
retrieval, reranking, LLM generation, evaluation pipeline, deployed with an API, with
clean README documentation, architecture diagrams, and clear explanations of design
decisions."*

### 3.1 Document ingestion — ⚠️ Partial
- **Now:** `POST /documents` accepts a JSON `{title, content}` — plaintext only. No file
  upload, no PDF/DOCX/HTML/Markdown parsing, no OCR, no URL ingestion.
- **Gap:** Real documents are files. A production system needs multi-format loaders,
  content extraction, encoding/mime detection, and idempotent re-ingestion (content
  hashing to avoid duplicate embeds).

### 3.2 Chunking strategy — ⚠️ Thin
- **Now:** One fixed recursive splitter (1000 chars / 200 overlap). No token-based sizing,
  no structure-aware (Markdown headers / PDF pages) or semantic chunking, no per-chunk
  metadata (page, section, source offsets).
- **Gap:** "Strategy" implies a *choice you can defend and measure*. Add ≥2 strategies and
  compare them in the eval harness (this is exactly the kind of design decision a Senior
  role wants to see justified with numbers).

### 3.3 Embedding — ✅ Good, minor gaps
- **Now:** Batched, cost/latency logged. Good.
- **Gap:** No embedding cache (re-embeds identical text), model/dim hardcoded to 1536 in
  the DB schema, no configurability for swapping embedding models.

### 3.4 Vector store — ❌ Critical gap (scalability)
- **Now:** `document_chunks.embedding` has **no ANN index** (confirmed: no ivfflat/HNSW in
  migrations). Every query is a full sequential scan + exact cosine — fine for a demo,
  **O(n) and unusable at scale**.
- **Gap:** Add an **HNSW** (or IVFFlat) index with the matching `vector_cosine_ops`
  operator class. This is the single highest-signal production fix.

### 3.5 Retrieval — ⚠️ Vector-only
- **Now:** Pure dense cosine top-k, filtered by `user_id`.
- **Gap:** No **hybrid search** (dense + BM25/`tsvector` keyword), no metadata filtering,
  no MMR/diversity, no query transformation (HyDE / multi-query). Hybrid + reranking is
  the standard production recipe and a strong talking point.

### 3.6 Reranking — ❌ Missing
- **Gap:** No second-stage reranker. Add a **cross-encoder / Cohere Rerank / LLM-as-reranker**
  or at minimum **Reciprocal Rank Fusion** over hybrid candidates. Retrieve wide (e.g. top-30),
  rerank to top-5. Measure the precision lift in the eval harness.

### 3.7 LLM generation — ✅ Solid
- **Now:** Grounded system prompt with anti-hallucination rules, source citations, cost
  tracking, provider fallback. Genuinely production-shaped.
- **Gap (polish):** No streaming responses, no citation-span mapping back to chunk offsets,
  answers not evaluated for faithfulness (see 3.8).

### 3.8 Evaluation pipeline — ❌ Missing (most important differentiator)
- **Gap:** Nothing exists. This is what separates a *demo* from an *engineered system*.
  Build an offline eval harness with:
  - A curated Q/A + gold-context dataset (`eval/dataset.jsonl`).
  - **Retrieval metrics:** hit-rate, recall@k, MRR, nDCG.
  - **Generation metrics:** faithfulness, answer relevance, context precision/recall
    (Ragas or an LLM-judge you write yourself).
  - A runnable `make eval` / CLI that prints a metrics table and writes a report, plus a
    regression gate in CI. Use it to justify chunking/reranking choices with numbers.

### 3.9 Deployment + API — ⚠️ API partial, deploy missing
- **Now:** FastAPI app with JWT auth; Postgres via docker-compose.
- **Gaps:**
  - No **Dockerfile** for the app; compose only runs Postgres.
  - No **CI/CD** (`.github/workflows` empty) — no lint/type/test/build automation.
  - **README claims "deployed on AWS"** but no IaC/deploy config exists — fix the claim or
    ship a real deploy (Render/Fly/Railway is fine and cheaper to demo than AWS).
  - API polish: `DELETE /documents/{id}`, ingestion status, health/readiness for the RAG
    subsystem, rate limiting, streaming query endpoint.

### 3.10 README, diagrams, design writeups — ❌ Missing
- **Now:** 2-line README; `docs/` covers the *booking agent*, not RAG.
- **Gap:** A portfolio README (quickstart, architecture, decisions, eval results, demo
  GIF), an **architecture diagram** (ingestion + query flows), and an
  Architecture-Decision-Record set explaining *why* pgvector, *why* HNSW, *why* this
  chunk size, *why* this reranker — backed by eval numbers.

---

## 4. Notable Code-Level Issues to Fix Along the Way

1. **N+1 queries in `DocumentService.query`** (`document_service.py:80` and `:95`): the
   parent `Document` is fetched inside two separate loops over chunks. Denormalize
   `document_title` onto `DocumentChunk`, or batch-load titles by id. `agent/tools.py::search_documents`
   has the same pattern.
2. **No vector index** — see 3.4. Highest priority.
3. **Embedding dim hardcoded** (`Vector(1536)`) couples the schema to one model; make it a
   config-driven constant referenced by both model and migration.
4. **Retrieval leaks no score** — `search_similar_chunks` discards the distance, so
   `similarity_rank` is positional only and can't be thresholded. Return the score.
5. **README over-claims** ("deployed on AWS", "custom MCP server") relative to what's in
   the repo — reconcile before it goes in a portfolio.

---

## 5. Prioritized Implementation Plan

Ordered by production-signal-per-effort. Each item lists the concrete files to add/change.

### Phase 1 — Make retrieval production-correct (highest signal)
- **Add HNSW index** — new Alembic migration:
  `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`. *(fixes 3.4)*
- **Return similarity scores** from `search_similar_chunks`; expose in `ChunkSource`.
  *(repository + schema)*
- **Fix N+1** — denormalize `document_title` onto `DocumentChunk` (migration + model +
  ingestion), simplify `query()` and `search_documents`.
- **Chunk metadata** — add `page`/`section`/`char_start`/`char_end` columns for future
  citation mapping.

### Phase 2 — Ingestion & chunking strategy
- **File loaders** — new `ingestion/loaders.py` (PDF via `pypdf`, DOCX, HTML via existing
  `beautifulsoup4`/`markdownify`, plain/Markdown). New `POST /documents/upload`
  (`multipart`, `UploadFile`) alongside the JSON route.
- **Content hashing** for idempotent ingestion (skip re-embedding unchanged content).
- **Pluggable chunkers** — `ingestion/chunking.py` with ≥2 strategies (recursive vs
  token-based vs structure-aware); make strategy a config choice.
- **Embedding cache** keyed by content hash.

### Phase 3 — Hybrid retrieval + reranking
- **Keyword search** — Postgres `tsvector` column + GIN index on chunk content.
- **Hybrid + RRF** — `retrieval/hybrid.py` fusing dense + keyword candidates.
- **Reranker** — `retrieval/reranker.py` (Cohere Rerank or a local cross-encoder;
  LLM-judge fallback). Retrieve top-30 → rerank → top-5.

### Phase 4 — Evaluation pipeline (portfolio centerpiece)
- `eval/dataset.jsonl` — curated questions with gold answers/contexts.
- `eval/retrieval_eval.py` — recall@k, MRR, nDCG.
- `eval/generation_eval.py` — faithfulness / relevance / context precision (Ragas or
  hand-written LLM-judge using the existing `LLMClient`).
- `eval/run.py` + `make eval` — prints metrics table, writes `eval/report.md`.
- Use results to A/B chunking strategies and quantify the reranking lift.

### Phase 5 — Deployment & API hardening
- **Dockerfile** (multi-stage, `uv`-based) + extend `docker-compose.yml` to run the app.
- **CI** — `.github/workflows/ci.yml`: ruff, mypy, pytest (with a pgvector service),
  build image. **Eval regression gate** on PRs.
- **Deploy** — pick one target (Render/Fly/Railway recommended for demo cost); or remove
  the AWS claim. Add deploy config + live URL in README.
- **API polish** — `DELETE /documents/{id}`, streaming `POST /documents/query/stream`,
  rate limiting, RAG readiness probe.

### Phase 6 — Documentation & diagrams
- **README rewrite** — problem framing, architecture diagram, quickstart, API examples,
  **eval results table**, live demo link/GIF, design-decisions section.
- **Architecture diagram** — ingestion pipeline + query/RAG pipeline (Mermaid or the
  existing matplotlib setup; the repo already renders `docs/*.png`).
- **ADRs** — `docs/adr/` short records: pgvector vs dedicated vector DB, HNSW params,
  chunk-size choice, reranker choice — each citing eval numbers.

---

## 6. Suggested Repository Shape (target)

```
src/paddington/
  ingestion/
    loaders.py        # PDF / DOCX / HTML / MD -> text + metadata
    chunking.py       # pluggable strategies
    pipeline.py       # load -> chunk -> embed -> store (idempotent)
  retrieval/
    dense.py          # pgvector cosine (indexed)
    keyword.py        # tsvector / BM25
    hybrid.py         # RRF fusion
    reranker.py       # cross-encoder / Cohere / LLM-judge
  ...
eval/
  dataset.jsonl
  retrieval_eval.py
  generation_eval.py
  run.py
  report.md           # generated
Dockerfile
.github/workflows/ci.yml
README.md             # portfolio-grade
docs/
  architecture.md + diagrams
  adr/
```

---

## 7. Portfolio Positioning (for a Senior AI Engineer application)

What reviewers will look for, and where this plan delivers it:

- **Systems thinking, not glue code:** the index fix, hybrid+rerank pipeline, and
  layered architecture show you understand *why* each stage exists.
- **Measurement culture:** the eval harness + CI regression gate is the strongest Senior
  signal — you make decisions with numbers, not vibes.
- **Production concerns:** Docker, CI, cost/latency logging (already present), retries and
  fallbacks (already present), auth (already present).
- **Communication:** README + diagrams + ADRs demonstrate you can justify trade-offs.

> **Recommendation:** consider extracting the RAG system into its **own repository** (or a
> clearly separated top-level module) so "Project 1" reads as a focused *production RAG
> system* rather than a feature inside a web-automation agent. Reconcile the README claims
> with reality first — accuracy is itself a Senior signal.

### Suggested sequencing if time-boxed
1. Phase 1 (correctness) + Phase 4 (eval) first — together they turn a demo into an
   *engineered, measured* system.
2. Phase 3 (rerank) — biggest quality lift you can then *prove* with the eval numbers.
3. Phases 2, 5, 6 to round out ingestion, deployment, and presentation.

---

_Generated as a planning document. No application code was modified._
