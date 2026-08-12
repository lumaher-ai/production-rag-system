# Technical Report — Production RAG System Roadmap

**Project:** Paddington RAG (Project 1)
**Author:** _(you)_
**Date:** 2026-07-15
**Purpose:** Gap analysis and implementation plan to turn the current RAG MVP into a
production-grade, portfolio-ready system suitable for a Senior AI Engineer application.

> ### ⚠️ Status as of 2026-08-07 — partially superseded
>
> **Phases 1 and 2 have shipped.** The current-state inventory (§2) and gap analysis (§3)
> below describe the repository as it stood on 2026-07-15 and are **no longer accurate**:
> the HNSW index, similarity scores, N+1 fix, chunk provenance columns, file loaders, and
> idempotent ingestion all exist now. Completed items are marked ✅ inline.
>
> **For the current gap analysis, read [`rag-production-decisions.md`](rag-production-decisions.md)** —
> it re-frames the remaining work as ~49 explicit engineering decisions (embedding model,
> vector store, index selection, fusion strategy, eval thresholds, …), verified against
> the source rather than against this document.
>
> This file is kept as a dated record of the original plan, so the before/after is legible.

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

> **Stale as of 2026-08-07.** The "Status" column reflects 2026-07-15. A "Now" column has been
> added showing verified current state; see `rag-production-decisions.md` §0 for the full
> claim-by-claim correction.

| Component | File | Status (2026-07-15) | Now (2026-08-07) |
|---|---|---|---|
| Chunking | `services/document_service.py` (`RecursiveCharacterTextSplitter`, 1000/200) | ✅ Works, single strategy | ⚠️ Still one strategy, now applied per structural segment |
| Embedding | `llm/embedding_service.py` (LiteLLM, `text-embedding-3-small`, 1536-d, batched) | ✅ Solid | ⚠️ Config-wired, but unbounded batch size; no chunk-level cache |
| Vector store | `models/document.py` (`Vector(1536)`), pgvector | ⚠️ No index | ✅ **HNSW index shipped** (`1188038e4c5b`) |
| Retrieval | `repositories/document_repository.py::search_similar_chunks` (cosine top-k) | ⚠️ Vector-only | ⚠️ Vector-only, **scores now returned**; filtered-ANN recall bug open |
| Generation | `services/document_service.py::query` (grounded prompt, sources, cost) | ✅ Solid | ✅ Unchanged; `retrieve()` split out for eval |
| Agent tools | `agent/tools.py` (`search_documents`, `list_documents`, `get_document_content`) | ✅ Works | ✅ N+1 resolved via denormalization |
| API | `routes/documents.py` (upload / query / list, JWT-auth) | ⚠️ Partial | ⚠️ `+ /documents/upload`; still no delete/stream/status |
| Ingestion input | `schemas/document.py` (raw `content` string only) | ❌ No file upload | ✅ **`ingestion/loaders.py`** — PDF/DOCX/HTML/MD via MIME registry |
| Reranking | — | ❌ Missing | ❌ Still missing |
| Evaluation | — | ❌ Missing | ❌ Still missing — now the top priority |
| Deployment | `docker-compose.yml` (Postgres only) | ❌ No app image | ❌ Unchanged |
| CI/CD | `.github/` | ❌ Empty | ❌ Unchanged |
| Docs | `README.md` (2 lines) | ❌ Missing | ✅ Portfolio README written |

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

1. ✅ **DONE — N+1 queries in `DocumentService.query`**: fixed by denormalizing
   `document_title` and `owner_id` onto `DocumentChunk` (`757e706cd39a`).
2. ✅ **DONE — No vector index**: HNSW with `vector_cosine_ops` shipped in `1188038e4c5b`.
3. ⬜ **OPEN — Embedding dim hardcoded** (`Vector(1536)`) couples the schema to one model.
   `settings.embedding_model` is now swappable but the column is not, so the two can silently
   disagree. See decisions doc **C2**.
4. ✅ **DONE — Retrieval leaks no score**: `search_similar_chunks` returns
   `(chunk, 1 - cosine_distance)`, surfaced as `ChunkSource.similarity_score`.
5. ✅ **DONE — README over-claims**: rewritten. (It now *under*-claims instead — corrected
   2026-08-07.)

**Defects found since, not in the original list** (full detail in the decisions doc, Tier 0):

6. **Filtered-ANN under-return** — pgvector post-filters the HNSW scan, so `WHERE owner_id`
   can return fewer than `top_k` rows with no error. Multi-tenant recall degrades silently.
   Decisions doc **D3** — the highest-priority correctness bug in the repo.
7. **Mixed embedding models rank silently** — no query-time guard that stored vectors were
   computed under the current model, and no backfill path. **C5**.
8. **Dead `filters` parameter** — accepted by `retrieve()`/`query()` and folded into the
   answer-cache key, but never applied to the query. **E4**.
9. **Unbounded `top_k` and context budget**; **default JWT secret boots in production**.
   **E8**, **H8**.

---

## 5. Prioritized Implementation Plan

Ordered by production-signal-per-effort. Each item lists the concrete files to add/change.

### Phase 1 — Make retrieval production-correct (highest signal) — ✅ COMPLETE
- ✅ **Add HNSW index** — `1188038e4c5b`, `vector_cosine_ops`, `m=16`, `ef_construction=64`
  set explicitly and documented. *(fixed 3.4)*
- ✅ **Return similarity scores** — `search_similar_chunks` returns `(chunk, similarity)`;
  exposed as `ChunkSource.similarity_score`.
- ✅ **Fix N+1** — `document_title` + `owner_id` denormalized onto `DocumentChunk`
  (`757e706cd39a`); `query()` and `search_documents` simplified.
- ✅ **Chunk metadata** — `page` / `section` / `char_start` / `char_end` added
  (`b5c1f0ed6025`) and populated per structural segment.

> **Follow-up now open:** the index exists but its params are untuned, `hnsw.ef_search` is
> never set, and the `owner_id` filter degrades ANN recall. See decisions doc **D2**/**D3**.

### Phase 2 — Ingestion & chunking strategy — ⚠️ MOSTLY COMPLETE
- ✅ **File loaders** — `ingestion/loaders.py` with a MIME→loader registry (PDF via `pypdf`,
  DOCX, HTML via `markdownify`, plain/Markdown), emitting `ExtractedSegment` with page/section
  provenance. `POST /documents/upload` (multipart) ships alongside the JSON route.
- ✅ **Content hashing** — `(user_id, source)` identity + `content_hash` + `chunker_version` +
  `embedding_model` gating; unchanged sources cost zero embedding calls, edited sources are
  replaced in place, concurrent uploads collapse via savepoint.
- ⬜ **Pluggable chunkers** — still one strategy. Decisions doc **B1**/**B2**.
- ⬜ **Embedding cache** — document-level idempotency exists; no chunk-level cache.
  Decisions doc **C4**, which also flags an unbounded `embed_batch` size as the more urgent
  half of that item.
- ✅ **Not in the original plan, now known to matter — all three since closed.** Ingestion was
  synchronous and held a transaction across the whole embed (**A2**, 2026-08-07); there was no
  normalization pass (**A3**, 2026-08-07); and a scanned PDF ingested silently as a near-empty
  document (**A6**, 2026-08-08 — a chars-per-page gate, a `failed_ingestions` dead-letter table,
  and Document AI OCR, which also brought `.xlsx`/`.pptx` support the original plan never had).

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

> **Updated sequencing (2026-08-07).** Phases 1–2 are done, and the recommendation above still
> holds for what remains — with one addition ahead of it. The verified sequence is now:
>
> **Tier 0 — six correctness bugs already live in the code** (D3 filtered-ANN recall, C5 mixed
> embedding models, C2 hardcoded dim, E4 dead filter param, E8 unbounded context, H8 default
> JWT secret). None need a new subsystem; all are latent production incidents.
> **Tier 1 — Phase 4 (eval).** Every recommendation in Phases 2–3 is a guess until this exists.
> **Tier 2 — Phase 3** (hybrid + RRF + reranker), now measurable.
> **Tier 3 — Phase 5** plus async ingestion, metrics, and cost controls.
>
> Full reasoning and per-decision detail: [`rag-production-decisions.md`](rag-production-decisions.md).

---

_Generated as a planning document. No application code was modified._
