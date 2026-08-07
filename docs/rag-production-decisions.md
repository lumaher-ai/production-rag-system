# Production RAG — Decision Gap Report

**Project:** production-rag
**Date:** 2026-08-07
**Supersedes the gap analysis in:** `docs/rag-production-roadmap.md`
**Reference framing:** Hampiholi, *Building Production-Grade RAG Systems* — "RAG is not about adding
context to an LLM. It is about engineering a retrieval system whose outputs can be trusted by a
generative model."

## How to read this document

The roadmap answered *"what is missing?"*. This report answers *"what must be **decided**?"* —
because a production RAG system is not a feature list, it is a **chain of engineering decisions**,
each of which has options, a cost, and a way to be validated with numbers.

Every heading below is one decision. Each carries:

| Field | Meaning |
|---|---|
| **State** | ✅ decided & implemented · ⚠️ implicitly decided (default, never justified/measured) · ❌ open |
| **Now** | What the repo actually does today (verified against source, not the roadmap) |
| **Options** | The realistic choices |
| **Call** | The recommended decision for *this* system, with the reason |
| **Proof** | The measurement that would let you defend the call in an interview or a postmortem |

An ⚠️ is not a bug. It means a default is in production without evidence — which is exactly the
class of thing that separates a demo from an engineered system. The dangerous entries are the
⚠️s, not the ❌s: an ❌ is a known hole, an ⚠️ is an *unknown* one.

---

## 0. Correction to the existing roadmap

`docs/rag-production-roadmap.md` describes a state the repo has outgrown. Verified as of this
commit:

| Roadmap claim | Reality |
|---|---|
| "No ANN index — every query is a sequential scan" | **Shipped.** `migrations/…1188038e4c5b` creates an HNSW index with `vector_cosine_ops`, `m=16`, `ef_construction=64` |
| "Retrieval leaks no score" | **Fixed.** `search_similar_chunks` returns `(chunk, 1 - cosine_distance)`; surfaced as `ChunkSource.similarity_score` |
| "N+1 queries in `query()`" | **Fixed.** `document_title` and `owner_id` denormalized onto `DocumentChunk` |
| "No file upload / plaintext only" | **Shipped.** `ingestion/loaders.py` (PDF/DOCX/HTML/Markdown/text) via a MIME registry, `POST /documents/upload` |
| "No chunk metadata" | **Shipped.** `char_start` / `char_end` / `page` / `section` columns, populated per segment |
| "No idempotent re-ingestion" | **Shipped.** `(user_id, source)` unique identity + `content_hash` + `chunker_version` + `embedding_model` gating; race-safe via savepoint |
| "Embedding dim hardcoded" | **Still true.** `Vector(1536)` in `models/document.py` is not derived from `settings.embedding_model` |
| "Embedding cache keyed by content hash" | **Still missing.** Document-level idempotency skips whole unchanged docs; a *chunk-level* cache does not exist |

**Also corrected 2026-08-07:** the README's TL;DR and status table claimed "no vector index (every
query is a sequential scan), no file loaders." That was false and understated the work — a README
that under-claims is as much an accuracy defect as one that over-claims. Both have been reconciled
against verified state, and the roadmap now carries a banner pointing here.

So the honest gap is narrower and sharper than the roadmap implies. **Phases 1 and 2 are done.**
What remains is: retrieval quality (Part E), the measurement layer that would justify any of it
(Part G), and the operational envelope (Part H).

---

# Part A — Ingestion decisions

> "Retrieval quality is determined long before a query is issued. It is determined during ingestion."

### A1. What are the authoritative sources, and how does content enter?

- **State:** ✅ **DECIDED & IMPLEMENTED (2026-08-07)** — upload + URL + Google Drive, one-shot pull,
  app-level credentials, `source` as a real URI.
- **Was:** One path — an authenticated user uploaded a file. `source` was a free-text bare filename.
- **Decision taken:** (a) + (b) + partial (c). Scope went past this report's original
  recommendation, deliberately: the connector interface is only proven by a second and third
  implementation, and the URI scheme is only load-bearing once more than one scheme exists.
  - **`upload://<user_id>/<filename>`** — minted server-side by `build_upload_uri`. The authority
    is the owner's **UUID, not their name**: `User.name` is mutable and non-unique and `User.email`
    is mutable PII, so either would change the source string on a rename and silently break
    `(user_id, source)` identity — the next upload of an unchanged file would duplicate rather than
    replace. Filenames are percent-encoded with `safe=""`, so `a/b.pdf` cannot forge path structure.
  - **`https://…`** via `HttpConnector` — redirects followed **by hand** so every hop is
    SSRF-checked (see below), response streamed against `max_upload_bytes`, plaintext `http://`
    refused unless explicitly enabled.
  - **`gdrive://<file_id>`** via `GoogleDriveConnector` — app-level service account. Google-native
    Docs are **exported as HTML**, not plaintext, so `HtmlLoader` recovers heading structure and the
    file keeps its `section` provenance; Slides export as text; other Google types are refused with
    a specific message rather than a generic failure.
  - **`s3://` deliberately not shipped.** It parses as a known-but-unsupported scheme and returns a
    422 naming what *is* supported, rather than a confusing 500.
- **Credentials: app-level, from `Settings`.** Every user's pull runs as the same server identity,
  so the system reaches only what the server can see. Per-user OAuth would need a token table,
  encryption at rest, and a consent flow — a scope boundary, stated rather than implied.
- **Sync: one-shot pull.** `POST /documents/ingest` fetches now and returns; re-syncing means
  calling again, which is cheap because the existing content-hash gate short-circuits unchanged
  content before any embedding spend. Scheduled/CDC sync stays blocked on **A2** (async ingestion) —
  polling on a synchronous request path would be the wrong foundation.
- **Security — this endpoint is an SSRF surface.** An authenticated caller naming an arbitrary URL
  makes the server fetch it and hands back the body as a document. `_assert_public_address` resolves
  the host and rejects private, loopback, link-local, reserved, multicast, and unspecified
  addresses — checking **every** A record, and re-checking **each redirect hop**, since automatic
  redirect-following would validate only the caller's original URL. The residual DNS-rebinding
  window is documented in the code rather than papered over. `allow_private_network_sources` exists
  solely so tests and local development can fetch loopback, and says so in its own description.
- **Migration:** `a1f7c2e94b31` deletes existing documents (chunks cascade) and clears `query_cache`.
  Pre-URI rows hold bare filenames that the new upload path cannot match, so they would duplicate on
  re-upload. With no production data, a clean reset is more honest than backfilling rows nobody
  depends on. `downgrade()` is a documented no-op — deleted data does not come back.
- **Proof:** 48 tests. The load-bearing ones: a re-ingest of the same URI returns the *same document
  id* with one row in the table; an upload and a URL fetch of the same filename stay *distinct*
  documents (the scheme is doing real work); six private-address forms are refused end-to-end
  through the route with a 502; percent-encoded and over-long filenames still round-trip through
  `parse_source_uri`.
- **Follow-ups this opened:** the `source` column is now structured data with an unenforced shape
  (a CHECK constraint or a parse-on-load validator would close that); and `A2` matters more now,
  since a URL fetch adds unbounded network time to an already-synchronous ingest transaction.

### A2. Is ingestion synchronous with the request, or queued?

- **State:** ✅ **DECIDED & IMPLEMENTED (2026-08-07)** — Redis-backed queue (**arq**), an
  `ingestion_jobs` table, batch checkpointing with resume, and `202 + job_id`.
- **Was:** `POST /documents/upload` ran load → chunk → embed *every* chunk → write inline in the
  HTTP request, in one transaction. Measured on a real 384-page PDF: 21s fetch + 12s parse +
  embedding, ~40–60s with the transaction held open throughout. A failure at chunk 847 of 1200
  discarded all 846 completed embeddings.
- **Decision taken:** (c). Both ingestion endpoints now record a job and return; a worker drives it.
  - **Queue: arq, not Celery.** The stack is async end to end, so arq tasks are plain `async def`
    and `IngestionService` is reused unchanged. Celery is sync-first and would need `asyncio.run()`
    per task or a gevent pool — an adapter layer wrapped around the actual work. Celery's name
    recognition is real; being able to say *why not Celery* is the better signal.
  - **Payload staging: Postgres `bytea` on the job row.** A worker cannot read the request's
    `UploadFile`. Bytes stage transactionally with the job and are cleared on success. Connector
    sources (`https://`, `gdrive://`) stage nothing — the worker re-fetches the URI, so a 100 MiB
    blob never touches the database for any source that has an address.
  - **Resume: batch checkpointing.** Chunks are embedded and committed 100 at a time, advancing
    `processed_chunks`. **Chunking is deterministic** for a given (content, `chunker_version`), so a
    retry re-derives the same chunk list and skips that many — resume needs no persisted chunk text,
    only the counter. `chunker_version` is stored per job; if it moves between attempts the cursor is
    meaningless and the job restarts from zero rather than splicing two chunkings together.
- **This also closed C4.** Batching bounds each embedding request; the previous code sent every
  chunk of a document in one call with no cap.
- **Two defects found by end-to-end testing, not by the test suite** — both worth more than the
  feature itself as evidence of what verification catches:
  1. **arq does not rescue a job whose worker was killed.** `retry_jobs` covers a task that *raises*;
     a `kill -9` raises nothing, so the row sat in `running` forever with no retry ever coming.
     Fixed with a `heartbeat_at` column and a `recover_stale_jobs` cron that re-queues jobs whose
     heartbeat has gone stale. Staleness is measured by heartbeat, not elapsed time — using
     `started_at` would hand a slow-but-healthy job to a second worker, and both would write the
     same chunks.
  2. **The arq `_job_id` dedup key silently dropped legitimate re-enqueues.** Set to
     `ingest:<job_id>`, it made arq treat a retry of an orphaned job as a duplicate and discard it —
     breaking the one path retries exist for. Removed: every request already mints a unique job row,
     and running a job twice is harmless because ingestion is idempotent and resumable.
- **Trade-off accepted:** ingestion is no longer atomic. Per-batch commits are what make resume
  possible, so a document under ingestion is partially visible to retrieval. Harmless for a new
  document (progressive availability); for a *replace*, old chunks drop on the first attempt so that
  document has reduced content until the job finishes. The clean fix is an `is_ready` flag filtered
  at retrieval, deferred because it forces a join into the hot ANN query and interacts with **D3**.
- **Proof (measured, not asserted):** upload of a 512 KB document returned in **0.083s** versus
  ~40–60s synchronously. Worker logs show batch commits at 100/200/…/667. Under `kill -9` at
  200/750, the job was reclaimed by the sweeper, resumed at 300 (not 0), and the final document held
  **750 chunks with 750 distinct indexes, 0→749** — no duplicates, no gaps, `attempts=2`, payload
  cleared. 19 tests cover the lifecycle, resume, chunker-change restart, orphan detection, and
  owner isolation.
- **Follow-ups this opened:** failed jobs retain their payload (so a retry needs no re-upload) and
  therefore need a retention policy; a dead-letter surface once jobs accumulate; job cancellation;
  progress via SSE instead of polling; and the A1 folder case — `gdrive://<folder_id>` fanning out
  to one job per file is natural now that a queue exists.

### A3. What normalization is applied before embedding?

- **State:** ✅ **DECIDED & IMPLEMENTED (2026-08-07)** — mechanical NFKC + whitespace, versioned into
  the idempotency gate, applied to documents *and* queries.
- **Was:** None. Loader output went straight into the splitter, so ligatures, non-breaking spaces,
  zero-width joiners, `\r\n`, and column-layout space runs all became embedded tokens.
- **Decision taken:** (b), in `ingestion/normalize.py`. Six ordered rules: NFKC → unify line endings
  → strip invisibles and control characters → collapse intra-line whitespace → cap blank runs at one
  → trim. Segments that normalize to empty are dropped. Rule order is load-bearing: **line-ending
  conversion must precede the control-character strip**, because `\r` *is* a control character, so
  stripping first deletes a lone `\r` and silently loses that line break. A unit test caught this.
- **Applied inside `build_chunks`, not as a step before it.** That follows the principle
  `ingestion/idempotency.py` already states about its key functions — a caller must have no way to
  skip it. It also keeps `char_start`/`char_end` truthful, since chunks and the stored
  `Document.content` come from the same normalized string.
- **Queries are normalized too**, under identical rules. Skipping that leaves an asymmetry that
  never errors: the corpus is NFKC-folded, so a question containing `ﬁ` would be embedded against
  text that no longer holds that character, and it would surface only as quietly worse recall.

**Why `normalizer_version` is load-bearing, precisely.** `content_hash` is computed over
*normalized* text, so a rules change that alters a document changes its hash and trips the gate on
its own — the hash is the more precise detector. The version is not redundant with it, for three
reasons the hash cannot cover:

1. **The query side is never hashed.** A normalized question is embedded and discarded, so the
   version is the only thing that can invalidate a cached answer when the rules move. It is
   therefore a mandatory keyword of `query_idempotency_key`.
2. **Targeted re-processing.** "Which documents predate the fix?" is an indexed column comparison
   (`DocumentRepository.find_stale`), not a re-normalize-the-corpus-and-diff.
3. **Robustness to a plausible refactor.** Hashing raw bytes instead of normalized text is a
   tempting optimization — it would let you skip re-parsing. The moment someone does that, the
   version is the only thing standing between them and silently mixed embeddings.

- **NFKC is lossy and that is accepted:** `x²→x2`, `½→1⁄2`, `𝐀→A`. Chosen because the retrieval win
  (a search for `file` matching a PDF's `ﬁle`) outweighs math fidelity for a prose-heavy corpus. A
  test pins this so a future change to it is deliberate; a math-heavy corpus would want NFC, which
  would be a different normalizer and therefore a different version string.
- **Existing rows were stamped `'none'`, not the current version** — the single easiest thing to get
  wrong here. Stamping them current would mark un-normalized content as up to date and permanently
  hide it from the gate, which is exactly the failure this column exists to prevent.
- **`reindex` CLI** — `find_stale` → one ingestion job per document, reusing A2's queue, resume, and
  status machinery rather than a second ingestion path. This needed a third branch in
  `_materialize`: a re-index of an `upload://` document has neither staged payload (cleared on
  success) nor a fetchable URI, so it falls back to the stored `Document.content` — which is the
  *joined* segment text, so **page/section provenance cannot be recovered**. The worker logs that per
  document, and distinguishes "provenance lost" from "there was none to lose."
- **Proof (measured on the real corpus):** all 6 documents re-indexed from `none` → `nfkc-ws-v1`.
  The arXiv paper and Drive playbook kept full page provenance through the re-fetch path (52/52 and
  22/22 chunks with `page` set); the three `upload://` `.txt` documents used the stored-body path and
  correctly logged that they had no provenance to lose. One document's chunk count dropped 233→227 —
  normalization collapsing whitespace, in the predicted direction. 179 tests pass, including the
  central regression: bump `NORMALIZER_VERSION`, re-ingest identical bytes, assert it re-embeds
  instead of short-circuiting.
- **A real hazard this exposed, worth keeping:** during verification a *stale worker process* from an
  earlier session — running code from before normalization existed — consumed one re-index job,
  applied the old gate, and reported `succeeded` without doing anything. Nothing in the queue, the
  job row, or the logs flagged it. This is the generic risk of any queue with rolling deploys: mixed
  code versions consuming the same jobs. It is also a live argument *for* the version column, since
  the affected document remained visibly stale (`normalizer_version='none'`) and the next
  `reindex --dry-run` surfaced it immediately. Worth a follow-up: workers should record their code
  version on the jobs they complete.
- **Open:** recall impact is unmeasured. The claim that normalization improves retrieval is
  reasonable and standard, but it is exactly the kind of thing **Part G** exists to prove.

### A4. What metadata is attached to a chunk, and where does it come from?

- **State:** ⚠️
- **Now:** Structural only — `page`, `section`, `char_start/end`, `document_title`, `owner_id`.
  Good provenance; **zero semantic or governance metadata.** No document type, date, author,
  language, entities, sensitivity/classification label.
- **Options:** (a) structural only; (b) + extractive (regex/heuristic: dates, doc type, language);
  (c) + model-derived (NER, classification, topic); (d) + LLM-generated summaries/keywords per chunk.
- **Call:** **(b) now, and add a `metadata JSONB` column** rather than more typed columns — you do
  not yet know which fields matter, and JSONB + a GIN index lets filtering evolve without a
  migration per field. Defer (c)/(d) until a filter or eval failure demands them.
- **Proof:** This is enabling infrastructure. Its payoff is measured in E4 (filtering) and H8
  (governance) — a chunk-level ACL is impossible without it.

### A5. What is a document's identity, and what does re-ingestion mean?

- **State:** ✅ — **the strongest decision in the repo.**
- **Now:** Identity is `(user_id, source)`, DB-enforced. Unchanged content under the same
  `chunker_version` + `embedding_model` is a genuine no-op with **no embedding spend**. Edited
  content replaces chunks in place, preserving the document id, inside one transaction. Concurrent
  uploads collapse via savepoint + `IntegrityError` recovery.
- **Gap:** No *cross-document* dedup (the same PDF uploaded under two filenames embeds twice, and
  both copies then compete for the same top-k, wasting context budget on duplicates). No deletion
  path (see H1), so the corpus only grows.
- **Call:** Keep the design. Add a `content_hash` lookup **across** a user's documents to warn on or
  link duplicates. Deletion is the more urgent sibling gap.
- **Proof:** Duplicate-chunk rate in retrieved top-k; embedding spend avoided per re-upload (already
  loggable from `document_ingest_skipped`).

### A6. What happens when a document fails to ingest?

- **State:** ❌
- **Now:** The exception propagates, the transaction rolls back, the user gets an HTTP error. Nothing
  is recorded. A scanned PDF that yields zero extractable text ingests "successfully" as an empty
  or near-empty document — a **silent** corpus hole.
- **Options:** (a) status quo; (b) quality gates at ingest (min extracted chars/page, non-empty
  segment count, detected-language check) that reject loudly; (c) + a dead-letter table and
  operator surface; (d) + OCR fallback for image-only PDFs.
- **Call:** **(b) immediately** — reject a PDF yielding < ~50 chars/page with a clear
  "this looks scanned; OCR is not supported" error. Silent partial ingestion is worse than a
  rejected upload because it corrupts recall invisibly. (c) follows naturally once A2 lands.
- **Proof:** Ingestion failure rate by cause; count of documents with anomalously low
  chars-per-page; empty-retrieval rate (H5) as the downstream symptom.

---

# Part B — Chunking decisions

### B1. Which chunking strategy?

- **State:** ⚠️ — **single strategy, never compared.**
- **Now:** `RecursiveCharacterTextSplitter`, separators `["\n\n", "\n", ". ", " ", ""]`, applied
  *per `ExtractedSegment`*. The per-segment application is genuinely good: it means chunks never
  straddle a PDF page or a Markdown heading, so `page`/`section` provenance is always truthful.
- **Options:** (a) recursive character (current); (b) token-aware recursive; (c) structure-aware
  (already half-achieved via segments); (d) semantic (embedding-similarity boundary detection);
  (e) hierarchical / parent-document (retrieve small, feed large).
- **Call:** Make the chunker **pluggable behind an interface** (`ingestion/chunking.py`, strategy
  selected by config, name folded into `CHUNKER_VERSION`) and ship **(b)** alongside (a). Then let
  the eval harness pick the winner. The roadmap is right that "strategy implies a choice you can
  defend" — but the defence has to be a number, and today there is nothing to produce one.
  **(e) is the highest-upside variant** for the PDF/DOCX corpora this system targets, and the
  `char_start`/`char_end` columns already make it implementable without schema change.
- **Proof:** Recall@10, nDCG@10, and mean context tokens per query, per strategy, on one golden set.

### B2. What size and overlap, in what unit?

- **State:** ⚠️
- **Now:** 1000 **characters**, 200 overlap. Never tuned. Note the unit mismatch: chunks are sized
  in characters but `token_count` is computed and stored in tokens, and the LLM context budget is
  in tokens — so the actual token size of a chunk drifts with content type (code and tables are far
  denser than prose).
- **Options:** (a) keep 1000/200 chars; (b) token-based ~500 tokens / 10% overlap (the article's
  baseline); (c) sweep the grid and pick empirically.
- **Call:** Move to **token units** — it is the unit that every downstream constraint is actually
  expressed in (embedding model max input, context budget, cost). Start at 512/50 and **sweep
  {256, 512, 1024} × {0, 10%, 20%}** once the harness exists. Do not hand-pick.
- **Proof:** The sweep table itself. This is the single most legible "I decide with numbers" artifact
  a portfolio reviewer can read in ten seconds. It belongs in the README.

### B3. Is a chunk embedded as-is, or enriched with context?

- **State:** ❌
- **Now:** As-is. A chunk from the middle of a document has no signal about which document it came
  from — `document_title` is stored on the row but is **not** part of the embedded text.
- **Options:** (a) raw; (b) prepend title/section to the embedded text; (c) contextual retrieval
  (an LLM writes a one-line "situating" preface per chunk at ingest time).
- **Call:** **(b) now — it is nearly free and the data is already on the row.** (c) is a real
  recall win but costs one LLM call per chunk at ingest; gate it behind eval evidence and A2's
  async pipeline. Critically: what gets *embedded* and what gets *shown to the LLM* need not be the
  same string — store both.
- **Proof:** Recall@10 delta from title-prepending alone; then cost-per-document vs. recall delta
  for (c).

### B4. How does the corpus migrate when chunking changes?

- **State:** ⚠️ — detection exists, execution does not.
- **Now:** `CHUNKER_VERSION` correctly *detects* staleness and forces re-embedding — but only when
  the user happens to re-upload that source. **There is no backfill.** Bump the constant today and
  the corpus is permanently split across two chunking regimes, silently.
- **Options:** (a) status quo (lazy, on next upload); (b) a re-index CLI that walks stale documents
  and rebuilds them; (c) online dual-write with a shadow index and atomic cutover.
- **Call:** **(b)** — `production_rag reindex --stale` reusing the stored `Document.content`, so no
  re-upload is needed. The `cli.py` module already exists as a home. This decision is what makes
  B1/B2 *actionable*: without it, you can measure a better chunking strategy but not adopt it.
- **Proof:** Count of documents whose `chunker_version` ≠ current — this should be a monitored
  gauge that alerts when non-zero, not a number you discover by hand.

---

# Part C — Embedding decisions

### C1. Which embedding model?

- **State:** ⚠️
- **Now:** `text-embedding-3-small` (1536-d), config-wired via `settings.embedding_model` and
  correctly folded into every idempotency key. Chosen by default, never benchmarked.
- **Options:** (a) keep 3-small; (b) `text-embedding-3-large` (3072-d); (c) an open model
  (BGE-M3, E5-Mistral, Nomic) — self-hosted, no per-token cost, and BGE-M3 emits sparse + dense
  from one pass; (d) a domain-specialized model.
- **Call:** **Keep 3-small as the baseline and justify it with a bake-off**, not with silence.
  The article's criteria apply directly: training-data alignment, max input length vs. your chunk
  size (8191 tokens — a non-constraint at 512), MTEB as a *filter* not an oracle, and the reminder
  that more dimensions ≠ better retrieval. 3-small is very likely the right answer on
  cost/latency/quality for a general English corpus; the gap is that you cannot currently *say why*.
- **Proof:** Recall@10 / MRR / nDCG@10 for 2–3 candidates on the golden set, tabled against
  cost-per-million-tokens and index size. That table is the ADR.

### C2. What dimensionality, and how are vectors stored?

- **State:** ⚠️ — and it carries a **latent correctness bug.**
- **Now:** `Vector(1536)`, hardcoded in `models/document.py`. `settings.embedding_model` is
  swappable; the column is not. Change the setting to `text-embedding-3-large` and ingestion fails
  at the DB layer — or worse, a model with a *coincidentally* compatible dim writes semantically
  incompatible vectors into the same index alongside the old ones.
- **Options:** (a) keep hardcoded; (b) derive the dim from a model registry constant referenced by
  both model and migration; (c) Matryoshka truncation (3-large → 1024-d) for size/quality tuning;
  (d) `halfvec` (fp16) to halve index memory; (e) binary quantization + rerank-on-full-vectors.
- **Call:** **(b) immediately** — a single `EMBEDDING_DIMS: dict[str, int]` plus a startup assertion
  that `settings.embedding_model`'s dim matches the live column. Fail loudly at boot, not silently
  at query time. Defer (c)–(e) until index memory is an actual constraint; at this corpus size they
  are premature.
- **Proof:** A test that asserts model↔column agreement, and a boot-time check. Later: index size
  in MB and P95 search latency per storage type.

### C3. Dense only, or dense + sparse?

- **State:** ❌ — see E2, of which this is the ingestion half.
- **Now:** Dense only. No `tsvector` column, no sparse vectors. **Exact-match retrieval is
  structurally impossible**: error codes, SKUs, proper nouns, and acronyms are precisely where dense
  embeddings are weakest, and there is no second channel to catch them.
- **Options:** (a) dense only; (b) dense + Postgres `tsvector`/BM25; (c) dense + learned sparse
  (SPLADE); (d) a multi-vector model (BGE-M3) producing both.
- **Call:** **(b)** — a generated `tsvector` column + GIN index is one migration and stays inside
  Postgres, preserving the "one datastore" property that makes this system operationally simple.
  Decided here because the column must exist at ingest time; consumed in E2.
- **Proof:** Recall@10 on a query slice deliberately loaded with exact-match terms (IDs, names,
  acronyms) — the slice where dense-only should visibly fail.

### C4. Is embedding work cached and rate-controlled?

- **State:** ⚠️
- **Now:** Document-level idempotency is excellent — an unchanged source costs zero embedding calls.
  But: **no chunk-level cache** (editing one paragraph of a 500-chunk document re-embeds all 500);
  `embed_batch` sends **every chunk in one call** with no batch-size cap, no concurrency limit, and
  no retry on the embedding path specifically. A large PDF can exceed provider request limits, and
  the whole ingest fails atomically.
- **Options:** (a) status quo; (b) chunk-level cache keyed by `sha256(text + model)`; (c) bounded
  batching (e.g. 96–256 inputs/request) with bounded concurrency and retry.
- **Call:** **(c) is a correctness fix, do it first; (b) is a cost optimization, do it second.**
  The roadmap listed the cache and missed the batching — but unbounded batch size is the one that
  produces a hard failure on a real document.
- **Proof:** Embedding cost per re-ingest of an edited document; ingest success rate for documents
  > 1000 chunks; provider 429/413 rate.

### C5. How does the system survive an embedding-model upgrade?

- **State:** ⚠️
- **Now:** `embedding_model` is stored per document and gates staleness — genuinely good design.
  But as with B4 there is no backfill, and **query-time embedding uses the current setting while
  stored vectors may be from the old model.** Mixing two models' vectors in one HNSW index does not
  error; it silently returns nonsense rankings.
- **Options:** (a) lazy re-embed on next upload; (b) offline re-index CLI (shares B4's machinery);
  (c) dual-index with shadow evaluation and cutover; (d) per-model index partitioning.
- **Call:** **(b) plus a hard guard**: retrieval must refuse to rank chunks whose `embedding_model`
  ≠ the query's. Cheap to implement — a `WHERE` clause via a denormalized column on the chunk — and
  it converts a silent quality collapse into a loud, correct failure. This is the article's
  "embedding model upgrade breaking similarity" failure mode, and it is live in this codebase today.
- **Proof:** A test that ingests under model A, switches config to model B, and asserts the query
  either re-embeds or refuses — never silently mis-ranks.

---

# Part D — Vector store & index decisions

### D1. Build, buy, or extend?

- **State:** ✅ — extend (Postgres + pgvector).
- **Now:** pgvector on `pgvector/pgvector:pg16`, same database as users/auth/cache.
- **Call:** **Correct, and defensible — keep it and write the ADR.** Single datastore means
  transactional consistency between a document and its chunks (which this codebase actively relies
  on: the in-place replace in `ingest_segments` is atomic *because* it is one transaction), one
  backup story, one connection pool, and no cross-store consistency bugs. A dedicated vector DB
  (Qdrant/Weaviate/Milvus/Pinecone) buys quantization, native hybrid, and horizontal sharding —
  none of which bind at this scale, all of which cost a distributed-consistency problem you do not
  have. Name the threshold at which you would switch: roughly **>10–50M chunks, or when
  filtered-ANN recall (D3) cannot be fixed inside Postgres.**
- **Proof:** P95 retrieval latency and Recall@k as corpus size grows — the curve that would tell you
  when the threshold arrives.

### D2. Which ANN index, with which build parameters?

- **State:** ✅ implemented, ⚠️ untuned.
- **Now:** HNSW, `vector_cosine_ops`, `m=16`, `ef_construction=64` — pgvector defaults, set
  explicitly and documented in the migration (good practice: the trade-off is written down rather
  than implicit). The operator class correctly matches the query's cosine distance.
- **Options:** HNSW (high recall, higher memory, slow build) · IVFFlat (fast build, needs training
  data, recall sensitive to `lists`) · IVF-PQ (scales past memory, lossy).
- **Call:** **HNSW is right** — the article's "HNSW for high-quality retrieval, IVF-PQ when scaling
  beyond memory" maps cleanly onto a corpus that fits in RAM. The open decision is `m` /
  `ef_construction`: defaults are a starting point, not a justified choice.
- **Proof:** A recall-vs-latency curve across `m ∈ {16, 32}` × `ef_construction ∈ {64, 128, 200}`,
  measuring ANN recall **against exact search** (run the same queries with the index disabled via
  `SET enable_indexscan = off` — that is ground truth). Index build time and size belong in the
  same table.

### D3. How is recall tuned at query time, and does filtering break it?

- **State:** ❌ — **the most technically interesting live gap in the system.**
- **Now:** `hnsw.ef_search` is never set, so every query runs at pgvector's default of 40. More
  importantly: retrieval is `WHERE owner_id = :user AND ORDER BY embedding <=> :q LIMIT k`. In
  pgvector, an HNSW scan walks the graph and **then** the filter is applied — so if the graph's
  first ~`ef_search` hits belong to other users, this query **returns fewer than `top_k` rows, or
  none at all**, while reporting perfect success. Recall silently collapses as the number of users
  grows. This is a multi-tenant correctness bug wearing the costume of a tuning parameter.
- **Options:** (a) raise `ef_search` per session (mitigates, never eliminates); (b) **partial
  indexes per tenant** (exact filtering, but does not scale past a few dozen tenants); (c)
  **partition `document_chunks` by `owner_id`** with a per-partition HNSW index — filter becomes
  partition pruning, so the ANN scan runs only inside the tenant's own vectors; (d) iterative scan
  (pgvector ≥ 0.8's `hnsw.iterative_scan = relaxed_order`, which re-scans until `LIMIT` is
  satisfied); (e) move to a store with native filtered ANN.
- **Call:** **(d) now** — a one-line session setting that makes the filter correct, available in the
  installed pgvector. **(c) as the scale answer**, since it also gives clean tenant isolation (H8)
  and cheap per-tenant deletion. Set `ef_search` explicitly per query (start at 100) regardless, so
  the recall/latency trade is a decision rather than a default.
- **Proof:** With N synthetic tenants sharing an index, measure returned-row count and Recall@10 for
  one tenant's queries as N grows — with and without the fix. **This single experiment is a better
  senior-engineer signal than any other item in this report**, because it demonstrates you found a
  correctness bug that only appears under production conditions and that no unit test would catch.

### D4. How are indexes built and maintained over time?

- **State:** ❌
- **Now:** The migration issues a plain `CREATE INDEX`, which takes an `ACCESS EXCLUSIVE` lock —
  fine on an empty table, a **full write outage** on a populated one. HNSW graphs degrade with heavy
  churn (this system deletes and rewrites every chunk on each document edit), and nothing monitors
  or rebuilds them. `maintenance_work_mem` is untuned, so a large build spills to disk.
- **Options:** (a) status quo; (b) `CREATE INDEX CONCURRENTLY` in future migrations + documented
  `maintenance_work_mem`; (c) scheduled `REINDEX CONCURRENTLY` on a churn threshold; (d) build
  offline and swap.
- **Call:** **(b)**, and record the rule for the team: *any index migration against a populated
  table uses CONCURRENTLY.* Revisit (c) only if measured recall drifts after sustained churn.
- **Proof:** Index bloat and build duration tracked over time; recall re-measured after a large
  re-ingest cycle.

### D5. What is the tenancy and isolation model at the storage layer?

- **State:** ⚠️
- **Now:** `owner_id` denormalized onto every chunk, indexed, and applied as a `WHERE` on every
  retrieval. Isolation is **enforced in application code only** — one forgotten predicate in a
  future query path is a cross-tenant data leak, and there is no second line of defence.
- **Options:** (a) app-level filter (current); (b) + Postgres **row-level security** as a backstop;
  (c) partition-per-tenant (converges with D3's recommendation); (d) database-per-tenant.
- **Call:** **(b)** — RLS is defence in depth for the failure mode with the worst blast radius, and
  it composes with (c). The article's "access control at chunk level / row-level security / tenant
  isolation" is exactly this. Note that `user_id` doubling as the tenant boundary is a real
  modelling decision (documented in the model, which is good) that will need revisiting the moment
  a document is shared between two users.
- **Proof:** A test that runs a query with a *deliberately omitted* app-level filter and asserts RLS
  still returns zero foreign rows.

---

# Part E — Retrieval decisions

### E1. Is the user's query used as-is?

- **State:** ❌
- **Now:** The raw question is embedded verbatim. No rewriting, expansion, decomposition, or
  conversational context resolution — a follow-up like "what about APAC?" retrieves against those
  four words alone.
- **Options:** (a) as-is; (b) rule-based normalization; (c) LLM rewrite (clarification, context
  injection, keyword normalization, noise reduction); (d) multi-query fan-out; (e) HyDE.
- **Call:** **Defer — deliberately, and say why.** The article is emphatic that query rewriting must
  not be assumed to help: it adds an LLM call to the critical path (latency + cost) and risks
  semantic drift. **You cannot evaluate it before Part G exists.** The one exception worth taking
  early is **context injection for multi-turn** — if the chat surface allows follow-ups, a bare
  pronoun query is not a tuning issue, it is broken retrieval.
- **Proof:** Recall@10 and P95 latency, with and without rewriting, on the same golden set. Report
  both; adopt only if recall gain exceeds the latency cost you are willing to pay.

### E2. Dense-only, or hybrid retrieval?

- **State:** ❌ — **the largest remaining quality gap.**
- **Now:** Dense cosine only. Per C3, exact-match queries have no working retrieval path.
- **Options:** (a) dense only; (b) dense + BM25/`tsvector`; (c) + learned sparse; (d) dense +
  metadata pre-filter only.
- **Call:** **(b).** Hybrid is the standard production recipe precisely because dense and lexical
  retrieval fail on *disjoint* query classes — dense misses exact tokens, BM25 misses paraphrase.
  Postgres full-text keeps it in one datastore. Implement as `retrieval/keyword.py` beside
  `retrieval/dense.py`, both returning ranked `(chunk_id, score)` lists.
- **Proof:** Recall@10 broken out by query type (paraphrase / exact-term / multi-hop). The value of
  hybrid is invisible in an aggregate number and obvious in a per-slice one — which is itself the
  lesson worth showing.

### E3. How are multiple ranked lists fused?

- **State:** ❌ (blocked on E2)
- **Options:** (a) **Reciprocal Rank Fusion** (rank-based, scale-free, one tunable `k`);
  (b) weighted score normalization (needs per-channel calibration, brittle); (c) learned fusion.
- **Call:** **RRF.** Cosine similarity and BM25 scores are not on comparable scales and their
  distributions shift with corpus content — rank-based fusion sidesteps calibration entirely. Start
  at `k=60`.
- **Proof:** nDCG@10 for dense-only vs. keyword-only vs. RRF. Sweep the RRF `k` and the per-channel
  candidate depth.

### E4. Can retrieval be filtered by metadata?

- **State:** ❌ — and there is a **dead parameter advertising it.**
- **Now:** `DocumentService.retrieve()` and `query()` both accept `filters: dict | None`. It is
  threaded into the answer-cache key (so different `filters` values produce different cache
  entries) but **never applied to the query** — the docstring says "reserved for forward-compat."
  Two callers passing different filters get correctly-separated cache entries containing
  *identically unfiltered* results. That is worse than not having the parameter.
- **Options:** (a) remove it until implemented; (b) implement over the A4 JSONB column; (c)
  implement and make it governance-bearing (H8).
- **Call:** **(a) today, (b) when A4 lands.** An unimplemented filter parameter on a multi-tenant
  retrieval API is a security-shaped footgun: the next engineer will reasonably assume passing
  `{"classification": "public"}` does something.
- **Proof:** A test asserting a filtered query cannot return a non-matching chunk. Once (b) lands,
  measure filtered-ANN recall — filtering interacts with D3 and can degrade it further.

### E5. Is there a second-stage reranker?

- **State:** ❌
- **Now:** Top-5 straight from ANN into the prompt. First-stage ranking is final.
- **Options:** (a) none; (b) cross-encoder (BGE-reranker, self-hosted, ~50–200 ms);
  (c) hosted API (Cohere Rerank — quality, but a network hop and a vendor);
  (d) LLM-as-reranker (best quality, worst latency/cost).
- **Call:** **(b).** A cross-encoder scores query and chunk *jointly* rather than comparing
  independently-computed vectors, which is why it reliably lifts Precision@K and MRR — the metrics
  that determine whether the right chunk is actually in the prompt. Self-hosting keeps latency
  predictable and cost fixed. Add `retrieval/reranker.py` behind an interface so (c)/(d) are
  swappable.
- **Proof:** MRR and Precision@5 before/after, **against the added P95 latency**. Report the
  trade-off, not just the win — an unreported 200 ms is a hidden cost.

### E6. How wide is the candidate pool, and when should the system refuse?

- **State:** ❌
- **Now:** `top_k=5` retrieved, all 5 used, **no score threshold anywhere.** A query with no relevant
  content still returns five chunks at cosine ~0.1, and the prompt presents them with equal
  authority. The refusal instruction in `RAG_SYSTEM_PROMPT` is the *only* defence, and it is a
  suggestion to a language model, not a control.
- **Options:** (a) status quo; (b) retrieve-wide → rerank → narrow (30 → 5); (c) + absolute
  similarity floor; (d) + reranker-score floor with explicit abstention.
- **Call:** **(b) + (d).** Retrieve 30, rerank, keep those above a threshold, cap at 5 — and if
  nothing clears the bar, **return an explicit "no relevant context found" without calling the LLM
  at all.** This is cheaper, faster, and more honest than paying for a generation that should say
  "I don't know." Calibrate the floor from the golden set's score distribution; never guess it.
- **Proof:** Abstention rate, and false-abstention rate on questions known to be answerable. Both
  matter — a system that refuses everything scores perfectly on faithfulness.

### E7. Is diversity managed, or can top-k be five copies of one passage?

- **State:** ❌
- **Now:** Nothing prevents it, and overlap makes it likely: consecutive chunks share 200 characters
  by construction, and A5 permits the same document under two filenames. The context budget is
  spent on redundancy while the answer's second required fact never appears.
- **Options:** (a) none; (b) MMR; (c) dedup by near-identical content hash; (d) cap chunks per
  document.
- **Call:** **(c) + (d) first** — both are a few lines and target the concrete causes present in
  *this* system. MMR is the more principled tool but introduces a relevance/diversity λ that needs
  its own tuning; the article's warning applies (over-weighting diversity admits irrelevant
  results). Do the cheap structural fixes, measure, then decide if λ is worth owning.
- **Proof:** Mean distinct documents per top-k; Recall@5 on multi-hop questions that require facts
  from two different documents.

### E8. How is the final context assembled?

- **State:** ⚠️
- **Now:** Chunks joined by `\n\n---\n\n` as `[Source i: title]`, in similarity order, **with no
  token budget.** `top_k` is caller-controlled and unbounded — `top_k=200` builds a prompt that
  either errors or costs a fortune. No parent-document expansion, no ordering strategy.
- **Options:** (a) status quo; (b) hard token budget with graceful truncation; (c) lost-in-the-middle
  ordering (strongest chunks at the head and tail); (d) parent-document expansion (retrieve the
  precise chunk, send its neighbours for context).
- **Call:** **(b) is non-negotiable** — an unbounded context budget is an unbounded cost and a
  latent 400 from the provider. Clamp `top_k` at the API boundary too. **(d) is the highest-value
  follow-up** and is already unlocked by the `char_start`/`char_end` columns plus the full document
  text stored on `Document.content` — small chunks retrieve precisely, large windows answer well.
- **Proof:** Context-token distribution per query (P50/P95/max); faithfulness score with and without
  parent expansion.

---

# Part F — Generation decisions

### F1. Which model generates, and what happens when it fails?

- **State:** ⚠️
- **Now:** `query()` defaults to a hardcoded `model="gpt-4.1-nano"` **in the method signature**,
  bypassing `settings.default_model` / `fallback_model` entirely. LiteLLM provides retries and
  provider fallback at the client layer, which is genuinely good — but the model choice itself is a
  literal in a function default, and callers can pass any string with no allow-list.
- **Options:** (a) status quo; (b) config-driven with a validated allow-list; (c) tiered routing
  (cheap model by default, escalate on low retrieval confidence).
- **Call:** **(b) now** — the config field exists and is being ignored, which is the kind of
  inconsistency that quietly makes an environment un-tunable. **(c) pairs naturally with E6's
  confidence score** and is a strong cost-control story once you can measure quality.
- **Proof:** Answer-correctness and cost-per-query across two or three model tiers on the golden set.

### F2. What is the grounding contract with the model?

- **State:** ✅ (with a gap)
- **Now:** A well-constructed system prompt: answer only from context, refuse when insufficient,
  cite, don't invent. Genuinely production-shaped.
- **Gap:** It is **unversioned and untested**. Editing the prompt silently changes system behaviour
  with no regression signal, and the answer cache does not include the prompt in its key — so a
  prompt change serves answers generated under the old one for up to an hour.
- **Call:** Add a `PROMPT_VERSION` constant, **fold it into the answer-cache key** alongside
  `chunker_version` and `embedding_model` (the pattern is already established and correct — the
  prompt is simply a missing input), and pin prompt behaviour with eval cases.
- **Proof:** Faithfulness and refusal-rate on adversarial "unanswerable" questions, tracked per
  prompt version.

### F3. How are claims attributed back to sources?

- **State:** ⚠️
- **Now:** Chunk-level attribution — the response lists which chunks were used, with titles,
  previews, ranks, and similarity scores. Solid, and better than most demos.
- **Gap:** No **claim-level** mapping. The user cannot tell which sentence of the answer came from
  which source, and cannot jump to the exact span — even though `char_start`/`char_end`/`page`/
  `section` are already populated and `Document.content` is stored.
- **Options:** (a) chunk-level (current); (b) inline citation markers the model emits, validated
  against the real source list; (c) post-hoc claim→span alignment.
- **Call:** **(b)** — require `[1]`-style markers and **reject or flag any marker referencing a
  source that was not retrieved.** A hallucinated citation is worse than no citation, and this is
  the cheapest place to catch it. The provenance columns exist specifically for this; using them is
  the payoff for having added them.
- **Proof:** Citation-validity rate (markers pointing at real retrieved sources) and citation
  precision on the golden set.

### F4. Streaming or blocking?

- **State:** ❌
- **Now:** Blocking. The user waits for embed + search + full generation with no feedback.
- **Call:** **Add `POST /documents/query/stream` (SSE).** Time-to-first-token is the latency the
  user actually experiences, and it is the article's named optimization for exactly this. Keep the
  blocking endpoint — evals and the answer cache both want a complete response — and note that the
  cache write must happen after the stream completes.
- **Proof:** P95 time-to-first-token vs. P95 total latency. The gap between them is the perceived
  win.

### F5. What guards sit on the output — and on the retrieved input?

- **State:** ❌ — **including one live security gap.**
- **Now:** The system prompt is the only guard. Retrieved chunks are interpolated directly into the
  system message. **A user can upload a document containing instructions and have them executed as
  system-level context** — indirect prompt injection, self-inflicted here (single-tenant blast
  radius) but a real vulnerability the moment documents are shared or ingested from URLs (A1).
  No PII detection on output, no post-hoc faithfulness check.
- **Options:** (a) prompt-only; (b) structural isolation of retrieved content (explicit delimiters,
  untrusted-data framing, put context in a user-role message rather than the system prompt);
  (c) input scanning at ingest; (d) output faithfulness check before returning.
- **Call:** **(b) now** — it is a prompt-architecture change, not a new dependency, and it closes
  the injection path that A1's roadmap would otherwise widen. **(d) for high-stakes answers**, using
  the same judge built in G3 (build it once, run it offline for eval and optionally online for
  guarding).
- **Proof:** An adversarial test set of documents containing embedded instructions; measure how many
  alter the answer. This belongs in the eval suite permanently, not as a one-off check.

---

# Part G — Evaluation decisions

> Nothing in Parts A–F can be *decided* without this part. Every "Proof" line above is currently
> unexecutable. **This is the highest-priority block in the report.**

### G1. What is the golden dataset, and where does it come from?

- **State:** ❌
- **Now:** Nothing. No dataset, no fixtures beyond unit-test text.
- **Options:** (a) hand-curated Q/A + gold chunk ids; (b) LLM-generated from your own corpus
  (question ← chunk, so the gold context is known by construction); (c) public benchmark;
  (d) production query logs, once they exist.
- **Call:** **(b) seeded, then (a) hand-audited** — generate ~150–200 Q/A/context triples from
  documents you actually ingest, then review them by hand and discard the bad ones. Deliberately
  stratify by query type: **paraphrase, exact-term, multi-hop, and unanswerable.** The unanswerable
  slice is the one most systems omit and the one that measures E6/F2 — without it you cannot
  distinguish "correctly refuses" from "never finds anything."
- **Proof:** The dataset is the instrument. Its own quality check is inter-rater agreement on a
  sample and confirmation that a trivial baseline does *not* score well on it.

### G2. Which retrieval metrics, at which thresholds?

- **State:** ❌
- **Call:** **Recall@k as the primary gate; nDCG@10 and MRR as ranking quality; Precision@5 as the
  prompt-noise proxy.** The article's framing is the right one to adopt and to state in the README:
  *Recall@k is the ceiling.* If the right chunk is not retrieved, no reranker, no prompt, and no
  model recovers it — every downstream metric is bounded by this number. Report all four; gate CI on
  Recall@10 and nDCG@10. Set thresholds from the first measured baseline, not from aspiration.
- **Proof:** Metrics stratified by the G1 query types, not just aggregated. An aggregate hides
  exactly the failures (exact-match, multi-hop) that hybrid and reranking exist to fix.

### G3. Which generation metrics, and who judges?

- **State:** ❌
- **Call:** **Faithfulness** (every claim traceable to retrieved context — the anti-hallucination
  metric and the one that justifies F2), **answer relevance**, and **correctness** against gold.
  Judge with an LLM, but **a different model than the generator** — self-judging inflates scores.
  `LLMClient` already provides multi-provider access, so this is close to free.
- **Proof:** Calibrate the judge itself: hand-label ~50 answers and measure judge-vs-human
  agreement. **An uncalibrated judge is a number, not a measurement** — and reporting judge
  agreement is a stronger signal than reporting a high faithfulness score.

### G4. Hand-rolled harness or a framework?

- **State:** ❌
- **Options:** (a) hand-written on `LLMClient`; (b) Ragas; (c) DeepEval; (d) LangSmith/TruLens.
- **Call:** **(a), deliberately.** For a portfolio piece, writing nDCG and faithfulness yourself
  demonstrates you understand what they measure — and the article's own point is that
  production-grade RAG is engineering judgment, not library selection. It also avoids a heavy
  dependency and gives you full control over the CI gate. ~300 lines. Cite Ragas as the alternative
  in the ADR and explain the trade.
- **Proof:** N/A — but the harness must be deterministic (fixed seeds, pinned judge model, cached
  judgments) or CI gating on it produces flaky failures that erode trust in the gate.

### G5. Where does eval run, and what does it block?

- **State:** ❌ (no CI at all — `.github/` is empty)
- **Options:** (a) local only; (b) CI on PR, advisory; (c) CI on PR, **blocking** on regression;
  (d) + nightly on the full set.
- **Call:** **(c) on a fast subset (~50 questions, mocked embeddings where possible), (d) nightly on
  the full set.** The regression gate is the single strongest senior signal in this entire report —
  it is the mechanism that converts "I measured it once" into "the system cannot silently get
  worse." Note the design constraint: `retrieve()` already exists as a separate, LLM-free method
  precisely so retrieval metrics are cheap and deterministic to compute. That refactor was the right
  call; G5 is what cashes it in.
- **Proof:** Metrics trend over commits, published in the README. A graph that goes up and to the
  right *with the commits that caused each jump labelled* is the artifact.

### G6. Is there an online feedback loop?

- **State:** ❌
- **Now:** No thumbs-up/down, no click capture, no correction path. Structured logs capture the
  question, chunks, and answer — the raw material exists.
- **Call:** **Defer, but instrument now.** Add a `query_feedback` table and a rating endpoint before
  you need them; retrofitting feedback onto answers you can no longer identify is impossible. The
  article's self-improving RAG (learning-to-rank, drift correction) is a real pattern but is
  premature without traffic. Say that explicitly rather than building it.
- **Proof:** Once traffic exists: thumbs-down rate by query type, and agreement between user
  feedback and the offline judge — the check on whether your eval measures what users care about.

---

# Part H — Serving, operations & governance decisions

### H1. What is the API contract?

- **State:** ⚠️
- **Now:** `POST /documents` (JSON), `POST /documents/upload` (multipart), `POST /documents/query`,
  `GET /documents`. Missing: **`DELETE /documents/{id}`** (no way to remove content — a
  right-to-erasure and cost problem), `GET /documents/{id}`, ingestion status (A2), streaming (F4),
  pagination beyond `limit=20`, and any `top_k` bound (E8).
- **Call:** Ship **DELETE first** — it is the only genuine data-lifecycle gap, and `ondelete=CASCADE`
  on the chunk FK means the storage half is already done. Then status, then streaming.
- **Proof:** Contract tests per endpoint; confirm a delete removes chunks *and* invalidates that
  user's cached answers (the existing `delete_by_user` invalidation pattern must extend to deletes,
  or the cache will serve answers citing deleted documents).

### H2. What is cached, at which layers?

- **State:** ✅ answer cache, ❌ everything else.
- **Now:** A well-designed **exact-match answer cache**: keyed on `(user, question, filters, top_k,
  chunker_version, embedding_model)`, TTL-bounded, invalidated on that user's ingest, and correctly
  bypassed by `use_cache=False` so evals never measure a warm cache. That last detail is a
  thoughtful piece of design.
- **Gaps:** the prompt version is missing from the key (F2); **no query-embedding cache** (the same
  question from two users embeds twice); no semantic/near-duplicate cache; the TTL is a fixed
  3600 s rather than reasoned from content volatility.
- **Call:** Add `PROMPT_VERSION` to the key, add a small **query-embedding cache** (cheap, high hit
  rate, and directly on the latency critical path). Skip semantic caching — near-miss cache hits
  return subtly wrong answers, and that risk is not worth the latency win at this scale.
- **Proof:** Cache hit rate, and P95 latency for hit vs. miss. Also: verify no answer outlives an
  ingest that should have invalidated it.

### H3. What is the latency budget, and how is it split?

- **State:** ❌
- **Now:** Embedding latency and LLM cost are logged per call, but **there is no end-to-end request
  timing and no target.** The article's decomposition —
  `T_total = T_rewrite + T_embed + T_search + T_rerank + T_assemble + T_generate` — cannot currently
  be reconstructed from the logs.
- **Call:** **Set a target before adding stages** (e.g. P95 < 3 s non-streaming, < 800 ms
  time-to-first-token) and instrument each stage. E1 and E5 both *spend* from this budget; deciding
  them without a budget means discovering the cost in production. Track P50/P95/P99 separately —
  a mean latency number hides the tail that users actually complain about.
- **Proof:** Per-stage latency histograms under realistic concurrency, not single-request timing.

### H4. What is the cost model, and what enforces it?

- **State:** ⚠️
- **Now:** Per-call cost is logged for both embedding and generation (genuinely ahead of most
  projects). But nothing **aggregates** it — no cost per query, per user, or per document; no
  budget; no rate limit. A user can upload 10 MiB repeatedly or issue unlimited queries.
- **Options:** (a) log only; (b) aggregate into a cost table with per-user rollups; (c) + quotas and
  rate limiting; (d) + tiered model routing (F1).
- **Call:** **(c).** Rate limiting is also the abuse control the API currently lacks entirely, so it
  pays twice. The five cost components to track are already all measurable here: embedding
  (ingest + query), storage, ANN compute, rerank (once E5 lands), and LLM tokens.
- **Proof:** Cost per query (P50/P95), cost per user, embedding-volume growth curve.

### H5. What is logged, traced, and alerted on?

- **State:** ⚠️
- **Now:** Good structured logging via `structlog` — `rag_query_completed` records question length,
  chunk sources with scores, the answer, tokens, and cost. Better than most.
- **Gaps:** no distributed tracing / request-id correlation across stages; **no metrics** (logs are
  not metrics — you cannot alert on a log line without an aggregation pipeline); and none of the
  article's retrieval-health signals are computed: **empty-retrieval rate**, similarity-score
  distribution, MRR/nDCG drift, abstention rate.
- **Call:** Add a request id threaded end-to-end, export Prometheus/OTel metrics, and treat
  **empty-retrieval rate and mean top-1 similarity as the two leading indicators** of silent
  retrieval decay. They are cheap, need no labels, and move before users complain. This is the
  article's core operational warning: without observability, retrieval quality degrades invisibly.
- **Proof:** An alert that fires on a deliberately induced regression (e.g. ingest garbage, or
  point at the wrong embedding model) before any human notices.

### H6. What are the known failure modes, and what is the fallback?

- **State:** ❌
- **Now:** No confidence scoring, no fallback retrieval path, no canary or staged rollout, no
  escalation. Live failure modes identified in this report: filtered-ANN under-return (D3),
  mixed-embedding-model ranking (C5), silent empty-text ingestion (A6), unbounded context (E8),
  prompt injection via retrieved content (F5).
- **Call:** Write them down as a **failure-mode table with detection and mitigation for each** —
  this document's D3/C5/A6 entries are the start. Then add the two cheapest controls: a retrieval
  confidence score (E6) and a fallback from dense to keyword when dense returns nothing. Reliability
  is engineered, not assumed.
- **Proof:** Fault-injection tests — each named failure mode gets a test that induces it and asserts
  the system degrades loudly rather than silently.

### H7. How is it packaged and deployed?

- **State:** ❌
- **Now:** `docker-compose.yml` runs **Postgres only**. No app Dockerfile, no CI, no deploy target,
  no migration-on-deploy step, no readiness probe (`/health` returns static config and does **not**
  check the database — a load balancer would route traffic to an app with a dead DB).
- **Call:** Multi-stage `uv` Dockerfile → compose runs app + Postgres → GitHub Actions
  (ruff, mypy, pytest against a pgvector service, image build, **and the G5 eval gate**) → one small
  deploy target (Fly/Render/Railway). Make `/health` shallow and add `/ready` that checks DB
  connectivity, pgvector availability, and the C2 embedding-dimension assertion.
- **Proof:** A green pipeline and a live URL. Note `mypy` is currently a *runtime* dependency in
  `pyproject.toml` rather than a dev one — move it before building an image.

### H8. What governs access, and what is auditable?

- **State:** ⚠️
- **Now:** JWT on every endpoint, argon2 password hashing, per-user scoping, refresh tokens, a role
  column on users. A solid foundation.
- **Gaps:** roles are not enforced at retrieval; no chunk-level classification (needs A4); no RLS
  (D5); no audit log of who retrieved which chunks; the JWT secret has an insecure default that
  will boot happily in production; documents are stored unencrypted at rest beyond disk-level.
- **Call:** In order — **(1)** refuse to start in `environment=production` with the default JWT
  secret (a one-line validator preventing the worst plausible incident); **(2)** RLS as the
  isolation backstop; **(3)** an audit log of retrieval events, since "who saw which document" is the
  question every compliance review asks and it cannot be answered retroactively; **(4)** chunk-level
  classification once A4 lands. Metadata-driven filtering is the mechanism that makes governance
  enforceable at all — which is why A4 is upstream of this.
- **Proof:** A test asserting a non-default secret is required in production; an RLS cross-tenant
  test; audit-log completeness against a known retrieval sequence.

### H9. What is the scale target, and what breaks first?

- **State:** ❌
- **Now:** Untested at any scale. No load test, no capacity model. Corpus size, QPS, and tenant count
  are all unstated, so no decision above has a stated operating point.
- **Call:** **Declare the target** — e.g. 10 tenants, 1M chunks, 10 QPS — and load-test to it.
  Prediction from the analysis above: **D3 (filtered-ANN recall) breaks first**, and it breaks
  *silently*, which is why it is the priority. Second is A2 (synchronous ingestion) under concurrent
  uploads holding long transactions.
- **Proof:** A load test at the declared target reporting P95 latency, Recall@10, and error rate
  simultaneously. Latency measured without recall is meaningless — a fast wrong answer is not a win.

---

# Part I — Advanced patterns: the decision is *when*, not *whether*

The article is explicit that advanced techniques must not be adopted prematurely, and gives the
maturity path: optimize chunking and embeddings → stabilize retrieval metrics → add reranking →
evaluate multi-stage retrieval → adopt hierarchical/graph only when justified → agentic loops last.

**This system is at step 1, with no way to verify it has finished step 1.** Every entry below is
therefore correctly deferred — but deferral is only a *decision* if the trigger is written down.

| Pattern | Adopt when | Cost if adopted early |
|---|---|---|
| **Hierarchical / parent-document retrieval** | Long structured documents where chunk-level recall is high but answers are incomplete | Low — this is the nearest and best-supported next step; `char_start`/`char_end` + `Document.content` already make it implementable |
| **GraphRAG** | Multi-hop questions traversing entity relationships fail persistently after hybrid + rerank | High — entity extraction pipeline, graph store, traversal logic, and a second thing to keep in sync |
| **Agentic RAG** | A measured slice of queries genuinely needs iterative retrieve→analyze→re-retrieve | High — multiplies latency and tokens; note the repo already has LangGraph and an agent loop, making this *tempting* rather than *justified* |
| **Self-improving / learning-to-rank** | Real traffic + G6 feedback data exist | Meaningless without traffic |

**Call:** State this maturity path in the README as a deliberate position. "We have not built
GraphRAG because our multi-hop failures are not yet measured" is a stronger engineering statement
than having built it.

---

# Prioritized decision sequence

Ordered by *what unblocks the most other decisions*, not by effort.

### Tier 0 — Correctness bugs live in the code today
1. **D3** — filtered-ANN under-return. Multi-tenant recall silently collapses as tenants grow.
2. **C5** — mixed embedding models rank silently and wrongly.
3. **C2** — hardcoded `Vector(1536)` vs. swappable `settings.embedding_model`.
4. **E4** — remove the unimplemented `filters` parameter.
5. **E8/H1** — bound `top_k` and the context budget.
6. **H8(1)** — refuse to boot in production with the default JWT secret.

*None of these need a new subsystem. All are latent production incidents.*

### Tier 1 — Build the instrument
7. **G1–G5** — golden dataset, retrieval metrics, generation metrics, harness, CI gate.

*Everything in Tiers 2–3 is a guess until this exists. It is also the strongest portfolio artifact
in the report.*

### Tier 2 — Retrieval quality, now measurable
8. **C3 + E2 + E3** — hybrid retrieval and RRF fusion.
9. **E5 + E6** — cross-encoder rerank, wide-then-narrow, abstention threshold.
10. **A3 + B3** — normalization and title-prepended embedding (cheapest recall wins available).
11. **B1 + B2** — chunking sweep, decided by numbers.

### Tier 3 — Operational envelope
12. **A2 + A6 + B4** — async ingestion, quality gates, re-index CLI.
13. **H7** — Dockerfile, CI, deploy, real readiness probe.
14. **H5 + H3 + H4** — metrics, latency budget, cost aggregation and rate limiting.
15. **F3 + F4 + F5** — citation validation, streaming, injection isolation.

### Tier 4 — Deliberately deferred
16. **E1** (query rewriting), **G6** (feedback), **Part I** — with triggers stated, not silently
    skipped.

---

## What this report changes about the project's story

The roadmap framed the work as *"a demo missing features."* The verified state says something
different and better: **the ingestion and storage layers are genuinely well-engineered** —
idempotency keyed on chunker and embedding-model version, race-safe concurrent upload, denormalized
tenancy, provenance columns, an HNSW index whose operator class and build parameters are documented
rather than defaulted, an answer cache that evals can bypass, and a `retrieve()` method split out
specifically so retrieval is measurable without an LLM. Those are the decisions of someone who has
thought about production.

What is missing is not features. It is **the measurement layer that turns the remaining choices
from opinions into decisions** — and a handful of correctness bugs that only appear under conditions
no current test creates. Tier 0 and Tier 1 are the whole story; Tiers 2–4 are execution once the
instrument exists.

That framing — *here is what I built, here is the bug I found in it that only shows up at scale,
here is the harness I built to prove the next change is an improvement* — is a considerably stronger
portfolio narrative than a longer feature list.
