# Document AI setup runbook

**Purpose:** everything that has to exist in Google Cloud before `OCR_ENABLED=true`
does anything, in the order it has to exist.

Document AI is not one API key. It is a *processor* — a versioned model instance you
create inside a project, in a region, addressed by an id that does not exist until you
create it, reachable only on that region's endpoint, and callable only by a principal
holding a specific role. Five things, each of which fails differently when it is
missing. This document is the order that makes each failure obvious.

**What it costs:** Layout Parser bills roughly **$10 per 1,000 pages** (verify on the
[pricing page](https://cloud.google.com/document-ai/pricing) — it moves). There is no
free tier for it. Failed requests (4xx/5xx) are not billed. Enterprise Document OCR is
far cheaper at ~$1.50 per 1,000 pages but **cannot read DOCX/XLSX/PPTX**, which is half
of why this system uses Layout Parser at all.

---

## 1. Project and billing

Layout Parser will not run in a project without billing enabled.

```bash
gcloud config set project PROJECT_ID
gcloud beta billing projects describe PROJECT_ID   # billingEnabled: true
```

This repo already carries a service-account key under `secrets/` for the Google Drive
connector. Reusing its project is the simple path; if that project has no billing, use
another and mint a separate key (`DOCUMENTAI_SERVICE_ACCOUNT_FILE` exists for exactly
this case and falls back to `GOOGLE_SERVICE_ACCOUNT_FILE` when unset).

## 2. Enable the APIs

```bash
gcloud services enable documentai.googleapis.com storage.googleapis.com
```

`storage.googleapis.com` is needed only for the batch path (step 5). Enabling it now
costs nothing and saves a confusing failure later on the first large document.

## 3. Create the processor

Console → **Document AI → Processor Gallery → Layout Parser → Create**. Or:

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://us-documentai.googleapis.com/v1/projects/PROJECT_ID/locations/us/processors" \
  -d '{"displayName":"production-rag-layout","type":"LAYOUT_PARSER_PROCESSOR"}'
```

Two choices here are permanent:

- **Region** (`us` or `eu`). It is baked into the endpoint *and* into the processor's
  resource name. Moving regions later means creating a new processor and re-ingesting
  anything you want re-parsed. Pick `eu` if the corpus is EU-resident.
- **Processor type.** `LAYOUT_PARSER_PROCESSOR` is the only general-purpose processor
  that reads OOXML — "HTML and OOXML support are only available with layout parser".
  `OCR_PROCESSOR` is cheaper and would handle scanned PDFs, but no spreadsheets.

The response's `name` ends in a hex id. That tail is `DOCUMENTAI_PROCESSOR_ID`.

## 4. Grant IAM

```bash
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:SA_EMAIL" \
  --role="roles/documentai.apiUser"
```

Note the scope difference from the Drive connector: Drive credentials are minted with
`drive.readonly`, which Document AI rejects. This system builds its Document AI
credentials separately with `https://www.googleapis.com/auth/cloud-platform` — that is
why the two live in different modules even when they read the same key file.

## 5. Create the batch bucket (large documents only)

Needed only for documents past `DOCUMENTAI_BATCH_THRESHOLD_PAGES` (default 60). Below
that, documents are sharded and sent inline and no bucket is touched.

```bash
gcloud storage buckets create gs://BUCKET --location=US --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding gs://BUCKET \
  --member="serviceAccount:SA_EMAIL" --role="roles/storage.objectAdmin"
```

The worker deletes its staging objects in a `finally`, so the bucket should stay empty.
Add a 1-day lifecycle rule anyway — it is the backstop for the case where the process
dies between upload and cleanup:

```bash
echo '{"rule":[{"action":{"type":"Delete"},"condition":{"age":1}}]}' > /tmp/lifecycle.json
gcloud storage buckets update gs://BUCKET --lifecycle-file=/tmp/lifecycle.json
```

## 6. Smoke-test before touching the application

Do this before setting anything in `.env`. It separates "my Google configuration is
wrong" from "my code is wrong", which is the entire reason to do it in this order —
those two failures look identical from inside the worker.

```bash
B64=$(base64 -i sample.pdf | tr -d '\n')
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://us-documentai.googleapis.com/v1/projects/PROJECT_ID/locations/us/processors/PROCESSOR_ID:process" \
  -d "{\"rawDocument\":{\"mimeType\":\"application/pdf\",\"content\":\"$B64\"}}" \
  | jq '.document.documentLayout.blocks | length'
```

A number greater than zero means all five things are in place.

| Symptom | Cause |
|---|---|
| `404` / `NOT_FOUND` | wrong region in the URL, or a processor id that does not exist |
| `403` / `PERMISSION_DENIED` | the caller lacks `roles/documentai.apiUser` |
| `INVALID_ARGUMENT` about pages | over 15 pages — expected for the raw API; the application shards |
| `billing` in the message | step 1 |

## 7. Configure the application

```bash
OCR_ENABLED=true
DOCUMENTAI_PROJECT_ID=PROJECT_ID
DOCUMENTAI_LOCATION=us
DOCUMENTAI_PROCESSOR_ID=<hex id from step 3>
DOCUMENTAI_PROCESSOR_VERSION=pretrained-layout-parser-v1.0-2024-06-03
DOCUMENTAI_SERVICE_ACCOUNT_FILE=secrets/your-key.json   # or leave empty to reuse the Drive key
DOCUMENTAI_GCS_BUCKET=BUCKET                            # only if step 5 was done
```

**On pinning the version.** `DOCUMENTAI_PROCESSOR_VERSION` names an exact model, not
"whatever is current". That is deliberate and is the same argument as
`NORMALIZER_VERSION` and `CHUNKER_VERSION`: extraction output is a silent input to every
stored embedding, so a model that moved underneath us would change what a re-ingest
produces while the source file looks identical — and nothing downstream could detect it.
Newer versions exist (`v1.5` is Gemini-2.5-backed and better on complex PDFs). Adopting
one is a deliberate change: bump `DOCAI_EXTRACTOR_VERSION` in
`src/production_rag/ingestion/document_ai.py` at the same time, so
`metadata->>'extraction_method'` still identifies which documents were parsed under
which rules.

## 8. Verify end to end

```bash
docker compose up -d
uv run alembic upgrade head
uv run arq production_rag.worker.WorkerSettings     # separate terminal
uv run uvicorn production_rag.main:app --reload     # separate terminal

curl -F file=@scanned.pdf -H "Authorization: Bearer $TOKEN" \
  localhost:8000/documents/upload
curl -H "Authorization: Bearer $TOKEN" localhost:8000/documents/jobs/$JOB_ID
```

Expect `succeeded`. Then confirm the document records where its text came from, and that
page provenance survived sharding:

```sql
SELECT metadata->>'extraction_method' FROM documents WHERE id = '<doc>';
-- docai-layout-v1

SELECT min(page), max(page), count(*) FROM document_chunks WHERE document_id = '<doc>';
-- max(page) must equal the PDF's real page count, not 15
```

Upload an `.xlsx` and a `.pptx` too — those have no local parser, so they exercise the
direct-to-Document-AI path rather than the fallback.

For a document past the batch threshold, confirm the bucket is empty afterwards:

```bash
gcloud storage ls -r gs://BUCKET/docai/    # should list nothing
```

---

## Operating notes

**Quotas.** 120 online requests/minute per processor per project; 5 concurrent batch
operations per project per region. `DOCUMENTAI_MAX_CONCURRENCY` (default 4) is set
against the first of these — four in flight at ~5-10s each is roughly 25-50 requests per
minute, well inside the limit with room for several workers.

**Limits that drive the design.** Online: 15 pages, 20 MB per request. Batch: 500 pages,
1 GB per file. PDFs are sharded client-side to fit the online limits; OOXML cannot be
sharded (you would have to rewrite the container), so an oversized one goes to batch.

**Cost control.** `DOCUMENTAI_MAX_PAGES` (default 500) refuses an oversized document
before the first billable call. Retries are automatic, so an unguarded 2,000-page scan
is ~$20 *per attempt*. A successful extraction is cached on the job row, so a job that
fails after extraction — while embedding, say — does not re-buy the OCR on retry.

**Watching spend.** The processor's page counter is in the Console under Document AI →
your processor. Compare it against pages actually submitted; a gap means the shard math
or the cache is wrong.

**What is deliberately off.** `enable_image_annotation` and `enable_table_annotation`
(Gemini-written descriptions of figures and tables) and Layout Parser's own chunker. The
chunker is non-negotiable — chunk boundaries are pinned by `CHUNKER_VERSION` and the
resume cursor depends on them being re-derivable. The annotations are the obvious next
lever for a figure-heavy corpus, and turning them on should come with a
`DOCAI_EXTRACTOR_VERSION` bump.
