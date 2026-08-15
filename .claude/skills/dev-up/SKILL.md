---
name: dev-up
description: Bring up the production-rag local stack — Postgres/pgvector and Redis via docker compose, Alembic migrations, the arq ingestion worker, and the uvicorn API. Use when asked to start, run, or boot the app locally.
disable-model-invocation: true
---

Start the stack in this order. Each step depends on the one before it.

**1. Infrastructure.** Postgres (pgvector/pg16) and Redis, both with healthchecks:

```bash
docker compose up -d
```

Wait for both to report healthy before continuing — `docker compose ps` shows the status. Migrating
against a Postgres that isn't accepting connections yet is the most common failure here.

**2. Migrations.**

```bash
uv run alembic upgrade head
```

**3. Ingestion worker.** Run in the background; it consumes the Redis queue. Without it, uploads
return `202` with a `job_id` that stays `pending` forever.

```bash
uv run arq production_rag.worker.WorkerSettings
```

**4. API.**

```bash
uv run uvicorn production_rag.main:app --reload
```

Then report: `http://127.0.0.1:8000` — interactive docs at `/docs`.

## Preconditions worth checking first

- `.env` must exist (`cp .env.example .env`) with a real `OPENAI_API_KEY`. Embedding calls fail at
  ingest time, not at boot, so a missing key looks like a working app until the first upload.
- `ANTHROPIC_API_KEY` is only needed for the fallback model.
- OCR (`OCR_ENABLED`) is off by default; scanned PDFs will be rejected by the quality gate and
  `.xlsx`/`.pptx` return 415 until Document AI is configured. That's expected, not a bug.

## Shutting down

`docker compose down` stops the containers and keeps the volumes. Only use `-v` if the intent is
genuinely to discard the database — confirm before doing so.
