# CLAUDE.md

Guidance for Claude Code working in this repo. Facts that are visible in the code do not belong
here — this file is for what cannot be inferred by reading it.

## Commands

Everything runs through `uv`. There is no `pip install` path and no activated-venv workflow; prefix
commands with `uv run` rather than calling `python`, `pytest`, or `ruff` directly.

```bash
uv sync --frozen                                  # install exactly uv.lock (what CI does)
uv add <pkg>                                      # add a dep — commit the uv.lock change with it
uv run pytest                                     # full suite
uv run ruff check . && uv run ruff format --check .
uv run mypy src                                   # src only; tests/ is not type-checked
```

Bring up the local stack in this order — the API assumes the schema exists, and ingestion silently
queues with no consumer if the worker isn't running:

```bash
docker compose up -d                              # postgres/pgvector + redis
uv run alembic upgrade head
uv run arq production_rag.worker.WorkerSettings   # ingestion worker
uv run uvicorn production_rag.main:app --reload   # API
```

Slash commands wrap the routine work: `/dev-up` (stack), `/verify` (full gate), `/pr`, `/decision`,
`/journal`. Prefer them — they encode conventions this file only summarizes.

## Environment

Copy `.env.example` to `.env`. Real keys are needed only for the paths that call out:
`OPENAI_API_KEY` (embeddings, `text-embedding-3-small`), `ANTHROPIC_API_KEY` (fallback model),
`DATABASE_URL`, `REDIS_URL`.

**Every setting must keep a working default.** A fresh clone with no `.env` has to boot, CI writes
no `.env` on purpose, and a test pins this. Adding a setting with no default breaks that invariant —
give it one, or expect CI to fail where the setting is read.

Document AI (`OCR_ENABLED`, `DOCUMENTAI_*`) is off by default and needs GCP credentials; see
`docs/document-ai-setup.md`. With it unconfigured, `POST /documents/upload` answers 415 for
XLSX/XLSM/PPTX that would otherwise be accepted — the supported-format list is a property of the
deployment, not of the codebase.

## Testing

`pytest` with `asyncio_mode = "auto"` — async tests need no marker.

The suite splits in two. Most tests run on in-memory SQLite and need nothing. The rest use the
opt-in `pg_engine` fixture (testcontainers + pgvector), which requires a running Docker daemon.
**With Docker down they report as `ERROR ... DockerException`, not as failures** — that is a blocked
test, not a regression, and should never be reported as a failing suite. CI has Docker, so that
third of the suite genuinely runs there.

Fixtures build schema with `Base.metadata.create_all`, not Alembic. Migrations are therefore never
exercised by the suite: a model change with no matching migration passes every test and breaks on a
real database. Generate the migration yourself when you touch `models/`.

## Code style

Ruff with `line-length = 100` and `select = ["E", "F", "I", "N", "UP", "B", "SIM"]`.

Tune the config, don't sprinkle `# noqa`. `B008` is already extended for FastAPI's `Depends`/`Query`
/`Body` sentinels because the mutable-default hazard it catches cannot occur there; a rule that
fires on every route gets filtered out by humans and stops working. If a rule is wrong here, it is
wrong repo-wide — say so in `pyproject.toml`.

Comments explain *why*, not *what*. The existing ones in `docker-compose.yml`, `pyproject.toml`, and
the CI workflow are the register to match: each documents a decision or a trap, not a mechanism.

## Architectural decisions

`docs/rag-production-decisions.md` is the source of truth. It is organized as lettered decisions
(`A1`, `E5`, `G5`…) and deliberately separates three states: **decided**, **implemented**, and
**proven**. Do not blur them — "it works" is not "it's measured", and claiming a gate passes when it
has only ever run locally is the specific failure mode that document exists to prevent.

Reference decision ids when work touches one. Changing a decision's state is `/decision`'s job,
which also moves the README status table; the two are expected to change together.

`docs/rag-production-roadmap.md` is partly superseded — trust the decisions doc where they disagree.

## Repository etiquette

- Branches: `feat/…`, `fix/…`, `docs/…`. Never commit to `main` directly.
- Commit subjects state behavior, not activity: `fix: CI resolves its actions and runs the gate`,
  not `fix: update ci.yml`. Lowercase, conventional prefix.
- PR descriptions are architectural reports — the decision, the alternative, the criterion, the cost
  accepted — not diff walkthroughs. Use `/pr`; it holds the full convention.
- Verify before claiming. If something was checked locally but not in CI, say exactly that.
- `CLAUDE.local.md` is gitignored: team tooling is committed, personal instructions are not.

## Gotchas

- **Root `main.py` is a leftover stub** that prints a greeting. The real app is
  `src/production_rag/main.py`, served as `production_rag.main:app`.
- **Postgres credentials are `paddington`**, not `production_rag` — this repo was split out of an
  older `paddington` project. Container names are prefixed `production-rag-` while volumes and the
  database keep the old name. Renaming a container without moving its volume makes the data look
  like it vanished.
- **The embedding dimension is hardcoded** as `Vector(1536)` in `models/document.py` and is not
  derived from `settings.embedding_model`. Changing the embedding model does not change the column,
  and nothing currently catches the mismatch.
- **GitHub Actions pins are not all interchangeable.** `astral-sh/setup-uv` publishes v10 releases
  but only maintains floating major tags through v7, so `@v10` does not resolve and fails the job
  before checkout. Pin its full release tag; verify any action bump actually resolves.
