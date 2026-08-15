---
name: verify
description: Run the full local check for production-rag — ruff lint, mypy on src, and the pytest suite — and report a single pass/fail summary with the real failures. Use before opening a PR, after a non-trivial change, or whenever asked to "verify", "check", or "make sure this passes".
---

Run all three checks. Do not stop at the first failure — the point is one complete picture.

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

Notes that change how you read the output:

- **Docker splits the suite.** ~222 tests run on in-memory SQLite and need nothing; ~84 use the
  opt-in `pg_engine` fixture (testcontainers + pgvector). With Docker down, those 84 report as
  `ERROR ... docker.errors.DockerException`, not as failures. Say "84 blocked on Docker" rather than
  reporting a suite failure — and never count them as regressions.
- `mypy` is scoped to `src` only. Type errors in `tests/` are not part of this gate.
- ruff is configured with `line-length = 100` and `select = ["E", "F", "I", "N", "UP", "B", "SIM"]`.
  `I` findings are import ordering and are auto-fixable with `uv run ruff check --fix .`.

## Reporting

Give one summary block:

```
ruff    ✅ clean  |  ❌ N findings
mypy    ✅ clean  |  ❌ N errors
pytest  ✅ N passed  |  ❌ N failed, N passed
```

Then, for anything red, list the actual failures with `file:line` and a one-line diagnosis each.
Do not paste full tracebacks unless a failure's cause isn't obvious from the assertion.

Do not fix anything unless asked. Report, then wait — per the plan-before-implementing preference.
