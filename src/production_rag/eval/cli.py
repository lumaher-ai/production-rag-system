"""Building and auditing the eval dataset (decision G1).

    uv run python -m production_rag.eval seed-corpus [--user-email ...]
    uv run python -m production_rag.eval generate [--seed N] [--dry-run] [--limit N]
    uv run python -m production_rag.eval trim [--dry-run]
    uv run python -m production_rag.eval stats
    uv run python -m production_rag.eval baseline [--k 10] [--with-retrieval]
    uv run python -m production_rag.eval audit-sheet [--n 50] [--run-id ...]
    uv run python -m production_rag.eval audit-apply [--sheet PATH] [--reviewer NAME]

Run them in that order. ``generate`` is resumable and idempotent: a unit already
attempted is not re-sent, so an interrupted run costs at most the calls that
were in flight, and a completed run re-invoked makes no API calls at all.
"""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv

# ``.env`` must be loaded before litellm reads provider keys out of the
# environment. ``main.py`` does this at import time for the API; an offline
# command that never imports it has to do it itself or every call fails
# authentication for reasons that look like a code bug.
load_dotenv()

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from production_rag.config import get_settings  # noqa: E402
from production_rag.database import close_db, get_session, init_db  # noqa: E402
from production_rag.eval import audit as audit_module  # noqa: E402
from production_rag.eval import baseline as baseline_module  # noqa: E402
from production_rag.eval.corpus import (  # noqa: E402
    AUDIT_DIR,
    CURATED_PATH,
    REJECTED_PATH,
    RUNS_DIR,
    SILVER_PATH,
    CorpusIndex,
    corpus_files,
    seed_corpus,
)
from production_rag.eval.generate import (  # noqa: E402
    QUOTAS,
    SilverGenerator,
    select_within_quota,
)
from production_rag.eval.metrics import GoldKey  # noqa: E402
from production_rag.eval.sampling import eligible  # noqa: E402
from production_rag.eval.schema import (  # noqa: E402
    RejectedRecord,
    append_jsonl,
    read_jsonl,
    read_unit_ids,
    write_jsonl,
)
from production_rag.llm.client import LLMClient  # noqa: E402
from production_rag.llm.embedding_service import EmbeddingService  # noqa: E402
from production_rag.repositories.document_repository import DocumentRepository  # noqa: E402
from production_rag.repositories.query_cache_repository import (  # noqa: E402
    QueryCacheRepository,
)
from production_rag.repositories.user_repository import (  # noqa: E402
    UserNotFoundError,
    UserRepository,
)
from production_rag.services.document_service import DocumentService  # noqa: E402
from production_rag.services.ingestion_service import IngestionService  # noqa: E402

DEFAULT_SEED = 20260810


async def _resolve_user(session: AsyncSession, email: str) -> UUID:
    """The owner of the eval corpus, or a clear instruction to create one.

    ``UserRepository.get_by_email`` raises rather than returning ``None``
    despite its ``| None`` annotation, so this catches instead of testing —
    checking for ``None`` here would be dead code that silently never fires.
    """
    try:
        user = await UserRepository(session).get_by_email(email)
    except UserNotFoundError:
        raise SystemExit(
            f"No user with email {email!r}.\n\n"
            f"The eval corpus wants an account of its own — retrieval is scoped by "
            f"owner_id, so sharing a user with real documents makes every\n"
            f"retrieval-based number depend on what else that account happens to "
            f"hold. Create one:\n\n"
            f"  uv run python -m production_rag.cli create-admin {email} Eval <password>\n\n"
            f"or point EVAL_USER_EMAIL at an existing account."
        ) from None
    assert user is not None
    return user.id


def _print_table(title: str, rows: list[tuple[str, object]]) -> None:
    print(f"\n{title}")
    print("-" * max(40, len(title)))
    for label, value in rows:
        print(f"{label:<38} {value}")


# ─── seed-corpus ───


async def cmd_seed_corpus(email: str) -> int:
    settings = get_settings()
    files = corpus_files()
    if not files:
        raise SystemExit("eval/corpus/ holds no .md files — nothing to seed.")

    await init_db()
    async with get_session() as session:
        user_id = await _resolve_user(session, email)
        documents = DocumentRepository(session)
        ingestion = IngestionService(
            document_repository=documents,
            embedding_service=EmbeddingService(model=settings.embedding_model),
            query_cache_repository=QueryCacheRepository(session),
            batch_size=settings.ingestion_batch_size,
        )
        print(f"Seeding {len(files)} document(s) as {email}:\n")
        seeded = await seed_corpus(ingestion, documents, user_id)
        index = await CorpusIndex.load(documents, user_id)

    await close_db()
    for item in seeded:
        state = "new" if item.was_new else ("re-embedded" if item.reembedded else "unchanged")
        print(f"  {item.document.title:<34} {item.document.chunk_count:>4} chunks   {state}")
    if not any(item.reembedded for item in seeded):
        print("\n  Nothing was re-embedded — the corpus already matches the current config.")
    print(f"\n{len(index)} chunk(s) available to the generator.")
    return 0


# ─── generate ───


async def cmd_generate(email: str, seed: int, dry_run: bool, limit: int | None) -> int:
    settings = get_settings()
    run_id = f"seed{seed}"

    await init_db()
    async with get_session() as session:
        user_id = await _resolve_user(session, email)
        documents = DocumentRepository(session)
        index = await CorpusIndex.load(documents, user_id)
        if not len(index):
            await close_db()
            raise SystemExit(
                "The eval corpus is empty. Run `seed-corpus` first."
            )

        refs = index.refs()
        usable = eligible(refs)
        if dry_run:
            await close_db()
            by_document: dict[str, int] = {}
            for ref in usable:
                by_document[ref.document_key] = by_document.get(ref.document_key, 0) + 1
            _print_table(
                "Corpus (dry run — no LLM calls made)",
                [
                    ("chunks total", len(refs)),
                    ("chunks eligible for generation", len(usable)),
                    *[(f"  {key}", count) for key, count in sorted(by_document.items())],
                    ("target split", ", ".join(f"{k}={v}" for k, v in QUOTAS.items())),
                    ("seed", seed),
                ],
            )
            print("\nRe-run without --dry-run to generate.")
            return 0

        embeddings = EmbeddingService(model=settings.embedding_model)
        service = DocumentService(
            repository=documents,
            embedding_service=embeddings,
            llm_client=LLMClient(),
            query_cache_repository=QueryCacheRepository(session),
            hnsw_ef_search=settings.hnsw_ef_search,
            hnsw_iterative_scan=settings.hnsw_iterative_scan,
        )
        already = read_unit_ids(SILVER_PATH, REJECTED_PATH)
        if already:
            print(f"{len(already)} generation unit(s) already attempted — skipping those.")

        # The quota belongs to the dataset, not to this run — otherwise a top-up
        # for one short stratum refills all four and the file doubles.
        existing_counts: dict[str, int] = {}
        for record in read_jsonl(SILVER_PATH):
            existing_counts[record.query_type] = existing_counts.get(record.query_type, 0) + 1
        if existing_counts:
            remaining = {
                key: QUOTAS[key] - existing_counts.get(key, 0)
                for key in QUOTAS
                if QUOTAS[key] > existing_counts.get(key, 0)
            }
            print(f"already on disk: {dict(sorted(existing_counts.items()))}")
            if not remaining:
                # Generating here would sample, prompt, gate and then discard
                # every record as over-quota — a full-price run that writes
                # nothing. Cheap to check, and it was not free to learn.
                await close_db()
                print("still wanted:    nothing — every stratum is at quota.")
                print("\nNo generation needed. Use --seed with `trim` if you want to reshuffle.")
                return 0
            print(f"still wanted:    {dict(sorted(remaining.items()))}")

        generator = SilverGenerator(
            llm=LLMClient(),
            index=index,
            settings=settings,
            run_id=run_id,
            seed=seed,
            document_service=service,
            user_id=user_id,
            completed_units=already,
            existing_counts=existing_counts,
        )
        if limit:
            usable = usable[:limit]

        records = await generator.generate(usable)

    await close_db()

    # A qid already on disk is the same question over the same gold chunks, so
    # appending it again would only duplicate a line. This is belt-and-braces
    # behind the unit-level skip above, which is what actually saves the money.
    existing_qids = {record.qid for record in read_jsonl(SILVER_PATH)}
    fresh = [record for record in records if record.qid not in existing_qids]
    for record in fresh:
        append_jsonl(SILVER_PATH, record)
    for rejected in generator.rejected:
        append_jsonl(REJECTED_PATH, rejected)

    stats = generator.stats
    _print_table(
        f"Generation run {run_id}",
        [
            ("records written", len(fresh)),
            ("units skipped (already generated)", stats.units_skipped),
            ("records already on disk", len(records) - len(fresh)),
            ("LLM calls", stats.llm_calls),
            ("measured cost (USD)", f"${stats.cost_usd:.4f}"),
        ],
    )
    _print_table("Kept, by query type", sorted(stats.generated.items()))
    if stats.dropped:
        _print_table("Dropped, by reason", sorted(stats.dropped.items()))

    requested = settings.eval_generator_model
    unexpected = {
        model: count for model, count in stats.served_models.items() if requested not in model
    }
    if unexpected:
        print(
            f"\n⚠️  {sum(unexpected.values())} call(s) were served by a model other than "
            f"{requested!r}: {unexpected}.\n"
            f"    LLMClient falls back on rate limits, so part of this dataset was written "
            f"by a different model.\n"
            f"    Every record names the model that actually served it — check "
            f"`generation.served_model` before describing the dataset."
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{run_id}.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": seed,
                "written": len(fresh),
                "llm_calls": stats.llm_calls,
                "cost_usd": round(stats.cost_usd, 6),
                "generated": stats.generated,
                "dropped": stats.dropped,
                "served_models": stats.served_models,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nsilver:   {SILVER_PATH}")
    print(f"rejected: {REJECTED_PATH}")
    print(f"manifest: {RUNS_DIR / f'{run_id}.json'}")
    return 0


# ─── trim ───


def cmd_trim(dry_run: bool) -> int:
    """Cut the silver file to exactly the G1 quota.

    Needed because generation runs accumulate: each run only knows what it
    produced, so a sequence of partial runs can overshoot a stratum. Trimming is
    a separate, free, deterministic step rather than something generation does
    implicitly — the over-quota records were paid for, and deleting them should
    be a decision somebody makes rather than a side effect they discover.
    """
    records = list(read_jsonl(SILVER_PATH))
    if not records:
        raise SystemExit(f"{SILVER_PATH} is empty — run `generate` first.")

    kept, dropped = select_within_quota(records, QUOTAS)
    counts: dict[str, int] = {}
    for record in kept:
        counts[record.query_type] = counts.get(record.query_type, 0) + 1

    _print_table(
        f"Trim {len(records)} → {len(kept)} record(s)",
        [
            (f"{key} (target {QUOTAS.get(key, '—')})", count)
            for key, count in sorted(counts.items())
        ],
    )
    if dropped:
        over: dict[str, int] = {}
        for record in dropped:
            over[record.query_type] = over.get(record.query_type, 0) + 1
        _print_table("Over quota, would be removed", sorted(over.items()))

    if dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    for record in dropped:
        append_jsonl(
            REJECTED_PATH,
            RejectedRecord(
                drop_reason=f"over_quota:{record.query_type}",
                qid=record.qid,
                unit_id=record.generation.unit_id,
                payload={"question": record.question},
            ),
        )
    write_jsonl(SILVER_PATH, kept)
    print(f"\n{SILVER_PATH} now holds {len(kept)} record(s).")
    print(f"{len(dropped)} over-quota record(s) moved to {REJECTED_PATH}.")
    return 0


# ─── stats ───


def cmd_stats(path: Path) -> int:
    records = list(read_jsonl(path))
    if not records:
        raise SystemExit(f"{path} is empty or missing.")

    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    warnings: dict[str, int] = {}
    for record in records:
        by_type[record.query_type] = by_type.get(record.query_type, 0) + 1
        by_status[record.audit.status] = by_status.get(record.audit.status, 0) + 1
        for warning in record.gates.warnings:
            warnings[warning] = warnings.get(warning, 0) + 1

    _print_table(
        f"{path.name} — {len(records)} record(s)",
        [
            (f"{key} (target {QUOTAS.get(key, '—')})", count)
            for key, count in sorted(by_type.items())
        ],
    )
    _print_table("Audit status", sorted(by_status.items()))
    if warnings:
        _print_table("Gate warnings", sorted(warnings.items()))

    negatives = [record for record in records if not record.answerable]
    bad = [record for record in negatives if record.gold]
    print(f"\nunanswerable records: {len(negatives)}, all with empty gold: {not bad}")
    if bad:
        print(f"  ⚠️  {len(bad)} unanswerable record(s) carry gold chunks — that is a bug.")
    return 1 if bad else 0


# ─── baseline ───


async def cmd_baseline(email: str, k: int, with_retrieval: bool, path: Path, seed: int) -> int:
    settings = get_settings()
    records = [record for record in read_jsonl(path) if record.answerable]
    if not records:
        raise SystemExit(f"{path} holds no answerable records to score.")

    await init_db()
    async with get_session() as session:
        user_id = await _resolve_user(session, email)
        documents = DocumentRepository(session)
        index = await CorpusIndex.load(documents, user_id)
        corpus_keys: list[GoldKey] = index.chunk_keys()

        results = [
            baseline_module.random_chunk_baseline(records, corpus_keys, k, seed),
            baseline_module.first_k_baseline(records, corpus_keys, k, seed),
            baseline_module.largest_document_baseline(records, corpus_keys, k),
        ]
        real = None
        if with_retrieval:
            service = DocumentService(
                repository=documents,
                embedding_service=EmbeddingService(model=settings.embedding_model),
                llm_client=LLMClient(),
                query_cache_repository=QueryCacheRepository(session),
                hnsw_ef_search=settings.hnsw_ef_search,
                hnsw_iterative_scan=settings.hnsw_iterative_scan,
            )
            real = await baseline_module.real_retrieval_baseline(
                records, service, user_id, index.key_for, k
            )
    await close_db()

    print(f"\nRecall@{k} over {len(records)} answerable question(s), {len(corpus_keys)} chunks")
    print("-" * 78)
    print(f"{'baseline':<20} {'expected':>10} {'measured':>10}   description")
    for result in [*results, *( [real] if real else [] )]:
        expected = f"{result.expected:.3f}" if result.expected is not None else "—"
        measured = f"{result.measured:.3f}" if result.measured is not None else "—"
        print(f"{result.name:<20} {expected:>10} {measured:>10}   {result.description}")

    problems = baseline_module.verdict(results, real)
    if problems:
        print("\nFAIL — this dataset is not ready to measure with:")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\nPASS — a trivial baseline does not score on this dataset.")
    if real is None:
        print("      (re-run with --with-retrieval to also check the margin over real retrieval)")
    return 0


# ─── audit ───


async def cmd_audit_sheet(email: str, n: int, run_id: str, seed: int) -> int:
    records = list(read_jsonl(SILVER_PATH))
    if not records:
        raise SystemExit(f"{SILVER_PATH} is empty — run `generate` first.")

    quotas = audit_module.AUDIT_QUOTAS
    if n != sum(quotas.values()):
        scale = n / sum(quotas.values())
        quotas = {key: max(1, round(value * scale)) for key, value in quotas.items()}
    selected = audit_module.select_for_audit(records, seed, quotas)

    await init_db()
    async with get_session() as session:
        user_id = await _resolve_user(session, email)
        index = await CorpusIndex.load(DocumentRepository(session), user_id)
        sheet = audit_module.render_sheet(selected, index, run_id)
    await close_db()

    path = audit_module.sheet_path(run_id, AUDIT_DIR)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sheet, encoding="utf-8")

    counts: dict[str, int] = {}
    for record in selected:
        counts[record.query_type] = counts.get(record.query_type, 0) + 1
    _print_table(f"Review sheet — {len(selected)} item(s)", sorted(counts.items()))
    print("\nEdit the `decision:` line in each verdict block, then:")
    print(f"  uv run python -m production_rag.eval audit-apply --sheet {path}")
    print(f"\n{path}")
    return 0


def cmd_audit_apply(sheet: Path, reviewer: str | None) -> int:
    if not sheet.exists():
        raise SystemExit(f"No sheet at {sheet}. Run `audit-sheet` first.")
    records = list(read_jsonl(SILVER_PATH))
    verdicts = audit_module.parse_sheet(sheet.read_text(encoding="utf-8"))
    if not verdicts:
        raise SystemExit(f"{sheet} contains no verdict blocks.")

    try:
        audit_module.validate_verdicts(verdicts, {record.qid for record in records})
    except audit_module.AuditError as exc:
        print(str(exc))
        return 1

    curated, warnings = audit_module.apply_verdicts(records, verdicts, reviewer=reviewer)
    write_jsonl(CURATED_PATH, curated)

    summary = audit_module.acceptance_summary(curated, verdicts)
    _print_table("Audit applied", sorted(summary.items(), key=lambda pair: pair[0]))
    for warning in warnings:
        print(f"  ⚠️  {warning}")
    print(f"\ncurated: {CURATED_PATH}")
    return 0


# ─── entry point ───


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="production_rag.eval")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_user(subparser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        subparser.add_argument("--user-email", default=settings.eval_user_email)
        return subparser

    with_user(sub.add_parser("seed-corpus", help="ingest eval/corpus/*.md"))

    generate = with_user(sub.add_parser("generate", help="generate the silver dataset"))
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument(
        "--dry-run", action="store_true", help="show the sampling plan without calling any model"
    )
    generate.add_argument("--limit", type=int, default=None, help="cap chunks sampled")

    trim = sub.add_parser("trim", help="cut the silver file to exactly the G1 quota")
    trim.add_argument("--dry-run", action="store_true", help="show what would go, write nothing")

    stats = sub.add_parser("stats", help="stratum counts for a dataset file")
    stats.add_argument("--path", type=Path, default=SILVER_PATH)

    base = with_user(sub.add_parser("baseline", help="prove a trivial retriever does not score"))
    base.add_argument("--k", type=int, default=10)
    base.add_argument("--with-retrieval", action="store_true")
    base.add_argument("--path", type=Path, default=SILVER_PATH)
    base.add_argument("--seed", type=int, default=DEFAULT_SEED)

    sheet = with_user(sub.add_parser("audit-sheet", help="render items for human review"))
    sheet.add_argument("--n", type=int, default=50)
    sheet.add_argument("--run-id", default=f"seed{DEFAULT_SEED}")
    sheet.add_argument("--seed", type=int, default=DEFAULT_SEED)

    apply_cmd = sub.add_parser("audit-apply", help="merge verdicts into the curated dataset")
    apply_cmd.add_argument(
        "--sheet", type=Path, default=AUDIT_DIR / f"audit-seed{DEFAULT_SEED}.md"
    )
    apply_cmd.add_argument("--reviewer", default=None)

    args = parser.parse_args()

    if args.command == "seed-corpus":
        raise SystemExit(asyncio.run(cmd_seed_corpus(args.user_email)))
    if args.command == "generate":
        raise SystemExit(
            asyncio.run(cmd_generate(args.user_email, args.seed, args.dry_run, args.limit))
        )
    if args.command == "trim":
        raise SystemExit(cmd_trim(args.dry_run))
    if args.command == "stats":
        raise SystemExit(cmd_stats(args.path))
    if args.command == "baseline":
        raise SystemExit(
            asyncio.run(
                cmd_baseline(args.user_email, args.k, args.with_retrieval, args.path, args.seed)
            )
        )
    if args.command == "audit-sheet":
        raise SystemExit(
            asyncio.run(cmd_audit_sheet(args.user_email, args.n, args.run_id, args.seed))
        )
    if args.command == "audit-apply":
        raise SystemExit(cmd_audit_apply(args.sheet, args.reviewer))


if __name__ == "__main__":
    main()
