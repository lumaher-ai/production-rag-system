"""Operational commands.

uv run python -m production_rag.cli create-admin <email> <name> <password>
uv run python -m production_rag.cli reindex [--dry-run] [--limit N]
"""

import argparse
import asyncio

from production_rag.config import get_settings
from production_rag.database import close_db, get_session, init_db
from production_rag.ingestion.normalize import NORMALIZER_VERSION
from production_rag.models.enums import UserRole
from production_rag.queue import ArqJobQueue, create_queue_pool
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.ingestion_job_repository import (
    IngestionJobRepository,
)
from production_rag.repositories.user_repository import UserRepository
from production_rag.services.auth_service import hash_password
from production_rag.services.document_service import CHUNKER_VERSION


async def create_admin(email: str, name: str, password: str) -> None:
    await init_db()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.create(
            name=name,
            email=email,
            hashed_password=hash_password(password),
        )
        await repo.update_role(user.id, UserRole.ADMIN.value)

    await close_db()
    print(f"Admin user created: {email}")


async def reindex(dry_run: bool = False, limit: int = 500) -> int:
    """Re-process documents whose stored vectors predate the current config.

    A document's ``normalizer_version`` / ``chunker_version`` / ``embedding_model``
    record the silent inputs its vectors were computed under. When any of them
    moves, existing rows keep working but are no longer consistent with how the
    same text would be processed today — and nothing surfaces that. This command
    is what makes such a change *actionable* instead of merely detectable.

    It enqueues one ingestion job per stale document rather than re-implementing
    ingestion: the worker's fetch, batching, resume, and status reporting all
    apply unchanged.

    Note the provenance caveat: documents whose source is fetchable
    (``https://``, ``gdrive://``) are re-fetched and keep their page/section
    values, while ``upload://`` documents are rebuilt from the stored body and
    lose them. The worker logs this per document; re-upload the original file to
    restore structure.
    """
    settings = get_settings()
    await init_db()

    async with get_session() as session:
        documents = DocumentRepository(session)
        stale = await documents.find_stale(
            normalizer_version=NORMALIZER_VERSION,
            chunker_version=CHUNKER_VERSION,
            embedding_model=settings.embedding_model,
            limit=limit,
        )

        if not stale:
            print("Everything is current — nothing to re-index.")
            await close_db()
            return 0

        print(f"{len(stale)} document(s) are stale:\n")
        print(f"{'title':<32} {'normalizer':<16} {'chunker':<20} source")
        print("-" * 100)
        for doc in stale:
            fetchable = not doc.source.startswith("upload://")
            marker = "" if fetchable else "  [rebuilt from stored body — loses page/section]"
            print(
                f"{doc.title[:31]:<32} {doc.normalizer_version:<16} "
                f"{doc.chunker_version:<20} {doc.source[:40]}{marker}"
            )

        print(
            f"\ncurrent: normalizer={NORMALIZER_VERSION} chunker={CHUNKER_VERSION} "
            f"embedding={settings.embedding_model}"
        )

        if dry_run:
            print("\n--dry-run: nothing enqueued.")
            await close_db()
            return len(stale)

        jobs = IngestionJobRepository(session)
        pool = await create_queue_pool(settings.redis_url)
        queue = ArqJobQueue(pool)
        try:
            for doc in stale:
                job = await jobs.create(
                    user_id=doc.user_id,
                    source=doc.source,
                    title=doc.title,
                )
                await queue.enqueue_ingestion(job.id)
                print(f"queued {job.id}  {doc.title}")
        finally:
            await pool.aclose()

    await close_db()
    print(f"\n{len(stale)} job(s) queued. Watch progress: GET /documents/jobs")
    return len(stale)


def main() -> None:
    parser = argparse.ArgumentParser(prog="production_rag.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    admin = sub.add_parser("create-admin", help="create an admin user")
    admin.add_argument("email")
    admin.add_argument("name")
    admin.add_argument("password")

    reindex_cmd = sub.add_parser(
        "reindex", help="re-process documents whose vectors predate the current config"
    )
    reindex_cmd.add_argument(
        "--dry-run", action="store_true", help="list stale documents without queueing"
    )
    reindex_cmd.add_argument("--limit", type=int, default=500)

    args = parser.parse_args()

    if args.command == "create-admin":
        asyncio.run(create_admin(args.email, args.name, args.password))
    elif args.command == "reindex":
        asyncio.run(reindex(dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    main()
