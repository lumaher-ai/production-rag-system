from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.models.query_cache import QueryCache


class QueryCacheRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, idempotency_key: str) -> QueryCache | None:
        """Return a live cache entry for the key, or None if missing/expired."""
        now = datetime.now(UTC)
        result = await self._session.execute(
            select(QueryCache).where(
                QueryCache.idempotency_key == idempotency_key,
                QueryCache.expires_at > now,
            )
        )
        return result.scalar_one_or_none()

    async def put(
        self,
        idempotency_key: str,
        user_id: UUID,
        response_json: dict[str, Any],
        expires_at: datetime,
    ) -> None:
        """Conditionally write a cache entry.

        A concurrent request may have inserted the same key already; the unique
        constraint turns that into an IntegrityError which we swallow (the
        portable equivalent of INSERT ... ON CONFLICT DO NOTHING). A SAVEPOINT
        keeps the outer transaction usable when that happens.
        """
        entry = QueryCache(
            idempotency_key=idempotency_key,
            user_id=user_id,
            response_json=response_json,
            expires_at=expires_at,
        )
        try:
            async with self._session.begin_nested():
                self._session.add(entry)
        except IntegrityError:
            pass  # key already cached by a concurrent request — nothing to do

    async def delete_by_user(self, user_id: UUID) -> None:
        """Invalidate all of a user's cached answers (called on new ingestion)."""
        await self._session.execute(delete(QueryCache).where(QueryCache.user_id == user_id))
