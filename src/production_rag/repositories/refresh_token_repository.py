import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.config import get_settings
from production_rag.exceptions import PaddingtonError
from production_rag.models.refresh_token import RefreshToken


class InvalidRefreshTokenError(PaddingtonError):
    status_code = 401


class RefreshTokenRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, user_id: UUID) -> RefreshToken:
        settings = get_settings()
        token = RefreshToken(
            token=secrets.token_urlsafe(64),
            user_id=user_id,
            expires_at=datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days),
        )
        self._session.add(token)
        await self._session.flush()
        return token

    async def get_valid_token(self, token_str: str) -> RefreshToken:
        result = await self._session.execute(
            select(RefreshToken).where(
                RefreshToken.token == token_str,
                RefreshToken.revoked == False,  # noqa: E712
                RefreshToken.expires_at > datetime.now(UTC),
            )
        )
        token = result.scalar_one_or_none()
        if token is None:
            raise InvalidRefreshTokenError("Invalid, expired, or revoked refresh token")
        return token

    async def revoke(self, token: RefreshToken) -> None:
        token.revoked = True
        await self._session.flush()

    async def revoke_all_for_user(self, user_id: UUID) -> None:
        result = await self._session.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == user_id,
                RefreshToken.revoked == False,  # noqa: E712
            )
        )
        tokens = result.scalars().all()
        for token in tokens:
            token.revoked = True
        await self._session.flush()
