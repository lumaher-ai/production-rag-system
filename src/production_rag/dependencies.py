from collections.abc import AsyncIterator, Callable
from uuid import UUID

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.config import get_settings
from production_rag.database import get_session
from production_rag.exceptions import ForbiddenError
from production_rag.llm.client import LLMClient
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.models import User
from production_rag.models.enums import UserRole
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.repositories.refresh_token_repository import RefreshTokenRepository
from production_rag.repositories.user_repository import UserRepository
from production_rag.services.auth_service import AuthService, InvalidTokenError, decode_access_token
from production_rag.services.document_service import DocumentService
from production_rag.services.user_service import UserService

security_scheme = HTTPBearer()


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with get_session() as session:
        yield session


def get_user_repository(
    session: AsyncSession = Depends(get_db_session),
) -> UserRepository:
    return UserRepository(session)


def get_user_service(
    repository: UserRepository = Depends(get_user_repository),
) -> UserService:
    return UserService(repository)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    service: UserService = Depends(get_user_service),
) -> User:
    payload = decode_access_token(credentials.credentials)

    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise InvalidTokenError("Token missing 'sub' claim")

    try:
        user_id = UUID(user_id_str)
    except ValueError as e:
        raise InvalidTokenError("Token 'sub' is not a valid UUID") from e

    return await service.get_user(user_id)


def require_role(*allowed_roles: UserRole) -> Callable:
    async def role_checker(
        current_user: User = Depends(get_current_user),
    ) -> User:
        if current_user.role not in [r.value for r in allowed_roles]:
            raise ForbiddenError(
                f"Role '{current_user.role}' does not have permission. "
                f"Required: {', '.join(r.value for r in allowed_roles)}"
            )
        return current_user

    return role_checker


def get_refresh_token_repository(
    session: AsyncSession = Depends(get_db_session),
) -> RefreshTokenRepository:
    return RefreshTokenRepository(session)


def get_auth_service(
    session: AsyncSession = Depends(get_db_session),
) -> AuthService:
    from production_rag.services.auth_service import AuthService

    return AuthService(
        user_repository=UserRepository(session),
        refresh_token_repository=RefreshTokenRepository(session),
    )


# It's a lazy singleton because creating the client involves reading the API keys
# and creating the OpenAI/Anthropic HTTP clients. We only want to do that once.

_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        # Model comes from config so it is the single source of truth shared by
        # ingestion, the idempotency key, and the persisted document row.
        _embedding_service = EmbeddingService(model=get_settings().embedding_model)
    return _embedding_service


def get_document_repository(
    session: AsyncSession = Depends(get_db_session),
) -> DocumentRepository:
    return DocumentRepository(session)


def get_query_cache_repository(
    session: AsyncSession = Depends(get_db_session),
) -> QueryCacheRepository:
    return QueryCacheRepository(session)


def get_checkpointer(request: Request) -> BaseCheckpointSaver | None:
    return getattr(request.app.state, "checkpointer", None)


def get_document_service(
    repository: DocumentRepository = Depends(get_document_repository),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    llm_client: LLMClient = Depends(get_llm_client),
    query_cache_repository: QueryCacheRepository = Depends(get_query_cache_repository),
) -> DocumentService:
    return DocumentService(
        repository=repository,
        embedding_service=embedding_service,
        llm_client=llm_client,
        query_cache_repository=query_cache_repository,
        query_cache_ttl_seconds=get_settings().query_cache_ttl_seconds,
    )
