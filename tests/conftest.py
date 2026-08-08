from collections.abc import AsyncIterator, Generator
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from production_rag.database import Base
from production_rag.dependencies import get_db_session, get_job_queue
from production_rag.main import app

# ─── Job queue fixture (no Redis needed) ───


class RecordingJobQueue:
    """Stands in for the arq queue, recording enqueued job ids.

    The suite asserts that the endpoint *hands work off* — actually running it
    is the worker's job and is tested by driving `ingest_document` directly. That
    split is what keeps the tests fast and Redis-free while still covering both
    halves of the handoff.
    """

    def __init__(self) -> None:
        self.enqueued: list[UUID] = []

    async def enqueue_ingestion(self, job_id: UUID) -> None:
        self.enqueued.append(job_id)


@pytest.fixture
def job_queue() -> Generator[RecordingJobQueue, None, None]:
    queue = RecordingJobQueue()
    app.dependency_overrides[get_job_queue] = lambda: queue
    yield queue
    app.dependency_overrides.pop(get_job_queue, None)


# ─── SQLite fixture (default, fast, no Docker needed) ───

SQLITE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(SQLITE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def client(test_session: AsyncSession) -> Generator[TestClient, None, None]:
    async def override():
        yield test_session

    app.dependency_overrides[get_db_session] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def admin_token(test_session: AsyncSession) -> str:
    from production_rag.models import User
    from production_rag.models.enums import UserRole
    from production_rag.services.auth_service import create_access_token, hash_password

    admin = User(
        name="Admin",
        email="admin@example.com",
        hashed_password=hash_password("adminpass123"),
        role=UserRole.ADMIN.value,
    )
    test_session.add(admin)
    await test_session.flush()
    return create_access_token(user_id=admin.id, email=admin.email, role=admin.role)


# ─── Postgres + pgvector fixture (opt-in, needs Docker) ───


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def pg_engine():
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(
        image="pgvector/pgvector:pg16",
        username="test",
        password="test",
        dbname="test_production_rag",
    ) as postgres:
        url = postgres.get_connection_url()
        # Convert psycopg2 URL to asyncpg URL
        async_url = url.replace("psycopg2", "asyncpg")

        engine = create_async_engine(async_url)

        # Enable pgvector extension and create tables
        async with engine.begin() as conn:
            await conn.execute(
                __import__("sqlalchemy").text("CREATE EXTENSION IF NOT EXISTS vector;")
            )
            await conn.run_sync(Base.metadata.create_all)

        yield engine

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def pg_session(pg_engine) -> AsyncIterator[AsyncSession]:
    session_factory = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()  # Clean up after each test


@pytest.fixture
def pg_client(pg_session: AsyncSession) -> Generator[TestClient, None, None]:
    async def override():
        yield pg_session

    app.dependency_overrides[get_db_session] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(loop_scope="module")
async def pg_async_client(pg_session: AsyncSession) -> AsyncIterator[AsyncClient]:
    async def override():
        yield pg_session

    app.dependency_overrides[get_db_session] = override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
