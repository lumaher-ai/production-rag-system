from collections.abc import AsyncIterator

import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from production_rag.database import Base

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def test_create_user_returns_201(client: TestClient) -> None:
    response = client.post(
        "/users",
        json={"name": "Test User", "email": "test@example.com"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test User"
    assert data["email"] == "test@example.com"
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data


def test_create_user_duplicate_email_returns_409(client: TestClient) -> None:
    client.post("/users", json={"name": "First", "email": "dup@example.com"})
    response = client.post(
        "/users",
        json={"name": "Second", "email": "dup@example.com"},
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()


def test_create_user_invalid_email_returns_422(client: TestClient) -> None:
    response = client.post(
        "/users",
        json={"name": "Test", "email": "not-an-email"},
    )
    assert response.status_code == 422


def test_get_user_returns_200(client: TestClient) -> None:
    create_response = client.post(
        "/users",
        json={"name": "Ana", "email": "ana@example.com"},
    )
    user_id = create_response.json()["id"]

    response = client.get(f"/users/{user_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Ana"


def test_get_user_not_found_returns_404(client: TestClient) -> None:
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/users/{fake_id}")
    assert response.status_code == 404


def test_list_users_returns_paginated_response(client: TestClient, admin_token: str) -> None:
    for i in range(3):
        client.post("/users", json={"name": f"User {i}", "email": f"user{i}@example.com"})

    response = client.get(
        "/users?limit=2&offset=0",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["users"]) == 2
    assert data["total"] == 4  # 3 created here + 1 admin from fixture
    assert data["limit"] == 2
    assert data["offset"] == 0


def test_list_users_rejects_invalid_limit(client: TestClient, admin_token: str) -> None:
    response = client.get(
        "/users?limit=500",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 422


def test_update_user_partial_returns_200(client: TestClient) -> None:
    create_response = client.post(
        "/users",
        json={"name": "Old Name", "email": "patch@example.com"},
    )
    user_id = create_response.json()["id"]

    response = client.patch(
        f"/users/{user_id}",
        json={"name": "New Name"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "New Name"
    assert response.json()["email"] == "patch@example.com"  # email no cambió


def test_update_user_not_found_returns_404(client: TestClient) -> None:
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = client.patch(f"/users/{fake_id}", json={"name": "Ghost"})
    assert response.status_code == 404


def test_delete_user_returns_204(client: TestClient, admin_token: str) -> None:
    create_response = client.post(
        "/users",
        json={"name": "To Delete", "email": "delete@example.com"},
    )
    user_id = create_response.json()["id"]

    response = client.delete(
        f"/users/{user_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 204

    get_response = client.get(f"/users/{user_id}")
    assert get_response.status_code == 404


def test_delete_user_not_found_returns_404(client: TestClient, admin_token: str) -> None:
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = client.delete(
        f"/users/{fake_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 404
