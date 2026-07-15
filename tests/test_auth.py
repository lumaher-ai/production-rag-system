from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession


def test_signup_returns_201(client: TestClient) -> None:
    response = client.post(
        "/auth/signup",
        json={
            "name": "Auth User",
            "email": "auth@example.com",
            "password": "securepass123",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "auth@example.com"
    assert "hashed_password" not in data  # Never expose password hash


def test_signup_short_password_returns_422(client: TestClient) -> None:
    response = client.post(
        "/auth/signup",
        json={"name": "Test", "email": "short@example.com", "password": "123"},
    )
    assert response.status_code == 422


def test_login_returns_token(client: TestClient) -> None:
    # First signup
    client.post(
        "/auth/signup",
        json={
            "name": "Login User",
            "email": "login@example.com",
            "password": "securepass123",
        },
    )

    # Then login
    response = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "securepass123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password_returns_401(client: TestClient) -> None:
    client.post(
        "/auth/signup",
        json={
            "name": "Wrong Pass",
            "email": "wrong@example.com",
            "password": "securepass123",
        },
    )
    response = client.post(
        "/auth/login",
        json={"email": "wrong@example.com", "password": "wrongpassword"},
    )
    assert response.status_code == 401


def test_get_me_returns_current_user(client: TestClient) -> None:
    # Signup + login
    client.post(
        "/auth/signup",
        json={
            "name": "Me User",
            "email": "me@example.com",
            "password": "securepass123",
        },
    )
    login_response = client.post(
        "/auth/login",
        json={"email": "me@example.com", "password": "securepass123"},
    )
    token = login_response.json()["access_token"]

    # Access protected endpoint
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"


def test_get_me_without_token_returns_401(client: TestClient) -> None:
    response = client.get("/users/me")
    assert response.status_code == 401


def test_get_me_with_invalid_token_returns_401(client: TestClient) -> None:
    response = client.get(
        "/users/me",
        headers={"Authorization": "Bearer invalid.token.here"},
    )
    assert response.status_code == 401


def test_login_returns_refresh_token(client: TestClient) -> None:
    client.post(
        "/auth/signup",
        json={"name": "Refresh User", "email": "refresh@example.com", "password": "securepass123"},
    )
    response = client.post(
        "/auth/login",
        json={"email": "refresh@example.com", "password": "securepass123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_refresh_returns_new_tokens(client: TestClient) -> None:
    client.post(
        "/auth/signup",
        json={
            "name": "Refresh Test",
            "email": "refreshtest@example.com",
            "password": "securepass123",
        },
    )
    login_response = client.post(
        "/auth/login",
        json={"email": "refreshtest@example.com", "password": "securepass123"},
    )
    old_refresh = login_response.json()["refresh_token"]
    old_access = login_response.json()["access_token"]

    refresh_response = client.post(
        "/auth/refresh",
        json={"refresh_token": old_refresh},
    )
    assert refresh_response.status_code == 200
    new_data = refresh_response.json()
    assert new_data["access_token"] != old_access
    assert new_data["refresh_token"] != old_refresh


def test_refresh_token_cannot_be_reused(client: TestClient) -> None:
    client.post(
        "/auth/signup",
        json={"name": "Reuse Test", "email": "reuse@example.com", "password": "securepass123"},
    )
    login_response = client.post(
        "/auth/login",
        json={"email": "reuse@example.com", "password": "securepass123"},
    )
    refresh_token = login_response.json()["refresh_token"]

    # First use works
    client.post("/auth/refresh", json={"refresh_token": refresh_token})

    # Second use fails — token was revoked after first use
    response = client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert response.status_code == 401


def test_refresh_with_invalid_token_returns_401(client: TestClient) -> None:
    response = client.post(
        "/auth/refresh",
        json={"refresh_token": "totally-fake-token"},
    )
    assert response.status_code == 401


def test_delete_user_as_regular_user_returns_403(client: TestClient) -> None:
    # Create a regular user
    signup_response = client.post(
        "/auth/signup",
        json={"name": "Regular", "email": "regular@example.com", "password": "securepass123"},
    )
    target_id = signup_response.json()["id"]

    # Login as that regular user
    login_response = client.post(
        "/auth/login",
        json={"email": "regular@example.com", "password": "securepass123"},
    )
    token = login_response.json()["access_token"]

    # Try to delete — should be forbidden
    response = client.delete(
        f"/users/{target_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_delete_user_as_admin_returns_204(client: TestClient, test_session: AsyncSession) -> None:
    import asyncio

    from production_rag.models import User
    from production_rag.models.enums import UserRole
    from production_rag.services.auth_service import hash_password

    # Create admin directly in DB
    admin = User(
        name="Admin",
        email="admin@example.com",
        hashed_password=hash_password("adminpass123"),
        role=UserRole.ADMIN.value,
    )

    async def setup():
        test_session.add(admin)
        await test_session.flush()

    asyncio.get_event_loop().run_until_complete(setup())

    # Create a regular user to delete
    signup_response = client.post(
        "/auth/signup",
        json={"name": "To Delete", "email": "todelete@example.com", "password": "securepass123"},
    )
    target_id = signup_response.json()["id"]

    # Login as admin
    login_response = client.post(
        "/auth/login",
        json={"email": "admin@example.com", "password": "adminpass123"},
    )
    token = login_response.json()["access_token"]

    # Delete as admin — should work
    response = client.delete(
        f"/users/{target_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 204
