from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from production_rag.llm.embedding_service import EmbeddingService


def _mock_embedding_service() -> EmbeddingService:
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_text.side_effect = lambda text: [0.1] * 1536
    mock.embed_batch.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    return mock


def test_agent_run_requires_auth(client: TestClient) -> None:
    response = client.post(
        "/agent/run",
        json={"message": "What documents do I have?"},
    )
    assert response.status_code == 401


def test_agent_run_validates_message(client: TestClient) -> None:
    # Signup + login
    client.post(
        "/auth/signup",
        json={"name": "Agent User", "email": "agent@example.com", "password": "securepass123"},
    )
    login = client.post(
        "/auth/login",
        json={"email": "agent@example.com", "password": "securepass123"},
    )
    token = login.json()["access_token"]

    # Empty message should fail validation
    response = client.post(
        "/agent/run",
        json={"message": ""},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422
