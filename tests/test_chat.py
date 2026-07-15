from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from production_rag.dependencies import get_llm_client
from production_rag.llm.client import LLMClient, LLMResponse
from production_rag.main import app


def test_chat_returns_response(client: TestClient) -> None:
    # Create a mock LLM client
    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.chat.return_value = LLMResponse(
        content="Paris is the capital of France.",
        model="gpt-4o-mini",
        input_tokens=15,
        output_tokens=8,
        total_tokens=23,
        cost_usd=0.000007,
        latency_ms=250.0,
        provider="openai",
    )

    # Override the dependency
    app.dependency_overrides[get_llm_client] = lambda: mock_llm

    # First signup and login to get a token
    client.post(
        "/auth/signup",
        json={"name": "Chat User", "email": "chat@example.com", "password": "securepass123"},
    )
    login_response = client.post(
        "/auth/login",
        json={"email": "chat@example.com", "password": "securepass123"},
    )
    token = login_response.json()["access_token"]

    # Call the chat endpoint
    response = client.post(
        "/chat",
        json={"message": "What is the capital of France?"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Paris is the capital of France."
    assert data["model"] == "gpt-4o-mini"
    assert data["cost_usd"] == 0.000007
    assert data["input_tokens"] == 15

    # Verify the mock was called correctly
    mock_llm.chat.assert_called_once()

    # Clean up
    app.dependency_overrides.pop(get_llm_client, None)


def test_chat_requires_auth(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "Hello"},
    )
    assert response.status_code == 401
