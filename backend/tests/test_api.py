from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Backend is running"}

@patch("app.api.chat.get_agent_executor")
def test_chat_api_success(mock_get_agent):
    # Setup mock agent executor
    mock_executor = MagicMock()
    mock_executor.invoke.return_value = {"output": "This is a mocked agent response"}
    mock_get_agent.return_value = mock_executor

    # Act
    payload = {
        "message": "Why are people unhappy?",
        "history": []
    }
    response = client.post("/api/chat", json=payload)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "This is a mocked agent response"
    assert data["metadata"]["toolUsed"] == "LangChain ReAct Agent"
    
    # Verify the mock was called correctly
    mock_executor.invoke.assert_called_once()
    called_args = mock_executor.invoke.call_args[0][0]
    assert called_args["input"] == "Why are people unhappy?"

@patch("app.api.chat.get_agent_executor")
def test_chat_api_missing_keys(mock_get_agent):
    # Setup mock to simulate missing keys (returns None)
    mock_get_agent.return_value = None

    # Act
    payload = {
        "message": "Hello",
        "history": []
    }
    response = client.post("/api/chat", json=payload)

    # Assert
    assert response.status_code == 500
    assert "Missing API keys" in response.json()["detail"]
