# tests/test_predict.py
"""
Basic integration tests for the FastAPI application.

These tests verify:
- the health endpoint,
- the predict endpoint,
- that /stats returns the expected structure.
"""

from fastapi.testclient import TestClient

from src.app import app


client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_predict_endpoint():
    payload = {"text": "I love this service!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "label" in data
    assert "score" in data
    assert "probabilities" in data
    assert isinstance(data["probabilities"], dict)


def test_stats_endpoint():
    # Trigger at least one prediction so that stats has data
    client.post("/predict", json={"text": "This is terrible."})

    response = client.get("/stats")
    assert response.status_code == 200

    data = response.json()
    assert "total_requests" in data
    assert "label_counts" in data
    assert "label_distribution" in data
    assert "time_series" in data
    assert isinstance(data["time_series"], list)
