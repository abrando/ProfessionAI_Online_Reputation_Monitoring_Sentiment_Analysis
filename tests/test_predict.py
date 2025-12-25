# tests/test_predict.py
"""
Basic integration tests for the FastAPI application.

These tests verify:
- the health endpoint,
- the predict endpoint,
- that /stats returns the expected monitoring structure for Grafana.
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

    # Core monitoring outputs
    assert "sentiment_counts" in data
    assert isinstance(data["sentiment_counts"], dict)

    # Expected keys for counters
    counts = data["sentiment_counts"]
    for k in ("positive", "neutral", "negative", "total"):
        assert k in counts
        assert isinstance(counts[k], int)

    # Moment trend (net counts over a rolling window)
    assert "sentiment_trend_moment" in data
    assert isinstance(data["sentiment_trend_moment"], list)

    # If we have at least one prediction, trend should not be empty
    assert len(data["sentiment_trend_moment"]) >= 1
    last_point = data["sentiment_trend_moment"][-1]
    for k in ("timestamp_utc", "positive_count", "neutral_count", "negative_count", "window_size"):
        assert k in last_point

    assert isinstance(last_point["positive_count"], int)
    assert isinstance(last_point["neutral_count"], int)
    assert isinstance(last_point["negative_count"], int)
    assert isinstance(last_point["window_size"], int)

    # model_eval is optional (file may not exist in fresh env), but if present must be structured
    assert "model_eval" in data
    assert isinstance(data["model_eval"], dict)
    assert "latest" in data["model_eval"]
    assert "series" in data["model_eval"]
    assert isinstance(data["model_eval"]["series"], list)
