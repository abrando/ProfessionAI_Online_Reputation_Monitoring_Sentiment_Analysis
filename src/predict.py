# src/predict.py
"""
Prediction logic.

This module defines functions that run sentiment analysis on single
texts or lists of texts. It is independent from FastAPI so it can be
reused from tests.
"""

from typing import Dict, List

from .model import load_sentiment_pipeline
from .monitoring import record_prediction


def predict_single(text: str) -> Dict:
    """Predict sentiment for a single text using the HF pipeline.
    Returns:
        A dictionary with:
        - label: predicted top sentiment (e.g. "positive")
        - score: confidence score for the top label
        - probabilities: dict with all labels and their scores
    """
    sentiment_pipeline = load_sentiment_pipeline()

    # Pipeline returns: [[{"label": "...", "score": ...}, ...]]
    outputs = sentiment_pipeline(text)[0]

    # Normalize labels to lowercase for consistency
    probabilities = {
        item["label"].lower(): float(item["score"]) for item in outputs
    }

    # Select the label with highest score
    top_label = max(probabilities, key=probabilities.get)
    top_score = probabilities[top_label]

    # Record for time-series monitoring and future Grafana dashboards
    record_prediction(label=top_label, score=top_score, text=text)

    return {
        "label": top_label,
        "score": top_score,
        "probabilities": probabilities,
    }


def predict_batch(texts: List[str]) -> List[Dict]:
    """
    Batch prediction over a list of texts.
    """
    return [predict_single(t) for t in texts]
