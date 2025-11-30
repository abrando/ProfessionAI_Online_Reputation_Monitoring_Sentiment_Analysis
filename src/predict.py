from typing import List, Dict
from .model import load_sentiment_pipeline

# Mapping model-specific labels to normalized labels
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "Negative": "negative",
    "Neutral": "neutral",
    "Positive": "positive",
}

clf = None  # Lazy-loaded pipeline


def get_pipeline():
    """Load sentiment pipeline only once."""
    global clf
    if clf is None:
        clf = load_sentiment_pipeline()
    return clf


def normalize_label(label: str) -> str:
    """Normalize labels returned by HuggingFace."""
    return LABEL_MAP.get(label, label.lower())


def predict_sentiment(texts: List[str]) -> List[Dict]:
    """Run prediction and return normalized outputs."""
    pipe = get_pipeline()
    outputs = pipe(texts)

    results = []
    for out in outputs:
        best = max(out, key=lambda x: x["score"]) if isinstance(out, list) else out
        results.append({
            "label": normalize_label(best["label"]),
            "score": float(best["score"])
        })
    return results