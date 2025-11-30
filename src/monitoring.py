from datetime import datetime, timedelta
from typing import List, Dict, Literal

# Allowed sentiment labels
SentimentLabel = Literal["positive", "neutral", "negative"]

# Simple in-memory log for stats aggregation
_LOG: List[Dict] = []


def log_prediction(label: SentimentLabel, score: float) -> None:
    """Store prediction event in memory."""
    _LOG.append({
        "timestamp": datetime.now(),
        "label": label,
        "score": score
    })


def get_stats(window_minutes: int = 60) -> Dict:
    """Aggregate sentiment stats for the last N minutes."""
    now = datetime.now()
    cutoff = now - timedelta(minutes=window_minutes)

    recent = [r for r in _LOG if r["timestamp"] >= cutoff]

    # Count totals by label
    totals = {"positive": 0, "neutral": 0, "negative": 0}
    for r in recent:
        totals[r["label"]] += 1

    # Aggregate by minute for time series
    buckets = {}
    for r in recent:
        t = r["timestamp"].replace(second=0, microsecond=0).isoformat() + "Z"
        buckets.setdefault(t, {"positive": 0, "neutral": 0, "negative": 0})
        buckets[t][r["label"]] += 1

    # Convert dict → sorted list
    series = [{"time": t, **vals} for t, vals in sorted(buckets.items())]

    return {
        "generated_at": now.isoformat() + "Z",
        "window_minutes": window_minutes,
        "totals": totals,
        "series": series
    }