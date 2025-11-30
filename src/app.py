from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal

from .predict import predict_sentiment
from .monitoring import log_prediction, get_stats
from .eval_monitor import run_tweeteval_evaluation, get_eval_runs


# Main FastAPI application
app = FastAPI(
    title="MachineInnovators Sentiment API",
    version="1.0.0",
    description="Sentiment analysis API with monitoring support for Grafana Infinity."
)

# ----- Request/Response Models -----

class SentimentRequest(BaseModel):
    # Input: list of texts to classify
    texts: List[str]


class SentimentResponseItem(BaseModel):
    # Output: single prediction result
    label: Literal["positive", "neutral", "negative"]
    score: float


class SentimentResponse(BaseModel):
    # Output: list of prediction results
    results: List[SentimentResponseItem]


class StatsResponse(BaseModel):
    # Monitoring stats returned to Grafana Infinity
    generated_at: str
    window_minutes: int
    totals: dict
    series: List[dict]

class EvalRun(BaseModel):
    time: str
    split: str
    num_samples: int
    accuracy: float
    f1_macro: float


class EvalStatsResponse(BaseModel):
    runs: List[EvalRun]


# ----- Endpoints -----

@app.get("/health")
def health():
    """Health check endpoint used by monitoring systems."""
    return {"status": "ok"}


@app.post("/predict", response_model=SentimentResponse)
def predict(req: SentimentRequest):
    """Run sentiment prediction on input texts."""
    preds = predict_sentiment(req.texts)

    # Log predictions for monitoring
    for p in preds:
        log_prediction(p["label"], p["score"])

    return SentimentResponse(
        results=[SentimentResponseItem(**p) for p in preds]
    )


@app.get("/stats", response_model=StatsResponse)
def stats(window_minutes: int = 60):
    """Return aggregate sentiment stats for Grafana Infinity."""
    return StatsResponse(**get_stats(window_minutes))


@app.post("/eval/tweeteval", response_model=EvalRun)
def eval_tweeteval(split: str = "validation", max_samples: int = 500):
    """
    Run an evaluation of the current model on TweetEval
    and return the metrics.
    """
    try:
        run = run_tweeteval_evaluation(split=split, max_samples=max_samples)
    except Exception as e:
        # This will show the real error message instead of generic 500
        raise HTTPException(status_code=500, detail=str(e))
    return EvalRun(**run)


@app.get("/eval/stats", response_model=EvalStatsResponse)
def eval_stats():
    """
    Return all stored evaluation runs for monitoring (Grafana Infinity).
    """
    data = get_eval_runs()
    return EvalStatsResponse(runs=[EvalRun(**r) for r in data["runs"]])

