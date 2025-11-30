# src/eval_monitor.py

from datetime import datetime
from typing import List, Dict, Optional

from datasets import load_dataset
import evaluate

from .predict import predict_sentiment

# Map normalized label → numeric id used in TweetEval
LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

# In-memory store for evaluation runs
_EVAL_RUNS: List[Dict] = []


def _predict_label_ids(texts: List[str]) -> List[int]:
    """Use predict_sentiment() and convert labels to TweetEval ids."""
    preds = predict_sentiment(texts)
    ids: List[int] = []
    for p in preds:
        lbl = p["label"]  # "negative"/"neutral"/"positive"
        ids.append(LABEL_TO_ID[lbl])
    return ids


def run_tweeteval_evaluation(
    split: str = "validation",
    max_samples: Optional[int] = 500,
) -> Dict:
    """
    Evaluate the current model on TweetEval (sentiment task) and
    store the result in memory for monitoring.

    Returns a dict like:
    {
      "time": "...Z",
      "split": "validation",
      "num_samples": 500,
      "accuracy": 0.72,
      "f1_macro": 0.68
    }
    """
    # Load TweetEval dataset
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")[split]

    # Optionally subsample for speed
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    texts = dataset["text"]
    labels = dataset["label"]  # list of ints 0/1/2

    batch_size = 32
    all_preds: List[int] = []

    # Predict in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_preds = _predict_label_ids(batch_texts)
        all_preds.extend(batch_preds)

    # Compute metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    acc_result = accuracy.compute(predictions=all_preds, references=labels)
    f1_result = f1.compute(predictions=all_preds, references=labels, average="macro")

    run = {
        "time": datetime.now().isoformat() + "Z",
        "split": split,
        "num_samples": len(labels),
        "accuracy": float(acc_result["accuracy"]),
        "f1_macro": float(f1_result["f1"]),
    }

    # Keep only last 50 runs
    _EVAL_RUNS.append(run)
    if len(_EVAL_RUNS) > 50:
        _EVAL_RUNS.pop(0)

    return run


def get_eval_runs() -> Dict:
    """Return all stored evaluation runs."""
    return {"runs": _EVAL_RUNS}
