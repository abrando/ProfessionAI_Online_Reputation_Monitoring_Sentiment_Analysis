# src/retrain.py
"""
Retraining plan.

This module does not actually fine-tune the model yet.
Instead, it:
- checks for new labeled data in data/new/*.csv;
- reports how many new samples are available;
- prints the planned steps for future retraining.

This is enough to show how retraining would be integrated
in an MLOps pipeline once labeled data is available.
"""

from pathlib import Path
from typing import List

from .data import list_new_data_files, load_labeled_data


def print_retraining_plan(files: List[Path]) -> None:
    """Print a simple summary of available new data and planned steps."""
    total_examples = 0
    for path in files:
        texts, labels = load_labeled_data(path)
        n = len(texts)
        total_examples += n
        print(f"Found {n} labeled examples in {path}")

    if total_examples == 0:
        print("New files were found, but they contain no labeled examples.")
        return

    print(f"\nTotal new labeled examples: {total_examples}\n")
    print(
        "Planned retraining steps (not executed yet):\n"
        "  1. Merge all new CSV files into a single dataset.\n"
        "  2. Split the dataset into train and validation sets.\n"
        "  3. Fine-tune the Hugging Face model on this labeled data.\n"
        "  4. Evaluate the new model and compare it to the current one.\n"
        "  5. If performance improves, promote the new model to production.\n"
    )


def main() -> None:
    """Entry point for scheduled retraining checks (GitHub Actions)."""
    files = list_new_data_files()

    if not files:
        print("✅ No new labeled data found in data/new. Skipping retraining.")
        return

    print(f"📥 Detected {len(files)} new data file(s) for retraining.")
    print_retraining_plan(files)


if __name__ == "__main__":
    main()
