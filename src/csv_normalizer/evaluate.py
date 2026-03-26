"""Evaluation script for the normalization model.

Compares model predictions against ground truth and computes
per-field accuracy metrics and category confusion matrix.

Usage: python -m csv_normalizer.evaluate predictions.json ground_truth.json
"""

import json
import sys
from collections import Counter


def compute_metrics(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Compute per-field accuracy metrics."""
    total = len(ground_truth)
    if total == 0:
        return {
            "total": 0,
            "date_accuracy": 0.0,
            "merchant_accuracy": 0.0,
            "amount_accuracy": 0.0,
            "category_accuracy": 0.0,
            "category_confusion": {},
        }

    date_correct = 0
    merchant_correct = 0
    amount_correct = 0
    category_correct = 0
    confusion: Counter = Counter()

    for pred, truth in zip(predictions, ground_truth):
        if pred["date"] == truth["date"]:
            date_correct += 1
        if pred["merchant"].lower() == truth["merchant"].lower():
            merchant_correct += 1
        if abs(pred["amount"] - truth["amount"]) < 0.01:
            amount_correct += 1
        if pred["category"] == truth["category"]:
            category_correct += 1
        confusion[(truth["category"], pred["category"])] += 1

    return {
        "total": total,
        "date_accuracy": date_correct / total,
        "merchant_accuracy": merchant_correct / total,
        "amount_accuracy": amount_correct / total,
        "category_accuracy": category_correct / total,
        "category_confusion": dict(confusion),
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\nEvaluation Results ({metrics['total']} transactions)")
    print(f"  Date accuracy:     {metrics['date_accuracy']:.1%}")
    print(f"  Merchant accuracy: {metrics['merchant_accuracy']:.1%}")
    print(f"  Amount accuracy:   {metrics['amount_accuracy']:.1%}")
    print(f"  Category accuracy: {metrics['category_accuracy']:.1%}")

    if metrics["category_confusion"]:
        print("\nCategory Confusion Matrix (truth -> predicted):")
        for (truth, pred), count in sorted(metrics["category_confusion"].items()):
            if truth != pred:
                print(f"  {truth} -> {pred}: {count}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m csv_normalizer.evaluate <predictions.json> <ground_truth.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        preds = json.load(f)
    with open(sys.argv[2]) as f:
        truth = json.load(f)

    metrics = compute_metrics(preds, truth)
    print_metrics(metrics)
