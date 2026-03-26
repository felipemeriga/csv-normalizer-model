from csv_normalizer.evaluate import compute_metrics


def test_perfect_scores():
    predictions = [
        {"date": "2024-03-15", "merchant": "Uber", "amount": -25.90, "category": "Transport"},
        {"date": "2024-03-20", "merchant": "Riachuelo", "amount": -159.90, "category": "Shopping"},
    ]
    ground_truth = [
        {"date": "2024-03-15", "merchant": "Uber", "amount": -25.90, "category": "Transport"},
        {"date": "2024-03-20", "merchant": "Riachuelo", "amount": -159.90, "category": "Shopping"},
    ]
    metrics = compute_metrics(predictions, ground_truth)
    assert metrics["date_accuracy"] == 1.0
    assert metrics["merchant_accuracy"] == 1.0
    assert metrics["amount_accuracy"] == 1.0
    assert metrics["category_accuracy"] == 1.0


def test_partial_scores():
    predictions = [
        {"date": "2024-03-15", "merchant": "uber", "amount": -25.90, "category": "Transport"},
        {"date": "2024-03-21", "merchant": "Riachuelo", "amount": -159.90, "category": "Food"},
    ]
    ground_truth = [
        {"date": "2024-03-15", "merchant": "Uber", "amount": -25.90, "category": "Transport"},
        {"date": "2024-03-20", "merchant": "Riachuelo", "amount": -159.90, "category": "Shopping"},
    ]
    metrics = compute_metrics(predictions, ground_truth)
    assert metrics["date_accuracy"] == 0.5
    assert metrics["merchant_accuracy"] == 1.0
    assert metrics["amount_accuracy"] == 1.0
    assert metrics["category_accuracy"] == 0.5


def test_category_confusion_matrix():
    predictions = [
        {"date": "2024-03-15", "merchant": "X", "amount": -1, "category": "Food"},
        {"date": "2024-03-15", "merchant": "X", "amount": -1, "category": "Transport"},
    ]
    ground_truth = [
        {"date": "2024-03-15", "merchant": "X", "amount": -1, "category": "Transport"},
        {"date": "2024-03-15", "merchant": "X", "amount": -1, "category": "Transport"},
    ]
    metrics = compute_metrics(predictions, ground_truth)
    cm = metrics["category_confusion"]
    assert cm[("Transport", "Food")] == 1
    assert cm[("Transport", "Transport")] == 1


def test_empty_inputs():
    metrics = compute_metrics([], [])
    assert metrics["date_accuracy"] == 0.0
    assert metrics["total"] == 0
